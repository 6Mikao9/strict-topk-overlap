"""PRR Table 3 interface/shape controls on available hardware, not reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import flashinfer
from flashinfer.sparse import BlockSparseAttentionWrapper

from strict_topk import gqa_hierarchical
from strict_topk.flashinfer_backend import FixedAttention
from benchmarks.critical_path import reference
from benchmarks.fusion_benchmark import capture
from benchmarks.paired_confirm import _interval_ms,_replay_interleaved,_warm_device
from tests.check_flashinfer_backend import metrics


def run_case(base_k,base_v,q,block,budget,seed):
    n,hkv,d=base_k.shape;h=q.shape[0];count=budget//block
    ids=torch.randperm(n//block,device='cuda')[:count].to(torch.int32)
    selected=ids[None].expand(h,-1).contiguous()
    # BSR path uses the sequence-major layout required by its public API.
    bsr=BlockSparseAttentionWrapper(torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8))
    bsr_ids=ids.clone()
    bsr.plan(torch.tensor([0,count],device='cuda',dtype=torch.int32),bsr_ids,
             1,n,1,block,h,hkv,d,q_data_type=q.dtype,kv_data_type=q.dtype)
    k=base_k.permute(1,0,2).contiguous().view(hkv,n//block,block,d)
    v=base_v.permute(1,0,2).contiguous().view_as(k)
    paged=FixedAttention(q,k,v,selected,shared=True)
    def bsr_run():return bsr.run(q[None],base_k,base_v).view_as(q)
    def paged_run():return paged.wrapper.run(q.view(hkv,h//hkv,d),paged.cache).view_as(q)
    functions={'blocksparse':bsr_run,'paged_grouped':paged_run,
               'triton_saved_exact':lambda:gqa_hierarchical.exact(q,k,v,selected)}
    graphs={name:(*capture(fn),fn) for name,fn in functions.items()}
    # Prove that replacement IDs reach a previously captured graph at fixed
    # cardinality. Table refresh is separate from kernel-only timing below.
    changed=(ids+1)%(n//block);selected.copy_(changed[None].expand_as(selected))
    bsr_ids.copy_(changed);paged.write_pages(selected)
    expected=reference(q,k,v,selected)
    mutations={}
    for name,(graph,out,fn) in graphs.items():
        graph.replay();torch.cuda.synchronize();mutations[name]=metrics(out,expected,q.dtype)
    assert all(x['passed'] for x in mutations.values()),mutations
    selected.copy_(ids[None].expand_as(selected));bsr_ids.copy_(ids);paged.write_pages(selected)
    expected=reference(q,k,v,selected)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    _warm_device(graphs['blocksparse'][0],graphs['paged_grouped'][0],start,end,1000,150.)
    comparisons=[('blocksparse','paged_grouped'),('blocksparse','triton_saved_exact')]
    raw={pair:[] for pair in comparisons};rng=random.Random(seed)
    for turn in range(5):
        order=comparisons.copy();rng.shuffle(order)
        for base,candidate in order:
            seq=[base,candidate,candidate,base] if rng.getrandbits(1) else [candidate,base,base,candidate]
            _replay_interleaved(graphs[base][0],graphs[candidate][0],30)
            times=[_interval_ms(graphs[x][0],start,end,100)/100 for x in seq]
            bt=statistics.mean(t for x,t in zip(seq,times) if x==base)
            ct=statistics.mean(t for x,t in zip(seq,times) if x==candidate)
            raw[(base,candidate)].append({'round':turn,'order':seq,'ms':times,
                'baseline_ms':bt,'candidate_ms':ct,'ratio':bt/ct})
    post={}
    for name,(graph,out,fn) in graphs.items():
        graph.replay();torch.cuda.synchronize();post[name]=metrics(out,expected,q.dtype)
    assert all(x['passed'] for x in post.values()),post
    return {'block_size':block,'token_budget':budget,'selected_blocks':count,
        'blocksparse_resolved_backend':str(bsr._backend),'blocksparse_tensor_cores':bool(bsr._use_tensor_cores),
        'mutation_checks':mutations,'post_timing_validation':post,'passed':True,
        'rows':[{'baseline':a,'candidate':b,'pairs':vv,
                 'median_paired_ratio':statistics.median(x['ratio'] for x in vv),
                 'median_baseline_us':1000*statistics.median(x['baseline_ms'] for x in vv),
                 'median_candidate_us':1000*statistics.median(x['candidate_ms'] for x in vv)}
                for (a,b),vv in raw.items()]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261109)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default='results/paper_baseline/benchmark_20261109.json')
    args=ap.parse_args();torch.manual_seed(args.seed)
    n,h,hkv,d=131072,32,8,128
    q=torch.randn((h,d),device='cuda',dtype=torch.float16)
    k=torch.randn((n,hkv,d),device='cuda',dtype=torch.float16);v=torch.randn_like(k)
    files=['benchmarks/paper_baseline.py','strict_topk/gqa_hierarchical.py','strict_topk/flashinfer_backend.py']
    data={'date':'2026-11-09','seed':args.seed,'gpu':torch.cuda.get_device_name(),
          'torch':torch.__version__,'flashinfer':flashinfer.__version__,'context':n,
          'q_heads':h,'kv_heads':hkv,'head_dim':d,'dtype':'float16','experiments':[],
          'scope':'PRR Table 3 nominal shapes and BlockSparseAttention interface on A100; NOT paper H100/FA3 reproduction',
          'limitations':['Paper FlashInfer version, exact table construction and timing boundary are not specified',
              'Shared random block IDs; no model, selector, TP2 or end-to-end decoding experiment',
              'Kernel-only graphs; planning, layout conversion and page-table preparation excluded for all methods',
              'Native API input layouts differ; zero-copy views used inside timing, physical layouts prepared outside',
              'Official PRR Hopper kernel is not executed; triton_saved_exact is our own strict serial implementation',
              'Five adjacent ABBA/BAAB pairs, each segment 100 graph replays; batch means not serving tail latency'],
          'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for b in ([32] if args.smoke else [16,32,64,128,256]):
      for budget in ([2048] if args.smoke else [1024,2048,4096,6144,8192]):
        case=run_case(k,v,q,b,budget,args.seed)
        data['experiments'].append(case);path.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({'block':b,'budget':budget,'ratios':[r['median_paired_ratio'] for r in case['rows']]}),flush=True)
    data['passed']=True;path.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
