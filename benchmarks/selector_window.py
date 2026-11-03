"""Serial stage ratios and calibrated selector-window decode simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import triton
import triton.language as tl

from strict_topk import gqa
from strict_topk import gqa_hierarchical
from benchmarks.fusion_benchmark import capture
from benchmarks.paired_confirm import _interval_ms, _replay_interleaved, _warm_device, confirm_case
from benchmarks.run_benchmark import environment
from tests.check_correctness import error_metrics
from benchmarks.critical_path import reference


@triton.jit
def _padding_kernel(OUT, iterations, WIDTH:tl.constexpr):
    lane=tl.arange(0,WIDTH)
    value=1.+lane.to(tl.float32)*.001
    for _ in range(iterations):
        value=value*1.000001+0.000001
    tl.store(OUT+tl.program_id(0)*WIDTH+lane,value)


def measure(graphs,rounds=7,replays=200,seed=123):
    rng=random.Random(seed)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    names=list(graphs)
    _warm_device(graphs[names[0]][0],graphs[names[-1]][0],start,end,1000,150.)
    raw=[]
    for turn in range(rounds):
        order=names.copy();rng.shuffle(order)
        row={'round':turn,'order':order,'ms':{}}
        for name in order:
            row['ms'][name]=_interval_ms(graphs[name][0],start,end,replays)/replays
        raw.append(row)
    return {'raw_rounds':raw,'median_ms':{n:statistics.median(r['ms'][n] for r in raw) for n in names},
            'replays_per_interval':replays}


def pair_stages(selector,attention,serial):
    # Keep closures and graph outputs alive for the full capture/measure span.
    funcs={'selector':selector,'attention':attention,'serial':serial}
    graphs={n:(*capture(fn),fn) for n,fn in funcs.items()}
    data=measure(graphs)
    data['selector_over_attention_by_round']=[r['ms']['selector']/r['ms']['attention'] for r in data['raw_rounds']]
    data['median_selector_over_attention']=statistics.median(data['selector_over_attention_by_round'])
    return data


def synthetic(kind,context,topk,seed,reference_indexer=False):
    torch.manual_seed(seed)
    if kind=='current':
        h,hkv,d,b=16,8,128,32
    else:
        h,hkv,d,b=24,2,256,4
    n=context//b
    q=torch.randn((h,d),device='cuda',dtype=torch.float16)
    k=torch.randn((hkv,n,b,d),device='cuda',dtype=torch.float16)
    v=torch.randn_like(k)
    if kind=='current':
        lo,hi=k.amin(2).float(),k.amax(2).float()
        def selector():return gqa.select_blocks(q,lo,hi,topk)
        selection='Current Quest-style per-head min/max bound plus torch.topk'
    else:
        iq=torch.randn((4,128),device='cuda',dtype=torch.float16)
        ik=torch.randn((n,128),device='cuda',dtype=torch.float16)
        # The model configuration and score formula only: no trained weights
        # or SGLang native kernels. One shared selection across all Q heads.
        def selector():
            scores=(torch.relu(iq.float()@ik.float().T).sum(0)/(128**.5)
                    if reference_indexer else qsa_reference_scores(iq,ik))
            ids=scores.topk(topk,sorted=False).indices.to(torch.int32)
            return ids[None].expand(h,-1).contiguous()
        fp64_scores=torch.relu(iq.double()@ik.double().T).sum(0)/(128**.5)
        measured_scores=qsa_reference_scores(iq,ik)
        assert torch.allclose(measured_scores.double(),fp64_scores,atol=2.e-5,rtol=2.e-5)
        selection=('PyTorch reference' if reference_indexer else 'Independent Triton QSA-formula')+' score=sum(ReLU(Q_indexer K_indexer^T))/sqrt(128), shared torch.topk; not native SGLang'
    return q,k,v,selector,{'kind':kind,'context':context,'topk':topk,'heads':h,'kv_heads':hkv,
                         'dim':d,'block_size':b,'selector':selection,'seed':seed}


def run_case(q,k,v,p,selector,metadata=None,**unused):
    backend=gqa_hierarchical if (metadata or {}).get('hierarchical') else gqa
    a=selector()
    oracle=reference(q,k,v,a)
    out=backend.exact(q,k,v,a)
    assert error_metrics(out,oracle)['passed']
    def attention():return backend.exact(q,k,v,a)
    def serial():return backend.exact(q,k,v,selector())
    base=pair_stages(selector,attention,serial)
    return {'config':metadata or {},'serial_probe':base,'passed':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/window/probe.json')
    ap.add_argument('--kind',choices=['current','qsa_shape'],default='current')
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--hierarchical',action='store_true')
    ap.add_argument('--reference-indexer',action='store_true')
    args=ap.parse_args()
    configs=[(16384,64),(32768,64),(16384,128),(16384,256)] if args.kind=='current' else [(8192,512),(32768,512)]
    env=environment()
    env.update(timing='CUDA Graph; shuffled stage order, 200 replays per event interval, seven rounds',
               selector='See per-case selector implementation',
               limitations=['Synthetic Q/K/V, no model run or trained indexer',
                            'Preparation and graph capture excluded; batch means are not request-tail samples'])
    result={'environment':env,'scope':'Synthetic one-query decode stage probe; not a model run',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'experiments':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for context,topk in configs:
        q,k,v,selector,meta=synthetic(args.kind,context,topk,args.seed,args.reference_indexer)
        meta['hierarchical']=args.hierarchical
        case=run_case(q,k,v,None,selector,meta)
        result['experiments'].append(case);path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'config':meta,'ms':case['serial_probe']['median_ms'],'S_over_A':case['serial_probe']['median_selector_over_attention']}),flush=True)
    result['passed']=True;path.write_text(json.dumps(result,indent=2)+'\n')


@triton.jit
def _indexer_scores(Q,K,SCORES,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
                    KEYS:tl.constexpr,DD:tl.constexpr):
    keys=tl.program_id(0)*KEYS+tl.arange(0,KEYS)
    heads,dims=tl.arange(0,H),tl.arange(0,DD)
    q=tl.load(Q+heads[:,None]*D+dims[None,:],dims[None,:]<D,0.).to(tl.float32)
    k=tl.load(K+keys[:,None]*D+dims[None,:],(keys[:,None]<N)&(dims[None,:]<D),0.).to(tl.float32)
    score=tl.sum(k[:,None,:]*q[None,:,:],2)
    score=tl.sum(tl.maximum(score,0.),1)*(D**-.5)
    tl.store(SCORES+keys,score,keys<N)


def qsa_reference_scores(q,k):
    scores=torch.empty(k.shape[0],device=q.device,dtype=torch.float32)
    _indexer_scores[(triton.cdiv(k.shape[0],8),)](q,k,scores,k.shape[0],q.shape[0],q.shape[1],8,
        triton.next_power_of_2(q.shape[1]),num_warps=4)
    return scores


if __name__=='__main__':main()


