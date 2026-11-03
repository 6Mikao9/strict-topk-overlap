"""Paired serial/speculative paths using FlashInfer for all regular attention."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import flashinfer

from strict_topk.flashinfer_backend import FixedAttention,Correction
from benchmarks.selector_window import synthetic
from benchmarks.run_benchmark import environment,overlap_run
from benchmarks.critical_path import reference,_prediction
from benchmarks.fusion_benchmark import capture
from benchmarks.paired_confirm import _interval_ms,_replay_interleaved,_warm_device
from benchmarks.vendor_baseline import PagedDecode
from tests.check_flashinfer_backend import metrics,union_reference


def run_case(q,k,v,a,p,selector,shared,seed):
    serial=FixedAttention(q,k,v,a,shared)
    predict=FixedAttention(q,k,v,p,shared)
    strict=Correction(q,k,v,a,p,shared,True)
    union=Correction(q,k,v,a,p,shared,False)
    legacy=PagedDecode(q,k,v,a)
    ss,ps=torch.cuda.Stream(),torch.cuda.Stream()
    prep=lambda:predict(q,p,True)
    finish=lambda ids,state:strict(q,k,v,ids,p,state)
    finish_union=lambda ids,state:union(q,k,v,ids,p,state)
    def serialized():
        ids=selector();state=prep()
        return finish(ids,state)
    functions={
        'serial_flashinfer':lambda:serial(q,selector()),
        'strict_flashinfer_serialized':serialized,
        'strict_flashinfer_overlap':lambda:overlap_run(selector,prep,finish,ss,ps),
        'union_flashinfer_overlap':lambda:overlap_run(selector,prep,finish_union,ss,ps),
        'legacy_virtual_heads_serial':lambda:legacy(q,selector()),
    }
    expected=reference(q,k,v,a);expected_union=union_reference(q,k,v,a,p)
    graphs={name:(*capture(fn),fn) for name,fn in functions.items()}
    def validate():
        checks={}
        for name,(graph,value,fn) in graphs.items():
            graph.replay();torch.cuda.synchronize()
            output=value.output if hasattr(value,'output') else value
            check=metrics(output,expected_union if name=='union_flashinfer_overlap' else expected,q.dtype)
            if hasattr(value,'fallback'):check['fallback_heads']=value.fallback.sum().item()
            if name=='union_flashinfer_overlap':
                check['relative_l2_vs_strict']=(torch.linalg.vector_norm(output.double()-expected)/
                                               torch.linalg.vector_norm(expected)).item()
            checks[name]=check
        assert all(v['passed'] for v in checks.values()),checks
        return checks
    checks=validate()
    pairs=[('serial_flashinfer','strict_flashinfer_overlap'),
           ('legacy_virtual_heads_serial','strict_flashinfer_overlap'),
           ('strict_flashinfer_serialized','strict_flashinfer_overlap'),
           ('serial_flashinfer','union_flashinfer_overlap'),
           ('legacy_virtual_heads_serial','serial_flashinfer')]
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    _warm_device(graphs[pairs[0][0]][0],graphs[pairs[0][1]][0],start,end,1000,150.)
    rng=random.Random(seed);raw={pair:[] for pair in pairs}
    for turn in range(5):
        order=pairs.copy();rng.shuffle(order)
        for base,candidate in order:
            seq=[base,candidate,candidate,base] if rng.getrandbits(1) else [candidate,base,base,candidate]
            _replay_interleaved(graphs[base][0],graphs[candidate][0],30)
            times=[_interval_ms(graphs[n][0],start,end,100)/100 for n in seq]
            bt=statistics.mean(t for n,t in zip(seq,times) if n==base)
            ct=statistics.mean(t for n,t in zip(seq,times) if n==candidate)
            raw[(base,candidate)].append({'round':turn,'sequence':seq,'ms':times,
                                         'baseline_ms':bt,'candidate_ms':ct,'ratio':bt/ct})
    rows=[{'baseline':b,'candidate':c,'pairs':values,
           'median_baseline_ms':statistics.median(v['baseline_ms'] for v in values),
           'median_candidate_ms':statistics.median(v['candidate_ms'] for v in values),
           'median_paired_ratio':statistics.median(v['ratio'] for v in values),
           'all_pairs_faster':all(v['ratio']>1 for v in values)} for (b,c),values in raw.items()]
    post=validate()
    # Stage diagnostics use independent graphs; these do not determine the
    # complete-path ratio. Keep state and closures alive throughout timing.
    state=prep()
    stage_functions={'selector':selector,'attention':lambda:serial(q,a),
                     'prediction':prep,'repair':lambda:finish(a,state)}
    stages={n:(*capture(fn),fn) for n,fn in stage_functions.items()}
    stage_ms={n:statistics.median(_interval_ms(g,start,end,100)/100 for _ in range(5))
              for n,(g,out,fn) in stages.items()}
    return {'rows':rows,'validation':checks,'post_timing_validation':post,
            'isolated_stage_ms':stage_ms,'passed':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--output',default='results/vendor_shared/benchmark_20261103.json')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    env=environment();env.update(flashinfer=flashinfer.__version__,
        timing='CUDA Graph, randomized adjacent ABBA/BAAB, 5 pairs x 100 replays',
        limitations=['Synthetic Q/K/V and controlled prediction IDs, no native model execution',
                    'Fixed-cardinality serial/prediction planning excluded for both paths',
                    'Repair uses version-pinned single-tile planning; dynamic CSR updates are timed',
                    'Union retains wrong predictions and has different attention semantics',
                    'Legacy-vs-new serial changes both tensor-core setting and grouped layout where shared',
                    'Empty correction slots execute one valid dummy page, excluded from merge',
                    'Strict signed subtraction has an empirical guard and parallel positive Triton fallback',
                    'Two fallback launches are timed even when no head requires recomputation'])
    files=['strict_topk/flashinfer_backend.py','benchmarks/flashinfer_overlap.py',
           'tests/check_flashinfer_backend.py']
    data={'environment':env,'seed':args.seed,'experiments':[],
          'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    configs=[('qsa_shape',8192,512)] if args.smoke else [
        ('qsa_shape',8192,512),('qsa_shape',32768,512),('current',16384,128)]
    for kind,context,topk in configs:
        q,k,v,selector,meta=synthetic(kind,context,topk,args.seed)
        a=selector();shared=kind=='qsa_shape'
        for fraction in ([.8] if args.smoke else [.5,.8,1.]):
          budget=int(topk*fraction)
          for precision in ([.8] if args.smoke else [.8,1.]):
            hits=round(budget*precision);p=_prediction(a,k.shape[1],budget,hits)
            case=run_case(q,k,v,a,p,selector,shared,args.seed)
            case['config']={**meta,'shared_grouped_gqa':shared,'prediction_budget':budget,
                            'precision':hits/budget,'coverage':hits/topk,'window':'natural'}
            data['experiments'].append(case)
            path.write_text(json.dumps(data,indent=2)+'\n')
            print(json.dumps({'config':case['config'],'ratios':[x['median_paired_ratio'] for x in case['rows']]}),flush=True)
    data['passed']=all(x['passed'] for x in data['experiments'])
    path.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
