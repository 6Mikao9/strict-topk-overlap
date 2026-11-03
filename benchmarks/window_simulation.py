"""Calibrated GPU selector windows; simulated inputs and prediction coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
from types import SimpleNamespace
import torch

from benchmarks.selector_window import synthetic,pair_stages,_padding_kernel
from benchmarks.fusion_benchmark import capture
from benchmarks.paired_confirm import _interval_ms,_replay_interleaved,_warm_device
from benchmarks.run_benchmark import environment,overlap_run
from benchmarks.critical_path import reference,_prediction
from benchmarks.vendor_baseline import PagedDecode
from strict_topk import gqa,gqa_hierarchical
from tests.check_correctness import error_metrics


def calibrate(selector,target_ms,base_ms,grid):
    if base_ms>=target_ms:
        return selector,{'status':'natural selector already at or above target; no padding',
                         'grid':grid,'iterations':0,'target_ms':target_ms,'base_ms':base_ms}
    scratch=torch.empty((grid,128),device='cuda',dtype=torch.float32)
    def make(iterations):
        def padded():
            ids=selector()
            _padding_kernel[(grid,)](scratch,iterations,128,num_warps=4)
            return ids
        return padded
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    trials=[]
    def trial(iters):
        fn=make(iters);graph,out=capture(fn)
        value=statistics.median(_interval_ms(graph,start,end,100)/100 for _ in range(3))
        trials.append({'iterations':iters,'ms':value})
        return value
    low,high=1,64
    while trial(high)<target_ms and high<1048576:high*=2
    for _ in range(8):
        mid=(low+high)//2
        if mid<=0:break
        if trial(mid)<target_ms:low=mid+1
        else:high=mid
    best=min(trials,key=lambda t:abs(t['ms']-target_ms))
    return make(best['iterations']),{'status':'calibrated synthetic GPU arithmetic padding',
        'grid':grid,'iterations':best['iterations'],'target_ms':target_ms,'base_ms':base_ms,'trials':trials}


def confirm(q,k,v,p,selector,backend,paged,seed):
    a=selector();expected=reference(q,k,v,a)
    ss,ps=torch.cuda.Stream(),torch.cuda.Stream()
    def prep():return backend.partials(q,k,v,p)
    def repair(selected,state):return backend.repair_saved(q,k,v,selected,p,state)
    functions={
        'serial_exact':lambda:backend.exact(q,k,v,selector()),
        'saved_overlap':lambda:overlap_run(selector,prep,repair,ss,ps),
        'zero_repair_overlap':lambda:overlap_run(selector,prep,lambda a,s:(a,s),ss,ps),
        'serial_flashinfer':lambda:paged(q,selector()),
    }
    graphs={name:(*capture(fn),fn) for name,fn in functions.items()}
    validation={}
    for name,(graph,out,fn) in graphs.items():
        graph.replay();torch.cuda.synchronize()
        validation[name]=({'passed':torch.equal(out[0],a)} if name=='zero_repair_overlap'
                          else error_metrics(out,expected))
    assert all(x['passed'] for x in validation.values()),validation
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    _warm_device(graphs['serial_exact'][0],graphs['saved_overlap'][0],start,end,1000,150.)
    comparisons=[('serial_exact','saved_overlap'),('serial_flashinfer','saved_overlap'),
                 ('serial_exact','zero_repair_overlap'),('serial_exact','serial_flashinfer')]
    raw={pair:[] for pair in comparisons};rng=random.Random(seed)
    for turn in range(5):
        order=comparisons.copy();rng.shuffle(order)
        for base,candidate in order:
            sequence=[base,candidate,candidate,base] if rng.getrandbits(1) else [candidate,base,base,candidate]
            _replay_interleaved(graphs[base][0],graphs[candidate][0],30)
            intervals=[_interval_ms(graphs[n][0],start,end,100) for n in sequence]
            bt=statistics.mean(t for n,t in zip(sequence,intervals) if n==base)/100
            ct=statistics.mean(t for n,t in zip(sequence,intervals) if n==candidate)/100
            raw[(base,candidate)].append({'round':turn,'sequence':sequence,'interval_ms':intervals,
                                         'baseline_ms':bt,'candidate_ms':ct,'ratio':bt/ct})
    post={}
    for name,(graph,out,fn) in graphs.items():
        graph.replay();torch.cuda.synchronize()
        post[name]=({'passed':torch.equal(out[0],a)} if name=='zero_repair_overlap'
                    else error_metrics(out,expected))
    assert all(x['passed'] for x in post.values()),post
    rows=[]
    for (base,candidate),pairs in raw.items():
        rows.append({'baseline':base,'candidate':candidate,'pairs':pairs,
                     'median_paired_ratio':statistics.median(x['ratio'] for x in pairs),
                     'median_baseline_ms':statistics.median(x['baseline_ms'] for x in pairs),
                     'median_candidate_ms':statistics.median(x['candidate_ms'] for x in pairs),
                     'all_pairs_faster':all(x['ratio']>1 for x in pairs)})
    return {'rows':rows,'validation':validation,'post_timing_validation':post,'passed':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--output',default='results/window/simulation_20261103.json')
    ap.add_argument('--current-only',action='store_true')
    args=ap.parse_args()
    configs=[('current',16384,128)]
    if not args.current_only:configs += [('qsa_shape',8192,512),('qsa_shape',32768,512)]
    files=['benchmarks/window_simulation.py','benchmarks/selector_window.py','strict_topk/gqa_hierarchical.py']
    env=environment()
    env.update(timing='CUDA Graph, warmed randomized adjacent ABBA/BAAB intervals; 100 replays per interval',
               selector='See each configuration: fused Quest bound or independent QSA-formula scorer plus torch.topk',
               limitations=['Synthetic Q/K/V and controlled prediction IDs; no model weights or task-quality run',
                            'Compilation, graph capture, metadata and prediction construction excluded',
                            'No CUPTI resource-traffic measurements; batch interval means are not request-tail samples'])
    data={'environment':env,'seed':args.seed,
          'scope':'Synthetic one-query decode window and prediction-budget simulation; not a model run',
          'model_configuration_reference':'https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json',
          'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
          'limitations':['No FlashNext weights, trained indexer or observed QSA rank trajectories',
             'Current proxy is per-head Quest-like; QSA-shape has shared IDs and independent random indexer Q/K',
             'Padding is synthetic GPU arithmetic at 1 or 64 CTAs, not a model of QSA memory contention',
             'Padding is included identically in serial and overlap paths; natural stages are never shortened',
             'FP64 checks and post-timing replay; zero-repair is incomplete diagnostic only',
             'FlashInfer virtual-request per-query-head adaptation; fixed plan excluded, GPU page-ID update included'],
          'calibrations':[],'experiments':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    def save():path.write_text(json.dumps(data,indent=2)+'\n')
    for kind,context,topk in configs:
        q,k,v,selector,meta=synthetic(kind,context,topk,args.seed)
        backend=(gqa if kind=='current' else SimpleNamespace(exact=gqa_hierarchical.exact,
            partials=gqa_hierarchical.partials,repair_saved=gqa_hierarchical.repair_saved_shared))
        a=selector();paged=PagedDecode(q,k,v,a)
        stages=pair_stages(selector,lambda:backend.exact(q,k,v,a),lambda:backend.exact(q,k,v,selector()))
        windows=[('natural',selector,{'status':'unmodified','iterations':0})]
        for grid in (1,64):
            padded,calibration=calibrate(selector,stages['median_ms']['attention'],stages['median_ms']['selector'],grid)
            if calibration['iterations']:
                verified=pair_stages(padded,lambda:backend.exact(q,k,v,a),lambda:backend.exact(q,k,v,padded()))
                calibration['verified_stages']=verified
                windows.append((f'pad_{grid}_ctas',padded,calibration))
            else:
                data['calibrations'].append({'config':meta,'window':f'pad_{grid}_ctas','calibration':calibration})
        for label,selected_fn,calibration in windows:
            data['calibrations'].append({'config':meta,'window':label,'natural_stages':stages,'calibration':calibration})
            for fraction in (.5,.8,1.):
                budget=max(1,int(topk*fraction))
                for precision in (.8,1.):
                    hits=round(budget*precision)
                    p=_prediction(a,k.shape[1],budget,hits)
                    if kind=='qsa_shape':
                        assert torch.equal(a,a[:1].expand_as(a)) and torch.equal(p,p[:1].expand_as(p))
                    case=confirm(q,k,v,p,selected_fn,backend,paged,args.seed)
                    case['config']={**meta,'window':label,'prediction_budget':budget,'budget_fraction':fraction,
                                    'controlled_precision':hits/budget,'controlled_coverage':hits/topk,
                                    'misses_per_head':topk-hits,'wrong_per_head':budget-hits}
                    data['experiments'].append(case);save()
            print(json.dumps({'kind':kind,'context':context,'window':label,'cases':len(data['experiments'])}),flush=True)
    data['passed']=all(x['passed'] for x in data['experiments']);save()


if __name__=='__main__':main()
