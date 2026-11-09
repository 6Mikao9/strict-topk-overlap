"""CUDA kernel timelines for library-backed serial and overlapping paths.

Profiler times are diagnostics, not benchmark speedups. No SM/DRAM bandwidth
counters are inferred from these timelines. Requires working CUPTI/Kineto.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import torch

from benchmarks.selector_window import synthetic
from benchmarks.critical_path import _prediction,reference
from benchmarks.fusion_benchmark import capture
from benchmarks.run_benchmark import overlap_run
from strict_topk.flashinfer_backend import FixedAttention,Correction
from tests.check_flashinfer_backend import metrics
from benchmarks.paired_confirm import _interval_ms


def timeline_summary(trace):
    events=trace.get('traceEvents',[])
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X' and e.get('dur',0)>0]
    totals=defaultdict(lambda:{'calls':0,'total_us':0.,'max_us':0.})
    edges=[]
    for e in kernels:
        row=totals[e['name']];row['calls']+=1;row['total_us']+=e['dur'];row['max_us']=max(row['max_us'],e['dur'])
        edges.extend([(e['ts'],1),(e['ts']+e['dur'],-1)])
    # Aggregate kernel-active intervals across streams, not utilization.
    active=0;previous=None;union=concurrent=0.
    for t,delta in sorted(edges):
        if previous is not None:
            if active>0:union+=t-previous
            if active>1:concurrent+=t-previous
        active+=delta;previous=t
    span=(max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)) if kernels else 0.
    launches=[e for e in events if 'GraphLaunch' in e.get('name','') and e.get('cat') in ('cuda_runtime','cuda_driver')]
    synchronizes=[e for e in events if 'DeviceSynchronize' in e.get('name','') and e.get('cat') in ('cuda_runtime','cuda_driver')]
    starts=sorted(e['ts'] for e in launches)
    return {'cuda_kernel_count':len(kernels),'sum_kernel_us':sum(e['dur'] for e in kernels),
            'kernel_active_union_us':union,'concurrent_kernel_interval_us':concurrent,
            'first_to_last_kernel_span_us':span,'no_kernel_interval_us':span-union,
            'reported_trace_lanes':sorted({str(e.get('args',{}).get('stream',e.get('tid'))) for e in kernels}),
            'host_graph_launch_calls':len(launches),'host_graph_launch_api_us':sum(e.get('dur',0) for e in launches),
            'host_graph_launch_start_gaps_us':[b-a for a,b in zip(starts,starts[1:])],
            'host_device_synchronize_us':sum(e.get('dur',0) for e in synchronizes),
            'kernels':dict(sorted(totals.items(),key=lambda item:-item[1]['total_us'])),
            'memcpy_event_count':sum(e.get('cat')=='gpu_memcpy' for e in events)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results/profiling')
    ap.add_argument('--seed',type=int,default=20261109);ap.add_argument('--replays',type=int,default=1)
    ap.add_argument('--context',type=int,default=8192);args=ap.parse_args()
    dest=Path(args.output_dir);dest.mkdir(parents=True,exist_ok=True)
    q,k,v,selector,meta=synthetic('qsa_shape',args.context,512,args.seed)
    a=selector();p=_prediction(a,k.shape[1],409,327)
    serial=FixedAttention(q,k,v,a,True);predict=FixedAttention(q,k,v,p,True)
    correction=Correction(q,k,v,a,p,True,True)
    ss,ps=torch.cuda.Stream(),torch.cuda.Stream()
    prep=lambda:predict(q,p,True)
    finish=lambda ids,state:correction(q,k,v,ids,p,state)
    def serialized():
        ids=selector();return finish(ids,prep())
    state=prep()
    functions={'serial':lambda:serial(q,selector()),'overlap':lambda:overlap_run(selector,prep,finish,ss,ps),
               'serialized_speculation':serialized,'repair_only':lambda:finish(a,state),
               'selector_only':selector,'prediction_only':prep}
    graphs={name:(*capture(fn),fn) for name,fn in functions.items()}
    expected=reference(q,k,v,a)
    result={'scope':'Profiler-instrumented CUDA Graph timelines; not performance speedup measurements',
            'config':meta,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'replays':args.replays,'profiles':{},'limitations':[
                'Graph replay CPU launch overhead and profiling overhead affect timeline spans',
                'Reported CUDA Graph trace lanes may not preserve logical stream identity',
                'All intervals aggregate the requested replays; default one replay isolates one GPU timeline',
                'Kernel-active interval is not SM utilization; no occupancy or hardware bandwidth counters',
                'Synthetic QSA shape and controlled prediction IDs, no real model or host KV transfer'],
            'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                ['benchmarks/profile_overlap.py','strict_topk/flashinfer_backend.py']}}
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    paired=[]
    for _ in range(5):
        seq=['serial','overlap','overlap','serial']
        times=[_interval_ms(graphs[x][0],start,end,100)/100 for x in seq]
        b,c=statistics.mean([times[0],times[3]]),statistics.mean([times[1],times[2]])
        paired.append({'baseline_ms':b,'candidate_ms':c,'ratio':b/c})
    result['unprofiled_full_path_pairs']=paired
    for name,(graph,value,fn) in graphs.items():
        for _ in range(20):graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False,profile_memory=False,with_stack=False) as prof:
            with torch.profiler.record_function('measured_'+name):
                for _ in range(args.replays):graph.replay()
                torch.cuda.synchronize()
        path=dest/(name+'.trace.json');prof.export_chrome_trace(str(path))
        trace=json.loads(path.read_text());summary=timeline_summary(trace)
        output=value.output if hasattr(value,'output') else value
        summary['validation']=({'passed':bool(torch.equal(output,a))} if name=='selector_only' else
                               metrics(output,reference(q,k,v,p) if name=='prediction_only' else expected,q.dtype))
        if hasattr(value,'fallback'):summary['fallback_heads']=value.fallback.sum().item()
        summary['trace_file']=path.name
        result['profiles'][name]=summary
        print(json.dumps({'name':name,'kernels':summary['cuda_kernel_count'],
                          'valid':summary['validation']['passed']}),flush=True)
        if not summary['cuda_kernel_count']:
            result['status']='CUPTI did not yield CUDA kernel events; CPU-only trace is insufficient'
            (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            raise RuntimeError(result['status'])
        assert summary['validation']['passed'],summary['validation']
    result['status']='cuda_timeline_collected'
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
