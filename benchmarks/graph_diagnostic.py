"""Check graph dependencies and submission order after a nonoverlapping trace.

Diagnostic only: same kernels, inputs, numerical guard and fixed planning.
No Nsight hardware counters or model-level speedup is claimed.
"""
import hashlib
import json
from pathlib import Path
import statistics

import torch

from benchmarks.critical_path import _prediction, reference
from benchmarks.paired_confirm import _interval_ms
from benchmarks.profile_overlap import timeline_summary
from benchmarks.run_benchmark import overlap_run
from benchmarks.selector_window import synthetic
from strict_topk.flashinfer_backend import FixedAttention, Correction
from tests.check_flashinfer_backend import metrics


def main():
    dest=Path('results/graph_diagnostic');dest.mkdir(parents=True,exist_ok=True)
    q,k,v,selector,meta=synthetic('qsa_shape',8192,512,20261109)
    a=selector();p=_prediction(a,k.shape[1],409,327)
    serial=FixedAttention(q,k,v,a,True);predict=FixedAttention(q,k,v,p,True)
    correction=Correction(q,k,v,a,p,True,True)
    ss,ps=torch.cuda.Stream(),torch.cuda.Stream()
    prep=lambda:predict(q,p,True)
    finish=lambda ids,state:correction(q,k,v,ids,p,state)
    def sequential():
        ids=selector();return finish(ids,prep())
    def prediction_first():
        main=torch.cuda.current_stream();ss.wait_stream(main);ps.wait_stream(main)
        with torch.cuda.stream(ps):state=prep()
        with torch.cuda.stream(ss):ids=selector()
        main.wait_stream(ss);main.wait_stream(ps)
        ids.record_stream(main);state.output.record_stream(main);state.lse.record_stream(main)
        return finish(ids,state)
    functions={'serial':lambda:serial(q,selector()),'sequential':sequential,
               'selection_first':lambda:overlap_run(selector,prep,finish,ss,ps),
               'prediction_first':prediction_first}
    graphs={}
    for name,fn in functions.items():
        for _ in range(5):fn()
        torch.cuda.synchronize()
        # PyTorch 2.8 clears has_graph_ after automatic instantiation even in
        # debug mode. Retain the graph explicitly so debug_dump writes a DOT.
        graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
        with torch.cuda.graph(graph):out=fn()
        graph.replay();torch.cuda.synchronize();graph.debug_dump(str(dest/(name+'.dot')))
        assert (dest/(name+'.dot')).is_file(),name
        graphs[name]=(graph,out)
    expected=reference(q,k,v,a)
    result={'config':meta,'gpu':torch.cuda.get_device_name(),'scope':'fixed-input graph scheduling diagnostic',
            'validation':{},'pairs':{},'profiles':{},'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ['benchmarks/graph_diagnostic.py','benchmarks/profile_overlap.py','benchmarks/run_benchmark.py','strict_topk/flashinfer_backend.py']}}
    for name,(graph,out) in graphs.items():
        graph.replay();torch.cuda.synchronize()
        result['validation'][name]=metrics(out.output if hasattr(out,'output') else out,expected,q.dtype)
        assert result['validation'][name]['passed']
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for base,candidate in [('serial','selection_first'),('sequential','selection_first'),('selection_first','prediction_first')]:
        rows=[]
        for i in range(5):
            order=[base,candidate,candidate,base] if i%2==0 else [candidate,base,base,candidate]
            times={base:[],candidate:[]}
            for name in order:times[name].append(_interval_ms(graphs[name][0],start,end,100)/100*1000)
            b,c=statistics.mean(times[base]),statistics.mean(times[candidate])
            rows.append({'baseline_us':b,'candidate_us':c,'ratio':b/c})
        result['pairs'][base+'/'+candidate]=rows
    for name in ['sequential','selection_first','prediction_first']:
        graph=graphs[name][0]
        for _ in range(20):graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            graph.replay();torch.cuda.synchronize()
        path=dest/(name+'.trace.json');prof.export_chrome_trace(str(path))
        result['profiles'][name]=timeline_summary(json.loads(path.read_text()))
        assert result['profiles'][name]['cuda_kernel_count']>0
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':result['validation'],'pairs':result['pairs']}),flush=True)


if __name__=='__main__':main()
