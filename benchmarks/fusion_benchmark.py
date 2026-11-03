"""Fusion ablation: complete paths plus post-selection repair critical path."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import torch
from strict_topk import kernels as ops
from strict_topk.fused import repair_fused
from benchmarks.run_benchmark import Config, setup, time_cuda, environment, overlap_run, output_of
from tests import check_correctness


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4): fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): result = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results/fusion.json')
    ap.add_argument('--repeats', type=int, default=60)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--selector', choices=['torch','fused'], default='torch')
    args = ap.parse_args()
    check_correctness.repair = repair_fused
    correctness = check_correctness.run(include_compact=True)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    result = {'environment':environment(), 'scope':'fusion ablation; fixed-data eager and CUDA Graph; repair stage excludes preparation and selector; full stage includes both',
              'correctness':correctness, 'rows':[]}
    path.write_text(json.dumps(result, indent=2)+'\n')
    if not correctness['passed']:
        print(json.dumps([r for r in correctness['rows'] if not r['passed']], indent=2))
        raise SystemExit(1)
    if args.check_only: return
    configs = [Config(blocks=1024, overlap=o) for o in [0., .5, .875, 1.]]
    configs += [Config(heads=32, blocks=1024, block_size=64, dim=128, topk=32, overlap=o) for o in [.5,.875,1.]]
    configs += [Config(blocks=1024,overlap=.5,heterogeneous=True)]
    for c in configs:
        c.selector=args.selector
        q,k,v,p,a,selector = setup(c)
        ref = check_correctness.reference(q,k,v,a)
        ss, ps = torch.cuda.Stream(), torch.cuda.Stream()
        methods = [('serial_exact','full',lambda:ops.exact_attention(q,k,v,selector()))]
        for mode in ['saved','hybrid','compact']:
            state = ops.speculate(q,k,v,p,mode=mode,mass_threshold=.2)
            for label, repair in [('split',ops.repair),('fused',repair_fused)]:
                def tail(state=state, repair=repair):
                    return repair(q,k,v,a,p,state)
                def prep(mode=mode): return ops.speculate(q,k,v,p,mode=mode,mass_threshold=.2)
                def finish(selected,state,repair=repair): return repair(q,k,v,selected,p,state)
                def full(prep=prep,finish=finish): return overlap_run(selector,prep,finish,ss,ps)
                methods += [(mode+'_'+label,'repair',tail),(mode+'_'+label,'full',full)]
        random.Random(c.seed).shuffle(methods)
        for name, stage, fn in methods:
            for timing in ['eager','graph']:
                if timing=='graph':
                    graph, out = capture(fn)
                    timed = graph.replay
                else:
                    out=fn(); timed=fn
                numeric=check_correctness.error_metrics(output_of(out),ref)
                row={'config':asdict(c),'method':name,'stage':stage,'timing':timing,
                     'numerics_vs_exact_topk':numeric,**time_cuda(timed,5,args.repeats)}
                if hasattr(out,'fallback'): row['fallback_heads']=int(out.fallback.sum())
                result['rows'].append(row)
                if timing=='graph': del timed, graph
                print(json.dumps({k:row[k] for k in ['method','stage','timing','median_ms','p95_ms']}),flush=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    result['passed']=all(r['numerics_vs_exact_topk']['passed'] for r in result['rows'])
    path.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__': main()
