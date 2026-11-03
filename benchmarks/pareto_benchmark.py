"""Accuracy/latency frontier for bounded extra attention mass, fixed CUDA Graph."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from strict_topk import kernels as ops
from strict_topk.fused import repair_fused
from strict_topk.selector import select_blocks
from benchmarks.run_benchmark import Config, setup, overlap_run, time_cuda, environment, output_of
from benchmarks.fusion_benchmark import capture
from tests.check_correctness import reference, error_metrics


def verify_mass(q,k,p,a,state,out):
    """Independent FP64 probability mass on selected UNION retained low wrongs."""
    values=[]
    flags=out.approximate_heads.tolist() if out.approximate_heads is not None else [False]*q.shape[0]
    for h,flag in enumerate(flags):
        if not flag:
            values.append(0.);continue
        lowwrong=[bid for bid,high in zip(p[h].tolist(),state.high[h].tolist()) if not high and bid not in a[h].tolist()]
        if not lowwrong:
            values.append(0.);continue
        chosen=k[h,a[h].long()].double().reshape(-1,q.shape[1]) @ q[h].double() / q.shape[1]**.5
        extra=k[h,lowwrong].double().reshape(-1,q.shape[1]) @ q[h].double() / q.shape[1]**.5
        values.append(torch.exp(torch.logsumexp(extra,0)-torch.logsumexp(torch.cat([chosen,extra]),0)).item())
    return values


def measure(q,k,v,p,a,selector,config,repeats=50):
    ref=reference(q,k,v,a)
    ss,ps=torch.cuda.Stream(),torch.cuda.Stream()
    rows=[]
    baseline=lambda: ops.exact_attention(q,k,v,selector())
    g,out=capture(baseline)
    base=time_cuda(g.replay,10,repeats)
    del g
    for epsilon in [0.,.001,.01,.05,.1,.2,.5,1.]:
        def prep(): return ops.speculate(q,k,v,p,mode='compact',mass_threshold=.2,keep_mass=True)
        def finish(a,state): return repair_fused(q,k,v,a,p,state,max_extra_mass=epsilon)
        def full(): return overlap_run(selector,prep,finish,ss,ps)
        state=prep()
        def tail(): return finish(a,state)
        for stage,fn in [('repair',tail),('full',full)]:
            graph,out=capture(fn)
            numeric=error_metrics(output_of(out),ref)
            row={'config':config,'epsilon':epsilon,'stage':stage,'timing':'graph',
                 'numeric':numeric,**time_cuda(graph.replay,10,repeats),
                 'fallback_heads':int(out.fallback.sum()),'serial_exact_median_ms':base['median_ms']}
            for attr in ['approximate_heads','extra_mass_bound']:
                if hasattr(out,attr) and getattr(out,attr) is not None:
                    row[attr]=getattr(out,attr).tolist()
            actual_mass=verify_mass(q,k,p,a,state,out)
            row['fp64_actual_extra_mass']=actual_mass
            row['independent_mass_bound_passed']=all(x<=epsilon+1e-5 and x<=b+1e-5 for x,b in zip(actual_mass,row.get('extra_mass_bound',[0.]*len(actual_mass))))
            row['speedup_vs_serial_exact']=base['median_ms']/row['median_ms'] if stage=='full' else None
            rows.append(row)
            del graph
            print(json.dumps({'epsilon':epsilon,'stage':stage,'median_ms':row['median_ms'],'error':numeric['max_abs_error']}),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results/pareto.json')
    ap.add_argument('--repeats',type=int,default=60)
    args=ap.parse_args()
    result={'environment':environment(),'scope':'approximate compact fused repair; epsilon bounds retained wrong probability mass in ideal arithmetic; quality measured as attention output error, not task score','rows':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for overlap in [.5,.875,1.]:
        for scale in [1.,3.]:
            c=Config(blocks=1024,overlap=overlap,selector='fused')
            q,k,v,p,a,selector=setup(c)
            # Scale Q in-place; selector closes over a float copy so rebuild it.
            q.mul_(scale)
            lo,hi=k.amin(2).float(),k.amax(2).float();qf=q.float()[:,None]
            def selector():return select_blocks(q,lo,hi,c.topk)
            a=selector()
            result['rows']+=measure(q,k,v,p,a,selector,{**asdict(c),'q_scale':scale},args.repeats)
            path.write_text(json.dumps(result,indent=2)+'\n')
    strict=[r for r in result['rows'] if r['epsilon']==0]
    result['strict_checks_passed']=all(r['numeric']['passed'] for r in strict)
    result['mass_budget_checks_passed']=all(r['independent_mass_bound_passed'] for r in result['rows'])
    path.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['strict_checks_passed'] and result['mass_budget_checks_passed'] else 1)


if __name__=='__main__':main()
