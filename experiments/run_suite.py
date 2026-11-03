"""Run serially on a preselected idle GPU; preserve stage status and source hashes."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model');args=ap.parse_args()
    out=Path('results');out.mkdir(exist_ok=True)
    stages=[
        ('selector','tests.check_selector',[]),
        ('correctness','tests.check_correctness',['--include-compact','--output','results/correctness.json']),
        ('eager','benchmarks.run_benchmark',['--include-compact','--profile','main','--warmup','5','--repeats','30','--output','results/eager.json']),
        ('graph','benchmarks.graph_benchmark',['--include-compact','--repeats','60','--output','results/graph.json']),
        ('guard','benchmarks.guard_ablation',['--output','results/guard_ablation.json']),
        ('fusion','benchmarks.fusion_benchmark',['--selector','fused','--repeats','60','--output','results/fusion.json']),
        ('pareto','benchmarks.pareto_benchmark',['--repeats','60','--output','results/pareto.json']),
    ]
    if args.model:
        stages.append(('model_trace','experiments.trace_replay',['--model',args.model,'--context','2048','--steps','8','--graph','--output','results/model_trace.json']))
    manifest={'date':'2026-11-03','gpu_allocation':'One idle A100-SXM4-80GB; final suite run on device 1, other work on device 0.',
              'source_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['strict_topk','tests','benchmarks','experiments'] for p in sorted(Path(folder).glob('*.py'))},'stages':[]}
    for name,module,arguments in stages:
        started=time.perf_counter()
        print('START '+name,flush=True)
        with Path(name+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'name':name,'module':module,'returncode':proc.returncode,'wall_seconds':time.perf_counter()-started})
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+name+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
