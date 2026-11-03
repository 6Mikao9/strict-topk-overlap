"""Serial GPU jobs for decode-window simulation and causal rank diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True)
    args=ap.parse_args();dest=Path('results/window');dest.mkdir(parents=True,exist_ok=True)
    stages=[
        ('current_probe','benchmarks.selector_window',['--output',str(dest/'current_probe.json')]),
        ('qsa_formula_probe','benchmarks.selector_window',['--kind','qsa_shape','--hierarchical','--output',str(dest/'qsa_formula_probe.json')]),
        ('qsa_old_probe','benchmarks.selector_window',['--kind','qsa_shape','--output',str(dest/'qsa_old_probe.json')]),
        ('predictor_rank_trace','experiments.predictor_rank_trace',['--model',args.model,'--model-label','Qwen/Qwen3-4B','--output',str(dest/'predictor_rank_trace.json')]),
    ]
    for seed in [20261103,20261104,20261105]:
        stages.append((f'simulation_{seed}','benchmarks.window_simulation',['--seed',str(seed),'--output',str(dest/f'simulation_{seed}.json')]))
    manifest={'date':'2026-11-03','gpu':'One prechecked idle A100-SXM4-80GB',
        'source_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['strict_topk','tests','benchmarks','experiments'] for p in sorted(Path(folder).glob('*.py'))},'stages':[]}
    for name,module,arguments in stages:
        print('START '+name,flush=True);start=time.perf_counter()
        with Path('window_'+name+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'name':name,'module':module,'returncode':proc.returncode,'wall_seconds':time.perf_counter()-start})
        (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+name+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
