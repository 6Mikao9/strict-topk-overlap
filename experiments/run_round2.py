"""Serial round-two jobs; execute only after checking the selected GPU is idle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-small',required=True)
    ap.add_argument('--model-large',required=True)
    args=ap.parse_args()
    dest=Path('results/round2');dest.mkdir(parents=True,exist_ok=True)
    stages=[('gqa_correctness','tests.check_gqa',[])]
    for seed in [20261103,20261104,20261105]:
        stages.append((f'synthetic_{seed}','benchmarks.critical_path',['--profile','main','--seed',str(seed),'--replays','100','--batches','5','--output',str(dest/f'synthetic_{seed}.json')]))
    stages += [
        ('trace_small','experiments.trace_round2',['--model',args.model_small,'--model-label','Qwen/Qwen3-0.6B-Base','--contexts','2048','16384','32760','--steps','6','--topk','32','64','--output',str(dest/'trace_small.json')]),
        ('trace_large','experiments.trace_round2',['--model',args.model_large,'--model-label','Qwen/Qwen3-4B','--contexts','16384','32768','--steps','4','--topk','64','--output',str(dest/'trace_large.json')]),
    ]
    manifest={'date':'2026-11-03','gpu':'One prechecked idle A100-SXM4-80GB device 1',
      'source_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['strict_topk','tests','benchmarks','experiments'] for p in sorted(Path(folder).glob('*.py'))},'stages':[]}
    for name,module,arguments in stages:
        start=time.perf_counter();print('START '+name,flush=True)
        with Path('round2_'+name+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'name':name,'module':module,'returncode':proc.returncode,'wall_seconds':time.perf_counter()-start})
        (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+name+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
