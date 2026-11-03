"""Follow-up adjacent paired comparisons; outputs are separate from first sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-small',required=True);ap.add_argument('--model-large',required=True)
    args=ap.parse_args();dest=Path('results/round2');dest.mkdir(parents=True,exist_ok=True)
    stages=[('light_correctness','tests.check_gqa_light',[])]
    for seed in [20261103,20261104,20261105]:
        stages.append((f'paired_{seed}','benchmarks.paired_confirm',['--seed',str(seed),'--output',str(dest/f'paired_{seed}.json')]))
    stages += [
        ('paired_idealized','benchmarks.paired_confirm',['--idealized-estimate','--output',str(dest/'paired_idealized.json')]),
        ('trace_paired_small','experiments.trace_paired',['--model',args.model_small,'--model-label','Qwen/Qwen3-0.6B-Base','--contexts','16384','32760','--steps','4','--topk','64','--output',str(dest/'trace_paired_small.json')]),
        ('trace_paired_large','experiments.trace_paired',['--model',args.model_large,'--model-label','Qwen/Qwen3-4B','--contexts','16384','32768','--steps','4','--topk','64','--output',str(dest/'trace_paired_large.json')]),
    ]
    manifest={'date':'2026-11-03','gpu':'A100 device 1, isolated from other GPU0 work',
      'source_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['strict_topk','tests','benchmarks','experiments'] for p in sorted(Path(folder).glob('*.py'))},'stages':[]}
    for name,module,arguments in stages:
        print('START '+name,flush=True);start=time.perf_counter()
        with Path('round2_'+name+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'name':name,'module':module,'returncode':proc.returncode,'wall_seconds':time.perf_counter()-start})
        (dest/'confirm_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+name+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
