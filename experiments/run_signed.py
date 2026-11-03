"""Serial signed-correction ablation suite on one prechecked idle GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-small',required=True)
    ap.add_argument('--model-large',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[20261103,20261104,20261105])
    args=ap.parse_args();dest=Path('results/round2');dest.mkdir(parents=True,exist_ok=True)
    stages=[]
    for seed in args.seeds:
        stages.append((f'signed_{seed}','benchmarks.signed_benchmark',['--seed',str(seed),'--output',str(dest/f'signed_{seed}.json')]))
    for size,model,label in [('small',args.model_small,'Qwen/Qwen3-0.6B-Base'),('large',args.model_large,'Qwen/Qwen3-4B')]:
        stages.append((f'trace_signed_{size}','experiments.trace_signed',['--model',model,'--model-label',label,'--contexts','16384','--steps','3','--topk','64','--output',str(dest/f'trace_signed_{size}.json')]))
    manifest={'date':'2026-11-03','gpu':'One prechecked idle A100-SXM4-80GB',
              'source_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['strict_topk','tests','benchmarks','experiments'] for p in sorted(Path(folder).glob('*.py'))},'stages':[]}
    for name,module,arguments in stages:
        print('START '+name,flush=True);start=time.perf_counter()
        with Path('round2_'+name+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m',module,*arguments],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'name':name,'module':module,'returncode':proc.returncode,'wall_seconds':time.perf_counter()-start})
        (dest/'signed_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+name+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
