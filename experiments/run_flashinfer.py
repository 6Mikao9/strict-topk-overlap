"""Bounded vendor-on-both-sides validation and three-seed paired sweep."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    dest=Path('results/vendor_shared');dest.mkdir(parents=True,exist_ok=True)
    stages=[('correctness','tests.check_flashinfer_backend',['--output',str(dest/'correctness.json')])]
    for seed in (20261103,20261104,20261105):
        stages.append((f'benchmark_{seed}','benchmarks.flashinfer_overlap',
                       ['--seed',str(seed),'--output',str(dest/f'benchmark_{seed}.json')]))
    files=['strict_topk/flashinfer_backend.py','tests/check_flashinfer_backend.py',
           'benchmarks/flashinfer_overlap.py','experiments/run_flashinfer.py']
    manifest={'date':'2026-11-03','device':'One prechecked idle A100-SXM4-80GB',
              'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},'stages':[]}
    for name,module,args in stages:
        print('START '+name,flush=True);start=time.perf_counter()
        with Path('vendor_'+name+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m',module,*args],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'name':name,'module':module,'returncode':proc.returncode,
                                   'wall_seconds':time.perf_counter()-start})
        (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+name+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
