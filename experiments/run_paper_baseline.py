"""Three seeded A100 sweeps of PRR's nominal Table 3 shapes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    dest=Path('results/paper_baseline');dest.mkdir(parents=True,exist_ok=True)
    files=['benchmarks/paper_baseline.py','experiments/run_paper_baseline.py',
           'strict_topk/gqa_hierarchical.py','strict_topk/flashinfer_backend.py']
    manifest={'date':'2026-11-09','source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},'stages':[]}
    for seed in (20261109,20261110,20261111):
        print('START '+str(seed),flush=True);start=time.perf_counter()
        with Path('paper_'+str(seed)+'.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m','benchmarks.paper_baseline','--seed',str(seed),
                                  '--output',str(dest/f'benchmark_{seed}.json')],stdout=log,stderr=subprocess.STDOUT)
        manifest['stages'].append({'seed':seed,'returncode':proc.returncode,'seconds':time.perf_counter()-start})
        (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END '+str(seed)+' code='+str(proc.returncode),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
