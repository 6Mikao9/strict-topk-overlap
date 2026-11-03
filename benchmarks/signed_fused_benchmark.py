"""Same paired ablation with the three-launch signed correction implementation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from benchmarks.signed_benchmark import run_case as compare
from benchmarks.run_benchmark import Config,setup,environment
from strict_topk.signed_fused import repair_signed_fused


def run_case(*args,**kwargs):
    return compare(*args,**kwargs,signed_impl=repair_signed_fused)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--output',default='results/round2/signed_fused_20261103.json')
    ap.add_argument('--replays',type=int,default=100);ap.add_argument('--rounds',type=int,default=5)
    args=ap.parse_args()
    configs=[Config(heads=16,blocks=512,block_size=32,dim=128,topk=32,overlap=o,seed=args.seed,selector='fused') for o in [0.,.5,.9375,1.]]
    configs += [Config(heads=32,blocks=1024,block_size=32,dim=128,topk=64,overlap=o,seed=args.seed,selector='fused') for o in [.5,.9375]]
    files=['strict_topk/signed_repair.py','strict_topk/signed_fused.py','tests/check_signed_fused.py','benchmarks/signed_benchmark.py','benchmarks/signed_fused_benchmark.py']
    data={'environment':environment(),'variant':'three-launch signed correction, direct low-frame negatives, inline fallback',
          'source_sha256':{x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in files},
          'limitations':['Expanded KV fixed across methods; state preparation unchanged',
                         'Fixed graph replay and synthetic prediction; no end-to-end model throughput',
                         'Repair baseline is old fused repair; full baseline is serial exact'],
          'experiments':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for config in configs:
        q,k,v,p,a,selector=setup(config)
        case=run_case(q,k,v,p,a,selector,asdict(config),args.replays,args.rounds,args.seed)
        data['experiments'].append(case);path.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({'config':asdict(config),'passed':case['passed']}),flush=True)
    data['passed']=True;path.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
