"""Host-side sampling to audit GPU sharing hidden by container PID namespaces.

Run on the host while benchmarks execute inside a container. This records only
counts and resource samples, never user names, PIDs or GPU UUIDs in the result.
Sampling cannot exclude sharing between polls; it does not stop any process.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiments.run_container_profile import inventory


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--container',required=True);ap.add_argument('--seconds',type=int,default=180)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result={'scope':'host-side polls separated by 1 second plus command time; may miss activity between polls',
            'gpu_index':args.gpu,'samples':[],'status':'running'}
    start=time.monotonic()
    while time.monotonic()-start<args.seconds:
        device=next(d for d in inventory() if d['index']==args.gpu)
        container_pids=set(subprocess.check_output(['docker','top',args.container,'-eo','pid'],text=True).splitlines()[1:])
        container_pids={p.strip() for p in container_pids}
        outside=set(device['pids'])-container_pids
        result['samples'].append({'unix_time':time.time(),'memory_mib':device['memory_mib'],
            'utilization':device['utilization'],'compute_process_count':len(device['pids']),
            'processes_outside_container':len(outside)})
        result['status']='sampling'
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        time.sleep(1)
    result['status']='complete'
    result['sharing_observed']=any(s['compute_process_count']>1 or s['processes_outside_container'] for s in result['samples'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'samples':len(result['samples']),'sharing_observed':result['sharing_observed']}))


if __name__=='__main__':main()
