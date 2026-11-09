"""Wait for a completely free GPU inside the training container, then profile.

No foreign job is stopped. If another visible process shares the device,
terminate only this script's own child and mark its results contaminated.
Container PID namespaces may hide outside processes: run host_gpu_audit on
the host as well. This container-only monitor cannot prove isolation.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    occupied={}
    for line in processes.splitlines():
        uuid,pid=[x.strip() for x in line.split(',')];occupied.setdefault(uuid,set()).add(pid)
    return [{'index':int(i),'uuid':u.strip(),'memory_mib':int(m),'utilization':int(s),
             'pids':sorted(occupied.get(u.strip(),set()))} for i,u,m,s in
            ([x.strip() for x in line.split(',')] for line in raw.splitlines())]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-seconds',type=int,default=1800)
    ap.add_argument('--stage',choices=['all','diagnostic'],default='all');args=ap.parse_args()
    dest=Path('results/profiling' if args.stage=='all' else 'results/graph_diagnostic');dest.mkdir(parents=True,exist_ok=True)
    state={'status':'waiting_for_free_gpu','scope':'training container only','stages':[]}
    def save(): (dest/'run_status.json').write_text(json.dumps(state,indent=2)+'\n')
    start=time.monotonic();stable={};chosen=None
    while time.monotonic()-start<args.wait_seconds:
        devices=inventory()
        state['last_check_unix']=time.time()
        state['devices']=[{k:v for k,v in d.items() if k not in ('uuid','pids')}|{'compute_process_count':len(d['pids'])} for d in devices]
        for d in devices:
            free=not d['pids'] and d['memory_mib']<100 and d['utilization']==0
            stable[d['index']]=stable.get(d['index'],0)+1 if free else 0
            if stable[d['index']]>=3:chosen=d;break
        save()
        if chosen:break
        time.sleep(10)
    if chosen is None:
        state['status']='deferred_all_gpus_occupied';save();print(state['status'],flush=True);return
    state['status']='running';state['gpu_index']=chosen['index'];save()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(chosen['index'])
    env['PYTHONPATH']=str(Path.cwd());env['TRITON_CACHE_DIR']=str(Path.cwd()/'.triton-cache')
    # Validate provenance in the requested container; rerun rather than reuse
    # the earlier potentially shared-device timing as an isolated result.
    stages=([('paper_baseline','experiments.run_paper_baseline',[]),
             ('profile','benchmarks.profile_overlap',[])] if args.stage=='all' else
            [('diagnostic','benchmarks.graph_diagnostic',[])])
    for label,module,args2 in stages:
        before=next(d for d in inventory() if d['uuid']==chosen['uuid'])
        if before['pids']:
            state['status']='deferred_device_became_occupied';save();return
        with (dest/(label+'.log')).open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m',module,*args2],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            # More than one process is allowed only for this runner's nested
            # CPU parent; each workload owns at most one CUDA process.
            shared=False
            while proc.poll() is None:
                current=next(d for d in inventory() if d['uuid']==chosen['uuid'])
                if len(current['pids'])>1:
                    shared=True
                    # The new session belongs only to this child and its
                    # descendants, including the runner's GPU subprocess.
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30);break
                time.sleep(2)
        state['stages'].append({'name':label,'returncode':proc.returncode,'shared_device_detected':shared})
        save()
        if shared or proc.returncode:
            state['status']='invalid_shared_device' if shared else 'failed';save();return
    state['status']='complete';save();print('complete',flush=True)


if __name__=='__main__':main()
