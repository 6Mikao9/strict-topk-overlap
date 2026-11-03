"""Paired add/erase fusion ablation on the unchanged round-one state layouts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch

from benchmarks.fusion_benchmark import capture
from benchmarks.paired_confirm import _interval_ms, _replay_interleaved, _warm_device
from benchmarks.run_benchmark import Config, setup, environment, output_of, overlap_run
from strict_topk import kernels as ops
from strict_topk.fused import repair_fused
from strict_topk.signed_repair import repair_signed
from tests.check_correctness import error_metrics, reference


def own_reference(q, k, v, a, p, state, result):
    if result.approximate_heads is None or not result.approximate_heads.any().item():
        return reference(q, k, v, a)
    rows = []
    for h in range(q.shape[0]):
        ids = a[h]
        if result.approximate_heads[h].item():
            wrong = ~(p[h, :, None] == a[h, None, :]).any(dim=1)
            extra = p[h, wrong & ~state.high[h]]
            ids = torch.cat((ids, extra))
        rows.append(reference(q[h:h+1], k[h:h+1], v[h:h+1], ids[None]))
    return torch.cat(rows)


def run_case(q, k, v, p, a, selector, config, replays=100, rounds=5, seed=20261103, signed_impl=None):
    exact = reference(q, k, v, a)
    ss, ps = torch.cuda.Stream(), torch.cuda.Stream()
    graphs, validation, diagnostics, comparisons = {}, {}, {}, []

    def add(name, fn, expected_state=None):
        graph, result = capture(fn)
        expected = exact if expected_state is None else own_reference(q,k,v,a,p,expected_state,result)
        validation[name] = error_metrics(output_of(result), expected)
        diagnostic = {'vs_strict_topk': error_metrics(output_of(result), exact)}
        if hasattr(result, 'fallback'):
            diagnostic['fallback_heads'] = int(result.fallback.sum())
        if getattr(result, 'approximate_heads', None) is not None:
            diagnostic['approximate_heads'] = int(result.approximate_heads.sum())
            diagnostic['max_extra_mass_bound'] = float(result.extra_mass_bound.max())
        if hasattr(result, 'positive_jobs'):
            diagnostic['positive_jobs_per_head'] = result.positive_jobs.tolist()
            diagnostic['negative_jobs_per_head'] = (result.total_jobs-result.positive_jobs).tolist()
        # CUDA Graph captures addresses, not Python ownership of external
        # tensors. Keep the closure (and its prediction state) alive until
        # every measurement finishes; otherwise later captures can recycle
        # a repair-only graph's inputs through the caching allocator.
        graphs[name] = (graph, result, fn, expected_state)
        diagnostics[name] = diagnostic

    add('serial_exact', lambda: ops.exact_attention(q,k,v,selector()))
    for mode in ('hybrid', 'compact'):
        for epsilon in (0., .05):
            state = ops.speculate(q,k,v,p,mode=mode,mass_threshold=.2,keep_mass=epsilon>0)
            variants = [('fused',repair_fused),('signed',repair_signed if signed_impl is None else signed_impl)]
            if epsilon == 0.: variants.append(('split',ops.repair))
            for variant, function in variants:
                kwargs = {'max_extra_mass': epsilon} if variant != 'split' else {}
                label = f'{mode}_{variant}_eps{epsilon:g}'
                def tail(function=function, state=state, kwargs=kwargs):
                    return function(q,k,v,a,p,state,**kwargs)
                def prepare(mode=mode,epsilon=epsilon):
                    return ops.speculate(q,k,v,p,mode=mode,mass_threshold=.2,keep_mass=epsilon>0)
                def finish(selected, prepared, function=function,kwargs=kwargs):
                    return function(q,k,v,selected,p,prepared,**kwargs)
                def full(prepare=prepare,finish=finish):
                    return overlap_run(selector,prepare,finish,ss,ps)
                for stage, fn in [('repair',tail),('full',full)]:
                    name = stage+'_'+label
                    add(name,fn,state if epsilon else None)
                    if stage == 'full':
                        comparisons.append(('serial_exact',name))
                    elif variant != 'fused':
                        comparisons.append((f'repair_{mode}_fused_eps{epsilon:g}',name))
    assert all(x['passed'] for x in validation.values()), validation
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    warmup = _warm_device(graphs['serial_exact'][0],graphs['full_hybrid_signed_eps0'][0],start,end,1000,200.)
    pairs = {name:[] for _,name in comparisons}
    rng = random.Random(seed)
    for round_index in range(rounds):
        order = comparisons.copy();rng.shuffle(order)
        for baseline,name in order:
            sequence=[baseline,name,name,baseline]
            if rng.getrandbits(1):sequence=[name,baseline,baseline,name]
            _replay_interleaved(graphs[baseline][0],graphs[name][0],30)
            raw=[_interval_ms(graphs[x][0],start,end,replays) for x in sequence]
            base_ms=statistics.mean(t for x,t in zip(sequence,raw) if x==baseline)/replays
            cand_ms=statistics.mean(t for x,t in zip(sequence,raw) if x==name)/replays
            pairs[name].append({'round':round_index,'sequence':sequence,'raw_interval_ms':raw,
                                'baseline_ms':base_ms,'candidate_ms':cand_ms,'ratio':base_ms/cand_ms})
    post_timing_validation={}
    for name,(graph,result,fn,expected_state) in graphs.items():
        graph.replay();torch.cuda.synchronize()
        expected=exact if expected_state is None else own_reference(q,k,v,a,p,expected_state,result)
        post_timing_validation[name]=error_metrics(output_of(result),expected)
    assert all(x['passed'] for x in post_timing_validation.values()),post_timing_validation
    rows=[]
    for baseline,name in comparisons:
        measurements=pairs[name];ratios=[x['ratio'] for x in measurements]
        rows.append({'method':name,'baseline':baseline,'pairs':measurements,
                     'median_paired_ratio':statistics.median(ratios),
                     'median_candidate_ms':statistics.median(x['candidate_ms'] for x in measurements),
                     'median_baseline_ms':statistics.median(x['baseline_ms'] for x in measurements),
                     'pairs_ratio_gt_one':sum(x>1 for x in ratios)})
    return {'config':config,'timing':{'replays':replays,'rounds':rounds,'warmup':warmup},
            'validation':validation,'post_timing_validation':post_timing_validation,
            'diagnostics':diagnostics,'rows':rows,'passed':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--output',default='results/round2/signed_20261103.json')
    ap.add_argument('--replays',type=int,default=100);ap.add_argument('--rounds',type=int,default=5)
    args=ap.parse_args()
    configs=[Config(heads=16,blocks=512,block_size=32,dim=128,topk=32,overlap=o,seed=args.seed,selector='fused') for o in [0.,.5,.9375,1.]]
    configs += [Config(heads=32,blocks=1024,block_size=32,dim=128,topk=64,overlap=o,seed=args.seed,selector='fused') for o in [.5,.9375]]
    sources=['strict_topk/signed_repair.py','tests/check_signed_repair.py','benchmarks/signed_benchmark.py']
    data={'environment':environment(),'source_sha256':{x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in sources},
          'limitations':['Round-one expanded KV layout held fixed to isolate repair changes',
                         'Fixed synthetic CUDA Graph replay; no full model or live request tail',
                         'Repair ratios use old fused repair; full ratios use serial exact selector plus attention',
                         'Approximate outputs checked against their own retained-low-block FP64 set'],
          'experiments':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    for config in configs:
        q,k,v,p,a,selector=setup(config)
        case=run_case(q,k,v,p,a,selector,asdict(config),args.replays,args.rounds,args.seed)
        data['experiments'].append(case)
        path.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({'config':asdict(config),'passed':case['passed']}),flush=True)
    data['passed']=all(x['passed'] for x in data['experiments'])
    path.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
