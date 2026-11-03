"""Capture real Qwen3 Q/K/V and replay block selection/repair, not sparse decoding.

Only public authored prompt and numeric summaries are exported. Model weights and
QKV tensors remain local. Dense generation is unchanged; no task-quality claim.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as modeling
from strict_topk import kernels as ops
from benchmarks.run_benchmark import time_cuda, overlap_run, environment
from tests.check_correctness import reference, error_metrics
from strict_topk.fused import repair_fused
from strict_topk.selector import select_blocks
from benchmarks.fusion_benchmark import capture as capture_graph

PROMPT = ("This research notebook studies numerical algorithms. A library stores books in numbered rooms. "
          "Room one contains astronomy, room two mathematics, and room three history. "
          "An accurate answer must distinguish an approximate candidate set from the selected set. "
          "Explain why removing an incorrect candidate can change a normalized weighted average.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--context', type=int, default=2048)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--output', default='results/model_trace.json')
    ap.add_argument('--graph', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(20261103)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
              torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    ids = tokenizer(PROMPT * (args.context // 30 + 1), return_tensors='pt').input_ids[:, :args.context].cuda()
    layers = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
    captured, counter = {}, [0]
    original = modeling.apply_rotary_pos_emb
    def capture(*a, **kw):
        q, k = original(*a, **kw)
        layer = counter[0] % model.config.num_hidden_layers
        if layer in layers:
            captured[layer] = q[0, :, -1].detach().contiguous()
        counter[0] += 1
        return q, k
    modeling.apply_rotary_pos_emb = capture
    previous, rows = {}, []
    selection_stream, prediction_stream = torch.cuda.Stream(), torch.cuda.Stream()
    try:
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            for step in range(args.steps):
                counter[0] = 0
                out = model(token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                for layer in layers:
                    q = captured[layer]
                    ck, cv = past.layers[layer].keys[0], past.layers[layer].values[0]
                    groups = q.shape[0] // ck.shape[0]
                    n = ck.shape[1] // 32
                    k = ck[:, :n*32].repeat_interleave(groups, 0).reshape(q.shape[0], n, 32, q.shape[1]).contiguous()
                    v = cv[:, :n*32].repeat_interleave(groups, 0).reshape_as(k).contiguous()
                    lo, hi = k.amin(2).float(), k.amax(2).float()
                    qf = q.float()[:, None]
                    budget = min(8, n)
                    def selector():
                        return select_blocks(q,lo,hi,budget)
                    selected = selector()
                    if layer in previous:
                        predicted = previous[layer]
                        expected = reference(q, k, v, selected)
                        hit = (predicted[:, :, None] == selected[:, None, :]).any(-1)
                        methods = {'serial_exact': lambda: ops.exact_attention(q, k, v, selector())}
                        for mode in ['saved', 'hybrid', 'compact']:
                            def prep(mode=mode):
                                return ops.speculate(q,k,v,predicted,mode=mode,mass_threshold=.2)
                            def finish(a, state):
                                return ops.repair(q,k,v,a,predicted,state,min_remaining=1e-3,cancellation_tol=1e-5)
                            methods[mode+'_overlap'] = lambda prep=prep, finish=finish: overlap_run(selector,prep,finish,selection_stream,prediction_stream)
                        for epsilon in [0.,.01,.05,.1,.2]:
                            def prep():
                                return ops.speculate(q,k,v,predicted,mode='compact',mass_threshold=.2,keep_mass=True)
                            def finish(a,state,epsilon=epsilon):
                                return repair_fused(q,k,v,a,predicted,state,max_extra_mass=epsilon)
                            methods[f'compact_fused_eps{epsilon:g}'] = lambda prep=prep,finish=finish: overlap_run(selector,prep,finish,selection_stream,prediction_stream)
                        for name, fn in methods.items():
                            if args.graph:
                                graph,result = capture_graph(fn)
                                timed=graph.replay
                            else:
                                result = fn()
                                timed=fn
                            tensor = result.output if hasattr(result, 'output') else result
                            row = {'step': step, 'layer': layer, 'method': name, 'blocks': n,
                                   'overlap_fraction': hit.float().mean().item(),
                                   'numeric': error_metrics(tensor, expected),
                                   'fallback_heads': int(result.fallback.sum()) if hasattr(result,'fallback') else 0}
                            row['epsilon'] = float(name.split('eps')[1]) if 'eps' in name else 0.
                            row['timing'] = 'graph' if args.graph else 'eager'
                            for attr in ['approximate_heads','extra_mass_bound']:
                                if hasattr(result,attr) and getattr(result,attr) is not None:
                                    row[attr]=getattr(result,attr).tolist()
                            row.update(time_cuda(timed, 2, 20))
                            rows.append(row)
                            if args.graph: del timed,graph
                    previous[layer] = selected.clone()
                print(json.dumps({'step_done':step}), flush=True)
    finally:
        modeling.apply_rotary_pos_emb = original
    result = {'environment':environment(), 'model':'Qwen/Qwen3-0.6B-Base',
              'context':args.context,'steps':args.steps,'layers':layers,'prompt':PROMPT,
              'selector_implementation':'fused minmax scores plus torch.topk',
              'scope':'Dense model QKV replay; previous-step selected IDs predict current IDs; only complete 32-token blocks; GQA KV expanded; no sparse model inference or quality evaluation; metadata construction excluded.',
              'passed':all(r['numeric']['passed'] for r in rows if r['epsilon']==0.),'rows':rows}
    path=Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'rows':len(rows)}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
