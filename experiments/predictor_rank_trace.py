"""Analyze ranked block-ID prediction on dense Qwen3 Q/K traces."""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as modeling

from experiments.trace_replay import PROMPT
from strict_topk.selector import block_scores


def ranked_ids(query, keys, block_size, topk):
    n_blocks = keys.shape[1] // block_size
    if n_blocks < topk:
        raise ValueError("not enough complete KV blocks for the requested top-k")
    groups = query.shape[0] // keys.shape[0]
    blocks = keys[:, :n_blocks * block_size].reshape(keys.shape[0], n_blocks, block_size, query.shape[1])
    lo = blocks.amin(dim=2).float().repeat_interleave(groups, dim=0)
    hi = blocks.amax(dim=2).float().repeat_interleave(groups, dim=0)
    scores = block_scores(query, lo, hi)
    # Resolve ties by ascending block ID so the prefix is defined at every rank.
    ids = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :topk]
    ranked_scores = scores.gather(1, ids)
    cutoff = ranked_scores[:, -1:]
    ties = {
        "adjacent_equal_scores_in_topk": (ranked_scores[:, 1:] == ranked_scores[:, :-1]).sum(dim=1).tolist(),
        "all_blocks_tied_at_cutoff": (scores == cutoff).sum(dim=1).tolist(),
        "selected_blocks_tied_at_cutoff": (ranked_scores == cutoff).sum(dim=1).tolist(),
    }
    return ids.tolist(), ties, n_blocks


def interleave(temporal, layer, budget):
    """Alternate accepted unique IDs while preserving each source's rank order."""
    sources = (iter(temporal), iter(layer))
    selected, seen = [], set()
    for slot in range(budget):
        source = sources[slot % 2]
        block = next(source)
        while block in seen:
            block = next(source)
        seen.add(block)
        selected.append(block)
    return selected


def rank_quartiles(ranking, retained, quartile_edges):
    bins = [sum(block in retained for block in ranking[lo:hi])
            for lo, hi in zip(quartile_edges, quartile_edges[1:])]
    rates = [hits / (hi - lo) for hits, lo, hi in zip(bins, quartile_edges, quartile_edges[1:])]
    return bins, rates


def metrics(target, candidate, quartile_edges):
    bins, rates = rank_quartiles(target, set(candidate), quartile_edges)
    hits = sum(bins)
    return {
        "hits": hits, "miss": len(target) - hits, "wrong": len(candidate) - hits,
        "precision": hits / len(candidate), "coverage": hits / len(target),
        "target_rank_quartile_hits": bins,
        "target_rank_quartile_retention": rates,
    }


def compare(target_ids, temporal_ids, layer_ids, budget, quartile_edges):
    rows = []
    for head, (target, temporal_ranking, layer_ranking) in enumerate(zip(target_ids, temporal_ids, layer_ids)):
        mixed = interleave(temporal_ranking, layer_ranking, budget)
        temporal, layer = temporal_ranking[:budget], layer_ranking[:budget]
        target_set, temporal_set, layer_set = set(target), set(temporal), set(layer)
        common = temporal_set & layer_set
        temporal_bins, temporal_rates = rank_quartiles(temporal_ranking, target_set, quartile_edges)
        layer_bins, layer_rates = rank_quartiles(layer_ranking, target_set, quartile_edges)
        rows.append({
            "head": head,
            "temporal": {**metrics(target, temporal, quartile_edges),
                         "source_rank_quartile_hits": temporal_bins,
                         "source_rank_quartile_retention": temporal_rates},
            "layer": {**metrics(target, layer, quartile_edges),
                      "source_rank_quartile_hits": layer_bins,
                      "source_rank_quartile_retention": layer_rates},
            "interleave": metrics(target, mixed, quartile_edges),
            "sources": {
                "prediction_intersection": len(common),
                "common_hits": len(common & target_set),
                "temporal_only_hits": len((temporal_set - layer_set) & target_set),
                "layer_only_hits": len((layer_set - temporal_set) & target_set),
            },
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local cached Qwen3 checkpoint")
    parser.add_argument("--model-label", default="Qwen/Qwen3-4B", help="Public label for JSON output")
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--output", default="results/window/predictor_rank_trace.json")
    args = parser.parse_args()
    block_size = 32
    if args.steps < 2 or args.topk < 4 or min(args.contexts) // block_size < args.topk:
        parser.error("require at least two decode steps, K >= 4, and at least K complete blocks")

    torch.manual_seed(20261103)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda").eval()
    layer_count = model.config.num_hidden_layers
    if layer_count < 4 or max(args.contexts) + args.steps > model.config.max_position_embeddings:
        raise ValueError("checkpoint must support the requested layers and context plus decode steps")
    targets = [1, layer_count // 2, layer_count - 1]
    layers = sorted(set(targets + [layer - 1 for layer in targets]))
    fractions = [(1, 2), (4, 5), (1, 1)]
    quartile_edges = [args.topk * i // 4 for i in range(5)]
    result = {
        "model": args.model_label,
        "model_max_positions": model.config.max_position_embeddings,
        "contexts": args.contexts, "decode_steps": args.steps,
        "target_layers": targets, "captured_layers": layers,
        "block_size": block_size, "topk": args.topk, "prompt": PROMPT,
        "target_rank_quartile_edges": quartile_edges,
        "scope": [
            "Dense greedy Qwen3 generation is unchanged. The external Quest-style key min/max bound ranks complete blocks.",
            "These rankings do not evaluate FlashNext's learned QSA selector or a deployed predictor. No latency or model-quality measurements.",
            "Query heads align by index across dense Qwen3 layers as a proxy. FlashNext QSA shares one selection across heads and occurs every fourth layer; the immediately preceding dense layer here is a different source.",
            "Step zero initializes temporal history. The current target ranking is used only as the comparison set.",
        ],
        "policies": {
            "rank": "Descending fused selector score, with ascending block ID to break exact score ties.",
            "temporal": "Previous decode step's ranking from the same layer, truncated to the budget.",
            "layer": "Current decode step's ranking from the immediately preceding layer, truncated to the budget.",
            "interleave": "Alternating unique admissions from temporal then preceding-layer ranking. Each source advances past duplicates in its own ranking. No target-driven fill.",
            "source_comparison": "Intersection and unique hits compare the two individually truncated predictions at equal budget.",
            "target_rank_quartile_retention": "Fraction of each current target rank quartile found in the budgeted prediction.",
            "source_rank_quartile_retention": "Fraction of each complete source ranking quartile found in the current target; independent of the prediction budget.",
        },
        "snapshots": [], "experiments": [],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(result, separators=(",", ":")) + "\n")

    original = modeling.apply_rotary_pos_emb
    captured = {}
    calls = 0
    capture_enabled = False

    def hook(*args, **kwargs):
        nonlocal calls
        q, k = original(*args, **kwargs)
        if capture_enabled:
            if calls in layers:
                captured[calls] = q[0, :, -1].detach().contiguous()
            calls += 1
        return q, k

    modeling.apply_rotary_pos_emb = hook
    try:
        with torch.inference_mode():
            for context in args.contexts:
                tokens = tokenizer(PROMPT * (context // 30 + 1), return_tensors="pt").input_ids[:, :context].cuda()
                if tokens.shape[1] != context:
                    raise RuntimeError("authored prompt did not yield the requested context length")
                past = None
                previous = {}
                for start in range(0, context, 1024):
                    out = model(tokens[:, start:start + 1024], past_key_values=past,
                                use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                print(json.dumps({"prefill_done": context}), flush=True)
                for step in range(args.steps):
                    calls = 0
                    captured.clear()
                    capture_enabled = True
                    out = model(token, past_key_values=past, use_cache=True, logits_to_keep=1)
                    capture_enabled = False
                    if calls != layer_count or set(captured) != set(layers):
                        raise RuntimeError("rotary capture did not match the expected Qwen3 layer order")
                    past = out.past_key_values
                    token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                    current, ties, complete_blocks = {}, {}, {}
                    for layer in layers:
                        current[layer], ties[layer], complete_blocks[layer] = ranked_ids(
                            captured[layer], past.layers[layer].keys[0], block_size, args.topk,
                        )
                    result["snapshots"].append({"context": context, "step": step,
                                                "ranked_ids_by_layer": current,
                                                "score_ties_by_layer": ties,
                                                "complete_blocks_by_layer": complete_blocks})
                    if previous:
                        for layer in targets:
                            for numerator, denominator in fractions:
                                budget = args.topk * numerator // denominator
                                result["experiments"].append({
                                    "context": context, "step": step, "layer": layer,
                                    "preceding_layer": layer - 1,
                                    "budget_fraction": numerator / denominator, "budget": budget,
                                    "heads": compare(current[layer], previous[layer], current[layer - 1],
                                                     budget, quartile_edges),
                                })
                    previous = current
                    save()
                    print(json.dumps({"context": context, "step_done": step,
                                      "cases": len(result["experiments"])}), flush=True)
                del out, past, tokens, token
                captured.clear()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        modeling.apply_rotary_pos_emb = original


if __name__ == "__main__":
    main()
