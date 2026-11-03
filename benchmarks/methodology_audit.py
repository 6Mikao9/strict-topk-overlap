"""Cross legacy and native GQA paths with single and batched replay timing."""

import argparse
import json
from pathlib import Path
import random

import torch

from benchmarks.critical_path import _capture, _prediction, _synthetic_inputs, _time_graph, reference
from benchmarks.run_benchmark import environment, output_of, overlap_run, time_cuda
from strict_topk import gqa, kernels
from strict_topk.fused import repair_fused
from strict_topk.selector import select_blocks
from tests.check_correctness import error_metrics


def run_case(context, topk, overlap, seed, replays, batches, single_replays):
    q, k, v, lo, hi = _synthetic_inputs(context, seed)
    groups = q.shape[0] // k.shape[0]
    expanded_k = k.repeat_interleave(groups, dim=0)
    expanded_v = v.repeat_interleave(groups, dim=0)
    expanded_lo = lo.repeat_interleave(groups, dim=0)
    expanded_hi = hi.repeat_interleave(groups, dim=0)

    def legacy_selector():
        return select_blocks(q, expanded_lo, expanded_hi, topk)

    def native_selector():
        return gqa.select_blocks(q, lo, hi, topk)

    selected = native_selector()
    if not torch.equal(selected, legacy_selector()):
        raise RuntimeError("legacy and native selectors returned different ordered block IDs")
    hit_count = round(topk * overlap)
    p = _prediction(selected, k.shape[1], topk, hit_count)
    expected = reference(q, k, v, selected)
    selector_stream, prediction_stream = torch.cuda.Stream(), torch.cuda.Stream()

    def overlap_graph(selector, prepare, finish):
        return lambda: overlap_run(selector, prepare, finish, selector_stream, prediction_stream)

    def legacy_saved_prepare():
        return kernels.speculate(q, expanded_k, expanded_v, p, mode="saved", mass_threshold=.2)

    def legacy_saved_repair(a, state):
        return kernels.repair(q, expanded_k, expanded_v, a, p, state)

    def legacy_compact_prepare():
        # Match the round-one Qwen replay at epsilon=0, including scalar stats.
        return kernels.speculate(q, expanded_k, expanded_v, p, mode="compact",
                                 mass_threshold=.2, keep_mass=True)

    def legacy_compact_repair(a, state):
        return repair_fused(q, expanded_k, expanded_v, a, p, state, max_extra_mass=0.)

    def native_prepare():
        return gqa.partials(q, k, v, p)

    def native_repair(a, state):
        return gqa.repair_saved(q, k, v, a, p, state)

    # Cross the new attention paths with the same selector used by the old
    # paths, and measure their native selector variants separately.
    methods = {
        "legacy_exact": ("expanded", "legacy", lambda: kernels.exact_attention(q, expanded_k, expanded_v, legacy_selector())),
        "legacy_saved_overlap": ("expanded", "legacy", overlap_graph(legacy_selector, legacy_saved_prepare, legacy_saved_repair)),
        "legacy_compact_fused_overlap": ("expanded", "legacy", overlap_graph(legacy_selector, legacy_compact_prepare, legacy_compact_repair)),
        "native_exact_shared_selector": ("native", "legacy", lambda: gqa.exact(q, k, v, legacy_selector())),
        "native_saved_overlap_shared_selector": ("native", "legacy", overlap_graph(legacy_selector, native_prepare, native_repair)),
        "native_exact": ("native", "native", lambda: gqa.exact(q, k, v, native_selector())),
        "native_saved_overlap": ("native", "native", overlap_graph(native_selector, native_prepare, native_repair)),
    }
    rng = random.Random(seed)
    order = list(methods)
    rng.shuffle(order)
    rows = []
    for name in order:
        layout, selector, fn = methods[name]
        graph, result = _capture(fn)
        graph.replay()
        torch.cuda.synchronize()
        numeric = error_metrics(output_of(result), expected)
        timing_order = ["single", "batched"]
        rng.shuffle(timing_order)
        timing = {}
        for style in timing_order:
            if style == "single":
                timing[style] = time_cuda(graph.replay, warmup=2, repeats=single_replays)
            else:
                timing[style] = _time_graph(graph, replays, batches)
        rows.append({
            "method": name, "kv_layout": layout, "selector": selector,
            "numerics_vs_exact_topk": numeric, "timing_order": timing_order, **timing,
            "single_vs_batched_median": timing["single"]["median_ms"] / timing["batched"]["median_ms"],
        })
        del graph, result

    return {
        "config": {
            "context": context, "blocks": k.shape[1], "block_size": k.shape[2],
            "dim": q.shape[1], "heads": q.shape[0], "kv_heads": k.shape[0],
            "topk": topk, "predict_k": topk, "hits_per_head": hit_count,
            "overlap": hit_count / topk, "seed": seed,
            "native_kv_bytes": (k.numel() + v.numel()) * k.element_size(),
            "expanded_kv_bytes": (expanded_k.numel() + expanded_v.numel()) * expanded_k.element_size(),
        },
        "ordered_selectors_match": True,
        "rows": rows,
        "passed": all(row["numerics_vs_exact_topk"]["passed"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--single-replays", type=int, default=20)
    parser.add_argument("--output", default="results/methodology_audit.json")
    args = parser.parse_args()
    if not 0 <= args.overlap <= 1 or min(args.replays, args.batches, args.single_replays) <= 0:
        parser.error("overlap must be in [0,1] and timing counts must be positive")
    env = environment()
    env["timing"] = "Same CUDA Graph per row. Single events enclose one replay; batched events enclose multiple replays."
    env["selector"] = "Same fused min/max bound and topk; legacy and native metadata layouts produce identical ordered IDs."
    env["limitations"] = [
        "Synthetic fixed Q/K/V and prediction IDs, not a recreation of model traces",
        "Compilation, capture, KV expansion, metadata and prediction construction excluded",
        "Replay timing can include host submission gaps and is not isolated kernel execution time",
        "Batched statistics summarize per-replay batch means, not individual requests",
    ]
    data = {"environment": env, "arguments": vars(args), "experiments": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for context, topk in ((2048, 8), (16384, 64), (65536, 128)):
        case = run_case(context, topk, args.overlap, args.seed, args.replays, args.batches, args.single_replays)
        data["experiments"].append(case)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({"config": case["config"], "passed": case["passed"]}), flush=True)
    data["passed"] = all(case["passed"] for case in data["experiments"])
    path.write_text(json.dumps(data, indent=2) + "\n")
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
