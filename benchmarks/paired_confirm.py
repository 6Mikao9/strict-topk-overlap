"""Randomized ABBA graph timings with an adjacent exact control in every pair."""

import argparse
import json
from pathlib import Path
import random
import statistics

import torch

from benchmarks.critical_path import _capture, _prediction, _synthetic_inputs, reference
from benchmarks.run_benchmark import environment, overlap_run
from strict_topk import gqa, gqa_light
from tests.check_correctness import error_metrics


def _replay_interleaved(first, second, repeats):
    for _ in range(repeats):
        first.replay()
        second.replay()


def _warm_device(baseline, saved, start, end, chunk_replays, warmup_ms):
    raw_ms = []
    elapsed = 0.
    while elapsed < warmup_ms:
        start.record()
        _replay_interleaved(baseline, saved, chunk_replays)
        end.record()
        end.synchronize()
        raw_ms.append(start.elapsed_time(end))
        elapsed += raw_ms[-1]
    return {"raw_chunk_ms": raw_ms, "total_ms": elapsed,
            "replays_per_graph": chunk_replays * len(raw_ms)}


def _interval_ms(graph, start, end, replays):
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def confirm_case(q, k, v, p, selector, metadata=None, *, replays=100, rounds=5,
                 warmup_ms=250., warmup_chunk_replays=2000, pair_warmup_replays=50,
                 expected_miss_fraction=1., seed=20261103):
    """Compare graphs on fixed data, with an external estimate for auto repair."""
    if min(replays, rounds, warmup_chunk_replays) <= 0 or min(warmup_ms, pair_warmup_replays) < 0:
        raise ValueError("invalid timing counts or warmup duration")
    if not 0 <= expected_miss_fraction <= 1:
        raise ValueError("expected_miss_fraction must be in [0, 1]")
    with torch.cuda.device(q.device):
        return _confirm_case(q, k, v, p, selector, metadata, replays, rounds,
                             warmup_ms, warmup_chunk_replays, pair_warmup_replays,
                             expected_miss_fraction, seed)


def _confirm_case(q, k, v, p, selector, metadata, replays, rounds,
                  warmup_ms, warmup_chunk_replays, pair_warmup_replays,
                  expected_miss_fraction, seed):
    a = selector()
    miss = ~(p[:, :, None] == a[:, None, :]).any(dim=1)
    hits = (~miss).sum(dim=1).tolist()
    exact = reference(q, k, v, a)
    union = reference(q, k, v, torch.cat((p, a), dim=1),
                      torch.cat((torch.ones_like(p, dtype=torch.bool), miss), dim=1))
    select_stream, predict_stream = torch.cuda.Stream(), torch.cuda.Stream()

    def prepare():
        return gqa.partials(q, k, v, p)

    def saved(selected, state):
        return gqa.repair_saved(q, k, v, selected, p, state)

    def light(selected, state):
        return gqa_light.repair_saved_light(q, k, v, selected, p, state)

    def auto(selected, state):
        return gqa_light.repair_saved_auto(q, k, v, selected, p, state, expected_miss_fraction)

    def union_repair(selected, state):
        return gqa.repair_union(q, k, v, selected, p, state)

    def concurrent(finish):
        return lambda: overlap_run(selector, prepare, finish, select_stream, predict_stream)

    candidates = {
        "serial_exact": (lambda: gqa.exact(q, k, v, selector()), exact),
        "saved_overlap": (concurrent(saved), exact),
        "light_overlap": (concurrent(light), exact),
        "auto_overlap": (concurrent(auto), exact),
        "union_overlap": (concurrent(union_repair), union),
        "zero_repair_overlap": (concurrent(lambda selected, state: (selected, state)), None),
    }
    captured = {}
    validation = {}
    for name, (fn, expected) in candidates.items():
        graph, output = _capture(fn)
        captured[name] = (graph, output)
        graph.replay()
        torch.cuda.synchronize()
        if expected is None:
            validation[name] = {"passed": torch.equal(output[0].sort(dim=1).values, a.sort(dim=1).values)}
        else:
            validation[name] = error_metrics(output, expected)

    baseline_name = "serial_exact"
    baseline = captured[baseline_name][0]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    warmup = _warm_device(baseline, captured["saved_overlap"][0], start, end,
                          warmup_chunk_replays, warmup_ms)
    rng = random.Random(seed)
    pairs = {name: [] for name in candidates if name != baseline_name}
    for round_index in range(rounds):
        order = list(pairs)
        rng.shuffle(order)
        for pair_index, name in enumerate(order):
            left, right = baseline_name, name
            if rng.getrandbits(1):
                left, right = right, left
            sequence = [left, right, right, left]
            _replay_interleaved(captured[left][0], captured[right][0], pair_warmup_replays)
            intervals = [_interval_ms(captured[label][0], start, end, replays) for label in sequence]
            baseline_ms = statistics.mean(elapsed for label, elapsed in zip(sequence, intervals)
                                          if label == baseline_name) / replays
            candidate_ms = statistics.mean(elapsed for label, elapsed in zip(sequence, intervals)
                                           if label == name) / replays
            pairs[name].append({"round": round_index, "order_in_round": pair_index,
                                "sequence": sequence, "raw_interval_ms": intervals,
                                "baseline_ms": baseline_ms, "candidate_ms": candidate_ms,
                                "baseline_over_candidate": baseline_ms / candidate_ms})

    summaries = []
    for name, measurements in pairs.items():
        ratios = [pair["baseline_over_candidate"] for pair in measurements]
        summaries.append({
            "method": name, "pairs": measurements,
            "median_baseline_ms": statistics.median(pair["baseline_ms"] for pair in measurements),
            "median_candidate_ms": statistics.median(pair["candidate_ms"] for pair in measurements),
            "median_paired_ratio": statistics.median(ratios),
            "min_paired_ratio": min(ratios), "max_paired_ratio": max(ratios),
            "pairs_ratio_gt_one": sum(ratio > 1. for ratio in ratios),
        })
    return {
        "config": {**(metadata or {}), "heads": q.shape[0], "kv_heads": k.shape[0],
                   "blocks": k.shape[1], "block_size": k.shape[2], "dim": q.shape[1],
                   "topk": a.shape[1], "predict_k": p.shape[1], "hits_per_head": hits,
                   "expected_miss_fraction": expected_miss_fraction,
                   "overlap_mean": sum(hits) / (len(hits) * a.shape[1])},
        "timing": {"replays_per_interval": replays, "rounds": rounds, "seed": seed,
                   "pair_warmup_replays_per_graph": pair_warmup_replays, "warmup": warmup},
        "validation": validation, "rows": summaries,
        "passed": all(row["passed"] for row in validation.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--warmup-ms", type=float, default=250.)
    parser.add_argument("--output", default="results/paired_confirm.json")
    estimate = parser.add_mutually_exclusive_group()
    estimate.add_argument("--expected-miss-fraction", type=float, default=.2,
                          help="Fixed miss estimate used by auto repair (default: 0.2)")
    estimate.add_argument("--idealized-estimate", action="store_true",
                          help="Use one minus requested synthetic overlap as the estimate")
    args = parser.parse_args()
    env = environment()
    env["timing"] = "Randomized ABBA/BAAB adjacent intervals of CUDA Graph replays; no clock locking"
    env["selector"] = "Native GQA min/max bound scores plus torch.topk; common to all paths"
    env["limitations"] = [
        "Fixed-input graph replays, with compilation, capture, metadata and prediction construction excluded",
        "Pair timings are replay batch means and can include host submission gaps",
        "Union computes attention on selected UNION predicted, and zero-repair emits no attention output",
        "Pairs within one process share its thermal, clock and cache conditions",
        "The optional idealized estimate uses synthetic overlap known during setup, which is unavailable to a live predictor",
    ]
    data = {"environment": env, "arguments": vars(args), "experiments": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = {
        512: [(32, .8125), (32, 1.), (64, .5), (64, 1.), (128, .8), (128, 1.)],
        1024: [(32, .5), (32, 1.), (64, .95)],
        2048: [(64, .8), (64, 1.), (128, .8)],
    }
    if args.profile == "smoke":
        configs = {512: [(32, .8125)]}
    for blocks, options in configs.items():
        context = blocks * 32
        q, k, v, lo, hi = _synthetic_inputs(context, args.seed)
        for topk, overlap in options:
            def selector():
                return gqa.select_blocks(q, lo, hi, topk)

            p = _prediction(selector(), blocks, topk, round(topk * overlap))
            miss_estimate = 1. - overlap if args.idealized_estimate else args.expected_miss_fraction
            meta = {"context": context, "requested_overlap": overlap,
                    "expected_miss_fraction_source": "idealized_requested_overlap" if args.idealized_estimate else "fixed"}
            case = confirm_case(q, k, v, p, selector, meta,
                                replays=args.replays, rounds=args.rounds, warmup_ms=args.warmup_ms,
                                expected_miss_fraction=miss_estimate, seed=args.seed)
            data["experiments"].append(case)
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps({"config": case["config"], "passed": case["passed"]}), flush=True)
    data["passed"] = all(case["passed"] for case in data["experiments"])
    path.write_text(json.dumps(data, indent=2) + "\n")
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
