"""Static-shape CUDA Graph latency; captures selector, prediction and repair.

Graph inputs and predictions are fixed across replays. No autoregressive model,
changing sequence length, or request scheduler is captured.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch

from strict_topk import kernels as ops
from benchmarks.run_benchmark import Config, setup, time_cuda, environment, overlap_run, output_of
from tests.check_correctness import reference, error_metrics


def run_case(c, repeats=50, include_compact=False):
    q, k, v, p, a, selector = setup(c)
    expected = reference(q, k, v, a)
    selector_stream, speculative_stream = torch.cuda.Stream(), torch.cuda.Stream()
    methods = {"serial_exact": lambda: ops.exact_attention(q, k, v, selector())}
    for mode in ["saved", "hybrid"] + (["compact"] if include_compact else []):
        def speculate(mode=mode):
            return ops.speculate(q, k, v, p, mode=mode, mass_threshold=.1)
        def finish(selected, state):
            return ops.repair(q, k, v, selected, p, state, min_remaining=1e-3, cancellation_tol=1e-5)
        def serial(speculate=speculate, finish=finish):
            selected = selector()
            state = speculate()
            return finish(selected, state)
        def overlap(speculate=speculate, finish=finish):
            return overlap_run(selector, speculate, finish, selector_stream, speculative_stream)
        methods[mode + "_serial"] = serial
        methods[mode + "_overlap"] = overlap
    methods["prefetch_overlap"] = lambda: overlap_run(selector, lambda: ops.touch_kv(k, v, p), lambda selected, unused: ops.exact_attention(q, k, v, selected), selector_stream, speculative_stream)
    order = list(methods)
    random.Random(c.seed).shuffle(order)
    rows = []
    for name in order:
        fn = methods[name]
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(4):
                fn()
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                result = fn()
        except RuntimeError as exc:
            rows.append({"config": asdict(c), "method": name, "capture_error": str(exc)})
            print(json.dumps(rows[-1]), flush=True)
            continue
        graph.replay()
        torch.cuda.synchronize()
        numeric = error_metrics(output_of(result), expected)
        row = {"config": asdict(c), "method": name, "numerics_vs_exact_topk": numeric,
               **time_cuda(graph.replay, warmup=10, repeats=repeats)}
        if hasattr(result, "fallback"):
            row["fallback_heads"] = result.fallback.int().sum().item()
        rows.append(row)
        print(json.dumps({"method": name, "config": asdict(c), "median_ms": row["median_ms"], "passed": numeric["passed"]}), flush=True)
        del graph, result
    baseline = next((r["median_ms"] for r in rows if r["method"] == "serial_exact" and "median_ms" in r), None)
    if baseline is not None:
        for row in rows:
            if "median_ms" in row:
                row["speedup_vs_serial_exact"] = baseline / row["median_ms"]
    return {"config": asdict(c), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/graph_benchmark.json")
    parser.add_argument("--include-compact", action="store_true")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    configs = [Config(blocks=n, overlap=o, seed=args.seed) for n in [512, 2048] for o in [.5, .875, 1.]]
    configs += [Config(heads=32, blocks=1024, block_size=64, dim=128, topk=32, overlap=.875, seed=args.seed)]
    result = {"environment": environment(), "scope": "Static CUDA Graph replay of complete selector + attention/repair, fixed Q/K/V and prediction IDs", "experiments": []}
    result["environment"]["timing"] = "CUDA events around single CUDA Graph replays; compilation and capture excluded"
    result["environment"]["limitations"] = [x for x in result["environment"]["limitations"] if x != "No CUDA graphs"]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        result["experiments"].append(run_case(config, args.repeats, args.include_compact))
        path.write_text(json.dumps(result, indent=2) + "\n")
    rows = [row for experiment in result["experiments"] for row in experiment["rows"]]
    result["validation"] = {"passed": all(r.get("numerics_vs_exact_topk", {}).get("passed", False) for r in rows),
                             "capture_failures": sum("capture_error" in r for r in rows)}
    path.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["validation"]["passed"] else 1)
