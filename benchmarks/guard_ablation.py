"""Expose cancellation failures and guard tradeoffs; failures without guards are data."""
import argparse
import json
from pathlib import Path

import torch

from strict_topk.kernels import speculate, repair
from tests.check_correctness import cases, reference, error_metrics
from benchmarks.run_benchmark import time_cuda, environment


def run(warmup=3, repeats=10):
    rows = []
    for name, q, k, v, p, a in cases(dtype=torch.float16):
        if name.startswith("random"):
            continue
        expected = reference(q, k, v, a)
        for threshold in [.05, .2, 1.]:
            state = speculate(q, k, v, p, mode="hybrid", mass_threshold=threshold)
            for min_remaining in [0., 1e-5, 1e-3, .1]:
                for cancellation_tol in [0., 1e-5]:
                    def fn():
                        return repair(q, k, v, a, p, state, min_remaining=min_remaining, cancellation_tol=cancellation_tol)
                    result = fn()
                    row = {"case": name, "mass_threshold": threshold, "min_remaining": min_remaining,
                           "cancellation_tol": cancellation_tol, "fallback_heads": result.fallback.int().sum().item(),
                           "remaining": result.remaining.tolist(), "high_blocks": state.high.int().sum().item(),
                           **error_metrics(result.output, expected), **time_cuda(fn, warmup, repeats)}
                    rows.append(row)
    return {"environment": environment(), "scope": "Repair-only timing; speculation excluded. Unguarded failures are expected exploratory results, not a pass/fail test.", "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/guard_ablation.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    result = run(args.warmup, args.repeats)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "failing_numerical_rows": sum(not r["passed"] for r in result["rows"])}))
