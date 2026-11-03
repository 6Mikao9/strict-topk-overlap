"""Native GQA Graph replay, with paired complete and zero-repair paths."""

import argparse
import json
import math
from pathlib import Path
import random

import torch

from benchmarks.run_benchmark import environment, overlap_run, percentiles, time_cuda
from strict_topk import gqa
from tests.check_correctness import error_metrics


def reference(q, k, v, ids, include=None):
    """FP64 attention using the original quantized inputs and native KV groups."""
    groups = torch.arange(q.shape[0], device=q.device) // (q.shape[0] // k.shape[0])
    keys = k[groups[:, None], ids.long()].double()
    values = v[groups[:, None], ids.long()].double()
    logits = torch.einsum("hd,hsbd->hsb", q.double(), keys) / math.sqrt(q.shape[1])
    if include is not None:
        logits = logits.masked_fill(~include[:, :, None], -torch.inf)
    weights = logits.flatten(1).softmax(-1).view_as(logits)
    return torch.einsum("hsb,hsbd->hd", weights, values)


def _time_graph(graph, replays, batches):
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    raw_batch_ms = []
    for _ in range(batches):
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        raw_batch_ms.append(start.elapsed_time(end))
    return {
        **percentiles([elapsed / replays for elapsed in raw_batch_ms]),
        "raw_batch_ms": raw_batch_ms,
        "replays_per_batch": replays,
        "batches": batches,
    }


def _capture(fn):
    main = torch.cuda.current_stream()
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(main)
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            fn()
    main.wait_stream(warm_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def run_case(q, k, v, p, selector, metadata=None, *, replays=100, batches=5, seed=20261103):
    """Measure a fixed Q/K/V snapshot with an already available prediction."""
    if replays <= 0 or batches <= 0:
        raise ValueError("replays and batches must be positive")
    with torch.cuda.device(q.device):
        return _run_case(q, k, v, p, selector, metadata, replays, batches, seed)


def _run_case(q, k, v, p, selector, metadata, replays, batches, seed):
    a = selector()
    matches = p[:, :, None] == a[:, None, :]
    hits = matches.any(dim=1).sum(dim=1)
    coverage = (hits / a.shape[1]).tolist()
    config = {
        **(metadata or {}),
        "heads": q.shape[0], "kv_heads": k.shape[0], "blocks": k.shape[1],
        "block_size": k.shape[2], "dim": q.shape[1], "topk": a.shape[1],
        "predict_k": p.shape[1], "kv_tokens": k.shape[1] * k.shape[2],
        "overlap_mean": sum(coverage) / len(coverage),
        "overlap_min": min(coverage), "overlap_max": max(coverage),
        "hits_per_head": hits.tolist(),
    }
    exact = reference(q, k, v, a)
    union = reference(q, k, v, torch.cat((p, a), dim=1),
                      torch.cat((torch.ones_like(p, dtype=torch.bool), ~matches.any(dim=1)), dim=1))
    state = gqa.partials(q, k, v, p)
    select_stream, predict_stream = torch.cuda.Stream(), torch.cuda.Stream()

    def prepare():
        return gqa.partials(q, k, v, p)

    def saved(selected, prediction):
        return gqa.repair_saved(q, k, v, selected, p, prediction)

    def union_repair(selected, prediction):
        return gqa.repair_union(q, k, v, selected, p, prediction)

    def join_only():
        main = torch.cuda.current_stream()
        select_stream.wait_stream(main)
        predict_stream.wait_stream(main)
        main.wait_stream(select_stream)
        main.wait_stream(predict_stream)

    methods = {
        "serial_exact": ("full", lambda: gqa.exact(q, k, v, selector()), exact),
        "saved_overlap": ("full", lambda: overlap_run(selector, prepare, saved, select_stream, predict_stream), exact),
        "union_overlap": ("full", lambda: overlap_run(selector, prepare, union_repair, select_stream, predict_stream), union),
        "zero_repair_overlap": ("diagnostic", lambda: overlap_run(selector, prepare, lambda selected, prediction: (selected, prediction), select_stream, predict_stream), None),
        "selector_only": ("stage", selector, None),
        "prepare_only": ("stage", prepare, None),
        "saved_repair_only": ("stage", lambda: saved(a, state), exact),
        "union_repair_only": ("stage", lambda: union_repair(a, state), union),
        "join_only": ("diagnostic", join_only, None),
        "empty_graph": ("diagnostic", lambda: None, None),
    }
    order = list(methods)
    random.Random(seed).shuffle(order)
    rows = []
    for method in order:
        stage, fn, expected = methods[method]
        row = {"method": method, "stage": stage}
        if method in ("union_overlap", "union_repair_only"):
            row["semantics"] = "attention(selected UNION predicted)"
        elif expected is not None:
            row["semantics"] = "attention(selected)"
        try:
            graph, output = _capture(fn)
        except RuntimeError as exc:
            row.update(capture_error=str(exc), passed=False)
            rows.append(row)
            continue
        graph.replay()
        torch.cuda.synchronize()
        row["passed"] = True
        if expected is not None:
            metric = error_metrics(output, expected)
            row["numerics_vs_own_semantics"] = metric
            row["passed"] = metric["passed"]
            if method == "union_overlap":
                row["numerics_vs_exact_topk"] = error_metrics(output, exact)
        if method in ("selector_only", "zero_repair_overlap"):
            selected = output[0] if method == "zero_repair_overlap" else output
            row["passed"] = torch.equal(selected.sort(dim=1).values, a.sort(dim=1).values)
        row.update(_time_graph(graph, replays, batches))
        if method in ("serial_exact", "empty_graph"):
            row["single_replay_audit"] = time_cuda(graph.replay, warmup=5, repeats=20)
            if row["median_ms"] > 0:
                row["single_vs_batched_median"] = row["single_replay_audit"]["median_ms"] / row["median_ms"]
        rows.append(row)
        del graph, output

    complete = {row["method"]: row for row in rows if "median_ms" in row}
    if "serial_exact" in complete:
        baseline_ms = complete["serial_exact"]["median_ms"]
        for row in complete.values():
            if row["stage"] == "full" and row["median_ms"] > 0:
                row["speedup_vs_serial_exact"] = baseline_ms / row["median_ms"]
        if "zero_repair_overlap" in complete:
            zero = complete["zero_repair_overlap"]
            zero["fraction_of_serial_exact"] = zero["median_ms"] / baseline_ms
            zero["headroom_ms"] = baseline_ms - zero["median_ms"]
    return {"config": config, "rows": rows, "passed": all(row["passed"] for row in rows)}


def _prediction(selected, blocks, predict_k, hits):
    rows = []
    for row in selected.cpu().tolist():
        selected_ids = set(row)
        wrong = [block for block in range(blocks) if block not in selected_ids]
        rows.append(row[:hits] + wrong[:predict_k - hits])
    return torch.tensor(rows, dtype=selected.dtype, device=selected.device)


def _synthetic_inputs(context, seed):
    heads, kv_heads, block_size, dim = 16, 8, 32, 128
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(heads, dim, generator=gen, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_heads, context // block_size, block_size, dim, generator=gen,
                    device=q.device, dtype=q.dtype)
    v = torch.randn(k.shape, generator=gen, device=q.device, dtype=q.dtype)
    return q, k, v, k.amin(dim=2).float(), k.amax(dim=2).float()


def _synthetic_cases(profile):
    contexts = [16 * 1024] if profile == "smoke" else [16 * 1024, 32 * 1024, 64 * 1024]
    for context in contexts:
        if profile == "smoke":
            yield context, [(32, 32, .8), (32, 32, 1.)]
        else:
            core = [(topk, topk, hit) for topk in (32, 64, 128) for hit in (.5, .8, .95, 1.)]
            # Hold hits constant to isolate the extra speculative blocks.
            yield context, core + [(64, 32, .5), (64, 96, .5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", default="results/critical_path.json")
    args = parser.parse_args()
    env = environment()
    env["timing"] = "CUDA events around batches of CUDA Graph replays; statistics of batch means, not request latency percentiles"
    env["selector"] = "shared native GQA min/max metadata, fused scoring and torch.topk; metadata construction excluded"
    env["limitations"] = [
        "Static synthetic Q/K/V and controlled overlap; no full decode or task quality measurements",
        "Compilation, capture, metadata and prediction ID construction excluded",
        "Join-only is an empty-work graph control; CUDA may elide event-only dependencies",
        "Empty graph and exact baseline include single-replay event audits; submission gaps can differ from batched timing",
        "Standalone stages use separate graphs; subtracting their times does not decompose a complete path",
    ]
    data = {"environment": env, "arguments": vars(args), "experiments": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for context, configs in _synthetic_cases(args.profile):
        q, k, v, lo, hi = _synthetic_inputs(context, args.seed)
        for topk, predict_k, overlap in configs:
            def selector():
                return gqa.select_blocks(q, lo, hi, topk)

            p = _prediction(selector(), k.shape[1], predict_k, round(topk * overlap))
            meta = {"context": context, "requested_overlap": overlap}
            case = run_case(q, k, v, p, selector, meta, replays=args.replays,
                            batches=args.batches, seed=args.seed)
            data["experiments"].append(case)
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps({"config": case["config"], "passed": case["passed"]}), flush=True)
    data["passed"] = all(case["passed"] for case in data["experiments"])
    path.write_text(json.dumps(data, indent=2) + "\n")
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
