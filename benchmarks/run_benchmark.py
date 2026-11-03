"""Reproducible eager CUDA microbenchmarks, including selector and stream joins.

Selection uses a block-key min/max bound plus torch.topk. This is a controlled
selector proxy, not a reproduction of the PRR selector or an LLM decode loop.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import torch
import triton

from strict_topk import kernels as ops
from tests.check_correctness import reference, error_metrics


@dataclass
class Config:
    heads: int = 16
    blocks: int = 512
    block_size: int = 32
    dim: int = 64
    topk: int = 16
    overlap: float = 0.5
    seed: int = 20261103
    selector: str = 'torch'
    heterogeneous: bool = False


def setup(config):
    c = config
    gen = torch.Generator(device="cuda").manual_seed(c.seed)
    q = torch.randn(c.heads, c.dim, generator=gen, device="cuda", dtype=torch.float16)
    k = torch.randn(c.heads, c.blocks, c.block_size, c.dim, generator=gen, device="cuda", dtype=torch.float16)
    v = torch.randn(k.shape, generator=gen, device="cuda", dtype=torch.float16)
    # Metadata construction is a prefill cost, deliberately outside decoding timing.
    lo, hi = k.amin(dim=2).float(), k.amax(dim=2).float()
    qf = q.float().unsqueeze(1)

    def selector():
        if c.selector == 'fused':
            from strict_topk.selector import select_blocks
            return select_blocks(q,lo,hi,c.topk)
        scores = torch.where(qf >= 0, hi * qf, lo * qf).sum(-1)
        return scores.topk(c.topk, sorted=False).indices.to(torch.int32)

    a = selector()
    overlap = round(c.topk * c.overlap)
    # Construct an exact controlled hit rate. Prediction itself is not timed;
    # this emulates an already available previous-step block list.
    predicted_rows = []
    for head_index,row in enumerate(a.cpu().tolist()):
        if c.heterogeneous:
            overlap = c.topk if head_index % 2 else 0
        not_selected = [j for j in range(c.blocks) if j not in row]
        predicted_rows.append(row[:overlap] + not_selected[:c.topk - overlap])
    p = torch.tensor(predicted_rows, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    return q, k, v, p, a, selector


def percentiles(xs):
    ordered = sorted(xs)
    def percentile(q):
        pos = (len(ordered) - 1) * q
        low = math.floor(pos)
        high = math.ceil(pos)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    return {"p10_ms": percentile(.1), "median_ms": statistics.median(xs), "p90_ms": percentile(.9), "p95_ms": percentile(.95), "mean_ms": statistics.mean(xs)}


def time_cuda(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    raw, wall = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        ts = time.perf_counter()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - ts) * 1000)
        raw.append(start.elapsed_time(end))
    return {**percentiles(raw), "raw_ms": raw, "wall_including_sync_ms": wall}


def output_of(result):
    return result.output if hasattr(result, "output") else result


def overlap_run(selector, speculative, finish, select_stream, predict_stream):
    main = torch.cuda.current_stream()
    select_stream.wait_stream(main)
    predict_stream.wait_stream(main)
    with torch.cuda.stream(select_stream):
        a = selector()
    with torch.cuda.stream(predict_stream):
        state = speculative()
    main.wait_stream(select_stream)
    main.wait_stream(predict_stream)
    # CUDA caching allocator must know that the main stream consumes allocations.
    a.record_stream(main)
    def record_tensors(obj):
        if isinstance(obj, torch.Tensor):
            obj.record_stream(main)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                record_tensors(value)
    record_tensors(state)
    return finish(a, state)


def prediction_storage_bytes(prediction):
    storages = {}
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                visit(value)
    visit(prediction)
    return sum(storages.values())


def run_config(c, warmup, repeats, include_compact=False):
    q, k, v, p, a, selector = setup(c)
    expected = reference(q, k, v, a)
    select_stream, predict_stream = torch.cuda.Stream(), torch.cuda.Stream()
    methods = {}
    methods["serial_exact"] = lambda: ops.exact_attention(q, k, v, selector())
    methods["attention_only"] = lambda: ops.exact_attention(q, k, v, a)

    modes = [("saved", .1), ("hybrid", .05), ("hybrid", .1), ("hybrid", .2)]
    if include_compact:
        modes += [("compact", .05), ("compact", .1), ("compact", .2)]
    for mode, threshold in modes:
        label = mode if mode == "saved" else f"{mode}_tau{threshold:g}"
        def speculative(mode=mode, threshold=threshold):
            return ops.speculate(q, k, v, p, mode=mode, mass_threshold=threshold)
        def finish(selected, state):
            return ops.repair(q, k, v, selected, p, state, min_remaining=1e-3, cancellation_tol=1e-5)
        def serial(speculative=speculative, finish=finish):
            selected = selector()
            state = speculative()
            return finish(selected, state)
        def concurrent(speculative=speculative, finish=finish):
            return overlap_run(selector, speculative, finish, select_stream, predict_stream)
        methods[label + "_serial"] = serial
        methods[label + "_overlap"] = concurrent

    # PRR-style set-union semantic control. Implementation is ours, not PRR code.
    # Retain all predictions, add only missing selected blocks; no wrong removal.
    def union_speculate():
        return ops.block_partials(q, k, v, p)
    def union_finish(selected, state):
        missing = ~(selected[:, :, None] == p[:, None, :]).any(-1)
        extra = ops.block_partials(q, k, v, selected, active=missing)
        combined = type(state)(m=torch.cat((state.m, extra.m), 1), l=torch.cat((state.l, extra.l), 1), u=torch.cat((state.u, extra.u), 1))
        return ops.merge_partials(combined)
    methods["prr_union_overlap"] = lambda: overlap_run(selector, union_speculate, union_finish, select_stream, predict_stream)

    if hasattr(ops, "touch_kv"):
        def prefetch_finish(selected, unused):
            return ops.exact_attention(q, k, v, selected)
        methods["prefetch_overlap"] = lambda: overlap_run(selector, lambda: ops.touch_kv(k, v, p), prefetch_finish, select_stream, predict_stream)

    results = []
    selector_timing = time_cuda(selector, warmup, repeats)
    method_order = list(methods)
    random.Random(c.seed).shuffle(method_order)
    for name in method_order:
        fn = methods[name]
        # First call compiles kernels outside recorded timing.
        result = fn()
        torch.cuda.synchronize()
        numeric = error_metrics(output_of(result), expected)
        row = {"config": asdict(c), "method": name, "numerics_vs_exact_topk": numeric}
        if hasattr(result, "fallback"):
            row["fallback_heads"] = result.fallback.int().sum().item()
            row["remaining_per_head"] = result.remaining.tolist()
        if name == "prr_union_overlap":
            row["semantics"] = "attention(selected UNION predicted); not exact Top-K"
            union_ids = torch.tensor([list(dict.fromkeys(ar + pr)) for ar, pr in zip(a.cpu().tolist(), p.cpu().tolist())], device="cuda", dtype=torch.int32)
            row["numerics_vs_own_semantics"] = error_metrics(output_of(result), reference(q, k, v, union_ids))
        else:
            row["semantics"] = "attention(selected)"
        row.update(time_cuda(fn, warmup, repeats))
        # This includes all temporary torch-allocator tensors, not just state.
        del result
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        torch.cuda.synchronize()
        row["incremental_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated() - before
        del result
        row["selected_kv_bytes"] = c.heads * c.topk * c.block_size * c.dim * 2 * 2
        # These are analytic logical element reads, not HBM traffic counters.
        hit_count = round(c.topk * c.overlap)
        pred_state = c.heads * c.topk * (c.dim + 2) * 4
        row["predicted_partial_state_bytes_estimate"] = pred_state if name.startswith(("saved", "hybrid", "prr")) else 0
        if name.startswith(("saved", "prr")):
            row["logical_kv_block_reads_estimate"] = c.heads * (2 * c.topk - hit_count)
        elif name in ["serial_exact", "attention_only"]:
            row["logical_kv_block_reads_estimate"] = c.heads * c.topk
        elif name == "prefetch_overlap":
            row["logical_kv_block_reads_estimate"] = c.heads * 2 * c.topk
        elif name.startswith(("hybrid", "compact")):
            mode = name.split("_")[0]
            threshold = float(name.split("tau")[1].split("_")[0])
            pred = ops.speculate(q, k, v, p, mode=mode, mass_threshold=threshold)
            hit = (p[:, :, None] == a[:, None, :]).any(-1)
            low_hit = (hit & ~pred.high).any(-1)
            reread = ((~hit) & (~pred.high) & low_hit[:, None]).int().sum().item()
            row["low_wrong_kv_block_rereads_estimate"] = reread
            row["high_state_blocks"] = pred.high.int().sum().item()
            row["logical_kv_block_reads_estimate"] = c.heads * (2 * c.topk - hit_count) + reread + row.get("fallback_heads", 0) * c.topk
            # Low aggregate (m, l, u), absolute numerator envelope, and high mask.
            row["predicted_partial_state_bytes_estimate"] += c.heads * ((2 * c.dim + 2) * 4 + c.topk)
            row["prediction_live_tensor_storage_bytes"] = prediction_storage_bytes(pred)
            if mode == "compact":
                # Compact preparation makes a K-only statistics pass then KV.
                row["logical_v_block_reads_estimate"] = row.pop("logical_kv_block_reads_estimate")
                row["logical_k_block_reads_estimate"] = row["logical_v_block_reads_estimate"] + c.heads * c.topk
                row.pop("predicted_partial_state_bytes_estimate", None)
        results.append(row)
        print(json.dumps({"method": name, "config": asdict(c), "median_ms": row["median_ms"], "max_abs_error": numeric["max_abs_error"]}), flush=True)
    baseline = next(r["median_ms"] for r in results if r["method"] == "serial_exact")
    for row in results:
        row["speedup_vs_serial_exact"] = baseline / row["median_ms"]
        row["selector_only_median_ms"] = selector_timing["median_ms"]
    return {"config": asdict(c), "selector": selector_timing, "rows": results}


def environment():
    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory, "compute_capability": [props.major, props.minor],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "timing": "eager CUDA events, one invocation per sample, includes stream waits and host submission gaps, excludes compilation",
        "selector": "torch key-min/max q-sign bound plus torch.topk(sorted=False); metadata construction excluded",
        "limitations": ["Synthetic independent random Q/K/V", "No full LLM decode or task-quality measurement", "No CUPTI HBM/L2 traffic counters", "No CUDA graphs", "Prediction construction cost excluded, previous-step IDs assumed available", "Per-block FP32 states materialized in hybrid preparation; no compact-state peak-memory claim", "Method order shuffled deterministically by seed; repeated independent jobs recommended", "Baseline is the same split-block prototype, not a vendor-optimized sparse-attention kernel"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "main"], default="main")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--include-compact", action="store_true", help="Include the compact two-pass optimization when implemented")
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    if args.profile == "smoke":
        configs = [Config(heads=4, blocks=128, topk=8, overlap=.5, seed=args.seed)]
    else:
        configs = [Config(blocks=n, overlap=o, seed=args.seed) for n in [512, 2048] for o in [0., .5, .875, 1.]]
        configs += [Config(heads=32, blocks=1024, block_size=64, dim=128, topk=32, overlap=o, seed=args.seed) for o in [.5, .875]]
        configs += [Config(heads=1, blocks=2048, overlap=.875, seed=args.seed)]
    result = {"environment": environment(), "arguments": vars(args), "experiments": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        result["experiments"].append(run_config(config, args.warmup, args.repeats, args.include_compact))
        path.write_text(json.dumps(result, indent=2) + "\n")
    failures = []
    for experiment in result["experiments"]:
        for row in experiment["rows"]:
            check = row.get("numerics_vs_own_semantics", row["numerics_vs_exact_topk"])
            if not check["passed"]:
                failures.append({"config": row["config"], "method": row["method"], "error": check})
    result["validation"] = {"passed": not failures, "unexpected_failures": failures,
                             "note": "PRR union is validated against union semantics; its difference from exact Top-K is intentional."}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}", flush=True)
    if failures:
        raise SystemExit("Unexpected numerical validation failures; see output JSON")


if __name__ == "__main__":
    main()
