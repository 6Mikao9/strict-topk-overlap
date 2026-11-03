"""Matched-selector CUDA Graph comparison with FlashAttention and FlashInfer.

FlashAttention gathers selected KV into a dense sequence per query head.
FlashInfer maps each query head to a virtual request into the original
shared GQA storage. Its fixed page plan is setup work and the page table
is rewritten on the GPU in every replay.
"""

import argparse
from importlib import import_module
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

from benchmarks.critical_path import _capture, _synthetic_inputs, reference
from benchmarks.run_benchmark import environment, percentiles
from strict_topk import gqa
from tests.check_correctness import error_metrics


@triton.jit
def _gather(K, V, IDS, KO, VO, N: tl.constexpr, G: tl.constexpr,
            S: tl.constexpr, BLOCK_ELEMENTS: tl.constexpr, TILE: tl.constexpr):
    h, s = tl.program_id(0), tl.program_id(1)
    i = tl.program_id(2) * TILE + tl.arange(0, TILE)
    p = tl.load(IDS + h * S + s)
    kv = h.to(tl.int64) // G
    src = ((kv * N + p) * BLOCK_ELEMENTS) + i
    dst = (h.to(tl.int64) * S + s) * BLOCK_ELEMENTS + i
    valid = i < BLOCK_ELEMENTS
    tl.store(KO + dst, tl.load(K + src, valid, 0.), valid)
    tl.store(VO + dst, tl.load(V + src, valid, 0.), valid)


@triton.jit
def _page_table(IDS, PAGE, S: tl.constexpr, N: tl.constexpr,
                G: tl.constexpr, SS: tl.constexpr):
    h = tl.program_id(0)
    i = tl.arange(0, SS)
    off = h * S + i
    bid = tl.load(IDS + off, i < S, 0)
    tl.store(PAGE + off, bid + (h // G) * N, i < S)


def _load_flash_attn():
    try:
        module = import_module("flash_attn")
    except (ImportError, OSError):
        return None, None
    fn = getattr(module, "flash_attn_func", None)
    return (fn, getattr(module, "__version__", None)) if callable(fn) else (None, None)


def vendor_attention(q, k, v, a, flash_attn_func=None):
    """Gather per-query-head selected pages and run FlashAttention or Flash SDPA."""

    h, d = q.shape
    hkv, n, b, _ = k.shape
    s = a.shape[1]
    kp = torch.empty((h, s * b, 1, d), device=k.device, dtype=k.dtype)
    vp = torch.empty_like(kp)
    tile = 1024
    _gather[(h, s, triton.cdiv(b * d, tile))](k, v, a, kp, vp, n, h // hkv, s, b * d, tile)
    query = q.view(h, 1, 1, d)
    if flash_attn_func is not None:
        return flash_attn_func(query, kp, vp, dropout_p=0., causal=False).view(h, d)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query, kp.transpose(1, 2), vp.transpose(1, 2), dropout_p=0., is_causal=False,
        ).view(h, d)


class PagedDecode:
    """Fixed-shape FlashInfer decode with mutable GPU page IDs."""

    def __init__(self, q, k, v, a):
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        h, self.d = q.shape
        hkv, self.n, b, _ = k.shape
        self.s = a.shape[1]
        self.group = h // hkv
        self.k = k.view(hkv * self.n, b, 1, self.d)
        self.v = v.view_as(self.k)
        self.indices = torch.empty((h * self.s,), device=q.device, dtype=torch.int32)
        indptr = torch.arange(0, (h + 1) * self.s, self.s, dtype=torch.int32)
        lengths = torch.full((h,), b, dtype=torch.int32)
        workspace = torch.zeros((128 * 1024 * 1024,), device=q.device, dtype=torch.uint8)
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_cuda_graph=True, use_tensor_cores=False,
            paged_kv_indptr_buffer=torch.empty_like(indptr, device=q.device),
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=torch.empty_like(lengths, device=q.device),
        )
        self._write_pages(a)
        self.wrapper.plan(
            indptr, self.indices.clone(), lengths, 1, 1, self.d, b,
            q_data_type=q.dtype, kv_data_type=k.dtype,
        )

    def _write_pages(self, a):
        _page_table[(a.shape[0],)](
            a, self.indices, self.s, self.n, self.group, triton.next_power_of_2(self.s),
        )

    def __call__(self, q, a):
        self._write_pages(a)
        return self.wrapper.run(q.view(-1, 1, self.d), (self.k, self.v)).view_as(q)


def _changed_ids_check(q, k, v, a, run):
    ids = a.clone()
    graph, out = _capture(lambda: run(ids))
    ids.copy_((a + 1) % k.shape[1])
    graph.replay()
    torch.cuda.synchronize()
    return error_metrics(out, reference(q, k, v, ids))


def run_case(q, k, v, lo, hi, topk, replays, batches, seed, use_paged, flash_attn_func=None):
    def select():
        return gqa.select_blocks(q, lo, hi, topk)

    a = select()
    expected = reference(q, k, v, a)
    gather_method = "gather_flash_attn" if flash_attn_func is not None else "gather_torch_flash"
    methods = {
        "serial_gqa": lambda: gqa.exact(q, k, v, select()),
        gather_method: lambda: vendor_attention(q, k, v, select(), flash_attn_func),
    }
    dynamic = {}
    if use_paged:
        paged = PagedDecode(q, k, v, a)
        dynamic["paged_flashinfer"] = _changed_ids_check(q, k, v, a, lambda ids: paged(q, ids))
        methods["paged_flashinfer"] = lambda: paged(q, select())

    graphs = {}
    rows = {}
    for name, fn in methods.items():
        graph, out = _capture(fn)
        graph.replay()
        torch.cuda.synchronize()
        numeric = error_metrics(out, expected)
        rows[name] = {"numerics_vs_fp64": numeric, "raw_batch_ms": []}
        if name in dynamic:
            rows[name]["changed_graph_ids_vs_fp64"] = dynamic[name]
        graphs[name] = (graph, out)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rng = random.Random(seed)
    for _ in range(batches):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graph, _ = graphs[name]
            for _ in range(10):
                graph.replay()
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            rows[name]["raw_batch_ms"].append(start.elapsed_time(end))
    for row in rows.values():
        row.update(percentiles([elapsed / replays for elapsed in row["raw_batch_ms"]]))
        row["passed"] = row["numerics_vs_fp64"]["passed"]
        if "changed_graph_ids_vs_fp64" in row:
            row["passed"] &= row["changed_graph_ids_vs_fp64"]["passed"]
    return {
        "heads": q.shape[0], "kv_heads": k.shape[0], "blocks": k.shape[1],
        "block_size": k.shape[2], "dim": q.shape[1], "topk": topk,
        "replays_per_batch": replays, "methods": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/vendor_baseline.json")
    parser.add_argument("--context", type=int, default=32 * 1024)
    parser.add_argument("--topk", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--skip-paged", action="store_true")
    args = parser.parse_args()
    if min(args.replays, args.batches, *args.topk) <= 0 or max(args.topk) > args.context // 32:
        parser.error("positive replay and batch counts and valid top-k values are required")
    flash_attn_func, flash_attn_version = _load_flash_attn()
    env = environment()
    env.update({
        "flash_attn": flash_attn_version,
        "flash_attn_available": flash_attn_func is not None,
        "selector": "shared native-GQA fused min/max scores followed by torch.topk",
        "timing": "CUDA Graph batches with per-batch shuffled method order; percentiles of batch means",
        "limitations": [
            "Fixed synthetic inputs, one query token per head; no full model or host decode latency",
            "Compilation, capture, metadata construction and fixed-shape FlashInfer planning excluded",
            "Dense Flash attention materializes selected KV. FlashInfer maps virtual requests to shared physical pages",
            "Both vendor layouts treat query heads as independent requests and do not fuse GQA heads",
            "FP16 vendor outputs are compared with FP64; custom split-GQA outputs FP32",
        ],
    })
    if not args.skip_paged:
        import flashinfer

        env["flashinfer"] = flashinfer.__version__
    data = {"environment": env, "arguments": vars(args), "experiments": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = _synthetic_inputs(args.context, args.seed)
    for topk in args.topk:
        case = run_case(
            *inputs, topk, args.replays, args.batches, args.seed, not args.skip_paged,
            flash_attn_func,
        )
        data["experiments"].append(case)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({"topk": topk, "passed": case["passed"], "median_ms": {
            name: row["median_ms"] for name, row in case["methods"].items()
        }}), flush=True)
    data["passed"] = all(case["passed"] for case in data["experiments"])
    path.write_text(json.dumps(data, indent=2) + "\n")
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
