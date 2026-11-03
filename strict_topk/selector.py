"""Fused min/max block scoring for the synthetic selector."""

import torch
import triton
import triton.language as tl


@triton.jit
def _score(Q, LO, HI, SCORE, N: tl.constexpr, D: tl.constexpr,
           NN: tl.constexpr, DD: tl.constexpr):
    h = tl.program_id(0)
    ns = tl.program_id(1) * NN + tl.arange(0, NN)
    ds = tl.arange(0, DD)
    q = tl.load(Q + h * D + ds, ds < D, 0.).to(tl.float32)
    row = h.to(tl.int64) * N + ns
    offset = row[:, None] * D + ds[None, :]
    ptr = tl.where(q[None, :] >= 0., HI + offset, LO + offset)
    bound = tl.load(ptr, (ns[:, None] < N) & (ds[None, :] < D), 0.)
    score = tl.sum(bound * q[None, :], 1)
    tl.store(SCORE + row, score, ns < N)


def block_scores(q, lo, hi):
    """Compute FP32 min/max bound scores for contiguous CUDA inputs."""
    if lo.ndim != 3 or hi.shape != lo.shape:
        raise ValueError("lo and hi must have the same [heads, blocks, dim] shape")
    h, n, d = lo.shape
    if q.shape != (h, d) or min(h, n, d) <= 0:
        raise ValueError("q must have shape [heads, dim], with nonempty input dimensions")
    if lo.dtype != torch.float32 or hi.dtype != torch.float32:
        raise ValueError("lo and hi must be FP32")
    if not q.is_cuda or any(x.device != q.device or not x.is_contiguous() for x in (q, lo, hi)):
        raise ValueError("q, lo and hi must be contiguous on the same CUDA device")
    scores = torch.empty((h, n), device=q.device, dtype=torch.float32)
    dd = triton.next_power_of_2(d)
    nn = max(1, min(32, 2048 // dd))
    _score[(h, triton.cdiv(n, nn))](q, lo, hi, scores, n, d, nn, dd, num_warps=4)
    return scores


def select_blocks(q, lo, hi, topk):
    """Return the int32 block IDs selected by the fused score and torch.topk."""
    return block_scores(q, lo, hi).topk(topk, sorted=False).indices.to(torch.int32)
