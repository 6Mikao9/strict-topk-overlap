"""Split-block sparse attention for native grouped-query KV layouts.

Q is [query_heads, dim]. K and V are [kv_heads, blocks, tokens, dim].
Adjacent query heads share a KV head. Each row of IDs contains a nonempty
set of unique, valid block IDs. Inputs are contiguous and finite.
"""

import torch
import triton
import triton.language as tl

from .kernels import _allocate_state, merge_partials


@triton.jit
def _bound_score(Q, LO, HI, SCORE, N: tl.constexpr, D: tl.constexpr,
                 G: tl.constexpr, NN: tl.constexpr, DD: tl.constexpr):
    h = tl.program_id(0)
    ns = tl.program_id(1) * NN + tl.arange(0, NN)
    ds = tl.arange(0, DD)
    q = tl.load(Q + h * D + ds, ds < D, 0.).to(tl.float32)
    kv = h.to(tl.int64) // G
    off = (kv * N + ns[:, None]) * D + ds[None, :]
    ptr = tl.where(q[None, :] >= 0., HI + off, LO + off)
    bound = tl.load(ptr, (ns[:, None] < N) & (ds[None, :] < D), 0.)
    tl.store(SCORE + h * N + ns, tl.sum(bound * q[None, :], 1), ns < N)


@triton.jit
def _partial(Q, K, V, A, P, SLOT, M, L, U,
             N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
             G: tl.constexpr, SA: tl.constexpr, SP: tl.constexpr,
             PP: tl.constexpr, BB: tl.constexpr, DD: tl.constexpr,
             REPAIR: tl.constexpr):
    h, s = tl.program_id(0), tl.program_id(1)
    row = h * SA + s
    bid = tl.load(A + row)
    slot = -1
    if REPAIR:
        ps = tl.arange(0, PP)
        p = tl.load(P + h * SP + ps, ps < SP, -1)
        slot = tl.max(tl.where((ps < SP) & (p == bid), ps, -1), 0)
        tl.store(SLOT + row, slot)
    if slot < 0:
        bs = tl.arange(0, BB)
        ds = tl.arange(0, DD)
        kv = h.to(tl.int64) // G
        off = ((kv * N + bid) * B + bs[:, None]) * D + ds[None, :]
        valid = (bs[:, None] < B) & (ds[None, :] < D)
        q = tl.load(Q + h * D + ds, ds < D, 0.).to(tl.float32)
        k = tl.load(K + off, valid, 0.).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * (D ** -0.5)
        score = tl.where(bs < B, score, -float("inf"))
        m = tl.max(score, 0)
        w = tl.exp(score - m)
        v = tl.load(V + off, valid, 0.).to(tl.float32)
        tl.store(M + row, m)
        tl.store(L + row, tl.sum(w, 0))
        tl.store(U + row * D + ds, tl.sum(w[:, None] * v, 0), ds < D)


@triton.jit
def _merge_repair(PM, PL, PU, AM, AL, AU, SLOT, OUT,
                  D: tl.constexpr, SA: tl.constexpr, SP: tl.constexpr,
                  AA: tl.constexpr, PP: tl.constexpr, DD: tl.constexpr,
                  UNION: tl.constexpr):
    h = tl.program_id(0)
    aa = tl.arange(0, AA)
    ds = tl.arange(0, DD)
    ar = h * SA + aa
    slot = tl.load(SLOT + ar, aa < SA, -1)
    if UNION:
        ps = tl.arange(0, PP)
        pr = h * SP + ps
        miss = (slot < 0) & (aa < SA)
        pm = tl.load(PM + pr, ps < SP, -float("inf"))
        am = tl.load(AM + ar, miss, -float("inf"))
        m = tl.maximum(tl.max(pm, 0), tl.max(am, 0))
        px = tl.exp(pm - m)
        ax = tl.exp(am - m)
        pl = tl.load(PL + pr, ps < SP, 0.)
        al = tl.load(AL + ar, miss, 0.)
        pu = tl.load(PU + pr[:, None] * D + ds[None, :],
                     (ps[:, None] < SP) & (ds[None, :] < D), 0.)
        au = tl.load(AU + ar[:, None] * D + ds[None, :],
                     miss[:, None] & (ds[None, :] < D), 0.)
        z = tl.sum(px * pl, 0) + tl.sum(ax * al, 0)
        u = tl.sum(px[:, None] * pu, 0) + tl.sum(ax[:, None] * au, 0)
    else:
        # Gather hit partials directly from prediction storage. The first
        # repair kernel writes Q/K/V partials only for actual misses.
        hit = slot >= 0
        pr = h * SP + slot
        mp = tl.where(hit, PM + pr, AM + ar)
        lp = tl.where(hit, PL + pr, AL + ar)
        up = tl.where(hit, PU + pr * D, AU + ar * D)
        mm = tl.load(mp, aa < SA, -float("inf"))
        xx = tl.exp(mm - tl.max(mm, 0))
        ll = tl.load(lp, aa < SA, 0.)
        uu = tl.load(up[:, None] + ds[None, :],
                     (aa[:, None] < SA) & (ds[None, :] < D), 0.)
        z = tl.sum(xx * ll, 0)
        u = tl.sum(xx[:, None] * uu, 0)
    tl.store(OUT + h * D + ds, u / z, ds < D)


def _validate_ids(q, ids):
    if (ids.ndim != 2 or ids.shape[0] != q.shape[0] or ids.shape[1] == 0
            or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("ids must be a nonempty int32/int64 matrix with one row per query head")
    if ids.device != q.device or not ids.is_contiguous():
        raise ValueError("ids must be contiguous and on the query's CUDA device")


def _shape(q, k, v, ids):
    if q.ndim != 2 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("expected Q[Hq,D] and K/V[Hkv,N,B,D]")
    hkv, n, b, d = k.shape
    hq, qd = q.shape
    if min(hq, hkv, n, b, d) <= 0 or qd != d or hq % hkv != 0:
        raise ValueError("expected nonempty shapes, matching head dimensions and Hq divisible by Hkv")
    if not q.is_cuda or any(x.device != q.device or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("Q/K/V must be contiguous and on the same CUDA device")
    _validate_ids(q, ids)
    return hq, n, b, d, hq // hkv, ids.shape[1]


def partials(q, k, v, ids):
    """Compute one FP32 softmax partial per query head and block."""
    hq, n, b, d, group, sa = _shape(q, k, v, ids)
    state = _allocate_state(hq, sa, d, q.device)
    _partial[(hq, sa)](
        q, k, v, ids, None, None, state.m, state.l, state.u,
        n, b, d, group, sa, 0, 1, triton.next_power_of_2(b),
        triton.next_power_of_2(d), False, num_warps=4,
    )
    return state


def exact(q, k, v, selected):
    """Attend to the selected blocks using parallel partials and a merge."""
    return merge_partials(partials(q, k, v, selected))


def select_blocks(q, lo, hi, topk):
    """Select min/max bound top-k using native FP32 metadata [Hkv,N,D]."""
    if q.ndim != 2 or lo.ndim != 3 or hi.shape != lo.shape:
        raise ValueError("expected Q[Hq,D] and LO/HI[Hkv,N,D]")
    hkv, n, d = lo.shape
    hq, qd = q.shape
    if min(hq, hkv, n, d) <= 0 or qd != d or hq % hkv != 0:
        raise ValueError("expected nonempty shapes, matching head dimensions and Hq divisible by Hkv")
    if lo.dtype != torch.float32 or hi.dtype != torch.float32:
        raise ValueError("lo and hi must be FP32")
    if not q.is_cuda or any(x.device != q.device or not x.is_contiguous() for x in (q, lo, hi)):
        raise ValueError("Q/LO/HI must be contiguous and on the same CUDA device")
    scores = torch.empty((hq, n), device=q.device, dtype=torch.float32)
    dd = triton.next_power_of_2(d)
    nn = max(1, min(32, 2048 // dd))
    _bound_score[(hq, triton.cdiv(n, nn))](q, lo, hi, scores, n, d, hq // hkv, nn, dd, num_warps=4)
    return scores.topk(topk, sorted=False).indices.to(torch.int32)


def _repair(q, k, v, selected, predicted, state, union):
    hq, n, b, d, group, sa = _shape(q, k, v, selected)
    _validate_ids(q, predicted)
    sp = predicted.shape[1]
    if state.m.shape != (hq, sp) or state.l.shape != (hq, sp) or state.u.shape != (hq, sp, d):
        raise ValueError("state must contain a softmax partial for every predicted block")
    scratch = _allocate_state(hq, sa, d, q.device)
    slot = torch.empty((hq, sa), device=q.device, dtype=torch.int32)
    out = torch.empty((hq, d), device=q.device, dtype=torch.float32)
    aa, pp = triton.next_power_of_2(sa), triton.next_power_of_2(sp)
    dd = triton.next_power_of_2(d)
    _partial[(hq, sa)](
        q, k, v, selected, predicted, slot, scratch.m, scratch.l, scratch.u,
        n, b, d, group, sa, sp, pp, triton.next_power_of_2(b), dd, True, num_warps=4,
    )
    _merge_repair[(hq,)](
        state.m, state.l, state.u, scratch.m, scratch.l, scratch.u, slot,
        out, d, sa, sp, aa, pp, dd, union, num_warps=4,
    )
    return out


def repair_saved(q, k, v, selected, predicted, state):
    """Reuse predicted hits and compute only selected misses."""
    return _repair(q, k, v, selected, predicted, state, union=False)


def repair_union(q, k, v, selected, predicted, state):
    """Attend to the union of selected and predicted blocks, once each."""
    return _repair(q, k, v, selected, predicted, state, union=True)
