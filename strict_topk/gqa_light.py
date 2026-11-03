"""Single-launch saved-state repair for native GQA with few late misses."""

import torch
import triton
import triton.language as tl

from . import gqa


@triton.jit
def _repair_light(Q, K, V, A, P, PM, PL, PU, OUT,
                  N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
                  G: tl.constexpr, SA: tl.constexpr, SP: tl.constexpr,
                  AA: tl.constexpr, PP: tl.constexpr,
                  BB: tl.constexpr, DD: tl.constexpr):
    h = tl.program_id(0)
    aa = tl.arange(0, AA)
    pp = tl.arange(0, PP)
    ds = tl.arange(0, DD)
    a = tl.load(A + h * SA + aa, aa < SA, -1)
    p = tl.load(P + h * SP + pp, pp < SP, -2)
    if N > 2147483647:
        a = a.to(tl.int64)
    match = (p[:, None] == a[None, :]) & (pp[:, None] < SP) & (aa[None, :] < SA)
    hit = tl.sum(match.to(tl.int32), 1) > 0
    miss = (tl.sum(match.to(tl.int32), 0) == 0) & (aa < SA)
    count = tl.sum(miss.to(tl.int32), 0)

    pm = tl.load(PM + h * SP + pp, hit, -float("inf"))
    pl = tl.load(PL + h * SP + pp, hit, 0.)
    pu = tl.load(PU + (h * SP + pp[:, None]) * D + ds[None, :],
                 hit[:, None] & (ds[None, :] < D), 0.)
    m = tl.max(pm, 0)
    px = tl.exp(pm - tl.where(m == -float("inf"), 0., m))
    z = tl.sum(px * pl, 0)
    u = tl.sum(px[:, None] * pu, 0)

    if count > 0:
        bs = tl.arange(0, BB)
        q = tl.load(Q + h * D + ds, ds < D, 0.).to(tl.float32)
        valid = (bs[:, None] < B) & (ds[None, :] < D)
        kv = h.to(tl.int64) // G
        last = tl.full((), -1, a.dtype)
        for _ in range(count):
            # Choose the next miss from register-resident IDs. With unique
            # IDs, one ascending pass visits every miss without a sort.
            bid = tl.min(tl.where(miss & (a > last), a, N), 0)
            off = ((kv * N + bid) * B + bs[:, None]) * D + ds[None, :]
            k = tl.load(K + off, valid, 0.).to(tl.float32)
            score = tl.sum(k * q[None, :], 1) * (D ** -0.5)
            score = tl.where(bs < B, score, -float("inf"))
            next_m = tl.maximum(m, tl.max(score, 0))
            keep = tl.exp(m - next_m)
            w = tl.exp(score - next_m)
            v = tl.load(V + off, valid, 0.).to(tl.float32)
            z = z * keep + tl.sum(w, 0)
            u = u * keep + tl.sum(w[:, None] * v, 0)
            m = next_m
            last = bid
    tl.store(OUT + h * D + ds, u / z, ds < D)


def repair_saved_light(q, k, v, selected, predicted, state):
    """Repair the strict selected set with one CTA per query head."""
    hq, n, b, d, group, sa = gqa._shape(q, k, v, selected)
    gqa._validate_ids(q, predicted)
    sp = predicted.shape[1]
    if state.m.shape != (hq, sp) or state.l.shape != (hq, sp) or state.u.shape != (hq, sp, d):
        raise ValueError("state must contain a softmax partial for every predicted block")
    out = torch.empty((hq, d), device=q.device, dtype=torch.float32)
    _repair_light[(hq,)](
        q, k, v, selected, predicted, state.m, state.l, state.u, out,
        n, b, d, group, sa, sp, triton.next_power_of_2(sa),
        triton.next_power_of_2(sp), triton.next_power_of_2(b),
        triton.next_power_of_2(d), num_warps=4,
    )
    return out


def repair_saved_auto(q, k, v, selected, predicted, state, expected_miss_fraction):
    """Choose a repair schedule using an external estimate from past steps.

    The threshold of two expected misses per head and at least 16 query
    heads is a static scheduling hypothesis. Both schedules compute the
    strict selected set even when the estimate is inaccurate.
    """
    if not 0 <= expected_miss_fraction <= 1:
        raise ValueError("expected_miss_fraction must be in [0, 1]")
    if (q.ndim == 2 and q.shape[0] >= 16 and selected.ndim == 2
            and selected.shape[1] * expected_miss_fraction <= 2):
        return repair_saved_light(q, k, v, selected, predicted, state)
    return gqa.repair_saved(q, k, v, selected, predicted, state)
