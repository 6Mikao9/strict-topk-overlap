"""Parallel signed corrections for saved, hybrid and compact prediction states."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .kernels import RepairResult, _allocate_state, _fallback, _shape


@dataclass
class SignedRepairResult(RepairResult):
    positive_jobs: torch.Tensor | None = None
    total_jobs: torch.Tensor | None = None


@triton.jit
def _schedule(A, P, HIGH, RIDS, PM, PL, LM, LL, MASS_M, MASS_L,
              QUEUE, POS, TOTAL, KEEP, LOW_HIT, APPROX, EXTRA,
              SA: tl.constexpr, SP: tl.constexpr, SC: tl.constexpr, CAP: tl.constexpr,
              AA: tl.constexpr, PP: tl.constexpr, CC: tl.constexpr,
              HYBRID: tl.constexpr, COMPACT: tl.constexpr, EPS: tl.constexpr):
    h = tl.program_id(0)
    aa, pp = tl.arange(0, AA), tl.arange(0, PP)
    a = tl.load(A + h * SA + aa, aa < SA, -1)
    p = tl.load(P + h * SP + pp, pp < SP, -2)
    matches = (p[:, None] == a[None, :]) & (pp[:, None] < SP) & (aa[None, :] < SA)
    hit = tl.sum(matches.to(tl.int32), 1) > 0
    miss = (tl.sum(matches.to(tl.int32), 0) == 0) & (aa < SA)
    miss_count = tl.sum(miss.to(tl.int32), 0)
    rank = aa
    if AA > 1:
        rank = tl.cumsum(miss.to(tl.int32), 0) - 1
    tl.store(QUEUE + h * CAP + rank, a, miss)
    tl.store(POS + h, miss_count)

    if HYBRID:
        high = tl.load(HIGH + h * SP + pp, pp < SP, False)
        low_hit = tl.sum((hit & ~high).to(tl.int32), 0) > 0
        wrong = low_hit & ~hit & ~high & (pp < SP)
        retained_hit = hit & high
        tl.store(LOW_HIT + h, low_hit)
    else:
        retained_hit = hit

    if COMPACT:
        cs = tl.arange(0, CC)
        r = tl.load(RIDS + h * SC + cs, cs < SC, -2)
        eq = (r[:, None] == a[None, :]) & (cs[:, None] < SC) & (aa[None, :] < SA)
        keep = tl.sum(eq.to(tl.int32), 1) > 0
    else:
        cs = pp
        keep = retained_hit
    tl.store(KEEP + h * SC + cs, keep, cs < SC)

    skip = False
    extra = 0.
    total = miss_count
    if HYBRID:
        if EPS > 0:
            if tl.sum(wrong.to(tl.int32), 0) > 0:
                lm = tl.load(LM + h)
                ll = tl.load(LL + h)
                pm = tl.load(PM + h * SC + cs, keep, -float("inf"))
                pl = tl.load(PL + h * SC + cs, keep, 0.)
                wm = tl.load(MASS_M + h * SP + pp, wrong, -float("inf"))
                wl = tl.load(MASS_L + h * SP + pp, wrong, 0.)
                ref = tl.maximum(lm, tl.max(pm, 0))
                denom = tl.exp(lm - ref) * ll + tl.sum(tl.exp(pm - ref) * pl, 0)
                extra = tl.sum(tl.exp(wm - ref) * wl, 0) / denom
                skip = extra <= EPS
            wrong = wrong & ~skip
        rank = pp
        if PP > 1:
            rank = tl.cumsum(wrong.to(tl.int32), 0) - 1
        tl.store(QUEUE + h * CAP + miss_count + rank, p, wrong)
        total += tl.sum(wrong.to(tl.int32), 0)
    tl.store(TOTAL + h, total)
    if EPS > 0:
        tl.store(APPROX + h, skip)
        tl.store(EXTRA + h, tl.where(skip, extra, 0.))


@triton.jit
def _signed_partials(Q, K, V, QUEUE, POS, TOTAL, M, L, U,
                     N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
                     CAP: tl.constexpr, BB: tl.constexpr, DD: tl.constexpr,
                     HYBRID: tl.constexpr):
    h, s = tl.program_id(0), tl.program_id(1)
    if s < tl.load(TOTAL + h):
        row = h * CAP + s
        bid = tl.load(QUEUE + row)
        bs, ds = tl.arange(0, BB), tl.arange(0, DD)
        q = tl.load(Q + h * D + ds, ds < D, 0.).to(tl.float32)
        off = ((h.to(tl.int64) * N + bid) * B + bs[:, None]) * D + ds[None, :]
        valid = (bs[:, None] < B) & (ds[None, :] < D)
        k = tl.load(K + off, valid, 0.).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * (D ** -0.5)
        score = tl.where(bs < B, score, -float("inf"))
        m = tl.max(score, 0)
        w = tl.exp(score - m)
        v = tl.load(V + off, valid, 0.).to(tl.float32)
        sign = 1.
        if HYBRID:
            sign = tl.where(s < tl.load(POS + h), 1., -1.)
        tl.store(M + row, m)
        tl.store(L + row, sign * tl.sum(w, 0))
        tl.store(U + row * D + ds, sign * tl.sum(w[:, None] * v, 0), ds < D)


@triton.jit
def _merge(PM, PL, PU, KEEP, LM, LL, LU, LOW_ABS, LOW_HIT,
           M, L, U, POS, TOTAL, OUT, BAD, REMAIN, APPROX, EXTRA,
           D: tl.constexpr, SC: tl.constexpr, CAP: tl.constexpr,
           CC: tl.constexpr, RC: tl.constexpr, DD: tl.constexpr,
           HYBRID: tl.constexpr, MIN_REMAIN: tl.constexpr, CANCEL: tl.constexpr,
           EPS: tl.constexpr):
    h = tl.program_id(0)
    cs, rs, ds = tl.arange(0, CC), tl.arange(0, RC), tl.arange(0, DD)
    keep = tl.load(KEEP + h * SC + cs, cs < SC, False)
    pm = tl.load(PM + h * SC + cs, keep, -float("inf"))
    pl = tl.load(PL + h * SC + cs, keep, 0.)
    pu = tl.load(PU + (h * SC + cs[:, None]) * D + ds[None, :],
                 keep[:, None] & (ds[None, :] < D), 0.)
    pos = tl.load(POS + h)
    active = rs < tl.load(TOTAL + h)
    cm = tl.load(M + h * CAP + rs, active, -float("inf"))
    cl = tl.load(L + h * CAP + rs, active, 0.)
    cu = tl.load(U + (h * CAP + rs[:, None]) * D + ds[None, :],
                 active[:, None] & (ds[None, :] < D), 0.)

    fraction = 1.
    bad = False
    m = tl.maximum(tl.max(pm, 0), tl.max(tl.where(rs < pos, cm, -float("inf")), 0))
    low_m = -float("inf")
    low_z = 0.
    low_u = tl.full((DD,), 0., tl.float32)
    if HYBRID:
        if tl.load(LOW_HIT + h):
            low_m = tl.load(LM + h)
            original_z = tl.load(LL + h)
            original_u = tl.load(LU + h * D + ds, ds < D, 0.)
            envelope = tl.load(LOW_ABS + h * D + ds, ds < D, 0.)
            wx = tl.exp(tl.where(rs >= pos, cm, -float("inf")) - low_m)
            delta_u = tl.sum(wx[:, None] * cu, 0)
            low_z = original_z + tl.sum(wx * cl, 0)
            low_u = original_u + delta_u
            fraction = low_z / original_z
            scale = envelope + tl.abs(delta_u)
            cancel = (tl.abs(low_u) < CANCEL * scale) & (scale > 0) & (ds < D)
            bad = (fraction < MIN_REMAIN) | (fraction != fraction)
            bad = bad | (tl.sum(cancel.to(tl.int32), 0) > 0)
        m = tl.maximum(m, low_m)

    px = tl.exp(pm - m)
    ax = tl.exp(tl.where(rs < pos, cm, -float("inf")) - m)
    lx = tl.exp(low_m - m)
    z = tl.sum(px * pl, 0) + tl.sum(ax * cl, 0) + lx * low_z
    u = tl.sum(px[:, None] * pu, 0) + tl.sum(ax[:, None] * cu, 0) + lx * low_u
    out = u / z
    if HYBRID:
        invalid = ((out != out) | (tl.abs(out) == float("inf"))) & (ds < D)
        bad = bad | (z <= 0) | (tl.abs(z) == float("inf"))
        bad = bad | (tl.sum(invalid.to(tl.int32), 0) > 0)
        if EPS > 0:
            if bad:
                tl.store(APPROX + h, False)
                tl.store(EXTRA + h, 0.)
    tl.store(OUT + h * D + ds, out, ds < D)
    tl.store(BAD + h, bad)
    tl.store(REMAIN + h, fraction)


def repair_signed(q, k, v, selected, predicted, prediction, min_remaining=1.e-3,
                  cancellation_tol=1.e-5, max_extra_mass=0.):
    """Repair using one parallel partials kernel for both signs of correction.

    The queue packs missed selected blocks first and then wrong low blocks.
    A guarded merge evaluates the subtraction in the original low exponent
    frame. The fallback recomputes the strict selected set on flagged heads.
    Positive and total job counts describe the correction queue before fallback.
    """
    h, n, b, d, sa = _shape(q, k, v, selected)
    _, _, _, _, sp = _shape(q, k, v, predicted)
    p = prediction.state
    sc = p.m.shape[1]
    hybrid = prediction.mode in ("hybrid", "compact")
    compact = prediction.mode == "compact"
    if not 0 <= max_extra_mass <= 1:
        raise ValueError("max_extra_mass must be in [0, 1]")
    if hybrid and max_extra_mass > 0 and (prediction.mass_m is None or prediction.mass_l is None):
        raise ValueError("positive max_extra_mass requires speculate(..., keep_mass=True)")

    capacity = sa + sp if hybrid else sa
    device = q.device
    id_dtype = torch.promote_types(selected.dtype, predicted.dtype)
    queue = torch.empty((h, capacity), device=device, dtype=id_dtype)
    positive = torch.empty((h,), device=device, dtype=torch.int32)
    total = torch.empty_like(positive)
    keep = torch.empty((h, sc), device=device, dtype=torch.bool)
    low_hit = torch.empty((h,), device=device, dtype=torch.bool) if hybrid else None
    correction = _allocate_state(h, capacity, d, device)
    out = torch.empty((h, d), device=device, dtype=torch.float32)
    fallback = torch.empty((h,), device=device, dtype=torch.bool)
    remaining = torch.empty((h,), device=device, dtype=torch.float32)
    approximate = torch.empty_like(fallback) if max_extra_mass > 0 else None
    extra = torch.empty_like(remaining) if max_extra_mass > 0 else None
    aa, pp, cc = (triton.next_power_of_2(size) for size in (sa, sp, sc))
    bb, dd = triton.next_power_of_2(b), triton.next_power_of_2(d)
    low = prediction.low
    lm, ll, lu = (low.m, low.l, low.u) if hybrid else (None, None, None)
    _schedule[(h,)](
        selected, predicted, prediction.high, prediction.retained_ids, p.m, p.l, lm, ll,
        prediction.mass_m, prediction.mass_l, queue, positive, total, keep, low_hit,
        approximate, extra, sa, sp, sc, capacity, aa, pp, cc,
        hybrid, compact, max_extra_mass, num_warps=4,
    )
    _signed_partials[(h, capacity)](
        q, k, v, queue, positive, total, correction.m, correction.l, correction.u,
        n, b, d, capacity, bb, dd, hybrid, num_warps=4,
    )
    _merge[(h,)](
        p.m, p.l, p.u, keep, lm, ll, lu, prediction.low_abs, low_hit,
        correction.m, correction.l, correction.u, positive, total, out, fallback,
        remaining, approximate, extra, d, sc, capacity, cc,
        triton.next_power_of_2(capacity), dd, hybrid, min_remaining, cancellation_tol,
        max_extra_mass, num_warps=4,
    )
    if hybrid:
        _fallback[(h,)](q, k, v, selected, fallback, out, n, b, d, sa, bb, dd, num_warps=4)
    return SignedRepairResult(out, fallback, remaining, approximate, extra, positive, total)
