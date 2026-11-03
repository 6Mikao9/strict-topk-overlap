"""Single-launch repair for the prediction states in kernels.py."""

import torch
import triton
import triton.language as tl

from .kernels import RepairResult, _shape


@triton.jit
def _accumulate(Q, K, V, IDS, h, positions, count, m, z, u,
                N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
                S: tl.constexpr, SS: tl.constexpr, BB: tl.constexpr,
                DD: tl.constexpr, TILE: tl.constexpr, FIXED_M: tl.constexpr):
    # Process only active blocks. Pack their positions before calling this
    # helper so the loop count follows the amount of late KV work.
    ds = tl.arange(0, DD)
    ts = tl.arange(0, TILE * BB)
    q = tl.load(Q + h * D + ds, (ds < D) & (count > 0), 0.).to(tl.float32)
    m = tl.cast(m, tl.float32)
    z = tl.cast(z, tl.float32)
    for start in range(0, count, TILE):
        i = start + tl.arange(0, TILE)
        pos = tl.gather(positions, tl.minimum(i, SS - 1), axis=0)
        ids = tl.load(IDS + h * S + pos, i < count, 0)
        bid = tl.broadcast_to(ids[:, None], (TILE, BB)).reshape((TILE * BB,))
        bs = ts % BB
        valid = (start + ts // BB < count) & (bs < B)
        offsets = ((h.to(tl.int64) * N + bid[:, None]) * B + bs[:, None]) * D + ds[None, :]
        mask = valid[:, None] & (ds[None, :] < D)
        kk = tl.load(K + offsets, mask, 0.).to(tl.float32)
        score = tl.sum(kk * q[None, :], 1) * (D ** -0.5)
        score = tl.where(valid, score, -float("inf"))
        if FIXED_M:
            new_m = m
            alpha = 1.
        else:
            new_m = tl.maximum(m, tl.max(score, 0))
            alpha = tl.exp(m - new_m)
        weight = tl.exp(score - new_m)
        vv = tl.load(V + offsets, mask, 0.).to(tl.float32)
        u = u * alpha + tl.sum(weight[:, None] * vv, 0)
        z = z * alpha + tl.sum(weight, 0)
        m = new_m
    return m, z, u


@triton.jit
def _repair(Q, K, V, A, P, PM, PL, PU, HIGH, RIDS, LM, LL, LU, LOW_ABS,
            MASS_M, MASS_L, O, BAD, REMAIN, APPROX, EXTRA,
            N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
            SA: tl.constexpr, SP: tl.constexpr, SC: tl.constexpr,
            AA: tl.constexpr, PP: tl.constexpr, CC: tl.constexpr,
            BB: tl.constexpr, DD: tl.constexpr,
            A_TILE: tl.constexpr, P_TILE: tl.constexpr,
            HYBRID: tl.constexpr, COMPACT: tl.constexpr,
            MIN_REMAIN: tl.constexpr, CANCEL: tl.constexpr, EPS: tl.constexpr):
    h = tl.program_id(0)
    aa = tl.arange(0, AA)
    pp = tl.arange(0, PP)
    ds = tl.arange(0, DD)
    a = tl.load(A + h * SA + aa, aa < SA, -1)
    p = tl.load(P + h * SP + pp, pp < SP, -2)
    match = (p[:, None] == a[None, :]) & (pp[:, None] < SP) & (aa[None, :] < SA)
    hit = tl.sum(match.to(tl.int32), 1) > 0
    missing = (tl.sum(match.to(tl.int32), 0) == 0) & (aa < SA)
    missing_count = tl.sum(missing.to(tl.int32), 0)
    fraction = 1.
    bad = False
    residual = tl.full((DD,), 0., tl.float32)
    mass = 0.
    lm = -float("inf")
    skip = False
    extra_bound = 0.
    if HYBRID:
        high = tl.load(HIGH + h * SP + pp, pp < SP, False)
        low_hit = tl.sum((hit & ~high).to(tl.int32), 0) > 0
        retained_hit = hit & high
    else:
        retained_hit = hit

    if COMPACT:
        cs = tl.arange(0, CC)
        r = tl.load(RIDS + h * SC + cs, cs < SC, -2)
        match = (r[:, None] == a[None, :]) & (cs[:, None] < SC) & (aa[None, :] < SA)
        keep = tl.sum(match.to(tl.int32), 1) > 0
    else:
        cs = pp
        keep = retained_hit

    pm = tl.load(PM + h * SC + cs, keep, -float("inf"))
    pl = tl.load(PL + h * SC + cs, keep, 0.)
    retained_m = tl.max(pm, 0)
    if HYBRID:
        if low_hit:
            lm = tl.load(LM + h)
            ll = tl.load(LL + h)
            lu = tl.load(LU + h * D + ds, ds < D, 0.)
            low_abs = tl.load(LOW_ABS + h * D + ds, ds < D, 0.)
            wrong = ~hit & ~high & (pp < SP)
            wrong_count = tl.sum(wrong.to(tl.int32), 0)
            removed = 0.
            removed_u = tl.full((DD,), 0., tl.float32)
            if EPS > 0:
                if wrong_count > 0:
                    ref = tl.maximum(retained_m, lm)
                    wm = tl.load(MASS_M + h * SP + pp, wrong, -float("inf"))
                    wl = tl.load(MASS_L + h * SP + pp, wrong, 0.)
                    wrong_mass = tl.sum(tl.exp(wm - ref) * wl, 0)
                    denom = tl.exp(lm - ref) * ll + tl.sum(tl.exp(pm - ref) * pl, 0)
                    extra_bound = wrong_mass / denom
                    skip = extra_bound <= EPS
            if (wrong_count > 0) & ~skip:
                wrong_positions = pp
                if PP > 1:
                    wrong_positions = tl.sort(tl.where(wrong, pp, SP), descending=False)
                _, removed, removed_u = _accumulate(
                    Q, K, V, P, h, wrong_positions, wrong_count, lm, 0., removed_u,
                    N, B, D, SP, PP, BB, DD, P_TILE, True,
                )
            mass = ll - removed
            residual = lu - removed_u
            fraction = mass / ll
            scale = low_abs + tl.abs(removed_u)
            cancel = (tl.abs(residual) < CANCEL * scale) & (scale > 0) & (ds < D)
            bad = (fraction < MIN_REMAIN) | (fraction != fraction)
            bad = bad | (tl.sum(cancel.to(tl.int32), 0) > 0)
    pu = tl.load(PU + (h * SC + cs[:, None]) * D + ds[None, :],
                 keep[:, None] & (ds[None, :] < D), 0.)
    m = tl.maximum(retained_m, lm)
    ref = tl.where(m == -float("inf"), 0., m)
    alpha = tl.exp(pm - ref)
    low_alpha = tl.exp(lm - ref)
    z = tl.sum(alpha * pl, 0) + low_alpha * mass
    u = tl.sum(alpha[:, None] * pu, 0) + low_alpha * residual

    if HYBRID:
        # Use the selected blocks once when the subtraction guard already
        # failed. Otherwise add only misses to the retained state.
        if bad:
            missing = aa < SA
            missing_count = SA
            m = -float("inf")
            z = 0.
            u = tl.full((DD,), 0., tl.float32)
    if missing_count > 0:
        positions = aa
        if AA > 1:
            positions = tl.sort(tl.where(missing, aa, SA), descending=False)
        m, z, u = _accumulate(
            Q, K, V, A, h, positions, missing_count, m, z, u,
            N, B, D, SA, AA, BB, DD, A_TILE, False,
        )
    out = u / z
    if HYBRID:
        bad_output = (z <= 0) | (tl.abs(z) == float("inf"))
        invalid = ((out != out) | (tl.abs(out) == float("inf"))) & (ds < D)
        bad_output = bad_output | (tl.sum(invalid.to(tl.int32), 0) > 0)
        if bad_output & ~bad:
            _, z, u = _accumulate(
                Q, K, V, A, h, aa, SA, -float("inf"), 0.,
                tl.full((DD,), 0., tl.float32), N, B, D, SA, AA, BB, DD,
                A_TILE, False,
            )
            out = u / z
        bad = bad | bad_output
    tl.store(O + h * D + ds, out, ds < D)
    tl.store(BAD + h, bad)
    tl.store(REMAIN + h, fraction)
    if EPS > 0:
        used_approx = skip & ~bad
        tl.store(APPROX + h, used_approx)
        tl.store(EXTRA + h, tl.where(used_approx, extra_bound, 0.))


def repair_fused(q, k, v, selected, predicted, prediction,
                 min_remaining=1.e-3, cancellation_tol=1.e-5,
                 max_extra_mass=0.):
    """Repair finite Q/K/V with valid, unique IDs, using one CTA per head.

    The cancellation and remaining-mass guards operate in the original
    low-group exponent frame, before scaling to the retained or new maximum.
    A positive max_extra_mass opts into retaining wrong low blocks if their
    estimated share of the output softmax is within that budget.
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
    low = prediction.low
    out = torch.empty((h, d), device=q.device, dtype=torch.float32)
    fallback = torch.empty((h,), device=q.device, dtype=torch.bool)
    remaining = torch.empty((h,), device=q.device, dtype=torch.float32)
    approximate = None
    extra_mass_bound = None
    if max_extra_mass > 0:
        approximate = torch.empty_like(fallback)
        extra_mass_bound = torch.empty_like(remaining)
    bb, dd = triton.next_power_of_2(b), triton.next_power_of_2(d)
    aa, pp = triton.next_power_of_2(sa), triton.next_power_of_2(sp)
    # Bound the live [tokens, dim] tile for the online softmax loop.
    tile = max(1, min(4, 8192 // (bb * dd)))
    _repair[(h,)](
        q, k, v, selected, predicted, p.m, p.l, p.u,
        prediction.high, prediction.retained_ids,
        low.m if hybrid else None, low.l if hybrid else None,
        low.u if hybrid else None, prediction.low_abs,
        prediction.mass_m, prediction.mass_l,
        out, fallback, remaining, approximate, extra_mass_bound, n, b, d, sa, sp, sc,
        aa, pp, triton.next_power_of_2(sc), bb, dd, min(tile, aa), min(tile, pp),
        hybrid, compact,
        min_remaining, cancellation_tol, max_extra_mass, num_warps=4,
    )
    return RepairResult(out, fallback, remaining, approximate, extra_mass_bound)
