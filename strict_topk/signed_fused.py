"""Signed block correction with a shared low-group scale and inline fallback.

The bookkeeping queue is shared with the parallel reference. The arithmetic
below fuses positive and negative KV work and folds fallback into finalization.
"""
import torch
import triton
import triton.language as tl

from .kernels import _allocate_state, _shape
from .signed_repair import SignedRepairResult, _schedule


@triton.jit
def _kv_tile(Q, K, V, head, block,
             N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
             BB: tl.constexpr, DD: tl.constexpr):
    tokens = tl.arange(0, BB)
    dims = tl.arange(0, DD)
    query = tl.load(Q + head * D + dims, dims < D, 0.).to(tl.float32)
    offsets = ((head.to(tl.int64) * N + block) * B + tokens[:, None]) * D + dims[None, :]
    mask = (tokens[:, None] < B) & (dims[None, :] < D)
    keys = tl.load(K + offsets, mask, 0.).to(tl.float32)
    logits = tl.sum(keys * query[None, :], 1) * (D ** -0.5)
    logits = tl.where(tokens < B, logits, -float("inf"))
    values = tl.load(V + offsets, mask, 0.).to(tl.float32)
    return logits, values


@triton.jit
def _corrections(Q, K, V, IDS, POS, COUNT, LOW_M, M, L, U,
                 N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
                 CAP: tl.constexpr, BB: tl.constexpr, DD: tl.constexpr,
                 HYBRID: tl.constexpr, COMPACT: tl.constexpr):
    head, job = tl.program_id(0), tl.program_id(1)
    if job < tl.load(COUNT + head):
        row = head * CAP + job
        block = tl.load(IDS + row)
        logits, values = _kv_tile(Q,K,V,head,block,N,B,D,BB,DD)
        negative = False
        if HYBRID:
            negative = job >= tl.load(POS + head)
        block_m = tl.max(logits, 0)
        scale = block_m
        if COMPACT:
            # Match compact preparation's per-token low-frame arithmetic.
            scale = tl.where(negative, tl.load(LOW_M + head), block_m)
        weights = tl.exp(logits - scale)
        sign = tl.where(negative, -1., 1.)
        denominator = tl.sum(weights, 0)
        numerator = tl.sum(weights[:, None] * values, 0)
        if HYBRID:
            if not COMPACT:
                # Hybrid first creates block-local partials. A tiny exp in
                # the group frame can underflow before multiplication by V;
                # reproduce the preparation order, then rescale the sum.
                target_m = tl.load(LOW_M + head)
                factor = tl.where(negative,tl.exp(block_m-target_m),1.)
                denominator *= factor
                numerator *= factor
                scale = tl.where(negative,target_m,block_m)
        denominator *= sign
        numerator *= sign
        dims = tl.arange(0, DD)
        tl.store(M + row, scale)
        tl.store(L + row, denominator)
        tl.store(U + row * D + dims, numerator, dims < D)


@triton.jit
def _finalize(Q,K,V,A, PM,PL,PU,KEEP, LM,LL,LU,ABS_LOW,LOW_HIT,
              CM,CL,CU,POS,COUNT,OUT,BAD,REMAIN,APPROX,EXTRA,
              N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,SA:tl.constexpr,
              SC:tl.constexpr,CAP:tl.constexpr,CC:tl.constexpr,RC:tl.constexpr,
              BB:tl.constexpr,DD:tl.constexpr,HYBRID:tl.constexpr,
              MIN_REMAIN:tl.constexpr,CANCEL:tl.constexpr,EPS:tl.constexpr):
    head = tl.program_id(0)
    slots, jobs, dims = tl.arange(0,CC),tl.arange(0,RC),tl.arange(0,DD)
    keep = tl.load(KEEP + head*SC + slots, slots<SC, False)
    pm = tl.load(PM + head*SC + slots, keep, -float("inf"))
    pl = tl.load(PL + head*SC + slots, keep, 0.)
    pu = tl.load(PU + (head*SC + slots[:,None])*D + dims[None,:],
                 keep[:,None] & (dims[None,:]<D), 0.)
    npos, count = tl.load(POS+head),tl.load(COUNT+head)
    active, positive = jobs<count,jobs<npos
    cm = tl.load(CM + head*CAP + jobs, active, -float("inf"))
    cl = tl.load(CL + head*CAP + jobs, active, 0.)
    cu = tl.load(CU + (head*CAP + jobs[:,None])*D + dims[None,:],
                 active[:,None] & (dims[None,:]<D), 0.)
    low_m, low_l = -float("inf"),0.
    low_u = tl.full((DD,),0.,tl.float32)
    bad, fraction = False,1.
    if HYBRID:
        if tl.load(LOW_HIT+head):
            low_m = tl.load(LM+head)
            original_l = tl.load(LL+head)
            original_u = tl.load(LU + head*D + dims,dims<D,0.)
            envelope = tl.load(ABS_LOW + head*D + dims,dims<D,0.)
            # Negative tiles were already evaluated in low_m's frame.
            negative = active & ~positive
            removed_u = tl.sum(tl.where(negative[:,None],cu,0.),0)
            low_l = original_l + tl.sum(tl.where(negative,cl,0.),0)
            low_u = original_u + removed_u
            fraction = low_l/original_l
            cancellation_scale = envelope + tl.abs(removed_u)
            cancellation = (tl.abs(low_u)<CANCEL*cancellation_scale) & (cancellation_scale>0) & (dims<D)
            bad = (fraction<MIN_REMAIN) | (fraction!=fraction)
            bad = bad | (tl.sum(cancellation.to(tl.int32),0)>0)

    max_positive = tl.max(tl.where(positive,cm,-float("inf")),0)
    common_m = tl.maximum(tl.maximum(tl.max(pm,0),max_positive),low_m)
    # Finite inputs, nonempty A guarantee some retained or new state.
    pw = tl.exp(pm-common_m)
    aw = tl.exp(tl.where(positive,cm,-float("inf"))-common_m)
    lw = tl.exp(low_m-common_m)
    denominator = tl.sum(pw*pl,0)+tl.sum(aw*cl,0)+lw*low_l
    numerator = tl.sum(pw[:,None]*pu,0)+tl.sum(aw[:,None]*cu,0)+lw*low_u
    output = numerator/denominator
    if HYBRID:
        invalid = ((output!=output) | (tl.abs(output)==float("inf"))) & (dims<D)
        bad = bad | (denominator<=0) | (tl.abs(denominator)==float("inf"))
        bad = bad | (tl.sum(invalid.to(tl.int32),0)>0)
        if bad:
            # A rare numerical fallback stays within this launch. It streams
            # the strict selected set using a positive online softmax; no
            # signed accumulator is reused after a failed subtraction guard.
            fm, fl = -float("inf"),0.
            fu = tl.full((DD,),0.,tl.float32)
            for slot in range(SA):
                block = tl.load(A+head*SA+slot)
                logits, values = _kv_tile(Q,K,V,head,block,N,B,D,BB,DD)
                tile_m = tl.max(logits,0)
                next_m = tl.maximum(fm,tile_m)
                old_weight = tl.exp(fm-next_m)
                weights = tl.exp(logits-next_m)
                fl = old_weight*fl + tl.sum(weights,0)
                fu = old_weight*fu + tl.sum(weights[:,None]*values,0)
                fm = next_m
            output = fu/fl
            if EPS>0:
                tl.store(APPROX+head,False)
                tl.store(EXTRA+head,0.)
    tl.store(OUT+head*D+dims,output,dims<D)
    tl.store(BAD+head,bad)
    tl.store(REMAIN+head,fraction)


def repair_signed_fused(q,k,v,selected,predicted,prediction,min_remaining=1.e-3,
                        cancellation_tol=1.e-5,max_extra_mass=0.):
    """One shared add/erase KV kernel, followed by guarded finalization.

    Uses three launches: queue metadata; parallel signed KV computation;
    output merge and optional strict fallback. The metadata helper is shared
    with repair_signed so this ablation isolates arithmetic and finalization.
    """
    h,n,b,d,sa = _shape(q,k,v,selected)
    _,_,_,_,sp = _shape(q,k,v,predicted)
    hybrid = prediction.mode in ('hybrid','compact')
    compact = prediction.mode=='compact'
    if not 0<=max_extra_mass<=1:
        raise ValueError('max_extra_mass must be in [0,1]')
    if hybrid and max_extra_mass>0 and (prediction.mass_m is None or prediction.mass_l is None):
        raise ValueError('positive max_extra_mass requires speculate(..., keep_mass=True)')
    state,low = prediction.state,prediction.low
    sc = state.m.shape[1]
    capacity = sa+sp if hybrid else sa
    kwargs = {'device':q.device}
    queue = torch.empty((h,capacity),dtype=torch.promote_types(selected.dtype,predicted.dtype),**kwargs)
    pos = torch.empty(h,dtype=torch.int32,**kwargs)
    total = torch.empty_like(pos)
    keep = torch.empty((h,sc),dtype=torch.bool,**kwargs)
    low_hit = torch.empty(h,dtype=torch.bool,**kwargs) if hybrid else None
    correction = _allocate_state(h,capacity,d,q.device)
    out = torch.empty((h,d),dtype=torch.float32,**kwargs)
    bad = torch.empty(h,dtype=torch.bool,**kwargs)
    remain = torch.empty(h,dtype=torch.float32,**kwargs)
    approx = torch.empty_like(bad) if max_extra_mass>0 else None
    extra = torch.empty_like(remain) if max_extra_mass>0 else None
    aa,pp,cc = map(triton.next_power_of_2,(sa,sp,sc))
    bb,dd = map(triton.next_power_of_2,(b,d))
    lm,ll,lu = (low.m,low.l,low.u) if hybrid else (None,None,None)
    _schedule[(h,)](selected,predicted,prediction.high,prediction.retained_ids,
        state.m,state.l,lm,ll,prediction.mass_m,prediction.mass_l,
        queue,pos,total,keep,low_hit,approx,extra,sa,sp,sc,capacity,aa,pp,cc,
        hybrid,compact,max_extra_mass,num_warps=4)
    _corrections[(h,capacity)](q,k,v,queue,pos,total,lm,correction.m,correction.l,correction.u,
        n,b,d,capacity,bb,dd,hybrid,compact,num_warps=4)
    _finalize[(h,)](q,k,v,selected,state.m,state.l,state.u,keep,lm,ll,lu,
        prediction.low_abs,low_hit,correction.m,correction.l,correction.u,pos,total,
        out,bad,remain,approx,extra,n,b,d,sa,sc,capacity,cc,
        triton.next_power_of_2(capacity),bb,dd,hybrid,min_remaining,cancellation_tol,
        max_extra_mass,num_warps=4)
    return SignedRepairResult(out,bad,remain,approx,extra,pos,total)
