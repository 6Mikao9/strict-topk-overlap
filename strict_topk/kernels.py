"""Forward-only FP32-state sparse attention research kernels.

Inputs are contiguous Q[H,D], K/V[H,N,B,D], unique block IDs[H,S].
All rows contain exactly S valid IDs; padding/GQA/causal masking are not
implemented. The caller must finish KV writes before issuing either stream.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class PartialState:
    m: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor


@dataclass
class Prediction:
    state: PartialState
    mode: str
    high: torch.Tensor | None = None
    low: PartialState | None = None
    retained_ids: torch.Tensor | None = None
    overflow: torch.Tensor | None = None
    low_abs: torch.Tensor | None = None
    mass_m: torch.Tensor | None = None
    mass_l: torch.Tensor | None = None


@dataclass
class RepairResult:
    output: torch.Tensor
    fallback: torch.Tensor
    remaining: torch.Tensor
    approximate_heads: torch.Tensor | None = None
    extra_mass_bound: torch.Tensor | None = None


@triton.jit
def _partial(Q, K, V, IDS, ACTIVE, M, L, U,
             N: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
             S: tl.constexpr, BB: tl.constexpr, DD: tl.constexpr,
             HAS_ACTIVE: tl.constexpr, ACTIVE_HEAD: tl.constexpr):
    h, s = tl.program_id(0), tl.program_id(1)
    enabled = True
    if HAS_ACTIVE:
        enabled = tl.load(ACTIVE + h if ACTIVE_HEAD else ACTIVE + h*S+s)
    ds = tl.arange(0, DD)
    if enabled:
        bid = tl.load(IDS+h*S+s)
        bs = tl.arange(0, BB)
        q = tl.load(Q+h*D+ds, ds<D, 0).to(tl.float32)
        off = ((h.to(tl.int64)*N+bid)*B+bs[:,None])*D+ds[None,:]
        kk = tl.load(K+off, (bs[:,None]<B)&(ds[None,:]<D), 0).to(tl.float32)
        score = tl.sum(kk*q[None,:], 1)*(D ** -0.5)
        score = tl.where(bs<B, score, -float('inf'))
        m = tl.max(score, 0)
        w = tl.exp(score-m)
        ll = tl.sum(w, 0)
        vv = tl.load(V+off, (bs[:,None]<B)&(ds[None,:]<D), 0).to(tl.float32)
        uu = tl.sum(w[:,None]*vv, 0)
    else:
        m = -float('inf')
        ll = 0.
        uu = tl.full((DD,), 0, tl.float32)
    tl.store(M+h*S+s, m)
    tl.store(L+h*S+s, ll)
    tl.store(U+(h*S+s)*D+ds, uu, ds<D)


@triton.jit
def _merge(M,L,U,INC,O,S:tl.constexpr,D:tl.constexpr,SS:tl.constexpr,
           DD:tl.constexpr,HAS_INC:tl.constexpr):
    h=tl.program_id(0)
    ss=tl.arange(0,SS)
    ds=tl.arange(0,DD)
    mask=ss<S
    if HAS_INC:
        mask=mask & tl.load(INC+h*S+ss,ss<S,False)
    mm=tl.load(M+h*S+ss,mask,-float('inf'))
    ll=tl.load(L+h*S+ss,mask,0.)
    m=tl.max(mm,0)
    m=tl.where(m == -float('inf'),0.,m)
    a=tl.exp(mm-m)
    z=tl.sum(a*ll,0)
    uu=tl.load(U+(h*S+ss[:,None])*D+ds[None,:],mask[:,None]&(ds[None,:]<D),0.)
    out=tl.sum(a[:,None]*uu,0)/tl.maximum(z,1.e-30)
    tl.store(O+h*D+ds,out,ds<D)


@triton.jit
def _group(M,L,U,HIGH,LM,LL,LU,L_ABS,S:tl.constexpr,D:tl.constexpr,
           SS:tl.constexpr,DD:tl.constexpr,THRESH:tl.constexpr):
    h=tl.program_id(0)
    ss=tl.arange(0,SS)
    ds=tl.arange(0,DD)
    mm=tl.load(M+h*S+ss,ss<S,-float('inf'))
    ll=tl.load(L+h*S+ss,ss<S,0.)
    global_m=tl.max(mm,0)
    mass=tl.exp(mm-global_m)*ll
    total=tl.sum(mass,0)
    high=(mass >= THRESH*total)&(ss<S)
    tl.store(HIGH+h*S+ss,high,ss<S)
    low=(ss<S)&(~high)
    low_m=tl.max(tl.where(low,mm,-float('inf')),0)
    low_m=tl.where(low_m == -float('inf'),0.,low_m)
    alpha=tl.where(low,tl.exp(mm-low_m),0.)
    zz=tl.sum(alpha*ll,0)
    uu=tl.load(U+(h*S+ss[:,None])*D+ds[None,:],(ss[:,None]<S)&(ds[None,:]<D),0.)
    uv=tl.sum(alpha[:,None]*uu,0)
    tl.store(LM+h,low_m)
    tl.store(LL+h,zz)
    tl.store(LU+h*D+ds,uv,ds<D)
    tl.store(L_ABS+h*D+ds,tl.sum(alpha[:,None]*tl.abs(uu),0),ds<D)


@triton.jit
def _membership(P,A,HIGH,HIT,MISS,WRONG,LOW_HIT,
                SP:tl.constexpr,SA:tl.constexpr,PP:tl.constexpr,
                AA:tl.constexpr,HYBRID:tl.constexpr):
    h=tl.program_id(0)
    ps=tl.arange(0,PP)
    aa=tl.arange(0,AA)
    p=tl.load(P+h*SP+ps,ps<SP,-1)
    a=tl.load(A+h*SA+aa,aa<SA,-2)
    eq=(p[:,None]==a[None,:])&(ps[:,None]<SP)&(aa[None,:]<SA)
    hit=tl.sum(eq.to(tl.int32),1)>0
    miss=tl.sum(eq.to(tl.int32),0)==0
    if HYBRID:
        high=tl.load(HIGH+h*SP+ps,ps<SP,False)
        any_low_hit=tl.sum((hit & (~high) & (ps<SP)).to(tl.int32),0)>0
        wrong=(~hit)&(~high)&any_low_hit
        hit=hit&high
    else:
        any_low_hit=False
        wrong=tl.full((PP,),False,tl.int1)
    tl.store(HIT+h*SP+ps,hit,ps<SP)
    tl.store(MISS+h*SA+aa,miss,aa<SA)
    tl.store(WRONG+h*SP+ps,wrong,ps<SP)
    tl.store(LOW_HIT+h,any_low_hit)


@triton.jit
def _saved_merge(PM,PL,PU,HIT,AM,AL,AU,O,BAD,REMAIN,
                 SP:tl.constexpr,SA:tl.constexpr,D:tl.constexpr,
                 PP:tl.constexpr,AA:tl.constexpr,DD:tl.constexpr):
    h=tl.program_id(0)
    ps=tl.arange(0,PP)
    aa=tl.arange(0,AA)
    ds=tl.arange(0,DD)
    hit=tl.load(HIT+h*SP+ps,ps<SP,False)
    pm=tl.load(PM+h*SP+ps,(ps<SP)&hit,-float('inf'))
    am=tl.load(AM+h*SA+aa,aa<SA,-float('inf'))
    m=tl.maximum(tl.max(pm,0),tl.max(am,0))
    ap=tl.exp(pm-m)
    ax=tl.exp(am-m)
    pl=tl.load(PL+h*SP+ps,(ps<SP)&hit,0.)
    al=tl.load(AL+h*SA+aa,aa<SA,0.)
    z=tl.sum(ap*pl,0)+tl.sum(ax*al,0)
    pu=tl.load(PU+(h*SP+ps[:,None])*D+ds[None,:],((ps<SP)&hit)[:,None]&(ds[None,:]<D),0.)
    au=tl.load(AU+(h*SA+aa[:,None])*D+ds[None,:],(aa[:,None]<SA)&(ds[None,:]<D),0.)
    u=tl.sum(ap[:,None]*pu,0)+tl.sum(ax[:,None]*au,0)
    tl.store(O+h*D+ds,u/z,ds<D)
    tl.store(BAD+h,False)
    tl.store(REMAIN+h,1.)


@triton.jit
def _hybrid_merge(PM,PL,PU,HIT,LM,LL,LU,L_ABS,LOW_HIT,WM,WL,WU,
                  AM,AL,AU,O,BAD,REMAIN,
                  SP:tl.constexpr,SA:tl.constexpr,D:tl.constexpr,SC:tl.constexpr,
                  PP:tl.constexpr,AA:tl.constexpr,DD:tl.constexpr,CC:tl.constexpr,
                  MIN_REMAIN:tl.constexpr,CANCEL:tl.constexpr):
    h=tl.program_id(0)
    ps=tl.arange(0,PP)
    cs=tl.arange(0,CC)
    aa=tl.arange(0,AA)
    ds=tl.arange(0,DD)
    low_hit=tl.load(LOW_HIT+h)
    hit=tl.load(HIT+h*SC+cs,cs<SC,False)
    pm=tl.load(PM+h*SC+cs,(cs<SC)&hit,-float('inf'))
    am=tl.load(AM+h*SA+aa,aa<SA,-float('inf'))
    wm=tl.load(WM+h*SP+ps,(ps<SP)&low_hit,-float('inf'))
    lm=tl.load(LM+h)
    m=tl.maximum(tl.max(pm,0),tl.max(am,0))
    m=tl.maximum(m,tl.where(low_hit,lm,-float('inf')))
    ap=tl.exp(pm-m)
    ax=tl.exp(am-m)
    aw=tl.exp(wm-m)
    lowalpha=tl.where(low_hit,tl.exp(lm-m),0.)
    ll=tl.load(LL+h,low_hit,0.)*lowalpha
    wl=tl.load(WL+h*SP+ps,(ps<SP)&low_hit,0.)
    removed=tl.sum(aw*wl,0)
    remaining=ll-removed
    fraction=tl.where(ll>0,remaining/tl.maximum(ll,1.e-30),1.)
    pl=tl.load(PL+h*SC+cs,(cs<SC)&hit,0.)
    al=tl.load(AL+h*SA+aa,aa<SA,0.)
    z=remaining+tl.sum(ap*pl,0)+tl.sum(ax*al,0)
    pu=tl.load(PU+(h*SC+cs[:,None])*D+ds[None,:],((cs<SC)&hit)[:,None]&(ds[None,:]<D),0.)
    au=tl.load(AU+(h*SA+aa[:,None])*D+ds[None,:],(aa[:,None]<SA)&(ds[None,:]<D),0.)
    wu=tl.load(WU+(h*SP+ps[:,None])*D+ds[None,:],(ps[:,None]<SP)&(ds[None,:]<D)&low_hit,0.)
    lu=tl.load(LU+h*D+ds,(ds<D)&low_hit,0.)*lowalpha
    removed_u=tl.sum(aw[:,None]*wu,0)
    residual=lu-removed_u
    # The envelope also exposes cancellation that happened during aggregation,
    # before the signed low-group and removed numerators became available.
    low_abs=tl.load(L_ABS+h*D+ds,(ds<D)&low_hit,0.)*lowalpha
    scale=low_abs+tl.abs(removed_u)
    cancel=(tl.abs(residual)<CANCEL*scale)&(scale>0)&(ds<D)&low_hit
    u=residual+tl.sum(ap[:,None]*pu,0)+tl.sum(ax[:,None]*au,0)
    out=u/tl.maximum(z,1.e-30)
    bad=(z<=0)|(fraction<MIN_REMAIN)|(fraction!=fraction)|(tl.abs(z)==float('inf'))
    bad=bad | (tl.sum(cancel.to(tl.int32),0)>0)
    bad=bad | (tl.sum((((out!=out)|(tl.abs(out)==float('inf')))&(ds<D)).to(tl.int32),0)>0)
    tl.store(O+h*D+ds,out,ds<D)
    tl.store(BAD+h,bad)
    tl.store(REMAIN+h,fraction)


@triton.jit
def _fallback(Q,K,V,IDS,BAD,O,N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,
              S:tl.constexpr,BB:tl.constexpr,DD:tl.constexpr):
    h=tl.program_id(0)
    if tl.load(BAD+h):
        bs=tl.arange(0,BB)
        ds=tl.arange(0,DD)
        q=tl.load(Q+h*D+ds,ds<D,0.).to(tl.float32)
        m=-float('inf')
        z=0.
        u=tl.full((DD,),0.,tl.float32)
        for s in range(S):
            bid=tl.load(IDS+h*S+s)
            off=((h.to(tl.int64)*N+bid)*B+bs[:,None])*D+ds[None,:]
            kk=tl.load(K+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
            score=tl.sum(kk*q[None,:],1)*(D**-0.5)
            score=tl.where(bs<B,score,-float('inf'))
            nm=tl.maximum(m,tl.max(score,0))
            alpha=tl.exp(m-nm)
            w=tl.exp(score-nm)
            vv=tl.load(V+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
            u=u*alpha+tl.sum(w[:,None]*vv,0)
            z=z*alpha+tl.sum(w,0)
            m=nm
        tl.store(O+h*D+ds,u/z,ds<D)


@triton.jit
def _touch(K,V,IDS,OUT,N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,S:tl.constexpr,
           TILE:tl.constexpr):
    h,s,t=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    bid=tl.load(IDS+h*S+s)
    x=t*TILE+tl.arange(0,TILE)
    off=(h.to(tl.int64)*N+bid)*B*D+x
    kk=tl.load(K+off,x<B*D,0).to(tl.float32)
    vv=tl.load(V+off,x<B*D,0).to(tl.float32)
    ntiles=tl.cdiv(B*D,TILE)
    tl.store(OUT+(h*S+s)*ntiles+t,tl.sum(kk+vv,0))


@triton.jit
def _mass_only(Q,K,IDS,M,L,N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,
               S:tl.constexpr,BB:tl.constexpr,DD:tl.constexpr):
    h,s=tl.program_id(0),tl.program_id(1)
    bs=tl.arange(0,BB)
    ds=tl.arange(0,DD)
    bid=tl.load(IDS+h*S+s)
    q=tl.load(Q+h*D+ds,ds<D,0.).to(tl.float32)
    off=((h.to(tl.int64)*N+bid)*B+bs[:,None])*D+ds[None,:]
    kk=tl.load(K+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
    score=tl.sum(kk*q[None,:],1)*(D**-0.5)
    score=tl.where(bs<B,score,-float('inf'))
    m=tl.max(score,0)
    l=tl.sum(tl.exp(score-m),0)
    tl.store(M+h*S+s,m)
    tl.store(L+h*S+s,l)


@triton.jit
def _compact_group(M,L,IDS,HIGH,RIDS,LM,LL,OVERFLOW,
                   S:tl.constexpr,C:tl.constexpr,SS:tl.constexpr,CC:tl.constexpr,
                   THRESH:tl.constexpr):
    h=tl.program_id(0)
    ss=tl.arange(0,SS)
    cs=tl.arange(0,CC)
    mm=tl.load(M+h*S+ss,ss<S,-float('inf'))
    ll=tl.load(L+h*S+ss,ss<S,0.)
    pid=tl.load(IDS+h*S+ss,ss<S,0)
    mass=tl.exp(mm-tl.max(mm,0))*ll
    high=(mass>=THRESH*tl.sum(mass,0))&(ss<S)
    rank=tl.cumsum(high.to(tl.int32),0)
    count=tl.sum(high.to(tl.int32),0)
    tl.store(OVERFLOW+h,count>C)
    # A numerical boundary overflow spills extra candidates to the low group.
    # Strict-set semantics remain guarded; no unbounded allocation is required.
    high=high&(rank<=C)
    tl.store(HIGH+h*S+ss,high,ss<S)
    match=high[:,None]&(rank[:,None]==(cs[None,:]+1))
    rid=tl.sum(tl.where(match,pid[:,None],0),0)
    rid=tl.where(cs<count,rid,-1)
    tl.store(RIDS+h*C+cs,rid,cs<C)
    low=(ss<S)&(~high)
    lm=tl.max(tl.where(low,mm,-float('inf')),0)
    lm=tl.where(lm==-float('inf'),0.,lm)
    l=tl.sum(tl.where(low,tl.exp(mm-lm)*ll,0.),0)
    tl.store(LM+h,lm)
    tl.store(LL+h,l)


@triton.jit
def _compact_values(Q,K,V,IDS,HIGH,RIDS,LM,LU,L_ABS,M,L,U,
                    N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,S:tl.constexpr,
                    C:tl.constexpr,BB:tl.constexpr,DD:tl.constexpr):
    h,slot=tl.program_id(0),tl.program_id(1)
    bs=tl.arange(0,BB)
    ds=tl.arange(0,DD)
    q=tl.load(Q+h*D+ds,ds<D,0.).to(tl.float32)
    if slot==0:
        lm=tl.load(LM+h)
        u=tl.full((DD,),0.,tl.float32)
        abs_u=tl.full((DD,),0.,tl.float32)
        for s in range(S):
            if not tl.load(HIGH+h*S+s):
                bid=tl.load(IDS+h*S+s)
                off=((h.to(tl.int64)*N+bid)*B+bs[:,None])*D+ds[None,:]
                kk=tl.load(K+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
                score=tl.sum(kk*q[None,:],1)*(D**-0.5)
                w=tl.where(bs<B,tl.exp(score-lm),0.)
                vv=tl.load(V+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
                weighted=w[:,None]*vv
                u+=tl.sum(weighted,0)
                abs_u+=tl.sum(tl.abs(weighted),0)
        tl.store(LU+h*D+ds,u,ds<D)
        tl.store(L_ABS+h*D+ds,abs_u,ds<D)
    else:
        c=slot-1
        bid=tl.load(RIDS+h*C+c)
        if bid>=0:
            off=((h.to(tl.int64)*N+bid)*B+bs[:,None])*D+ds[None,:]
            kk=tl.load(K+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
            score=tl.sum(kk*q[None,:],1)*(D**-0.5)
            score=tl.where(bs<B,score,-float('inf'))
            m=tl.max(score,0)
            w=tl.exp(score-m)
            l=tl.sum(w,0)
            vv=tl.load(V+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
            u=tl.sum(w[:,None]*vv,0)
        else:
            m=-float('inf')
            l=0.
            u=tl.full((DD,),0.,tl.float32)
        tl.store(M+h*C+c,m)
        tl.store(L+h*C+c,l)
        tl.store(U+(h*C+c)*D+ds,u,ds<D)


@triton.jit
def _compact_membership(P,A,HIGH,RIDS,HIT,MISS,WRONG,LOW_HIT,
                        SP:tl.constexpr,SA:tl.constexpr,SC:tl.constexpr,
                        PP:tl.constexpr,AA:tl.constexpr,CC:tl.constexpr):
    h=tl.program_id(0)
    ps=tl.arange(0,PP)
    aa=tl.arange(0,AA)
    cs=tl.arange(0,CC)
    p=tl.load(P+h*SP+ps,ps<SP,-1)
    a=tl.load(A+h*SA+aa,aa<SA,-2)
    eq=(p[:,None]==a[None,:])&(ps[:,None]<SP)&(aa[None,:]<SA)
    phit=tl.sum(eq.to(tl.int32),1)>0
    miss=tl.sum(eq.to(tl.int32),0)==0
    high=tl.load(HIGH+h*SP+ps,ps<SP,False)
    low_hit=tl.sum((phit&(~high)&(ps<SP)).to(tl.int32),0)>0
    wrong=(~phit)&(~high)&low_hit
    r=tl.load(RIDS+h*SC+cs,cs<SC,-1)
    rhit=tl.sum(((r[:,None]==a[None,:])&(cs[:,None]<SC)&
                 (aa[None,:]<SA)&(r[:,None]>=0)).to(tl.int32),1)>0
    tl.store(HIT+h*SC+cs,rhit,cs<SC)
    tl.store(MISS+h*SA+aa,miss,aa<SA)
    tl.store(WRONG+h*SP+ps,wrong,ps<SP)
    tl.store(LOW_HIT+h,low_hit)


@triton.jit
def _wrong_aggregate(Q,K,V,IDS,ACTIVE,LM,M,L,U,N:tl.constexpr,
                     B:tl.constexpr,D:tl.constexpr,S:tl.constexpr,
                     BB:tl.constexpr,DD:tl.constexpr):
    h=tl.program_id(0)
    bs=tl.arange(0,BB)
    ds=tl.arange(0,DD)
    q=tl.load(Q+h*D+ds,ds<D,0.).to(tl.float32)
    m=tl.load(LM+h)
    z=0.
    u=tl.full((DD,),0.,tl.float32)
    for s in range(S):
        if tl.load(ACTIVE+h*S+s):
            bid=tl.load(IDS+h*S+s)
            off=((h.to(tl.int64)*N+bid)*B+bs[:,None])*D+ds[None,:]
            kk=tl.load(K+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
            score=tl.sum(kk*q[None,:],1)*(D**-0.5)
            w=tl.where(bs<B,tl.exp(score-m),0.)
            vv=tl.load(V+off,(bs[:,None]<B)&(ds[None,:]<D),0.).to(tl.float32)
            z+=tl.sum(w,0)
            u+=tl.sum(w[:,None]*vv,0)
    tl.store(M+h,m)
    tl.store(L+h,z)
    tl.store(U+h*D+ds,u,ds<D)


def _shape(q,k,v,ids):
    assert q.is_cuda and k.is_cuda and v.is_cuda and ids.is_cuda
    assert q.ndim==2 and k.ndim==4 and v.shape==k.shape
    h,n,b,d=k.shape
    assert q.shape==(h,d) and ids.ndim==2 and ids.shape[0]==h
    assert ids.dtype in (torch.int32,torch.int64) and ids.shape[1]>0
    assert all(x.is_contiguous() for x in (q,k,v,ids))
    return h,n,b,d,ids.shape[1]


def _allocate_state(h,s,d,device):
    m=torch.empty((h,s),device=device,dtype=torch.float32)
    return PartialState(m,torch.empty_like(m),torch.empty((h,s,d),device=device,dtype=torch.float32))


def block_partials(q,k,v,ids,active=None):
    """Each block's m, sum exp(logit-m), sum exp(logit-m)*V (all FP32)."""
    h,n,b,d,s=_shape(q,k,v,ids)
    state=_allocate_state(h,s,d,q.device)
    if active is not None:
        assert active.is_contiguous() and active.shape in ((h,), (h,s))
    _partial[(h,s)](q,k,v,ids,active,state.m,state.l,state.u,n,b,d,s,triton.next_power_of_2(b),
                    triton.next_power_of_2(d),active is not None,
                    active is not None and active.ndim==1,num_warps=4)
    return state


def merge_partials(state,include=None):
    h,s,d=state.u.shape
    out=torch.empty((h,d),device=state.u.device,dtype=torch.float32)
    _merge[(h,)](state.m,state.l,state.u,include,out,s,d,triton.next_power_of_2(s),
                 triton.next_power_of_2(d),include is not None,num_warps=4)
    return out


def exact_attention(q,k,v,selected):
    """Split-block, selected-set baseline, with the same FP32 accumulation."""
    return merge_partials(block_partials(q,k,v,selected))


def speculate(q,k,v,predicted,mode='saved',mass_threshold=0.1,*,keep_mass=False):
    """Speculate using CURRENT Q. Threshold uses actual predicted-set mass.

    Hybrid currently materializes all partials before grouping: it does NOT
    claim lower peak storage than saved states. Only low-group states are
    consumed by repair, but the allocated prediction tensor remains live.

    keep_mass exposes the FP32 block maxima and normalizers in predicted-ID
    order. Compact predictions retain their two scalar scratch matrices;
    saved and hybrid predictions reference their existing block states.
    """
    if mode not in ('saved','hybrid','compact'):
        raise ValueError(mode)
    if not 0 <= mass_threshold <= 1:
        raise ValueError('mass_threshold must be in [0,1]')
    if mode=='compact':
        return _speculate_compact(q,k,v,predicted,mass_threshold,keep_mass)
    state=block_partials(q,k,v,predicted)
    mass_m,mass_l=(state.m,state.l) if keep_mass else (None,None)
    if mode=='saved':
        return Prediction(state,mode,mass_m=mass_m,mass_l=mass_l)
    h,s,d=state.u.shape
    high=torch.empty((h,s),device=q.device,dtype=torch.bool)
    low=_allocate_state(h,1,d,q.device)
    low_abs=torch.empty((h,d),device=q.device,dtype=torch.float32)
    _group[(h,)](state.m,state.l,state.u,high,low.m,low.l,low.u,low_abs,s,d,
                 triton.next_power_of_2(s),triton.next_power_of_2(d),mass_threshold,num_warps=4)
    return Prediction(state,mode,high,low,low_abs=low_abs,mass_m=mass_m,mass_l=mass_l)


def _speculate_compact(q,k,v,predicted,mass_threshold,keep_mass):
    """Bounded retained-state version: an extra QK pass avoids full U storage."""
    h,n,b,d,s=_shape(q,k,v,predicted)
    # One spare slot handles thresholds lying close to FP32 mass boundaries.
    # The grouping kernel has an explicit overflow-to-low policy regardless.
    c=s if mass_threshold<=1/s else min(s,int(1/mass_threshold)+1)
    pm=torch.empty((h,s),device=q.device,dtype=torch.float32)
    pl=torch.empty_like(pm)
    state=_allocate_state(h,c,d,q.device)
    low=_allocate_state(h,1,d,q.device)
    low_abs=torch.empty((h,d),device=q.device,dtype=torch.float32)
    high=torch.empty((h,s),device=q.device,dtype=torch.bool)
    retained_ids=torch.empty((h,c),device=q.device,dtype=predicted.dtype)
    overflow=torch.empty((h,),device=q.device,dtype=torch.bool)
    _mass_only[(h,s)](q,k,predicted,pm,pl,n,b,d,s,triton.next_power_of_2(b),
                      triton.next_power_of_2(d),num_warps=4)
    _compact_group[(h,)](pm,pl,predicted,high,retained_ids,low.m,low.l,overflow,
                         s,c,triton.next_power_of_2(s),triton.next_power_of_2(c),mass_threshold,num_warps=4)
    _compact_values[(h,c+1)](q,k,v,predicted,high,retained_ids,low.m,low.u,low_abs,state.m,state.l,state.u,
                             n,b,d,s,c,triton.next_power_of_2(b),triton.next_power_of_2(d),num_warps=4)
    mass_m,mass_l=(pm,pl) if keep_mass else (None,None)
    return Prediction(state,'compact',high,low,retained_ids,overflow,low_abs,mass_m,mass_l)


def repair(q,k,v,selected,predicted,prediction,min_remaining=1.e-3,cancellation_tol=1.e-5):
    """Restore exact selected-set semantics; conservative guards recompute QKV.

    Remaining is residual low-group denominator / original low denominator.
    A wholly rejected low group is discarded directly without subtraction.
    cancellation_tol checks each residual numerator coordinate against a
    low-group absolute sum. This is a heuristic, not a forward-error certificate.
    """
    h,n,b,d,sa=_shape(q,k,v,selected)
    sp=predicted.shape[1]
    sc=prediction.state.m.shape[1]
    hit=torch.empty((h,sc),device=q.device,dtype=torch.bool)
    miss=torch.empty_like(selected,dtype=torch.bool)
    wrong=torch.empty_like(predicted,dtype=torch.bool)
    low_hit=torch.empty((h,),device=q.device,dtype=torch.bool)
    hybrid=prediction.mode in ('hybrid','compact')
    if prediction.mode=='compact':
        _compact_membership[(h,)](predicted,selected,prediction.high,prediction.retained_ids,
                                   hit,miss,wrong,low_hit,sp,sa,sc,triton.next_power_of_2(sp),
                                   triton.next_power_of_2(sa),triton.next_power_of_2(sc),num_warps=4)
    else:
        _membership[(h,)](predicted,selected,prediction.high,hit,miss,wrong,low_hit,
                           sp,sa,triton.next_power_of_2(sp),triton.next_power_of_2(sa),hybrid,num_warps=4)
    a=block_partials(q,k,v,selected,miss)
    p=prediction.state
    out=torch.empty((h,d),device=q.device,dtype=torch.float32)
    fallback=torch.empty((h,),device=q.device,dtype=torch.bool)
    remaining=torch.empty((h,),device=q.device,dtype=torch.float32)
    if not hybrid:
        _saved_merge[(h,)](p.m,p.l,p.u,hit,a.m,a.l,a.u,out,fallback,remaining,sp,sa,d,
                           triton.next_power_of_2(sp),triton.next_power_of_2(sa),triton.next_power_of_2(d),num_warps=4)
    else:
        low=prediction.low
        if prediction.mode=='compact':
            w=_allocate_state(h,1,d,q.device)
            _wrong_aggregate[(h,)](q,k,v,predicted,wrong,low.m,w.m,w.l,w.u,n,b,d,sp,
                                   triton.next_power_of_2(b),triton.next_power_of_2(d),num_warps=4)
        else:
            w=block_partials(q,k,v,predicted,wrong)
        sw=w.m.shape[1]
        _hybrid_merge[(h,)](p.m,p.l,p.u,hit,low.m,low.l,low.u,prediction.low_abs,low_hit,w.m,w.l,w.u,
                            a.m,a.l,a.u,out,fallback,remaining,sw,sa,d,sc,
                            triton.next_power_of_2(sw),triton.next_power_of_2(sa),triton.next_power_of_2(d),triton.next_power_of_2(sc),
                            min_remaining,cancellation_tol,num_warps=4)
        _fallback[(h,)](q,k,v,selected,fallback,out,n,b,d,sa,triton.next_power_of_2(b),
                        triton.next_power_of_2(d),num_warps=4)
    return RepairResult(out,fallback,remaining)


def touch_kv(k,v,predicted):
    """Read predicted KV with a checksum side effect; cache residency not assured."""
    h,n,b,d=k.shape
    s=predicted.shape[1]
    nt=triton.cdiv(b*d,1024)
    scratch=torch.empty((h,s,nt),device=k.device,dtype=torch.float32)
    _touch[(h,s,nt)](k,v,predicted,scratch,n,b,d,s,1024,num_warps=4)
    return scratch
