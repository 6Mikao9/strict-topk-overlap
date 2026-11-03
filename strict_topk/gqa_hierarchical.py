"""Bounded-tile reduction for large per-block GQA state arrays."""
import torch
import triton
import triton.language as tl

from . import gqa
from .kernels import _allocate_state,merge_partials


@triton.jit
def _match_ids(A,P,SLOT,SA:tl.constexpr,SP:tl.constexpr,TILE:tl.constexpr,PP:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    aa=tile*TILE+tl.arange(0,TILE)
    pp=tl.arange(0,PP)
    a=tl.load(A+head*SA+aa,aa<SA,-1)
    p=tl.load(P+head*SP+pp,pp<SP,-2)
    equal=(a[:,None]==p[None,:]) & (aa[:,None]<SA) & (pp[None,:]<SP)
    slot=tl.max(tl.where(equal,pp[None,:],-1),1)
    tl.store(SLOT+head*SA+aa,slot,aa<SA)


@triton.jit
def _packed_partials(Q,K,V,IDS,SLOT,M,L,U,
                     N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,GROUP:tl.constexpr,
                     S:tl.constexpr,PACK:tl.constexpr,
                     BB:tl.constexpr,DD:tl.constexpr,REPAIR:tl.constexpr,SHARED:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    positions=tile*PACK+tl.arange(0,PACK)
    tokens,dims=tl.arange(0,BB),tl.arange(0,DD)
    valid=positions<S
    ids=tl.load(IDS+head*S+positions,valid,0)
    enabled=valid
    if REPAIR:
        slot_head=0 if SHARED else head
        slot=tl.load(SLOT+slot_head*S+positions,valid,-1)
        enabled=valid & (slot<0)
    query=tl.load(Q+head*D+dims,dims<D,0.).to(tl.float32)
    offsets=(((head.to(tl.int64)//GROUP)*N+ids[:,None,None])*B+tokens[None,:,None])*D+dims[None,None,:]
    mask=enabled[:,None,None] & (tokens[None,:,None]<B) & (dims[None,None,:]<D)
    keys=tl.load(K+offsets,mask,0.).to(tl.float32)
    scores=tl.sum(keys*query[None,None,:],2)*(D**-.5)
    scores=tl.where(enabled[:,None] & (tokens[None,:]<B),scores,-float('inf'))
    maximum=tl.max(scores,1)
    reference_m=tl.where(enabled,maximum,0.)
    weights=tl.exp(scores-reference_m[:,None])
    values=tl.load(V+offsets,mask,0.).to(tl.float32)
    denominator=tl.sum(weights,1)
    numerator=tl.sum(weights[:,:,None]*values,1)
    tl.store(M+head*S+positions,maximum,enabled)
    tl.store(L+head*S+positions,denominator,enabled)
    tl.store(U+(head*S+positions[:,None])*D+dims[None,:],numerator,
             enabled[:,None] & (dims[None,:]<D))


def partials(q,k,v,selected):
    h,n,b,d,group,sa=gqa._shape(q,k,v,selected)
    state=_allocate_state(h,sa,d,q.device)
    pack=8 if b<=8 else 1
    _packed_partials[(h,triton.cdiv(sa,pack))](q,k,v,selected,None,state.m,state.l,state.u,
        n,b,d,group,sa,pack,triton.next_power_of_2(b),triton.next_power_of_2(d),False,False,num_warps=4)
    return state


@triton.jit
def _groups(M,L,U, PM,PL,PU,SLOT, OM,OL,OU,
            D:tl.constexpr,S:tl.constexpr,P:tl.constexpr,G:tl.constexpr,
            TILE:tl.constexpr,DD:tl.constexpr,REPAIR:tl.constexpr,SHARED:tl.constexpr):
    head,group=tl.program_id(0),tl.program_id(1)
    positions=group*TILE+tl.arange(0,TILE)
    dims=tl.arange(0,DD)
    valid=positions<S
    read_current=valid
    if REPAIR:
        slot_head=0 if SHARED else head
        slots=tl.load(SLOT+slot_head*S+positions,valid,-1)
        hit=valid & (slots>=0)
        read_current=valid & ~hit
    m=tl.load(M+head*S+positions,read_current,-float('inf'))
    z=tl.load(L+head*S+positions,read_current,0.)
    u=tl.load(U+(head*S+positions[:,None])*D+dims[None,:],
              read_current[:,None] & (dims[None,:]<D),0.)
    if REPAIR:
        pm=tl.load(PM+head*P+slots,hit,-float('inf'))
        pz=tl.load(PL+head*P+slots,hit,0.)
        pu=tl.load(PU+(head*P+slots[:,None])*D+dims[None,:],
                   hit[:,None] & (dims[None,:]<D),0.)
        m=tl.where(hit,pm,m)
        z=tl.where(hit,pz,z)
        u=tl.where(hit[:,None],pu,u)
    maximum=tl.max(m,0)
    weight=tl.exp(m-maximum)
    denominator=tl.sum(weight*z,0)
    numerator=tl.sum(weight[:,None]*u,0)
    tl.store(OM+head*G+group,maximum)
    tl.store(OL+head*G+group,denominator)
    tl.store(OU+(head*G+group)*D+dims,numerator,dims<D)


def _reduce(state,prediction=None,slot=None,shared=False):
    h,s,d=state.u.shape
    tile=32
    groups=triton.cdiv(s,tile)
    reduced=_allocate_state(h,groups,d,state.u.device)
    pred=prediction if prediction is not None else state
    _groups[(h,groups)](state.m,state.l,state.u,pred.m,pred.l,pred.u,slot,
        reduced.m,reduced.l,reduced.u,d,s,pred.m.shape[1],groups,tile,
        triton.next_power_of_2(d),prediction is not None,shared,num_warps=4)
    return merge_partials(reduced)


def exact(q,k,v,selected):
    return _reduce(partials(q,k,v,selected))


def _repair(q,k,v,selected,predicted,state,shared):
    h,n,b,d,group,sa=gqa._shape(q,k,v,selected)
    gqa._validate_ids(q,predicted)
    sp=predicted.shape[1]
    if state.u.shape!=(h,sp,d) or state.m.shape!=(h,sp) or state.l.shape!=(h,sp):
        raise ValueError('prediction state shape does not match predicted blocks')
    scratch=_allocate_state(h,sa,d,q.device)
    slot_heads=1 if shared else h
    slot=torch.empty((slot_heads,sa),device=q.device,dtype=torch.int32)
    _match_ids[(slot_heads,triton.cdiv(sa,16))](selected,predicted,slot,sa,sp,16,
        triton.next_power_of_2(sp),num_warps=4)
    pack=8 if b<=8 else 1
    _packed_partials[(h,triton.cdiv(sa,pack))](q,k,v,selected,slot,
        scratch.m,scratch.l,scratch.u,n,b,d,group,sa,pack,
        triton.next_power_of_2(b),triton.next_power_of_2(d),True,shared,num_warps=4)
    return _reduce(scratch,state,slot,shared)


def repair_saved(q,k,v,selected,predicted,state):
    return _repair(q,k,v,selected,predicted,state,False)


def repair_saved_shared(q,k,v,selected,predicted,state):
    """QSA-style shared selection: every head must have identical ordered IDs.

    Both selected and predicted arrays must obey this caller-validated
    contract. Only the first row is used to construct the membership map.
    Attention arithmetic and states remain independent for every query head.
    """
    return _repair(q,k,v,selected,predicted,state,True)
