"""FlashInfer attention on both sides of speculative Top-K execution.

Pinned to 0.3.1. Dynamic repair requests hold at most 128 KV tokens, so the
FA2 planner and runtime always use one KV tile per request. Empty requests
execute one valid dummy page and are excluded from state merging. This is
a version-specific scheduling invariant, not a general mutable-plan API.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class State:
    output: torch.Tensor
    lse: torch.Tensor


@dataclass
class Result:
    output: torch.Tensor
    fallback: torch.Tensor
    counts: torch.Tensor


def _version():
    import flashinfer
    if flashinfer.__version__ != '0.3.1':
        raise RuntimeError('Dynamic one-tile repair is validated only for FlashInfer 0.3.1')


@triton.jit
def _pages(IDS, PAGE, S: tl.constexpr, N: tl.constexpr,
           ROW_GROUP: tl.constexpr, KV_GROUP: tl.constexpr, SS: tl.constexpr):
    r = tl.program_id(0)
    s = tl.arange(0, SS)
    block = tl.load(IDS + r * ROW_GROUP * S + s, s < S, 0)
    tl.store(PAGE + r * S + s, block + (r // KV_GROUP) * N, s < S)


class FixedAttention:
    """Fixed selected cardinality; GPU page-ID updates are part of run().

    shared=True requires identical ordered ID rows across all query heads.
    Each KV head is a request with its actual group of query heads, avoiding
    KV copies and preserving native GQA in the FA2 tensor-core backend.
    """
    def __init__(self, q, k, v, selected, shared=False):
        _version()
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
        h, d = q.shape
        hkv, n, b, _ = k.shape
        if h % hkv or selected.shape[0] != h:
            raise ValueError('Invalid GQA shape')
        if shared and not torch.equal(selected, selected[:1].expand_as(selected)):
            raise ValueError('shared=True requires shared selected IDs')
        self.shared, self.s, self.n, self.d = shared, selected.shape[1], n, d
        self.g = h // hkv if shared else 1
        self.r = hkv if shared else h
        self.kv_group = 1 if shared else h // hkv
        self.cache = (k.view(hkv * n, b, 1, d), v.view(hkv * n, b, 1, d))
        self.indices = torch.empty(self.r * self.s, device=q.device, dtype=torch.int32)
        indptr = torch.arange(self.r + 1, dtype=torch.int32) * self.s
        lengths = torch.full((self.r,), b, dtype=torch.int32)
        self.workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, kv_layout='NHD', use_cuda_graph=True, use_tensor_cores=True,
            paged_kv_indptr_buffer=torch.empty_like(indptr, device=q.device),
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=torch.empty_like(lengths, device=q.device))
        self.write_pages(selected)
        self.wrapper.plan(indptr, self.indices.clone(), lengths, self.g, 1, d, b,
                          q_data_type=q.dtype, kv_data_type=k.dtype)

    def write_pages(self, ids):
        _pages[(self.r,)](ids, self.indices, self.s, self.n, self.g, self.kv_group,
                         triton.next_power_of_2(self.s))

    def __call__(self, q, ids, return_lse=False):
        self.write_pages(ids)
        value = self.wrapper.run(q.view(self.r, self.g, self.d), self.cache,
                                 return_lse=return_lse)
        if return_lse:
            return State(value[0].view_as(q), value[1].view(q.shape[0]))
        return value.view_as(q)


@triton.jit
def _classify(A, P, FLAGS, SA: tl.constexpr, SP: tl.constexpr, G: tl.constexpr,
              AT: tl.constexpr, PP: tl.constexpr, AA: tl.constexpr,
              TILE: tl.constexpr, SIGNED: tl.constexpr):
    r, tile = tl.program_id(0), tl.program_id(1)
    x = tl.arange(0, TILE)
    if tile < AT:
        pos = tile * TILE + x
        pp = tl.arange(0, PP)
        av = tl.load(A + r * G * SA + pos, pos < SA, -1)
        pv = tl.load(P + r * G * SP + pp, pp < SP, -2)
        hit = tl.sum((av[:, None] == pv[None, :]).to(tl.int32), 1) > 0
        tl.store(FLAGS + r * (SA + SP) + pos, ~hit, pos < SA)
    elif SIGNED:
        pos = (tile - AT) * TILE + x
        aa = tl.arange(0, AA)
        pv2 = tl.load(P + r * G * SP + pos, pos < SP, -1)
        av2 = tl.load(A + r * G * SA + aa, aa < SA, -2)
        hit = tl.sum((pv2[:, None] == av2[None, :]).to(tl.int32), 1) > 0
        tl.store(FLAGS + r * (SA + SP) + SA + pos, ~hit, pos < SP)


@triton.jit
def _compact(A, P, FLAGS, QUEUE, COUNTS, SA: tl.constexpr, SP: tl.constexpr,
             G: tl.constexpr, R: tl.constexpr, AA: tl.constexpr, PP: tl.constexpr):
    r, sign = tl.program_id(0), tl.program_id(1)
    if sign == 0:
        apos = tl.arange(0, AA)
        atake = tl.load(FLAGS + r * (SA + SP) + apos, apos < SA, 0)
        arank = tl.cumsum(atake, 0) - 1
        aids = tl.load(A + r * G * SA + apos, apos < SA, 0)
        tl.store(QUEUE + r * (SA + SP) + arank, aids, atake != 0)
        count = tl.sum(atake, 0)
    else:
        ppos = tl.arange(0, PP)
        ptake = tl.load(FLAGS + r * (SA + SP) + SA + ppos, ppos < SP, 0)
        prank = tl.cumsum(ptake, 0) - 1
        pids = tl.load(P + r * G * SP + ppos, ppos < SP, 0)
        tl.store(QUEUE + r * (SA + SP) + SA + prank, pids, ptake != 0)
        count = tl.sum(ptake, 0)
    tl.store(COUNTS + sign * R + r, count)


@triton.jit
def _tables(Q, QUEUE, COUNTS, INDPTR, INDICES, QQ, ACTIVE,
            R: tl.constexpr, G: tl.constexpr, KV_GROUP: tl.constexpr,
            N: tl.constexpr, D: tl.constexpr, SA: tl.constexpr, SP: tl.constexpr,
            C: tl.constexpr, TA: tl.constexpr, TP: tl.constexpr,
            JOBS: tl.constexpr, JJ: tl.constexpr, CC: tl.constexpr,
            GG: tl.constexpr, DD: tl.constexpr):
    job = tl.program_id(0)
    all_jobs = tl.arange(0, JJ)
    negative = all_jobs >= R * TA
    rr = tl.where(negative, (all_jobs - R * TA) // TP, all_jobs // TA)
    chunk = tl.where(negative, (all_jobs - R * TA) % TP, all_jobs % TA)
    counts = tl.load(COUNTS + negative.to(tl.int32) * R + rr, all_jobs < JOBS, 0)
    actual = tl.minimum(tl.maximum(counts - chunk * C, 0), C)
    allocated = tl.where(all_jobs < JOBS, tl.maximum(actual, 1), 0)
    offset = tl.sum(tl.where(all_jobs < job, allocated, 0), 0)
    size = tl.sum(tl.where(all_jobs == job, allocated, 0), 0)
    active = tl.sum(tl.where(all_jobs == job, actual, 0), 0)
    tl.store(INDPTR + job, offset)
    if job == JOBS - 1:
        tl.store(INDPTR + JOBS, offset + size)
    tl.store(ACTIVE + job, active)
    neg = job >= R * TA
    r = tl.where(neg, (job - R * TA) // TP, job // TA)
    part = tl.where(neg, (job - R * TA) % TP, job % TA)
    pages = tl.arange(0, CC)
    ids = tl.load(QUEUE + r * (SA + SP) + tl.where(neg, SA, 0) + part * C + pages,
                  pages < active, 0)
    tl.store(INDICES + offset + pages, ids + (r // KV_GROUP) * N, pages < size)
    head, dim = tl.arange(0, GG), tl.arange(0, DD)
    query = tl.load(Q + (r * G + head[:, None]) * D + dim[None, :],
                    (head[:, None] < G) & (dim[None, :] < D), 0)
    tl.store(QQ + (job * G + head[:, None]) * D + dim[None, :], query,
             (head[:, None] < G) & (dim[None, :] < D))


@triton.jit
def _merge(Q,K,V,A,PO,PS,CO,CS,ACTIVE,COUNTS,OUT,BAD,
           R:tl.constexpr,G:tl.constexpr,KV_GROUP:tl.constexpr,N:tl.constexpr,
           B:tl.constexpr,D:tl.constexpr,SA:tl.constexpr,SP:tl.constexpr,
           TA:tl.constexpr,TP:tl.constexpr,TT:tl.constexpr,BB:tl.constexpr,
           DD:tl.constexpr,SIGNED:tl.constexpr,UNIT:tl.constexpr):
    head=tl.program_id(0)
    r,gh=head//G,head%G
    t,dim=tl.arange(0,TT),tl.arange(0,DD)
    neg=t>=TA
    request=tl.where(neg,R*TA+r*TP+t-TA,r*TA+t)
    valid=t<(TA+TP if SIGNED else TA)
    active=tl.load(ACTIVE+request,valid,0)>0
    ls=tl.load(CS+request*G+gh,active,-float('inf'))
    co=tl.load(CO+(request[:,None]*G+gh)*D+dim[None,:],
               active[:,None] & (dim[None,:]<D),0.).to(tl.float32)
    keep_prediction=True
    if SIGNED:
        keep_prediction=tl.load(COUNTS+R+r)<SP
        active=active & (~neg | keep_prediction)
        ls=tl.where(active,ls,-float('inf'))
    ps=tl.load(PS+head)
    po=tl.load(PO+head*D+dim,dim<D,0.).to(tl.float32)
    ps=tl.where(keep_prediction,ps,-float('inf'))
    scale=tl.maximum(ps,tl.max(ls,0))
    pw=tl.exp2(ps-scale)
    w=tl.exp2(ls-scale)
    sign=tl.where(neg & SIGNED,-1.,1.)
    denominator=pw+tl.sum(w*sign,0)
    numerator=pw*po+tl.sum((w*sign)[:,None]*co,0)
    out=numerator/denominator
    bad=False
    if SIGNED:
        positive=pw+tl.sum(tl.where(~neg,w,0.),0)
        # Guard low-precision vendor O subtraction and poorly conditioned
        # mass. This heuristic is tested adversarially, not an error proof.
        envelope=pw*tl.abs(po)+tl.sum(w[:,None]*tl.abs(co),0)
        total=pw+tl.sum(w,0)
        lse_magnitude=tl.maximum(tl.where(keep_prediction,tl.abs(ps),0.),
                                 tl.max(tl.where(active,tl.abs(ls),0.),0))
        lse_error=2.e-6+lse_magnitude*(2.**-21)
        error=(2.*UNIT*envelope+lse_error*(envelope+total*tl.abs(out)))/denominator
        unsafe=(error>2.e-4+2.e-4*tl.abs(out)) & (dim<D)
        bad=(denominator<=0) | (denominator/positive<.05)
        bad=bad | (tl.sum(unsafe.to(tl.int32),0)>0)
        bad=bad | (tl.sum(((out!=out)|(tl.abs(out)==float('inf'))).to(tl.int32),0)>0)
    tl.store(OUT+head*D+dim,out,dim<D)
    tl.store(BAD+head,bad)


@triton.jit
def _fallback_parts(Q,K,V,A,BAD,M,L,U,N:tl.constexpr,B:tl.constexpr,D:tl.constexpr,
                    GROUP:tl.constexpr,SA:tl.constexpr,PARTS:tl.constexpr,
                    TOKENS:tl.constexpr,DD:tl.constexpr):
    head,part=tl.program_id(0),tl.program_id(1)
    if tl.load(BAD+head):
        token=part*TOKENS+tl.arange(0,TOKENS)
        dim=tl.arange(0,DD)
        valid=token<SA*B
        ids=tl.load(A+head*SA+token//B,valid,0)
        offsets=(((head.to(tl.int64)//GROUP)*N+ids[:,None])*B+token[:,None]%B)*D+dim[None,:]
        mask=valid[:,None] & (dim[None,:]<D)
        query=tl.load(Q+head*D+dim,dim<D,0.).to(tl.float32)
        keys=tl.load(K+offsets,mask,0.).to(tl.float32)
        values=tl.load(V+offsets,mask,0.).to(tl.float32)
        logits=tl.sum(keys*query[None,:],1)*(D**-.5)
        logits=tl.where(valid,logits,-float('inf'))
        maximum=tl.max(logits,0)
        weights=tl.exp(logits-maximum)
        denominator=tl.sum(weights,0)
        numerator=tl.sum(weights[:,None]*values,0)
        tl.store(M+head*PARTS+part,maximum)
        tl.store(L+head*PARTS+part,denominator)
        tl.store(U+(head*PARTS+part)*D+dim,numerator,dim<D)


@triton.jit
def _fallback_merge(M,L,U,BAD,OUT,PARTS:tl.constexpr,D:tl.constexpr,
                    PP:tl.constexpr,DD:tl.constexpr):
    head=tl.program_id(0)
    if tl.load(BAD+head):
        part,dim=tl.arange(0,PP),tl.arange(0,DD)
        m=tl.load(M+head*PARTS+part,part<PARTS,-float('inf'))
        l=tl.load(L+head*PARTS+part,part<PARTS,0.)
        u=tl.load(U+(head*PARTS+part[:,None])*D+dim[None,:],
                  (part[:,None]<PARTS)&(dim[None,:]<D),0.)
        weights=tl.exp(m-tl.max(m,0))
        out=tl.sum(weights[:,None]*u,0)/tl.sum(weights*l,0)
        tl.store(OUT+head*D+dim,out,dim<D)


class Correction:
    """Pack misses and wrong predictions into one library attention call."""
    def __init__(self,q,k,v,selected,predicted,shared=False,signed=True):
        _version()
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
        h,d=q.shape
        hkv,n,b,_=k.shape
        if 128 % b or b>128:
            raise ValueError('This version requires a block size dividing 128')
        if shared and any(not torch.equal(x,x[:1].expand_as(x)) for x in (selected,predicted)):
            raise ValueError('shared=True requires shared selected and predicted IDs')
        self.r,self.g=(hkv,h//hkv) if shared else (h,1)
        self.kv_group=1 if shared else h//hkv
        self.sa,self.sp=selected.shape[1],predicted.shape[1]
        self.c=128//b
        self.ta,self.tp=triton.cdiv(self.sa,self.c),triton.cdiv(self.sp,self.c)
        self.jobs=self.r*(self.ta+(self.tp if signed else 0))
        self.n,self.b,self.d,self.signed=n,b,d,signed
        self.cache=(k.view(hkv*n,b,1,d),v.view(hkv*n,b,1,d))
        kw={'device':q.device,'dtype':torch.int32}
        self.flags=torch.empty((self.r,self.sa+self.sp),**kw)
        self.queue=torch.empty_like(self.flags)
        self.counts=torch.zeros(2*self.r,**kw)
        self.active=torch.empty(self.jobs,**kw)
        self.indptr=torch.empty(self.jobs+1,**kw)
        self.indices=torch.zeros(self.jobs*self.c,**kw)
        self.qq=torch.empty((self.jobs,self.g,d),device=q.device,dtype=q.dtype)
        qo=torch.arange(self.jobs+1,dtype=torch.int32)
        ptr=qo*self.c
        lengths=torch.full((self.jobs,),b,dtype=torch.int32)
        self.workspace=torch.empty(128*1024*1024,device=q.device,dtype=torch.uint8)
        self.wrapper=BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,kv_layout='NHD',use_cuda_graph=True,backend='fa2',
            qo_indptr_buf=torch.empty_like(qo,device=q.device),
            paged_kv_indptr_buf=self.indptr,paged_kv_indices_buf=self.indices,
            paged_kv_last_page_len_buf=torch.empty_like(lengths,device=q.device))
        self.wrapper.plan(qo,ptr,self.indices.clone(),lengths,self.g,1,d,b,
                          q_data_type=q.dtype,kv_data_type=k.dtype,causal=False)

    def __call__(self,q,k,v,selected,predicted,state):
        r,g,sa,sp=self.r,self.g,self.sa,self.sp
        at,pt=triton.cdiv(sa,16),triton.cdiv(sp,16)
        _classify[(r,at+(pt if self.signed else 0))](selected,predicted,self.flags,
            sa,sp,g,at,triton.next_power_of_2(sp),triton.next_power_of_2(sa),16,self.signed)
        _compact[(r,2 if self.signed else 1)](selected,predicted,self.flags,self.queue,
            self.counts,sa,sp,g,r,triton.next_power_of_2(sa),triton.next_power_of_2(sp))
        _tables[(self.jobs,)](q,self.queue,self.counts,self.indptr,self.indices,self.qq,self.active,
            r,g,self.kv_group,self.n,self.d,sa,sp,self.c,self.ta,self.tp,self.jobs,
            triton.next_power_of_2(self.jobs),triton.next_power_of_2(self.c),
            triton.next_power_of_2(g),triton.next_power_of_2(self.d))
        co,cs=self.wrapper.run(self.qq,self.cache,return_lse=True)
        out=torch.empty_like(q,dtype=torch.float32)
        bad=torch.empty(q.shape[0],device=q.device,dtype=torch.bool)
        _merge[(q.shape[0],)](q,k,v,selected,state.output,state.lse,co,cs,self.active,self.counts,
            out,bad,r,g,self.kv_group,self.n,self.b,self.d,sa,sp,self.ta,self.tp,
            triton.next_power_of_2(self.ta+(self.tp if self.signed else 0)),
            triton.next_power_of_2(self.b),triton.next_power_of_2(self.d),self.signed,
            2.**(-11 if q.dtype==torch.float16 else -8),num_warps=4)
        if self.signed:
            parts=triton.cdiv(sa*self.b,128)
            m=torch.empty((q.shape[0],parts),device=q.device,dtype=torch.float32)
            l=torch.empty_like(m)
            u=torch.empty((q.shape[0],parts,self.d),device=q.device,dtype=torch.float32)
            _fallback_parts[(q.shape[0],parts)](q,k,v,selected,bad,m,l,u,self.n,self.b,self.d,
                g*self.kv_group,sa,parts,128,triton.next_power_of_2(self.d),num_warps=4)
            _fallback_merge[(q.shape[0],)](m,l,u,bad,out,parts,self.d,
                triton.next_power_of_2(parts),triton.next_power_of_2(self.d),num_warps=4)
        return Result(out,bad,self.counts)
