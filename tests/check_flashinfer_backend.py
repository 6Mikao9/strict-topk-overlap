"""Version-pinned vendor correction, changing IDs/counts under one graph."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from strict_topk.flashinfer_backend import FixedAttention,Correction
from benchmarks.fusion_benchmark import capture
from benchmarks.critical_path import reference


def metrics(out,expected,dtype):
    diff=(out.double()-expected).abs()
    # Vendor O is rounded to input dtype. Keep tolerances explicit and also
    # report errors; BF16 tolerance accounts for this unavoidable final cast.
    atol,rtol=(3e-4,3e-4) if dtype==torch.float16 else (2e-3,4e-3)
    return {'passed':bool(torch.isfinite(out).all() and (diff<=atol+rtol*expected.abs()).all()),
            'max_abs_error':diff.max().item(),'atol':atol,'rtol':rtol}


def union_reference(q,k,v,a,p):
    outputs=[]
    group=q.shape[0]//k.shape[0]
    for head,(x,y) in enumerate(zip(a,p)):
        ids=torch.unique(torch.cat((x,y)))[None]
        kv=head//group
        outputs.append(reference(q[head:head+1],k[kv:kv+1],v[kv:kv+1],ids))
    return torch.cat(outputs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/vendor_shared/correctness.json')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.manual_seed(20261103)
    rows=[]
    shapes=[(True,24,2,256,4,128,64,51),(False,4,2,128,32,48,17,13)]
    if not args.smoke:shapes.append((True,24,2,256,4,1100,512,409))
    for dtype in ([torch.float16] if args.smoke else [torch.float16,torch.bfloat16]):
      for shared,h,hkv,d,b,n,sa,sp in shapes:
        q=torch.randn((h,d),device='cuda',dtype=dtype)
        k=torch.randn((hkv,n,b,d),device='cuda',dtype=dtype);v=torch.randn_like(k)
        p=torch.arange(sp,device='cuda',dtype=torch.int32)[None].expand(h,-1).contiguous()
        a=torch.arange(sa,device='cuda',dtype=torch.int32)[None].expand(h,-1).contiguous()
        fixed=FixedAttention(q,k,v,a,shared)
        predict=FixedAttention(q,k,v,p,shared)
        strict=Correction(q,k,v,a,p,shared,True)
        union=Correction(q,k,v,a,p,shared,False)
        # A single graph must survive all-hit, no-hit, ragged final chunks,
        # differing head IDs, and repeated transitions without stale states.
        def run():
            state=predict(q,p,True)
            return fixed(q,a),strict(q,k,v,a,p,state),union(q,k,v,a,p,state)
        graph,out=capture(run)
        for hit in [sp,0,1,sp//2,sp-1,0,sp]:
            aa=torch.cat((torch.arange(hit),torch.arange(sp,sp+sa-hit))).to(device='cuda',dtype=torch.int32)
            a.copy_(aa[None].expand_as(a))
            if not shared:
                shifts=torch.arange(h,device='cuda')[:,None]*3
                a.copy_((a+shifts)%n)
                p.copy_((torch.arange(sp,device='cuda')[None]+shifts)%n)
            q.normal_()
            graph.replay();torch.cuda.synchronize()
            expected=reference(q,k,v,a)
            checks={'serial':metrics(out[0],expected,dtype),
                    'strict':metrics(out[1].output,expected,dtype),
                    'union':metrics(out[2].output,union_reference(q,k,v,a,p),dtype)}
            row={'dtype':str(dtype),'shared':shared,'shape':[h,hkv,d,b,n,sa,sp],
                 'hits':hit,'checks':checks,'fallback_heads':out[1].fallback.sum().item()}
            rows.append(row)
            assert all(x['passed'] for x in checks.values()),row
        if not shared:
            for head in range(h):
                hit=(head*5)%(sp+1)
                aa=torch.cat((torch.arange(hit),torch.arange(sp,sp+sa-hit)))
                a[head].copy_((aa.to(device='cuda')+head*3)%n)
            graph.replay();torch.cuda.synchronize()
            expected=reference(q,k,v,a)
            checks={'strict':metrics(out[1].output,expected,dtype),
                    'union':metrics(out[2].output,union_reference(q,k,v,a,p),dtype)}
            rows.append({'dtype':str(dtype),'adversary':'unequal_counts_across_requests',
                         'checks':checks,'fallback_heads':out[1].fallback.sum().item()})
            assert all(x['passed'] for x in checks.values()),rows[-1]
        if shared:
            # Wrong predictions dominate mass, or signed V cancellation.
            a.copy_(torch.arange(sp-1,sp-1+sa,device='cuda')[None].expand_as(a))
            for adversary in ('wrong_mass','value_cancellation'):
                q.zero_();q[:,0]=1;k.zero_();v.normal_()
                if adversary=='wrong_mass':k[:,:sp-1,:,0]=80*(d**.5)
                else:
                    v.fill_(.01);v[:,:sp-1]=60000 if dtype==torch.float16 else 1.e8
                graph.replay();torch.cuda.synchronize()
                check=metrics(out[1].output,reference(q,k,v,a),dtype)
                row={'dtype':str(dtype),'shape':[h,hkv,d,b,n,sa,sp],
                     'adversary':adversary,'checks':{'strict':check},
                     'fallback_heads':out[1].fallback.sum().item()}
                rows.append(row);assert check['passed'],row
        del graph,out,run,fixed,predict,strict,union
        torch.cuda.empty_cache()
    # Huge common logits erase log2(cardinality) from FP32 vendor LSE.
    q=torch.zeros((24,256),device='cuda',dtype=torch.float16);q[:,0]=65504
    k=torch.zeros((2,4,4,256),device='cuda',dtype=torch.float16);k[:,:,:,0]=65504
    v=torch.tensor([.0625,-.03125,-.03125,.0625],device='cuda',dtype=torch.float16)
    v=v[None,:,None,None].expand_as(k).contiguous()
    p=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32).expand(24,-1).contiguous()
    a=torch.tensor([[0,3]],device='cuda',dtype=torch.int32).expand(24,-1).contiguous()
    predict=FixedAttention(q,k,v,p,True);strict=Correction(q,k,v,a,p,True,True)
    graph,out=capture(lambda:strict(q,k,v,a,p,predict(q,p,True)))
    graph.replay();torch.cuda.synchronize()
    check=metrics(out.output,reference(q,k,v,a),q.dtype)
    rows.append({'adversary':'large_common_logit_lse_rounding','checks':{'strict':check},
                 'fallback_heads':out.fallback.sum().item()})
    assert check['passed'] and bool(out.fallback.all()),rows[-1]
    files=['strict_topk/flashinfer_backend.py','tests/check_flashinfer_backend.py']
    data={'passed':True,'cases':len(rows),'checks':sum(len(x['checks']) for x in rows),'rows':rows,
          'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'passed':True,'checks':data['checks']}))


if __name__=='__main__':main()
