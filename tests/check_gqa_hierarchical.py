"""FP64 checks for bounded-tile GQA reduction and changing captured IDs."""
import torch
import hashlib
import json
from pathlib import Path
from strict_topk import gqa,gqa_hierarchical as impl
from benchmarks.critical_path import reference,_prediction
from benchmarks.fusion_benchmark import capture
from tests.check_correctness import error_metrics


def main():
    torch.manual_seed(317)
    checks=0
    shapes=[(24,2,1024,4,256,512),(6,2,96,5,96,37)]
    for dtype,h,hkv,n,b,d,sa in [(dtype,*shape) for dtype in (torch.float16,torch.bfloat16) for shape in shapes]:
        q=torch.randn((h,d),device='cuda',dtype=dtype)
        k=torch.randn((hkv,n,b,d),device='cuda',dtype=dtype)
        v=torch.randn_like(k)
        a=torch.stack([torch.randperm(n,device='cuda')[:sa] for _ in range(h)]).int()
        target=reference(q,k,v,a)
        assert error_metrics(impl.exact(q,k,v,a),target)['passed'];checks+=1
        for hit in [0,sa//2,sa-1,sa]:
            p=_prediction(a,n,sa,hit)
            state=impl.partials(q,k,v,p)
            output=impl.repair_saved(q,k,v,a,p,state)
            assert error_metrics(output,target)['passed'];checks+=1
        def fn():return impl.repair_saved(q,k,v,a,p,impl.partials(q,k,v,p))
        graph,result=capture(fn)
        a.copy_((a+1)%n)
        graph.replay();torch.cuda.synchronize()
        assert error_metrics(result,reference(q,k,v,a))['passed'];checks+=1
        shared=a[:1].expand(h,-1).contiguous()
        for hit in (0,sa//2,sa):
            pred=_prediction(shared,n,max(1,sa//2),min(hit,sa//2))
            out=impl.repair_saved_shared(q,k,v,shared,pred,impl.partials(q,k,v,pred))
            assert error_metrics(out,reference(q,k,v,shared))['passed'];checks+=1
    report={'passed':True,'checks':checks,'dtypes':['float16','bfloat16'],
            'oracle':'FP64 on identical quantized inputs and exact selected IDs',
            'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                             ['strict_topk/gqa_hierarchical.py','tests/check_gqa_hierarchical.py']}}
    output=Path('results/window/correctness.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
