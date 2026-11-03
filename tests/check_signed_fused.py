"""Independent FP64 and graph data-change checks for three-launch signed repair."""
import json
from pathlib import Path
import torch

from strict_topk.kernels import speculate
from strict_topk.signed_fused import repair_signed_fused
from tests import check_signed_repair as shared
from tests.check_correctness import reference,error_metrics
from benchmarks.fusion_benchmark import capture


def main():
    shared.repair_signed=repair_signed_fused
    shared.check_strict()
    shared.check_budget()
    # exp(-104) rounds to zero in FP32, but a block-local sum multiplied
    # by exp(-80) retains a measurable contribution from finite BF16 V.
    # Hybrid erasure must reproduce its preparation's order of operations.
    q=torch.ones((1,1),device='cuda',dtype=torch.bfloat16)
    k=torch.full((1,3,8192,1),-200.,device='cuda',dtype=q.dtype)
    k[:,:,0,0]=0.;k[:,2,:,0]=-104.;k[:,2,0,0]=-80.
    v=torch.zeros_like(k);v[:,2,1:,0]=3.e38
    p=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32)
    a=torch.tensor([[0,1]],device='cuda',dtype=torch.int32)
    pred=speculate(q,k,v,p,mode='hybrid',mass_threshold=1.)
    result=repair_signed_fused(q,k,v,a,p,pred)
    underflow_check=error_metrics(result.output,reference(q,k,v,a))
    assert underflow_check['passed'],underflow_check
    assert result.fallback.item()
    gen=torch.Generator(device='cuda').manual_seed(817)
    rows=[]
    for dtype in (torch.float16,torch.bfloat16):
        q=torch.randn((3,7),device='cuda',dtype=dtype,generator=gen)
        k=torch.randn((3,17,5,7),device='cuda',dtype=dtype,generator=gen)
        v=torch.randn(k.shape,device='cuda',dtype=dtype,generator=gen)
        p=torch.tensor([[0,1,2,3,4],[3,4,5,6,7],[8,9,10,11,12]],device='cuda',dtype=torch.int64)
        a=torch.tensor([[1,2,8],[4,9,15],[0,5,16]],device='cuda',dtype=torch.int32)
        for mode in ('saved','hybrid','compact'):
            def fn():
                pred=speculate(q,k,v,p,mode=mode,mass_threshold=.2)
                return repair_signed_fused(q,k,v,a,p,pred)
            graph,result=capture(fn)
            for scale in (1.,-.5,2.):
                v.mul_(scale)
                graph.replay();torch.cuda.synchronize()
                metric=error_metrics(result.output,reference(q,k,v,a))
                assert metric['passed'],(dtype,mode,scale,metric)
                rows.append({'dtype':str(dtype),'mode':mode,'value_multiplier':scale,**metric})
    report={'passed':True,'shared_checks':'All strict adversarial and epsilon budget checks passed',
            'hybrid_underflow_regression':underflow_check,
            'graph_mutation_checks':rows}
    path=Path('results/round2/signed_fused_correctness.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'graph_mutation_checks':len(rows)}),flush=True)


if __name__=='__main__':main()
