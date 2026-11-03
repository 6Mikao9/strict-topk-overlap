"""FP64 checks for the shared positive and negative correction kernel."""

import torch

from strict_topk.kernels import speculate
from strict_topk.signed_repair import repair_signed
from tests.check_correctness import cases, error_metrics, reference


def check_strict():
    for dtype in (torch.float16, torch.bfloat16):
        for name, q, k, v, p, a in cases(dtype=dtype):
            expected = reference(q, k, v, a)
            for mode in ("saved", "hybrid", "compact"):
                thresholds = (0.2,) if mode == "saved" else (0.05, 0.2, 1.)
                for threshold in thresholds:
                    prediction = speculate(q, k, v, p, mode=mode, mass_threshold=threshold)
                    result = repair_signed(q, k, v, a, p, prediction)
                    metric = error_metrics(result.output, expected)
                    assert metric["passed"], (name, dtype, mode, threshold, metric)
                    assert result.approximate_heads is None
                    if mode != "saved" and threshold == 1.:
                        if name == "signed_low_aggregate_cancellation":
                            assert result.fallback.all().item()
                        if name == "discard_high_logit_low_group":
                            assert not result.fallback.any().item()


def check_budget():
    q = torch.ones((1, 1), device="cuda", dtype=torch.float16)
    k = torch.tensor([3., 0., -3., 0.], device=q.device, dtype=q.dtype).view(1, 4, 1, 1)
    v = torch.tensor([100., 1., 8., 2.], device=q.device, dtype=q.dtype).view_as(k)
    p = torch.tensor([[0, 1, 2]], device=q.device, dtype=torch.int64)
    a = torch.tensor([[1, 3]], device=q.device, dtype=torch.int32)
    extra_set = torch.tensor([[1, 3, 2]], device=q.device, dtype=torch.int32)
    strict = reference(q, k, v, a)
    with_extra = reference(q, k, v, extra_set)
    extra_mass = k[0, extra_set[0].long(), 0, 0].double().softmax(dim=0)[-1].item()
    for mode in ("hybrid", "compact"):
        state = speculate(q, k, v, p, mode=mode, mass_threshold=.2, keep_mass=True)
        low_budget = repair_signed(q, k, v, a, p, state, max_extra_mass=.01)
        assert not low_budget.approximate_heads.item()
        assert low_budget.extra_mass_bound.item() == 0.
        assert low_budget.positive_jobs.item() == 1 and low_budget.total_jobs.item() == 2
        assert error_metrics(low_budget.output, strict)["passed"]

        high_budget = repair_signed(q, k, v, a, p, state, max_extra_mass=.1)
        assert high_budget.approximate_heads.item()
        assert extra_mass <= high_budget.extra_mass_bound.item() <= .1
        assert high_budget.positive_jobs.item() == 1 and high_budget.total_jobs.item() == 1
        assert error_metrics(high_budget.output, with_extra)["passed"]

        forced = repair_signed(q, k, v, a, p, state, cancellation_tol=2., max_extra_mass=.1)
        assert forced.fallback.item() and not forced.approximate_heads.item()
        assert forced.extra_mass_bound.item() == 0.
        assert error_metrics(forced.output, strict)["passed"]


if __name__ == "__main__":
    check_strict()
    check_budget()
    print("Parallel signed repair FP64 and budget checks passed")
