"""CUDA checks for fused block scoring and score-equivalent top-k ties."""

import torch

from strict_topk.selector import block_scores, select_blocks


def main():
    gen = torch.Generator(device="cuda").manual_seed(20261103)
    q = torch.randn((3, 70), device="cuda", dtype=torch.float16, generator=gen)
    keys = torch.randn((3, 37, 7, 70), device="cuda", dtype=torch.float16, generator=gen)
    lo, hi = keys.amin(dim=2).float(), keys.amax(dim=2).float()
    q[0].zero_()
    q_ref = q.double()[:, None, :]
    reference = torch.where(q_ref >= 0, hi.double(), lo.double()).mul(q_ref).sum(-1)
    old = torch.where(q_ref >= 0, hi * q.float()[:, None], lo * q.float()[:, None]).sum(-1)
    scores = block_scores(q, lo, hi)
    torch.testing.assert_close(old.double(), reference, atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(scores.double(), reference, atol=2e-5, rtol=2e-6)

    for topk in (1, 11, 37):
        ids = select_blocks(q, lo, hi, topk)
        assert ids.shape == (3, topk) and ids.is_contiguous() and ids.dtype == torch.int32
        sorted_ids = ids.sort(-1).values
        assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all().item()
        selected_scores = reference.gather(-1, ids.long()).sort(-1, descending=True).values
        torch.testing.assert_close(selected_scores, reference.topk(topk).values, atol=4e-5, rtol=4e-6)
    print("Fused selector scores and top-k checks passed")


if __name__ == "__main__":
    main()
