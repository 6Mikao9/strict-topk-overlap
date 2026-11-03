"""FP64 oracle checks for exact and predicted-union native GQA."""

import math

import torch

from strict_topk import gqa


def reference(q, k, v, ids):
    group = q.shape[0] // k.shape[0]
    kv_heads = torch.arange(q.shape[0], device=q.device)[:, None] // group
    keys = k[kv_heads, ids.long()].double()
    values = v[kv_heads, ids.long()].double()
    logits = torch.einsum("hd,hsbd->hsb", q.double(), keys) / math.sqrt(q.shape[-1])
    weights = logits.flatten(1).softmax(-1).view_as(logits)
    return torch.einsum("hsb,hsbd->hd", weights, values)


def check(q, k, v, p, a):
    is_hit = (a[:, :, None] == p[:, None, :]).any(-1)
    union = torch.cat((p, a[~is_hit].reshape(a.shape[0], -1)), dim=1)
    expected = reference(q, k, v, a)
    expected_union = reference(q, k, v, union)
    state = gqa.partials(q, k, v, p)
    for out, target in [
        (gqa.exact(q, k, v, a), expected),
        (gqa.repair_saved(q, k, v, a, p, state), expected),
        (gqa.repair_union(q, k, v, a, p, state), expected_union),
    ]:
        assert out.shape == q.shape and out.dtype == torch.float32
        torch.testing.assert_close(out.double(), target, atol=3e-4, rtol=3e-4)


def main():
    gen = torch.Generator(device="cuda").manual_seed(20261103)
    q = torch.randn((6, 13), device="cuda", dtype=torch.float16, generator=gen)
    k = torch.randn((2, 23, 7, 13), device="cuda", dtype=torch.float16, generator=gen)
    v = torch.randn(k.shape, device="cuda", dtype=torch.float16, generator=gen)
    perms = torch.stack([torch.randperm(23, device="cuda", generator=gen) for _ in range(6)])
    for sp, sa, overlap in [(6, 6, 0), (6, 6, 3), (6, 6, 6), (7, 5, 2), (3, 7, 2)]:
        p = perms[:, :sp].contiguous()
        a = torch.cat((perms[:, :overlap], perms[:, sp:sp + sa - overlap]), dim=1)
        check(q, k, v, p, a.flip(1).int())

    q[0].zero_()  # Tied scores can select any of the equivalent block IDs.
    lo, hi = k.amin(2).float(), k.amax(2).float()
    ids = gqa.select_blocks(q, lo, hi, 5)
    kv_heads = torch.arange(q.shape[0], device=q.device) // (q.shape[0] // k.shape[0])
    q_ref = q.double()[:, None, :]
    scores = (torch.where(q_ref >= 0, hi[kv_heads].double(), lo[kv_heads].double()) * q_ref).sum(-1)
    assert ids.shape == (q.shape[0], 5) and ids.dtype == torch.int32 and ids.is_contiguous()
    selected_scores = scores.gather(1, ids.long()).sort(1, descending=True).values
    torch.testing.assert_close(selected_scores, scores.topk(5).values, atol=2e-5, rtol=2e-6)

    q.zero_()
    q[:, 0] = 1
    k[..., 0] = -5000
    k[:, 1, :, 0] = 2000
    k[:, 3, :, 0] = 5000
    p = torch.tensor([[0, 3, 2]] * 6, device="cuda", dtype=torch.int32)
    a = torch.tensor([[2, 1, 0]] * 6, device="cuda", dtype=torch.int64)
    check(q, k, v, p, a)

    q = torch.ones((2, 1), device="cuda", dtype=torch.float16)
    k = torch.tensor([[[[2.]], [[-3.]]]], device="cuda", dtype=torch.float16)
    v = torch.tensor([[[[8.]], [[-5.]]]], device="cuda", dtype=torch.float16)
    p = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    check(q, k, v, p, p)
    check(q, k, v, p, 1 - p)
    print("Native GQA exact and union FP64 checks passed")


if __name__ == "__main__":
    main()
