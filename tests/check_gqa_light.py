"""FP64 checks for single-launch and statically dispatched GQA repair."""

import torch

from strict_topk import gqa, gqa_light
from tests.check_gqa import reference


def check(q, k, v, p, a):
    expected = reference(q, k, v, a)
    state = gqa.partials(q, k, v, p)
    # Exercise both auto estimates regardless of the actual miss count.
    outputs = [
        gqa_light.repair_saved_light(q, k, v, a, p, state),
        gqa_light.repair_saved_auto(q, k, v, a, p, state, 0.),
        gqa_light.repair_saved_auto(q, k, v, a, p, state, 1.),
    ]
    for output in outputs:
        assert output.shape == q.shape and output.dtype == torch.float32
        torch.testing.assert_close(output.double(), expected, atol=3e-4, rtol=3e-4)


def main():
    gen = torch.Generator(device="cuda").manual_seed(20261103)
    q = torch.randn((16, 13), device="cuda", dtype=torch.float16, generator=gen)
    k = torch.randn((4, 23, 7, 13), device="cuda", dtype=torch.float16, generator=gen)
    v = torch.randn(k.shape, device="cuda", dtype=torch.float16, generator=gen)
    perms = torch.stack([torch.randperm(23, device="cuda", generator=gen) for _ in range(16)])
    for sp, sa, overlap in [(6, 6, 0), (6, 6, 3), (6, 6, 5), (6, 6, 6), (7, 5, 2), (3, 7, 2)]:
        p = perms[:, :sp].contiguous()
        a = torch.cat((perms[:, :overlap], perms[:, sp:sp + sa - overlap]), dim=1)
        check(q, k, v, p, a.flip(1).int())

    q.zero_()
    q[:, 0] = 1
    k[..., 0] = -5000
    k[:, 1, :, 0] = 2000
    k[:, 3, :, 0] = 5000
    p = torch.tensor([[0, 3, 2]] * 16, device="cuda", dtype=torch.int32)
    a = torch.tensor([[2, 1, 0]] * 16, device="cuda", dtype=torch.int64)
    check(q, k, v, p, a)

    q = torch.ones((2, 1), device="cuda", dtype=torch.float16)
    k = torch.tensor([[[[2.]], [[-3.]]]], device="cuda", dtype=torch.float16)
    v = torch.tensor([[[[8.]], [[-5.]]]], device="cuda", dtype=torch.float16)
    p = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    check(q, k, v, p, p)
    check(q, k, v, p, 1 - p)
    print("Native GQA light and auto exact FP64 checks passed")


if __name__ == "__main__":
    main()
