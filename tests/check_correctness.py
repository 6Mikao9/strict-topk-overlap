"""GPU correctness/adversarial suite; the oracle accumulates quantized inputs in FP64."""
import argparse
import json
import math
from pathlib import Path

import torch

from strict_topk.kernels import exact_attention, speculate, repair


def reference(q, k, v, ids):
    h = torch.arange(q.shape[0], device=q.device)[:, None]
    ks, vs = k[h, ids.long()].double(), v[h, ids.long()].double()
    logits = torch.einsum("hd,hkbd->hkb", q.double(), ks) / math.sqrt(q.shape[-1])
    weights = logits.flatten(1).softmax(-1).view_as(logits)
    return torch.einsum("hkb,hkbd->hd", weights, vs)


def error_metrics(output, expected):
    diff = (output.double() - expected).abs()
    tolerance = 3e-4 + 3e-4 * expected.abs()
    return {
        "max_abs_error": diff.max().item(),
        "max_head_l2_relative": (
            diff.square().sum(-1).sqrt() / expected.square().sum(-1).sqrt().clamp_min(1e-12)
        ).max().item(),
        "per_head_max_abs_error": diff.amax(-1).tolist(),
        "finite": bool(torch.isfinite(output).all()),
        "passed": bool((diff <= tolerance).all() and torch.isfinite(output).all()),
        "tolerance": "abs(error) <= 3e-4 + 3e-4 * abs(FP64 reference)",
    }


def cases(device="cuda", dtype=torch.float16):
    gen = torch.Generator(device=device).manual_seed(20261103)
    # Ordinary tests cover arbitrary block lengths, multiple heads and D=128.
    for h, n, b, d, count in [(1, 19, 16, 32, 5), (4, 32, 32, 64, 8), (8, 24, 64, 128, 8)]:
        q = torch.randn(h, d, generator=gen, device=device, dtype=dtype)
        k = torch.randn(h, n, b, d, generator=gen, device=device, dtype=dtype)
        v = torch.randn(h, n, b, d, generator=gen, device=device, dtype=dtype)
        p = torch.arange(count, device=device, dtype=torch.int32).expand(h, -1).contiguous()
        for overlap in [0, count // 2, count]:
            a = torch.arange(count - overlap, 2 * count - overlap, device=device, dtype=torch.int32)
            yield f"random_h{h}_b{b}_d{d}_overlap{overlap}of{count}", q, k, v, p, a.expand(h, -1).contiguous()

    # Non-power-of-two tiles, unequal prediction/selection cardinality, and
    # independent unsorted unique IDs per head stress masking/membership.
    h, n, b, d = 4, 31, 17, 48
    q = torch.randn(h, d, generator=gen, device=device, dtype=dtype)
    k = torch.randn(h, n, b, d, generator=gen, device=device, dtype=dtype)
    v = torch.randn(h, n, b, d, generator=gen, device=device, dtype=dtype)
    for sp, sa, overlap in [(8, 5, 0), (8, 5, 2), (8, 5, 5), (3, 7, 2)]:
        preds, sels = [], []
        for _ in range(h):
            perm = torch.randperm(n, generator=gen, device=device)
            pi = perm[:sp]
            ai = torch.cat((pi[:overlap], perm[sp:sp + sa - overlap]))
            preds.append(pi[torch.randperm(sp, generator=gen, device=device)])
            sels.append(ai[torch.randperm(sa, generator=gen, device=device)])
        p, a = torch.stack(preds).int(), torch.stack(sels).int()
        yield f"random_unordered_b17_d48_p{sp}_a{sa}_hit{overlap}", q, k, v, p, a

    for name in ["high_wrong_mass", "many_low_wrong_mass", "signed_value_cancellation", "extreme_wrong_logits"]:
        h, n, b, d, count = 2, 24, 16, 32, 8
        q = torch.zeros(h, d, device=device, dtype=dtype)
        q[:, 0] = 1
        logits = torch.zeros(h, n, device=device)
        if name == "high_wrong_mass":
            logits[:, 0] = 20
        elif name == "many_low_wrong_mass":
            logits.fill_(-8)
            logits[:, :7] = 10
        elif name == "extreme_wrong_logits":
            logits.fill_(-40)
            logits[:, :7] = 80
        k = torch.zeros(h, n, b, d, device=device, dtype=dtype)
        k[..., 0] = (logits * math.sqrt(d))[:, :, None]
        v = torch.randn(h, n, b, d, generator=gen, device=device, dtype=dtype)
        if name == "signed_value_cancellation":
            signs = torch.where(torch.arange(d, device=device) % 2 == 0, 1., -1.)
            # Deliberately ill-conditioned: large finite FP16 wrong values
            # cancel against the aggregate while retained values are small.
            v[:] = signs * .01
            v[:, :7] = signs * 60000.
        p = torch.arange(count, device=device, dtype=torch.int32).expand(h, -1).contiguous()
        a = torch.arange(7, 15, device=device, dtype=torch.int32).expand(h, -1).contiguous()
        yield name, q, k, v, p, a

    # The removed blocks cancel each other exactly. A guard using only the
    # signed low and removed sums cannot detect the lost middle contribution.
    q = torch.ones(1, 1, device=device, dtype=dtype)
    k = torch.zeros(1, 4, 1, 1, device=device, dtype=dtype)
    v = torch.tensor([60000., .01, -60000., 1.], device=device, dtype=dtype).view_as(k)
    p = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    a = torch.tensor([[1]], device=device, dtype=torch.int32)
    yield "signed_low_aggregate_cancellation", q, k, v, p, a

    # In compact mode the empty wrong aggregate carries the low-group maximum.
    # A rejected low group must discard that scale before exponentiating.
    k = k.clone()
    k[:, :3] = 100
    a = torch.tensor([[3]], device=device, dtype=torch.int32)
    yield "discard_high_logit_low_group", q, k, v, p, a


def run(device="cuda", include_compact=False):
    rows = []
    for dtype in [torch.float16, torch.bfloat16]:
        for name, q, k, v, p, a in cases(device, dtype):
            expected = reference(q, k, v, a)
            row = {"case": name, "dtype": str(dtype), "method": "exact", **error_metrics(exact_attention(q, k, v, a), expected)}
            rows.append(row)
            modes = [("saved", 0.1), ("hybrid", 0.05), ("hybrid", 0.2), ("hybrid", 1.0)]
            if include_compact:
                modes += [("compact", 0.05), ("compact", 0.2), ("compact", 1.0)]
            for mode, threshold in modes:
                state = speculate(q, k, v, p, mode=mode, mass_threshold=threshold)
                result = repair(q, k, v, a, p, state, min_remaining=1e-3, cancellation_tol=1e-5)
                row = {
                    "case": name, "dtype": str(dtype), "method": mode, "mass_threshold": threshold,
                    "fallback_heads": result.fallback.int().sum().item(),
                    "remaining": result.remaining.tolist(),
                    **error_metrics(result.output, expected),
                }
                if state.overflow is not None:
                    row["overflow_heads"] = state.overflow.int().sum().item()
                if mode in ("hybrid", "compact") and threshold == 1.0:
                    expected_fallback = {
                        "signed_low_aggregate_cancellation": 1,
                        "discard_high_logit_low_group": 0,
                    }.get(name)
                    if expected_fallback is not None:
                        row["expected_fallback_heads"] = expected_fallback
                        row["passed"] &= row["fallback_heads"] == expected_fallback
                rows.append(row)
    return {"oracle": "FP64 attention on identical quantized Q/K/V and exact selected IDs", "passed": all(r["passed"] for r in rows), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/correctness.json")
    parser.add_argument("--include-compact", action="store_true")
    args = parser.parse_args()
    result = run(include_compact=args.include_compact)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    failed = [r for r in result["rows"] if not r["passed"]]
    print(json.dumps({"passed": result["passed"], "checks": len(result["rows"]), "failures": failed}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
