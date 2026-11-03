# Fused repair

`strict_topk.fused.repair_fused` accepts the same inputs as `repair` and
returns FP32 attention output, a per-head fallback flag, and a per-head
remaining low-group mass fraction. It accepts predictions produced by
`speculate` in saved, hybrid, or compact mode. The input and stream
requirements in [KERNELS.md](KERNELS.md) apply.

Each head uses one Triton program for membership, retained-state merge,
late Q/K/V reads, normalization, and any guarded recomputation. It packs
the active block positions within that program and processes them in
tiles. Missing selected blocks enter an online softmax reduction.
Hybrid and compact repair accumulate wrong low blocks at the original
low-group maximum, then subtract their denominator and signed numerator.
Heads with no selected low blocks discard that aggregate directly.

The remaining fraction and the absolute-envelope cancellation test run
in the low group's own exponent frame. A much larger retained or missed
logit therefore cannot erase the quantities needed by those checks.
A failed subtraction check sends all selected blocks through the online
reduction once. A nonfinite or invalid result from the repair reduction
also triggers exact selected-set recomputation in the same program.
Reduction order and guard decisions near a threshold can differ from
those of the split repair.

The fused path launches one repair kernel for each prediction mode.
The split path launches three for saved repair and five for hybrid or
compact repair. Fusion eliminates global repair intermediates for
membership and block partials. Its per-head output and diagnostics use
`H * (4 * D + 5)` tensor bytes before allocator rounding. The prediction
stage and its allocations are shared with the split path.

The kernel trades block-level parallelism for fewer launches and
intermediate writes. The amount of late work depends on the number of
misses, wrong low blocks, and fallback heads. Report repair-only timings
separately from full selector/prediction timings, and measure the same
fixed inputs through both CUDA Graph replay and eager dispatch.

## Optional extra softmax mass

`max_extra_mass=0` uses strict repair. A positive budget lets a head
leave its wrong low blocks in the aggregate when their mass is small
enough. Prepare hybrid or compact predictions with
`speculate(..., keep_mass=True)` to expose the predicted block maxima
and denominators for this decision. In hybrid mode those tensors alias
existing state. In compact mode they add `8 * H * P` persistent tensor
bytes by keeping the already computed first-pass statistics alive.

Let `E` be the summed exponential mass of wrong low blocks, `L` the
mass of the entire low aggregate, and `H_hit` the mass of retained high
hits, all in the same exponent frame. The kernel estimates
`E / (L + H_hit)` and skips wrong-block Q/K/V reads only if the estimate
fits the budget. The selected misses still enter the reduction and
increase the denominator. In real arithmetic the eventual extra
softmax share is at most the estimated ratio's exact counterpart.
The signed cancellation check remains active, and a fallback recomputes
the selected set and clears the approximation diagnostics.

Positive budgets return per-head `approximate_heads` and
`extra_mass_bound` tensors. `extra_mass_bound` is the FP32 estimate
used for the decision, or zero on heads that used exact repair. It
does not certify the final output error. When the extra share is
`r`, the real-arithmetic output perturbation is `r` times the difference
between the softmax means over the extra and selected blocks. FP64
comparison therefore matters even when a budget is small.
