# Kernel contract and numerical policy

The implementation is forward-only, inference-only, and assumes finite inputs.
Q has shape `[heads, dimension]`; K and V have shape
`[heads, blocks, tokens_per_block, dimension]`. Tensors must be contiguous and
on the same CUDA device. Block IDs have shape `[heads, selected_blocks]`, must
be unique within a row, and must refer to valid blocks. IDs have no padding;
empty sets, causal masks, GQA broadcast, and backward are not implemented.
Shape checks do not synchronize the GPU to validate ID values or uniqueness.
Calling code owns stream/event dependencies, including completion of KV writes.

All speculative states use the **current** query. No attention state from an
older query is reused; only the predicted block IDs may come from a prior step.

## States and exact baseline

Each block computes FP32 `m = max(logits)`, `l = sum(exp(logits-m))`, and
`u = sum(exp(logits-m) * V)`. K, Q, and V are converted to FP32 for the reduction.
The baseline uses a parallel block kernel followed by a merge kernel. Merging
rescales both the numerator and denominator to a common maximum. Outputs are
FP32. "Exact" denotes identical selected-set semantics, not bitwise equality
across different floating-point reduction orders, nor dense-attention semantics.

## Saved-state repair

Membership comparisons identify predicted hits and selected misses. Only misses
read KV again. The final merge uses hit states and miss states. This is a
required baseline because saving block states already permits strict-set
filtering without subtraction; the technique alone is not a novelty claim.

## Hybrid repair

The block's mass is measured under the predicted-set softmax. States with mass
at least `mass_threshold` are retained individually; others form a low-mass
aggregate. A high-mass wrong block is omitted without re-reading KV. A low-mass
wrong block is recomputed using the current query and subtracted in the same
scale as the aggregate. Selected misses are then added. If the entire low group
is rejected, it is discarded directly, avoiding cancellation and KV rereads.

The `hybrid` grouping kernel operates after materialization of **all** per-block
states. The `Prediction` object still owns those tensors. Therefore this version
does **not** reduce peak state allocation and does not demonstrate the potential
memory savings of a fused, bounded-retention implementation. It adds grouping,
membership, subtraction, and guard overhead; benchmark timings must include it.
Per-block speculative state storage is `4 * H * P * (D+2)` bytes. The hybrid
prediction additionally holds its low aggregate, a Boolean mask, and an FP32
absolute numerator envelope of shape `[H,D]`. Miss/wrong scratch, outputs,
and allocator overhead are separate.

## Bounded retained-state optimization (`compact`)

This separate implementation avoids materializing the full per-predicted-block
numerator. Its first pass reads Q/K and computes only `m,l`. The second kernel
classifies actual predicted-set mass and packs high-block IDs into capacity
`C = min(P, floor(1 / threshold) + 1)` (or `P` for a zero threshold). The spare
slot protects ordinary rounding at the threshold boundary. The implementation
still has an explicit overflow policy: any excess high candidates are put into
the low group and the per-head `Prediction.overflow` flag is set. This preserves
bounded allocation and strict-set repair, but an overflow means the requested
high-mass separation was not fully achieved and must be reported.

The values pass reads all predicted Q/K/V again. Retained high blocks have
parallel individual CTAs; one CTA per head aggregates low blocks sequentially.
Only the retained high blocks and one low aggregate have signed numerator
vectors. The low absolute numerator envelope adds one FP32 vector per head.
Compact repair similarly recomputes low wrong blocks into one aggregate rather
than materializing their individual numerator vectors. This trades additional
Q/K reads and less parallelism for reduced state storage; it is not necessarily
faster. Miss states still allocate `[H, selected_blocks, D]`, including inactive
slots. Memory reporting must distinguish persistent prediction tensors,
transient scalar statistics, repair scratch, CUDA allocator rounding, and graph
capture pools.

For ID width `I` bytes, compact prediction tensor storage is
`H * (4 * (C+1) * (D+2) + 4 * D + P + I * C + 1)` bytes. The two temporary
statistics matrices add `8 * H * P` bytes during preparation. Compact prediction
storage can exceed saved-state storage when the cap is close to `P`.

Nominal KV element reads before repair are `3 * H * P * B * D` in `compact`
(K-only plus K/V), compared with `2 * H * P * B * D` for `saved` and `hybrid`.
These are algorithmic counts, not measured DRAM traffic or L2 cache misses.

## Guards

The low-group remaining denominator fraction is checked against `min_remaining`.
The total denominator must be positive and finite. For every numerator
coordinate of a participating low group, repair is considered poorly
conditioned if

```
abs(low_u - removed_u) < cancellation_tol * (low_abs_u + abs(removed_u))
```

`low_abs_u` is the sum of absolute weighted token contributions in `compact`
and the sum of absolute rescaled block numerators in `hybrid`. This catches cases
where signed cancellation precedes subtraction. For example, adding a small
selected contribution between two large opposite-sign wrong contributions can
round the small term during aggregation even when the wrong terms sum to zero.
A nonfinite output also triggers fallback. Flagged heads recompute all selected
blocks with a single conditional online-softmax kernel, which launches even if
no head falls back. The threshold is a heuristic. The check does not bound all
errors in logits, softmax weights, individual block reductions, or the fallback
reduction. FP64 comparisons characterize the tested inputs.

## Cache warming baseline

`touch_kv` reads all predicted K and V elements and emits tile checksums so the
loads have a visible side effect. It does not perform attention or guarantee
that the data remain in L2. Concurrent selector traffic can evict those lines.
Without hardware-counter profiling, no cache-hit-rate claim should be made.

## Scope of optimization

The prototype parallelizes block attention, skips KV loads for inactive repair
blocks, discards wholly rejected low groups, and writes diagnostics in the merge
kernel rather than launching separate initialization kernels. CUDA Graph timing
can distinguish GPU execution overhead from Python dispatch overhead. None of
these kernels implements PRR's official CUDA implementation or full-model decode.
