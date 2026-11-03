# Compact mode review

## Selected-set semantics

With unique valid IDs, compact repair splits the selected set into retained
prediction hits, low-group prediction hits, and selected misses. The first
group uses packed states and the last group reads selected KV. The low group
contributes its aggregate minus its wrong blocks, with both terms evaluated at
the low-group maximum. When the low group has no hit, the merge discards its
state and scale. In particular, a very large maximum from a rejected group
cannot produce an `inf * 0` contribution from the empty wrong aggregate.
Spilling excess high candidates to the low group changes this partition and
leaves the selected set intact.

## Numerical regression

For one head with one token per block and one value coordinate, consider equal
logits, FP16 values `[60000, 0.01, -60000]`, and only the middle block selected.
A mass threshold of `1` puts all three predicted blocks in the low group.
Sequential FP32 addition in compact mode rounds the low numerator to
`0.01171875` while the exact quantized middle value is `0.01000213623046875`.
The removed numerator is zero and the remaining denominator fraction is `1/3`,
so the signed numerator and denominator checks alone miss the error.

The absolute numerator envelope exposes the large contributions and triggers
selected-only fallback. The regression suite checks that fallback occurs for
both grouping modes on this input. A separate disjoint case checks that a fully
rejected high-logit low group takes the direct discard path. These are targeted
checks of the guard policy. The guard remains heuristic and can also flag
legitimate small residuals.

## Optional extra softmax mass budget

Let `S` be the selected set, `L` the predicted low group, and `H_hit` the
retained high blocks that are also selected. When `L` has a selected hit, an
approximate repair can keep all of `L` and still discard every wrong high
block. Write `E = L \ S` for the extra blocks and `M = S \ prediction` for
the selected misses. For a block set `X`, write `Z(X)` for its unnormalized
softmax mass under the current query. All masses in the following ratios can
use the same logit shift for stable computation.

Before reading selected misses, the retained aggregate and high hits give
`Z0 = Z(L) + Z(H_hit)`. Per-block `(m,l)` statistics provide `Z(E)` without
reading wrong KV. If `Z(E) / Z0 <= epsilon`, keeping `E` satisfies the
following bound in real arithmetic.

```text
beta = Z(E) / (Z(S) + Z(E)) = Z(E) / (Z0 + Z(M)) <= epsilon
```

Selected misses add nonnegative mass. If there are no misses, the proxy and
final fraction coincide. If the whole low group is rejected, it can be
discarded directly and contributes zero extra mass.

Let `y_S` and `y_E` be the normalized attention outputs on the selected and
extra blocks. The augmented output is `(1 - beta) * y_S + beta * y_E`.
For any output coordinate, its absolute difference from `y_S` is bounded by
`epsilon` times the range of that coordinate among selected and extra values.
The total variation between the exact and augmented token distributions is
`beta`. These statements describe the attention computation in real
arithmetic. FP32 evaluation can move the decision near the threshold, and the
mass budget does not bound downstream model quality.

The optional `keep_mass=True` prediction setting retains the two FP32 scalar
arrays already produced by compact's statistics pass, adding `8 * H * P`
bytes of live prediction storage and no additional KV pass. Hybrid exposes
references to the existing per-block arrays, so its tensor storage is
unchanged. With `keep_mass=False`, compact releases the arrays after
preparation.

## Measurement boundaries

The live prediction storage formula in [Kernel contract](KERNELS.md) includes
the envelope, masks, retained IDs, and overflow flag. Peak allocator usage
also includes scalar scratch during speculation, dense selected-miss slots
during repair, and selector allocations. CUDA Graph capture pools and hardware
cache traffic are separate measurements. The compact grouping kernel uses a
padded `[P,C]` temporary tile, while membership still compares padded `[P,A]`
and `[C,A]` tiles. Tensor-storage savings do not establish scaling of compiler
resources or latency for larger candidate lists.

Eager timings include selector launches and joins on the completion path. CUDA
Graph timings use fixed inputs and omit capture. Synthetic overlap controls
the ID intersection size. Mass distribution and fallback rates can vary at
the same ID overlap, so the overlap percentage alone cannot explain timing or
quality on a model trace.
