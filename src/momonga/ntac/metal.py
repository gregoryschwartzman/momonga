"""The Apple-GPU kernels of the NTAC steps, through MLX: the seed-selection gains and the pruned nearest-centre
search, for one profile matrix resident on the GPU (MetalProfiles).

Same algebra as kernels.jaccard:

    smax[u, c] = sum_j max(0, profile[u, j] - centre[c, j])
    D[u, c]    = (centre_sum[c] - profile_sum[u] + 2 smax) / (centre_sum[c] + smax)

One thread per (vertex, centre); the centres are passed transposed (width x n_centres) so neighbouring threads,
which differ in the centre, read neighbouring addresses.

Precision: Metal has no float64 while the original NTAC implementation accumulates smax in float64.  The accumulator here is a
double-float (two float32 limbs, about 48 bits of mantissa), which reproduces the original exactly when the
centres are integer-valued and is within one float32 ulp otherwise.  The distances are therefore bit-identical to
the CPU's; the seed-selection gains are reduced per block of vertices into float32 and summed on the host in
float64, so an argmax over them can differ from the CPU at a near-tie.
"""
from __future__ import annotations

import numpy as np

from ..kernels.profiles import row_sums
from .kernels import candidate_ranges

# The distance from vertex u to the centre in column `col` of centres_t, given that centre's sum centre_sum and
# the vertex's profile in CSR; the positive part of (profile - centre) is accumulated as a double-float
# (smax_hi + smax_lo), and the quotient is refined once by Newton's step.
_DISTANCE = r"""
        uint start = (uint)indptr[u];
        uint end   = (uint)indptr[u + 1];
        float profile_sum = profile_sums[u];
        float smax_hi = 0.0f, smax_lo = 0.0f;
        for (uint p = start; p < end; ++p) {
            float difference = data[p] - centres_t[(uint)indices[p] * K + col];
            if (difference > 0.0f) {
                float sum = smax_hi + difference;                                       // two-sum
                float error = (fabs(smax_hi) >= fabs(difference)) ? ((smax_hi - sum) + difference)
                                                                   : ((difference - sum) + smax_hi);
                smax_lo += error;
                smax_hi = sum;
            }
        }
        float renormalised = smax_hi + smax_lo;
        smax_lo = smax_lo - (renormalised - smax_hi);
        smax_hi = renormalised;
        float denominator_hi = centre_sum + smax_hi;
        float denominator_lo = ((centre_sum - denominator_hi) + smax_hi) + smax_lo;
        float sum_difference = centre_sum - profile_sum;
        float numerator_hi = sum_difference + 2.0f * smax_hi;
        float numerator_lo = ((sum_difference - numerator_hi) + 2.0f * smax_hi) + 2.0f * smax_lo;
        float distance;
        if (denominator_hi + denominator_lo != 0.0f) {
            distance = numerator_hi / denominator_hi;
            distance = distance + ((numerator_hi - distance * denominator_hi)
                                   + (numerator_lo - distance * denominator_lo)) / (denominator_hi + denominator_lo);
        } else {
            distance = 1.0f;
        }
"""

# The seed-selection gains: each thread walks a block of BS vertices for one candidate and accumulates
# max(0, own_distance - distance) directly, so the (n x n_candidates) distance block is never materialised.
# Per-thread sums cover only BS terms; the NB partials are added in float64 on the host.
_GAINS = r"""
    uint i = thread_position_in_grid.x;      // candidate within the chunk
    uint b = thread_position_in_grid.y;      // block of vertices
    if (i >= K || b >= NB) return;
    uint u0 = b * BS;
    uint u1 = min(u0 + BS, (uint)N);
    uint sorted_position = i + OFF;
    uint col = i;
    float centre_sum = centre_sums[i];
    float gain = 0.0f;
    for (uint u = u0; u < u1; ++u) {
        if (sorted_position < range_start[u] || sorted_position >= range_end[u]) continue;   // a zero term
""" + _DISTANCE + r"""
        float improvement = own_distances[u] - distance;
        if (improvement > 0.0f) gain += improvement;
    }
    partials[b * K + i] = gain;
"""

# The pruned nearest-centre search, the centres sorted by row sum so that each vertex's surviving candidates are a
# contiguous range and whole simdgroups exit together.  Results are written back to the original centre position,
# so the argmin keeps numpy's first-index tie-break.
_NEAREST_SORTED = r"""
    uint t = thread_position_in_grid.x;      // position in the sorted order
    uint u = thread_position_in_grid.y;
    if (t >= K || u >= N) return;
    uint original = (uint)order[t];
    if (t < range_start[u] || t >= range_end[u]) { distances[u * K + original] = INFINITY; return; }
    uint col = t;
    float centre_sum = centre_sums[t];
""" + _DISTANCE + r"""
    distances[u * K + original] = distance;
"""

_KERNELS = {}


def _kernel(name, source, inputs, outputs):
    if name not in _KERNELS:
        import mlx.core as mx

        _KERNELS[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source)
    return _KERNELS[name]


class MetalProfiles:
    """One profile matrix resident on the GPU.  The CSR arrays are uploaded here; the centres are uploaded per call."""

    def __init__(self, F):
        import mlx.core as mx

        self.mx = mx
        self.n, self.width = F.shape
        self.indptr = mx.array(F.indptr)
        self.indices = mx.array(F.indices)
        self.data = mx.array(F.data)
        self.row_sums = row_sums(F).astype(np.float32)
        self.profile_sums = mx.array(self.row_sums)

    def candidate_ranges(self, sorted_sums, own_distances):
        """[start, end) per vertex of the candidates, sorted by row sum, that can contribute to its gain.

        A term max(0, d_u - d(u, c)) is nonzero only when d(u, c) < d_u, and d(u, c) >= |S_u - S_c| / max(S_u, S_c),
        so the survivors satisfy S_u (1 - d_u) < S_c < S_u / (1 - d_u): a contiguous range once sorted.
        """
        return candidate_ranges(sorted_sums, self.row_sums, own_distances)

    def gains(self, candidates, own_distances, range_start, range_end, offset, block_size=64, threadgroup=256):
        """gain[c] = sum_u max(0, d_u - d(u, c)) for the dense candidate rows `candidates` (a chunk of the sorted
        candidates beginning at `offset`), the vertices' ranges [range_start, range_end) pruning the terms."""
        mx = self.mx
        n_candidates = candidates.shape[0]
        n_blocks = int(np.ceil(self.n / block_size))
        (partials,) = _kernel("gains", _GAINS,
                              ["indptr", "indices", "data", "profile_sums", "centres_t", "centre_sums",
                               "range_start", "range_end", "own_distances"], ["partials"])(
            inputs=[self.indptr, self.indices, self.data, self.profile_sums,
                    mx.array(np.ascontiguousarray(candidates)).T, mx.array(candidates.sum(axis=1).astype(np.float32)),
                    mx.array(range_start), mx.array(range_end), mx.array(np.asarray(own_distances, dtype=np.float32))],
            template=[("K", n_candidates), ("N", self.n), ("NB", n_blocks), ("BS", int(block_size)),
                      ("OFF", int(offset))],
            grid=(n_candidates, n_blocks, 1), threadgroup=(min(threadgroup, n_candidates), 1, 1),
            output_shapes=[(n_blocks, n_candidates)], output_dtypes=[mx.float32])
        mx.eval(partials)
        return np.array(partials, copy=False).astype(np.float64).sum(axis=0)

    def nearest_centres_sorted(self, centres, centre_sums, bound, threadgroup=256):
        """The nearest centre of every vertex, with the centres sorted by row sum so that the candidates within
        `bound` of each vertex are contiguous."""
        mx = self.mx
        n_centres = centres.shape[0]
        order = np.argsort(centre_sums, kind="stable").astype(np.int32)
        range_start, range_end = candidate_ranges(centre_sums[order].astype(np.float64), self.row_sums, bound)
        (distances,) = _kernel("nearest_sorted", _NEAREST_SORTED,
                               ["indptr", "indices", "data", "profile_sums", "centres_t", "centre_sums", "order",
                                "range_start", "range_end"], ["distances"])(
            inputs=[self.indptr, self.indices, self.data, self.profile_sums,
                    mx.array(np.ascontiguousarray(centres[order])).T, mx.array(centre_sums[order]), mx.array(order),
                    mx.array(range_start), mx.array(range_end)],
            template=[("K", n_centres), ("N", self.n)],
            grid=(n_centres, self.n, 1), threadgroup=(min(threadgroup, n_centres), 1, 1),
            output_shapes=[(self.n, n_centres)], output_dtypes=[mx.float32])
        nearest = mx.argmin(distances, axis=1)
        mx.eval(nearest)
        return np.array(nearest, copy=False).astype(np.int64)
