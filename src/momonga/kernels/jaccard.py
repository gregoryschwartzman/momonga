"""The weighted-Jaccard distance from every vertex profile to every centre, tiled, and its argmin.

The original NTAC implementation loops vertex -> centre -> edge and reads `M[i, indices[p]]`: for every vertex it re-streams the
whole dense centre matrix with scattered loads.  Here the loops are vertex -> edge -> centre over a transposed,
compacted tile of centres, so the tile stays in cache and the innermost loop is contiguous.

Bit-identity with the original: it accumulates the positive part of (profile - centre) from `0.0`, which numba
types as float64 even though the profiles and centres are float32, and computes the numerator
(centre sum - profile sum) + 2 smax left to right, so the first difference is a float32 subtraction and the rest is
float64.  Both are mirrored exactly below.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

TILE_BYTES = 1 << 20


@njit(parallel=True, fastmath=False, cache=True)
def distance_tile(indptr, indices, data, profile_sums, centres_t, centre_sums, out):
    """out[u, t]: the distance from profile row u to centre t of the tile `centres_t` (the centres transposed,
    (width, n_centres))."""
    n = indptr.shape[0] - 1
    n_centres = centres_t.shape[1]
    for u in prange(n):
        profile_sum = profile_sums[u]
        positive = np.zeros(n_centres, dtype=np.float64)         # matches numba's `smax = 0.0`
        for p in range(indptr[u], indptr[u + 1]):
            value = data[p]
            centre_row = centres_t[indices[p]]
            for t in range(n_centres):
                difference = value - centre_row[t]                # float32 subtraction
                positive[t] += max(difference, np.float32(0.0))  # branchless; adds into float64
        for t in range(n_centres):
            smax = positive[t]
            centre_sum = centre_sums[t]
            denominator = centre_sum + smax
            out[u, t] = ((centre_sum - profile_sum) + 2.0 * smax) / denominator if denominator != 0.0 else 1.0


@njit(parallel=True, cache=True)
def _argmin_tile(distances, offset, best, nearest):
    """Fold a tile of distances into a running (minimum, argmin).  Ties keep the first index, as numpy does; a NaN
    is never selected (it fails `value < best`)."""
    n, n_centres = distances.shape
    for u in prange(n):
        best_u = best[u]
        nearest_u = nearest[u]
        for t in range(n_centres):
            value = distances[u, t]
            if value < best_u:
                best_u = value
                nearest_u = offset + t
        best[u] = best_u
        nearest[u] = nearest_u


def tile_width(width, itemsize):
    """How many centres of `width` columns fit a tile of TILE_BYTES."""
    return max(32, TILE_BYTES // (width * itemsize))


def prepare_centres(profiles, centres):
    """(dtype, centres, profile_sums, centre_sums, data): the inputs of distance_tile.  Integer centres are promoted
    to float32 so that a distance in [0, 1) is never truncated to 0 by an integer output array."""
    dtype = np.result_type(centres.dtype, np.float32)
    centres = np.ascontiguousarray(centres, dtype=dtype)
    profile_sums = np.asarray(profiles.sum(axis=1)).ravel().astype(dtype)
    centre_sums = centres.sum(axis=1).astype(dtype)
    data = profiles.data.astype(dtype)
    return dtype, centres, profile_sums, centre_sums, data


def nearest_centres(profiles, centres):
    """For every row of the CSR `profiles`, the index of the nearest row of the dense `centres`, without the full
    distance matrix."""
    n = profiles.shape[0]
    n_centres, width = centres.shape
    dtype, centres, profile_sums, centre_sums, data = prepare_centres(profiles, centres)
    best = np.full(n, np.inf, dtype=dtype)
    nearest = np.zeros(n, dtype=np.int64)
    tile = min(n_centres, tile_width(width, dtype.itemsize))
    for start in range(0, n_centres, tile):
        end = min(start + tile, n_centres)
        centres_t = np.ascontiguousarray(centres[start:end].T)
        distances = np.empty((n, end - start), dtype=dtype)
        distance_tile(profiles.indptr, profiles.indices, data, profile_sums, centres_t, centre_sums[start:end],
                      distances)
        _argmin_tile(distances, start, best, nearest)
    return nearest
