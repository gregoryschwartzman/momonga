"""Each vertex's weighted-Jaccard distance to its own cluster's centre.

`own_distances` reproduces the reference solver's summation order exactly: it walks every column, in order, in
float64 for the numerator and float32 for the denominator.  `assigned_distances` is the support-only form,
O(nnz) per vertex, used only as an upper bound for the accelerators' pruning, so it need not match that order.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..kernels.profiles import row_sums


@njit(parallel=True, cache=True)
def _own_distances(indptr, indices, data, centres, cluster_ids, width, out):
    n = indptr.shape[0] - 1
    for u in prange(n):
        centre = centres[cluster_ids[u]]
        numerator = 0.0                      # float64, like numba's np.linalg.norm
        denominator = np.float32(0.0)        # float32, like numba's ndarray.sum()
        p = indptr[u]
        end = indptr[u + 1]
        for j in range(width):
            a = np.float32(0.0)
            if p < end and indices[p] == j:
                a = data[p]
                p += 1
            b = centre[j]
            numerator += abs(a - b)
            denominator += max(a, b)
        out[u] = numerator / denominator if denominator != 0 else 1.0


def own_distances(F, centres, cluster_ids):
    """The distance of every row of the CSR F to its own cluster's centre.

    F must have sorted indices: the kernel walks the columns in order and advances the CSR cursor only on a match,
    so an unsorted row would silently give a wrong distance rather than failing.
    """
    if not F.has_sorted_indices:
        raise ValueError("own_distances needs sorted CSR indices; call F.sort_indices() first")
    out = np.empty(F.shape[0], dtype=np.float64)
    _own_distances(F.indptr, F.indices, F.data, centres, np.asarray(cluster_ids, np.int64), F.shape[1], out)
    return out


@njit(parallel=True, cache=True)
def _assigned_distances(indptr, indices, data, row_sum, centres, centre_sum, cluster_ids, out):
    n = indptr.shape[0] - 1
    for u in prange(n):
        c = cluster_ids[u]
        centre = centres[c]
        overlap = 0.0
        for p in range(indptr[u], indptr[u + 1]):
            a = data[p]
            b = centre[indices[p]]
            overlap += a if a < b else b
        denominator = row_sum[u] + centre_sum[c] - overlap
        out[u] = 1.0 - overlap / denominator if denominator > 0.0 else 1.0


def assigned_distances(F, centres, centre_sums, cluster_ids):
    """The distance of every row of F to its own cluster's centre, by the support-only form: an upper bound."""
    out = np.empty(F.shape[0], dtype=np.float32)
    _assigned_distances(F.indptr, F.indices, F.data, row_sums(F).astype(np.float32), centres, centre_sums,
                        np.asarray(cluster_ids, np.int64), out)
    return out
