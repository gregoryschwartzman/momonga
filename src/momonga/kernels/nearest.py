"""Nearest neighbours between sparse profiles by weighted Jaccard, over the shared columns only.

    nearest_by_distance    the shattering's links: the nearest OTHER row by weighted-Jaccard distance, in the
                           float32 arithmetic of the original NTAC implementation (a row at distance one from every other row
                           links to row 0, or row 1 for row 0)
    nearest_other_side     the twin search of the bilateral profiles: the most similar row on the other side by
                           weighted-Jaccard similarity in float64, or -1 when no row shares a column

The two walk the same postings but keep different arithmetic and different tie and no-partner rules, so they are
two kernels: merging them would change which neighbour wins at a near-tie, and with it the recorded results.
"""
import numpy as np
from numba import get_num_threads, njit, prange
from scipy import sparse


@njit(parallel=True, cache=True)
def _by_distance(indptr, indices, data, sum_32, sum_64, t_indptr, t_indices, t_data, n_threads, out):
    n = sum_32.size
    overlap = np.zeros((n_threads, n))
    seen = np.zeros((n_threads, n), np.bool_)
    touched = np.empty((n_threads, n), np.int64)
    for t in prange(n_threads):
        for i in range(t, n, n_threads):
            n_touched = 0
            for p in range(indptr[i], indptr[i + 1]):
                column = indices[p]
                a = data[p]
                for q in range(t_indptr[column], t_indptr[column + 1]):
                    j = t_indices[q]
                    if j == i:
                        continue
                    if not seen[t, j]:
                        seen[t, j] = True
                        touched[t, n_touched] = j
                        n_touched += 1
                    b = t_data[q]
                    overlap[t, j] += a if a < b else b
            sum_i = sum_32[i]
            best = np.float32(np.inf)
            best_j = -1
            for k in range(n_touched):
                j = touched[t, k]
                smax = sum_64[i] - overlap[t, j]
                sum_j = sum_32[j]
                denominator = sum_j + smax
                distance = (np.float32(((sum_j - sum_i) + 2.0 * smax) / denominator) if denominator != 0.0
                            else np.float32(1.0))
                if distance < best or (distance == best and j < best_j):
                    best = distance
                    best_j = j
                overlap[t, j] = 0.0
                seen[t, j] = False
            if not (best < np.float32(1.0)):
                best_j = 1 if i == 0 else 0
            out[i] = best_j


def nearest_by_distance(F):
    """For every row u of the CSR F (integer counts), the nearest other row by weighted-Jaccard distance; the
    smallest index on a tie, and row 0 (row 1 for row 0) for a row at distance one from every other row."""
    if not np.all(np.isfinite(F.data) & (F.data >= 0) & (F.data == np.round(F.data))):
        raise ValueError("profiles must contain finite non-negative integer counts")
    if F.shape[0] <= 1:
        return np.arange(F.shape[0], dtype=np.int64)
    if np.asarray(F.astype(np.float64).sum(1)).max() >= 2 ** 24:
        raise ValueError("row totals must be below float32's exact-integer limit")
    F = F.astype(np.float32).tocsr()
    F.sum_duplicates()
    F.sort_indices()
    sum_64 = np.asarray(F.astype(np.float64).sum(1)).ravel()
    sum_32 = sum_64.astype(np.float32)
    T = F.T.tocsr()
    T.sort_indices()
    out = np.empty(F.shape[0], np.int64)
    _by_distance(F.indptr.astype(np.int64), F.indices.astype(np.int64), F.data.astype(np.float64), sum_32, sum_64,
                 T.indptr.astype(np.int64), T.indices.astype(np.int64), T.data.astype(np.float64),
                 get_num_threads(), out)
    return out


@njit(parallel=True, cache=True)
def _other_side(indptr, indices, data, sums, t_indptr, t_indices, t_data, side, n_threads, out):
    """For every row i, the row j != i on the other side with the highest similarity overlap / (|i| + |j| - overlap),
    in float64 over shared columns only; the lower index on ties; -1 if no such row shares a column."""
    n = sums.size
    overlap = np.zeros((n_threads, n))
    seen = np.zeros((n_threads, n), np.bool_)
    touched = np.empty((n_threads, n), np.int64)
    for t in prange(n_threads):
        for i in range(t, n, n_threads):
            n_touched = 0
            for p in range(indptr[i], indptr[i + 1]):
                column = indices[p]
                a = data[p]
                for q in range(t_indptr[column], t_indptr[column + 1]):
                    j = t_indices[q]
                    if j == i or side[j] == side[i] or side[j] > 1 or side[i] > 1:
                        continue
                    if not seen[t, j]:
                        seen[t, j] = True
                        touched[t, n_touched] = j
                        n_touched += 1
                    b = t_data[q]
                    overlap[t, j] += a if a < b else b
            best = -1.0
            best_j = -1
            for k in range(n_touched):
                j = touched[t, k]
                shared = overlap[t, j]
                similarity = shared / (sums[i] + sums[j] - shared)
                if similarity > best or (similarity == best and j < best_j):
                    best = similarity
                    best_j = j
                overlap[t, j] = 0.0
                seen[t, j] = False
            out[i] = best_j


def nearest_other_side(F, side):
    """For every row of the sparse matrix F, the index of its most similar row on the other side (side 0 or 1; a
    row with another code has no partner) by weighted-Jaccard similarity in float64, or -1."""
    F = sparse.csr_matrix(F, dtype=np.float64)
    F.sort_indices()
    T = F.T.tocsr()
    T.sort_indices()
    out = np.empty(F.shape[0], np.int64)
    _other_side(F.indptr.astype(np.int64), F.indices.astype(np.int64), F.data, np.asarray(F.sum(1)).ravel(),
                T.indptr.astype(np.int64), T.indices.astype(np.int64), T.data, np.asarray(side, dtype=np.int64),
                get_num_threads(), out)
    return out
