"""The kernel behind GraphData.confidence: each vertex's weighted-Jaccard distance to its own cluster's profile and
to the nearest other cluster's, from sparse postings, without an n x n_clusters distance matrix."""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _own_and_nearest_other(indptr, indices, data, c_indptr, c_indices, c_data, row_sum, cluster_sum, partition):
    n, n_clusters = len(partition), len(cluster_sum)
    own = np.empty(n)
    other = np.empty(n)
    for u in prange(n):
        overlap = np.zeros(n_clusters)
        for p in range(indptr[u], indptr[u + 1]):
            column, value = indices[p], data[p]
            for q in range(c_indptr[column], c_indptr[column + 1]):
                overlap[c_indices[q]] += min(value, c_data[q])
        best = np.inf
        for c in range(n_clusters):
            denominator = row_sum[u] + cluster_sum[c] - overlap[c]
            distance = 1.0 - overlap[c] / denominator if denominator > 0 else 1.0
            if c == partition[u]:
                own[u] = distance
            elif distance < best:
                best = distance
        other[u] = best
    return own, other


def own_and_nearest_other(vertex_profiles, cluster_profiles, partition):
    """(own, other): each vertex's distance to its own cluster's profile and to the nearest other cluster's."""
    F = vertex_profiles.tocsr()
    C = cluster_profiles.tocsc()
    return _own_and_nearest_other(F.indptr, F.indices, F.data, C.indptr, C.indices, C.data,
                                  np.asarray(F.sum(1)).ravel(), np.asarray(C.sum(1)).ravel(), partition)
