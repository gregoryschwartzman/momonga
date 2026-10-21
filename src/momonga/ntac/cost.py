"""NTAC's own cost: the mean weighted-Jaccard distance of each vertex to its cluster's median profile.

This is the 'jac' the reference solver reports; NTAC's answer is the partition of its trajectory that minimises
it.  It is used nowhere else: the NTAC step assigns vertices to cluster means, and the refinement uses the
description length.

Two observations make it cheap without densifying every cluster:

  1. a column's median over a cluster is zero unless at least half the members are nonzero there, because the
     sorted column is (size - count) zeros followed by the nonzeros; sparse profiles leave few such columns.
  2. min(a, b) = 0 wherever a = 0, so the overlap needs only the vertex's own nonzeros, and
     sum(max) = sum(vertex) + sum(median) - overlap.

The medians are bit-identical to the reference's; the mean is accumulated in float32 like the reference's.
"""
from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from ..kernels.profiles import row_sums
from .kernels import trusted_groups


@njit(cache=True)
def _median_one_cluster(indptr, indices, data, members, width, count, touched, buffer, buffer_start, out_row):
    """Column medians for one cluster, writing only the nonzero ones."""
    size = members.shape[0]
    n_touched = 0
    for r in range(size):
        u = members[r]
        for p in range(indptr[u], indptr[u + 1]):
            j = indices[p]
            if count[j] == 0:
                touched[n_touched] = j
                n_touched += 1
            count[j] += 1
    # a column can only have a nonzero median if 2 * count >= size
    n_qualifying = 0
    for q in range(n_touched):
        j = touched[q]
        if 2 * count[j] >= size:
            buffer_start[j] = n_qualifying
            n_qualifying += count[j]
        else:
            buffer_start[j] = -1
    # scatter the qualifying columns' values into a compact buffer
    fill = np.zeros(n_touched, dtype=np.int32)
    position = np.empty(width, dtype=np.int32)
    for q in range(n_touched):
        position[touched[q]] = q
    for r in range(size):
        u = members[r]
        for p in range(indptr[u], indptr[u + 1]):
            j = indices[p]
            base = buffer_start[j]
            if base >= 0:
                buffer[base + fill[position[j]]] = data[p]
                fill[position[j]] += 1
    for q in range(n_touched):
        j = touched[q]
        base = buffer_start[j]
        c = count[j]
        count[j] = 0                    # reset for the next cluster
        if base < 0:
            continue
        values = buffer[base:base + c]
        values.sort()
        n_zeros = size - c              # the implicit zeros, all smallest
        if size % 2 == 1:
            index = (size - 1) // 2
            out_row[j] = np.float32(0.0) if index < n_zeros else values[index - n_zeros]
        else:
            i1 = size // 2 - 1
            i2 = size // 2
            v1 = np.float32(0.0) if i1 < n_zeros else values[i1 - n_zeros]
            v2 = np.float32(0.0) if i2 < n_zeros else values[i2 - n_zeros]
            out_row[j] = (v1 + v2) / np.float32(2)


@njit(parallel=True, cache=True)
def _medians(indptr, indices, data, members, offsets, width, out, bounds, nnz):
    for b in prange(bounds.shape[0] - 1):
        count = np.zeros(width, dtype=np.int32)
        touched = np.empty(width, dtype=np.int32)
        buffer_start = np.empty(width, dtype=np.int64)
        buffer = np.empty(nnz, dtype=data.dtype)
        for c in range(bounds[b], bounds[b + 1]):
            start, end = offsets[c], offsets[c + 1]
            if end > start:
                _median_one_cluster(indptr, indices, data, members[start:end], width, count, touched, buffer,
                                    buffer_start, out[c])


def cluster_medians(F, members, offsets):
    """The column medians of every cluster of the CSR profiles F, as a dense (n_clusters, width) array.

    members, offsets   the members of every cluster, flattened, and where each cluster's run starts
    """
    n_clusters, width = offsets.shape[0] - 1, F.shape[1]
    out = np.zeros((n_clusters, width), dtype=F.dtype)
    if n_clusters == 0:
        return out
    n_bounds = max(1, min(n_clusters, get_num_threads() * 4))
    bounds = np.linspace(0, n_clusters, n_bounds + 1).astype(np.int64)
    _medians(F.indptr, F.indices, F.data, members, offsets, width, out, bounds, F.nnz)
    return out


@njit(parallel=True, cache=True)
def _jaccard_to_median(indptr, indices, data, row_sum, medians, median_sum, cluster_ids, out):
    """overlap / union per vertex, against its own cluster's median."""
    n = indptr.shape[0] - 1
    for u in prange(n):
        c = cluster_ids[u]
        row = medians[c]
        overlap = np.float32(0.0)
        for p in range(indptr[u], indptr[u + 1]):
            a = data[p]
            b = row[indices[p]]
            overlap += a if a < b else b
        union = row_sum[u] + median_sum[c] - overlap
        out[u] = overlap / union if union > 0 else np.float32(0.0)


def partition_cost(context, partition):
    """1 - the mean over vertices of Jaccard(vertex profile, its cluster's median profile)."""
    cluster_ids = partition.cluster_ids
    n_clusters = partition.n_clusters
    F = context.profiles(cluster_ids, n_clusters)
    members, offsets = trusted_groups(cluster_ids, cluster_ids, np.zeros(cluster_ids.size, np.bool_), n_clusters)
    medians = cluster_medians(F, members, offsets)
    similarity = np.empty(F.shape[0], dtype=F.dtype)
    _jaccard_to_median(F.indptr, F.indices, F.data, row_sums(F), medians, medians.sum(axis=1).astype(F.dtype),
                       np.asarray(cluster_ids, np.int64), similarity)
    return 1.0 - float(np.mean(similarity))
