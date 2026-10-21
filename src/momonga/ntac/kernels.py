"""Kernels only the NTAC steps use: cluster means, trusted groups, candidate rows, seed gains, and the row-sum
bound the accelerators prune with.

Means sum each cluster's rows in cluster order into a float32 accumulator, which is what the reference solver's
nanmean over a densified block does, so every mean is bit-identical to it.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..kernels.jaccard import distance_tile, prepare_centres, tile_width
from ..kernels.profiles import thread_chunks


# -- cluster means --

@njit(parallel=True, cache=True)
def _means(indptr, indices, data, members, offsets, width, out, bounds):
    """Parallel over clusters.  Each cluster's rows are summed in cluster order into its own float32 accumulator,
    so every mean is bit-identical to a serial pass.  An empty cluster gets a row of NaN."""
    for chunk in prange(bounds.shape[0] - 1):
        total = np.zeros(width, dtype=out.dtype)
        for c in range(bounds[chunk], bounds[chunk + 1]):
            start, end = offsets[c], offsets[c + 1]
            n_rows = end - start
            if n_rows == 0:
                for j in range(width):
                    out[c, j] = np.nan
                continue
            for r in range(start, end):
                u = members[r]
                for p in range(indptr[u], indptr[u + 1]):
                    total[indices[p]] += data[p]
            divisor = np.float32(n_rows)
            for j in range(width):
                out[c, j] = total[j] / divisor
            for r in range(start, end):
                u = members[r]
                for p in range(indptr[u], indptr[u + 1]):
                    total[indices[p]] = np.float32(0.0)


def means_from_groups(F, members, offsets):
    """The mean profile of every group of rows of F: the members of every group, flattened, and where each group's
    run starts."""
    n_groups, width = offsets.shape[0] - 1, F.shape[1]
    out = np.empty((n_groups, width), dtype=F.dtype)
    if n_groups == 0:
        return out
    _means(F.indptr, F.indices, F.data, members, offsets, width, out, thread_chunks(n_groups))
    return out


def means_from_clusters(F, clusters):
    """The mean profile of every cluster, given as a list of member-index arrays."""
    n_clusters, width = len(clusters), F.shape[1]
    if n_clusters == 0:
        return np.zeros((0, width), dtype=F.dtype)
    sizes = np.array([len(members) for members in clusters], dtype=np.int64)
    members = (np.concatenate([np.asarray(c, dtype=np.int64) for c in clusters]) if sizes.sum()
               else np.zeros(0, np.int64))
    offsets = np.zeros(n_clusters + 1, np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return means_from_groups(F, members, offsets)


# -- trusted groups: a seeded NTAC iteration keeps only the vertices that did not move --

@njit(cache=True)
def _trusted_groups(cluster_ids, old_cluster_ids, is_seed, n_clusters, offsets, members):
    """Vertices still in the cluster they had last iteration (seeds always kept), grouped by cluster, each group in
    ascending vertex order: the order the reference's `good[labels[u]].append(u)` loop produced."""
    n = cluster_ids.shape[0]
    for c in range(n_clusters + 1):
        offsets[c] = 0
    for u in range(n):
        if cluster_ids[u] == old_cluster_ids[u] or is_seed[u]:
            offsets[cluster_ids[u] + 1] += 1
    for c in range(n_clusters):
        offsets[c + 1] += offsets[c]
    position = offsets[:n_clusters].copy()
    for u in range(n):
        if cluster_ids[u] == old_cluster_ids[u] or is_seed[u]:
            c = cluster_ids[u]
            members[position[c]] = u
            position[c] += 1
    return offsets[n_clusters]


def trusted_groups(cluster_ids, old_cluster_ids, is_seed, n_clusters):
    """(members, offsets): the trusted members of every cluster, flattened, and where each cluster's run starts."""
    offsets = np.zeros(n_clusters + 1, dtype=np.int64)
    members = np.empty(cluster_ids.shape[0], dtype=np.int64)
    total = _trusted_groups(cluster_ids, old_cluster_ids, is_seed, n_clusters, offsets, members)
    return members[:total], offsets


# -- a dense block of candidate rows --

@njit(parallel=True, cache=True)
def _dense_rows(indptr, indices, data, rows, out):
    for r in prange(rows.shape[0]):
        u = rows[r]
        for p in range(indptr[u], indptr[u + 1]):
            out[r, indices[p]] = data[p]


def dense_rows(F, rows):
    """The rows `rows` of the CSR F as a dense block."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros((rows.size, F.shape[1]), dtype=F.dtype)
    _dense_rows(F.indptr, F.indices, F.data, rows, out)
    return out


# -- the row-sum bound the accelerators prune candidates with --

@njit(parallel=True, cache=True)
def _candidate_ranges(sorted_sums, row_sums, distance, slack, start, end):
    """Per vertex, the range [start, end) of candidates (sorted by row sum) that can be closer than `distance`:
    those with S_u * (1 - d) < S_c < S_u / (1 - d).  Two binary searches per vertex, in parallel."""
    k = sorted_sums.shape[0]
    for u in prange(row_sums.shape[0]):
        keep = 1.0 - distance[u]
        if keep < 1e-12:
            keep = 1e-12
        v_lo = row_sums[u] * keep / slack
        v_hi = row_sums[u] / keep * slack if distance[u] < 1.0 else np.inf
        a, b = 0, k                        # first index with sorted_sums > v_lo
        while a < b:
            mid = (a + b) // 2
            if sorted_sums[mid] <= v_lo:
                a = mid + 1
            else:
                b = mid
        start[u] = a
        c, e = 0, k                        # first index with sorted_sums >= v_hi
        while c < e:
            mid = (c + e) // 2
            if sorted_sums[mid] < v_hi:
                c = mid + 1
            else:
                e = mid
        end[u] = c if c > a else a


def candidate_ranges(sorted_sums, row_sums, distance, slack=1.0 + 1e-5):
    """[start, end) per vertex into `sorted_sums`.  A range can be empty for an arbitrary `distance`; the callers
    pass each vertex's distance to its own centre (times the slack), which is always inside."""
    n = row_sums.shape[0]
    start = np.empty(n, dtype=np.uint32)
    end = np.empty(n, dtype=np.uint32)
    _candidate_ranges(np.asarray(sorted_sums, np.float64), np.asarray(row_sums, np.float64),
                      np.asarray(distance, np.float64), slack, start, end)
    return start, end


# -- seed-selection gains on the CPU --

@njit(parallel=True, cache=True)
def _reduce_gains(distances, own_distances, out, offset):
    """out[offset + t] = sum_u max(own_distances[u] - distances[u, t], 0), summed over u ascending."""
    n, n_candidates = distances.shape
    for t in prange(n_candidates):
        gain = 0.0
        for u in range(n):
            improvement = own_distances[u] - distances[u, t]
            if improvement > 0.0:
                gain += improvement
        out[offset + t] = gain


def gains(F, candidates, own_distances):
    """Seed-selection gains on the CPU: gain[c] = sum_u max(0, d_u - d(u, candidates[c])) for the dense candidate
    rows `candidates`, d_u being own_distances[u]."""
    n, width = F.shape
    n_candidates = candidates.shape[0]
    dtype, candidates, profile_sums, candidate_sums, data = prepare_centres(F, candidates)
    out = np.empty(n_candidates, dtype=np.float64)
    tile = min(n_candidates, tile_width(width, dtype.itemsize))
    for start in range(0, n_candidates, tile):
        end = min(start + tile, n_candidates)
        candidates_t = np.ascontiguousarray(candidates[start:end].T)
        distances = np.empty((n, end - start), dtype=dtype)
        distance_tile(F.indptr, F.indices, data, profile_sums, candidates_t, candidate_sums[start:end], distances)
        _reduce_gains(distances, own_distances, out, start)
    return out
