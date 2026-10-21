"""Unseeded NTAC: grow a partition one cluster at a time.

One step is `step()`:

    representatives        a few central members of every cluster
    next_seed              the vertex whose promotion to a new cluster lowers the cost most
    seeded_iterations      seeded NTAC iterations with trusted means, from those centres

On backend "cpu" this reproduces the reference solver exactly.
"""
from __future__ import annotations

import math

import numpy as np

from ..kernels.jaccard import nearest_centres
from .distances import assigned_distances, own_distances
from .kernels import dense_rows, gains, means_from_clusters, means_from_groups, trusted_groups
from .partition import Partition


def representatives(context, partition, n_representatives, vertex_profiles):
    """Up to n_representatives members per cluster, nearest their cluster mean, capped at half the cluster."""
    means = means_from_clusters(vertex_profiles, partition.members())
    distances = own_distances(vertex_profiles, means, partition.cluster_ids)   # each vertex's distance to its mean
    return [members[np.argsort(distances[members])][: min(n_representatives, (len(members) + 1) // 2)]
            for members in partition.members()]


def next_seed(context, partition, centres, candidate_fraction, batch_size, vertex_profiles):
    """The candidate whose promotion to its own cluster lowers the cost most, or None.

    centres              the representatives of every cluster
    candidate_fraction   the fraction of vertices tried as candidates
    batch_size           how many candidates are scored at a time (memory only)

    Candidates are drawn furthest-first from every cluster, in turn, until ceil(candidate_fraction * n) are
    collected.  The gain of a candidate c is the sum over vertices u of max(0, d_u - d(u, c)), d_u being u's distance
    to its own cluster's centre: how much every vertex would gain by moving to c.
    """
    n_candidates = math.ceil(candidate_fraction * context.n)
    means = means_from_clusters(vertex_profiles, centres)
    distances = own_distances(vertex_profiles, means, partition.cluster_ids)
    singletons = set().union(*(set(members.tolist()) for members in centres if len(members) == 1))
    furthest_first = [members[np.argsort(-distances[members])] for members in partition.members()]
    candidates = []
    rank = 0
    # round-robin the rank-th furthest member of every cluster; the clusters can run out before n_candidates is
    # reached, so stop when a full pass adds nothing
    while len(candidates) < n_candidates:
        added = 0
        for members in furthest_first:
            if len(members) > rank:
                candidates.append(members[rank])
                added += 1
        rank += 1
        if added == 0:
            break
    candidates = list(set(int(c) for c in candidates) - singletons)
    if not candidates:
        return None
    if context.accelerated:
        device = context.accelerator(vertex_profiles)
        candidate_array = np.asarray(candidates)
        # sort the candidates by row sum so each vertex's surviving candidates are contiguous
        candidate_row_sums = device.row_sums[candidate_array]
        order = np.argsort(candidate_row_sums, kind="stable")
        sorted_candidates = candidate_array[order]
        # per vertex, the range of sorted candidates whose row sum can give it a positive gain
        range_start, range_end = device.candidate_ranges(candidate_row_sums[order].astype(np.float64), distances)
        sorted_gains = np.empty(len(sorted_candidates), dtype=np.float64)
        for offset in range(0, len(sorted_candidates), batch_size):
            batch = sorted_candidates[offset:offset + batch_size]
            block = dense_rows(vertex_profiles, batch)
            sorted_gains[offset:offset + len(batch)] = device.gains(block, distances, range_start, range_end,
                                                                     offset)
        candidate_gains = np.empty_like(sorted_gains)
        candidate_gains[order] = sorted_gains
    else:
        batches = np.array_split(candidates, (len(candidates) + batch_size - 1) // batch_size)
        candidate_gains = np.hstack([gains(vertex_profiles, dense_rows(vertex_profiles, batch), distances)
                                     for batch in batches])
    return candidates[int(np.argmax(candidate_gains))]


def trusted_means_iteration(context, partition, old_cluster_ids, centres, is_seed):
    """One seeded NTAC iteration: centres from the vertices that did not move last time (seeds always count), then
    every vertex to its nearest centre, the seeds pinned.  Returns the new cluster ids."""
    vertex_profiles = context.profiles(partition.cluster_ids, partition.n_clusters)
    cluster_ids = partition.cluster_ids
    n_clusters = partition.n_clusters
    # the trusted members of every cluster, flattened, and where each cluster's run starts
    trusted, offsets = trusted_groups(cluster_ids, np.asarray(old_cluster_ids), is_seed, n_clusters)
    if not np.all(np.diff(offsets) > 0):
        raise RuntimeError("a cluster lost every trusted vertex; its mean would be undefined")
    means = means_from_groups(vertex_profiles, trusted, offsets)
    if context.accelerated:
        device = context.accelerator(vertex_profiles)
        mean_sums = means.sum(axis=1).astype(np.float32)
        # each vertex's distance to its own centre, a little above it: no closer centre can lie beyond it
        bound = assigned_distances(vertex_profiles, means, mean_sums, cluster_ids) * np.float32(1 + 1e-5)
        new_cluster_ids = device.nearest_centres_sorted(means, mean_sums, bound)
    else:
        new_cluster_ids = nearest_centres(vertex_profiles, means)
    for cluster, members in enumerate(centres):
        new_cluster_ids[members] = cluster
    return new_cluster_ids


def seeded_iterations(context, centres, n_iterations):
    """Seeded NTAC from the given centre groups: `n_iterations` trusted-means iterations."""
    n = context.n
    is_seed = np.zeros(n, dtype=np.bool_)
    old_cluster_ids = np.zeros(n, dtype=np.int32)
    for cluster, members in enumerate(centres):
        members = np.asarray(members, dtype=np.int64)
        is_seed[members] = True
        old_cluster_ids[members] = cluster
    partition = Partition(old_cluster_ids)
    for _ in range(n_iterations):
        new_cluster_ids = trusted_means_iteration(context, partition, old_cluster_ids, centres, is_seed)
        old_cluster_ids = partition.cluster_ids.copy()
        partition = Partition(new_cluster_ids)
    return partition


def step(context, partition, *, n_representatives, candidate_fraction, batch_size, n_seeded_iterations):
    """One unseeded step: the partition with one more cluster, or None if no seed is left."""
    vertex_profiles = context.profiles(partition.cluster_ids, partition.n_clusters)
    centres = representatives(context, partition, n_representatives, vertex_profiles)
    seed = next_seed(context, partition, centres, candidate_fraction, batch_size, vertex_profiles)
    if seed is None:
        return None
    seed_cluster = int(partition.cluster_ids[seed])
    centres[seed_cluster] = centres[seed_cluster][centres[seed_cluster] != seed]   # the seed leaves its cluster
    centres.append(np.array([seed]))                                                # and founds its own
    return seeded_iterations(context, centres, n_seeded_iterations)
