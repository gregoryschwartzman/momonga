"""The refinement (`refine`): shatter the starting partition, merge, then rounds of split and merge while a
round shortens the description length.  Each stage is a function below.

On a graph holding both sides of an animal (the central brain, the VNC), twin vertices (the same type, opposite
sides) can have different profiles wherever the starting clusters are one-sided.  The shattering then uses
bilateral profiles (`twins`, `bilateral_profiles`), which put their partners in comparable coordinates.
Sides come from GraphData.side ("left", "right", "center", "unknown"); only left and right vertices take part.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..kernels.split import cut_clusters
from ..kernels.nearest import nearest_by_distance, nearest_other_side
from ..kernels.profiles import profiles
from ..utils import callback, relabel
from .merge import merge


# -- bilateral profiles --

def side_codes(side):
    """0 left, 1 right, 2 neither, per vertex; None unless the graph has both left and right vertices."""
    if side is None:
        return None
    side = np.asarray(side).astype(str)
    codes = np.where(side == "left", 0, np.where(side == "right", 1, 2)).astype(np.int64)
    return codes if (codes == 0).any() and (codes == 1).any() else None


def twins(vertex_profiles, side):
    """Each vertex's twin on the other side, or -1, from the profiles (one row per vertex): repeatedly pair the
    vertices that are each other's nearest other-side vertex, remove them, and pair the rest, until a round pairs
    no one."""
    twin = np.full(vertex_profiles.shape[0], -1, np.int64)
    unmatched = np.flatnonzero(side < 2)
    vertex_profiles = sparse.csr_matrix(vertex_profiles)
    while True:
        partner = nearest_other_side(vertex_profiles[unmatched], side[unmatched])
        has_partner = np.flatnonzero(partner >= 0)
        mutual = has_partner[partner[partner[has_partner]] == has_partner]
        mutual = mutual[side[unmatched[mutual]] == 0]                      # each mutual pair once, from its left end
        if mutual.size == 0:
            return twin
        left_vertices, right_vertices = unmatched[mutual], unmatched[partner[mutual]]
        twin[left_vertices] = right_vertices
        twin[right_vertices] = left_vertices
        unmatched = unmatched[twin[unmatched] < 0]


def _fold(adjacency, side, twin, rows_side):
    """The adjacency with the partners of every row on `rows_side` replaced by their twins, where they have one."""
    entries = adjacency.tocoo()
    folded = (side[entries.row] == rows_side) & (twin[entries.col] >= 0)
    columns = np.where(folded, twin[entries.col], entries.col)
    return sparse.csr_matrix((entries.data, (entries.row, columns)), shape=adjacency.shape)


def bilateral_profiles(data, partition, side, twin):
    """The bilateral profile of every vertex, 4 x n_clusters wide: the profile with the right side's partners folded
    onto their twins, then the one with the left side's partners folded, so that twins' profiles stay similar."""
    n_clusters = int(partition.max()) + 1
    A, A_t = data.adjacency, data.adjacency_t
    folded = [profiles(_fold(A, side, twin, rows_side), _fold(A_t, side, twin, rows_side), partition, n_clusters)
              for rows_side in (1, 0)]
    return sparse.hstack(folded, format="csr")


# -- the stages --

def shatter(data, partition, log=None):
    """The connected components of the nearest-neighbour links between vertices, under their profiles against
    `partition` (their bilateral profiles when the graph has both sides)."""
    log = callback(log)
    vertex_profiles = data.profiles(partition)
    side = side_codes(data.side)
    if side is not None:
        twin = twins(vertex_profiles, side)
        vertex_profiles = bilateral_profiles(data, partition, side, twin)
        log(f"  bilateral profiles: {int((twin >= 0).sum()) // 2} left-right twins")
    nearest = nearest_by_distance(vertex_profiles)
    n = data.n
    links = sparse.coo_matrix((np.ones(n), (np.arange(n), nearest)), shape=(n, n))
    _, components = connected_components(links, directed=False)
    return components.astype(np.int32)


def split(data, partition):
    """Cut every cluster of `partition` in two; each cut cluster's second half gets a new id.

    Returns (partition, number of clusters cut, number whose Lloyd loop stopped at the iteration cap).
    """
    n_clusters = int(partition.max()) + 1
    vertex_profiles = data.profiles(partition)
    partition = np.asarray(partition, dtype=np.int64)
    order = np.argsort(partition, kind="stable").astype(np.int64)         # vertices grouped by cluster
    cluster_start = np.zeros(n_clusters + 1, np.int64)                    # cluster c is order[start:end]
    cluster_start[1:] = np.cumsum(np.bincount(partition, minlength=n_clusters))
    half = np.zeros(partition.size, np.int64)
    was_cut, hit_cap = cut_clusters(vertex_profiles.indptr.astype(np.int64), vertex_profiles.indices.astype(np.int64),
                                    vertex_profiles.data.astype(np.float64), order, cluster_start,
                                    vertex_profiles.shape[1], half)
    result = partition.astype(np.int32)
    next_id = n_clusters
    for cluster in np.flatnonzero(was_cut):
        members = order[cluster_start[cluster]:cluster_start[cluster + 1]]
        result[members[half[members] == 1]] = next_id
        next_id += 1
    return result, int(was_cut.sum()), int(hit_cap.sum())


def refine(data, start, on_stage=None, log=None, *, verbose=False):
    """Refine the partition `start` of the GraphData `data`.  Returns (partition, description_length).

    log        log(line) gets one line per stage
    verbose    also log every merge pass
    on_stage   on_stage(name, partition, description_length) at every stage: start, shatter, merge, round1, ...;
               description_length is None where there is none yet
    """
    log, on_stage = callback(log), callback(on_stage)
    pass_log = log if verbose else None
    started = time.perf_counter()
    A, min_edge_weight = data.adjacency, data.min_edge_weight
    partition = relabel(start)
    on_stage("start", partition, None)
    partition = shatter(data, partition, log)
    on_stage("shatter", partition, None)
    log(f"  shatter: {partition.max() + 1} clusters")
    partition, length = merge(A, partition, min_edge_weight=min_edge_weight, log=pass_log)
    on_stage("merge", partition, length)
    log(f"  merge: {partition.max() + 1} clusters, L={length:,.0f}  ({time.perf_counter() - started:.0f}s)")
    kept_stage = "merge"
    round_number = 0
    while True:
        round_number += 1
        split_partition, n_cut, n_capped = split(data, partition)
        if verbose:
            log(f"  round {round_number}: split {partition.max() + 1} -> {split_partition.max() + 1} ({n_cut} cuts; "
                f"{n_capped} clusters at the iteration cap)")
        candidate, candidate_length = merge(A, split_partition, min_edge_weight=min_edge_weight, log=pass_log)
        if candidate_length >= length:
            log(f"  round {round_number}: split to {split_partition.max() + 1}, merge to {candidate.max() + 1}, "
                f"L={candidate_length:,.0f} not kept; the result is {kept_stage}  "
                f"({time.perf_counter() - started:.0f}s)")
            break
        partition, length, kept_stage = candidate, candidate_length, f"round{round_number}"
        on_stage(kept_stage, partition, length)
        log(f"  round {round_number}: split to {split_partition.max() + 1}, merge to {partition.max() + 1}, "
            f"L={length:,.0f} kept")
    return partition, length
