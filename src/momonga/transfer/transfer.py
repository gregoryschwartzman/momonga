"""Carrying one graph's partition, the donor's, into another graph, the recipient (`transfer_partition`).

    match       align the two cluster graphs: each cluster matrix, rows normalised plus columns normalised, the
                smaller zero-padded to the size of the larger, matched by FAQ (scipy's quadratic_assignment, 60
                iterations) from one start; then improve the matching by maximum-weight matching on the overlap of
                the aligned cluster profiles, until it repeats.  Every recipient cluster takes its matched donor id;
                an unmatched one takes the donor cluster it overlaps most.
    reassign    move recipient vertices between the donor's clusters: first each to the nearest donor mean profile
                (weighted Jaccard), then repeatedly to the nearest mean of the donor and recipient means, until the
                partition repeats.
    candidates  `n_starts` starts of `match`: the barycentre, then randomised ones.  Each goes through `reassign`,
                and the one whose two graphs together have the shortest description length is kept.
    moves       single recipient vertices moved while that description length falls, the donor fixed and the
                number of clusters fixed (transfer.moves.MoveState.to_fixed_point).

Choosing which graph donates is done by momonga.momonga.transfer.
Any loop that revisits an earlier state stops there and says so.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy import sparse
from scipy.optimize import quadratic_assignment

from ..kernels.jaccard import nearest_centres
from ..kernels.overlap import overlaps
from ..objective import description_length
from ..graph import cluster_graph, cluster_means
from .moves import MoveState
from ..utils import callback, max_weight_matching

FAQ_ITERATIONS = 60

# one alignment start carried through reassignment, with the joint description length it reached
Candidate = namedtuple("Candidate", ["length", "start", "matching", "outcome", "aligned", "reassigned"])


def _normalised(C):
    """The cluster matrix C with its rows normalised by their totals, and C transposed likewise (that is, C with its
    columns normalised)."""
    outgoing = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
    incoming = C.T / np.maximum(C.T.sum(1, keepdims=True), 1e-12)
    return outgoing, incoming


def match(recipient_clusters, donor_clusters, start=-1):
    """Align the recipient's clusters with the donor's, from one start.

    recipient_clusters, donor_clusters   the two cluster matrices (graph.cluster_graph)
    start                                -1: FAQ from the barycentre; >= 0: a randomised start with that seed

    Returns (matching, overlap, outcome): matching[c] is the donor cluster matched to recipient cluster c, or -1;
    overlap the profile overlap of every pair of clusters; outcome how the improvement loop ended.
    """
    n_recipient, n_donor = recipient_clusters.shape[0], donor_clusters.shape[0]
    size = max(n_recipient, n_donor)
    recipient_out, recipient_in = _normalised(recipient_clusters)
    donor_out, donor_in = _normalised(donor_clusters)
    recipient_matrix = np.zeros((size, size))
    recipient_matrix[:n_recipient, :n_recipient] = recipient_out + recipient_in.T
    donor_matrix = np.zeros((size, size))
    donor_matrix[:n_donor, :n_donor] = donor_out + donor_in.T
    # FAQ maximises sum_ij A_ij B_pi(i)pi(j), which is the minimisation of sum_ij (A_ij - B_pi(i)pi(j))^2: the two
    # squared norms do not depend on pi
    options = {"maximize": True, "maxiter": FAQ_ITERATIONS}
    if start >= 0:
        options.update({"P0": "randomized", "rng": start})
    faq = quadratic_assignment(recipient_matrix, donor_matrix, method="faq", options=options)
    matching = np.where(faq.col_ind[:n_recipient] < n_donor, faq.col_ind[:n_recipient], -1)
    seen = [matching.copy()]
    overlap = np.zeros((n_recipient, n_donor))
    rounds = 0
    while True:
        donor_out_aligned = np.zeros((n_donor, n_recipient))
        donor_in_aligned = np.zeros((n_donor, n_recipient))
        matched = matching >= 0
        donor_out_aligned[:, matched] = donor_out[:, matching[matched]]
        donor_in_aligned[:, matched] = donor_in[:, matching[matched]]
        overlaps(recipient_out, recipient_in, donor_out_aligned, donor_in_aligned, overlap)
        recipient_ids, donor_ids = max_weight_matching(overlap)
        updated = np.full(n_recipient, -1)
        updated[recipient_ids] = donor_ids
        rounds += 1
        if (updated == matching).all():
            return matching, overlap, f"converged after {rounds} rounds"
        if any((updated == earlier).all() for earlier in seen):
            return matching, overlap, f"stopped: the matching revisited an earlier one after {rounds} rounds"
        matching = updated
        seen.append(matching.copy())


def _nearest(vertex_profiles, centres):
    """The index of the nearest centre to every profile, by weighted Jaccard."""
    return np.asarray(nearest_centres(vertex_profiles, np.ascontiguousarray(centres, dtype=np.float32)),
                      dtype=np.int64)


def reassign(recipient, partition, donor_means, log=None):
    """Move the recipient's vertices between the donor's clusters, from the partition `partition` in donor ids.

    recipient     the recipient GraphData
    donor_means   the donor's mean profile of each cluster, one row per donor cluster

    Returns the partition the passes end at.
    """
    log = callback(log)
    n_donor = donor_means.shape[0]
    pass_number = 0
    seen = {partition.tobytes()}
    while True:
        pass_number += 1
        current = _nearest(recipient.profiles(partition, n_donor), donor_means)       # the donor update
        inner = {current.tobytes()}
        steps = 0
        while True:                                                                 # the joint updates
            vertex_profiles = recipient.profiles(current, n_donor)
            recipient_means, recipient_sizes = cluster_means(vertex_profiles, current, n_donor)
            centres = np.where((recipient_sizes > 0)[:, None], (recipient_means + donor_means) / 2, donor_means)
            updated = _nearest(vertex_profiles, centres)
            steps += 1
            if (updated == current).all():
                break
            if updated.tobytes() in inner:
                log(f"   pass {pass_number}: stopped, the joint steps revisited an earlier partition after "
                    f"{steps} steps")
                current = updated
                break
            current = updated
            inner.add(current.tobytes())
        converged = bool((current == partition).all())
        partition = current
        log(f"   pass {pass_number}: {steps} joint steps")
        if converged:
            break
        if partition.tobytes() in seen:
            log(f"   passes stopped: pass {pass_number} revisited an earlier partition")
            break
        seen.add(partition.tobytes())
    return partition


def transfer_partition(recipient, recipient_partition, donor, donor_partition, n_starts=11, log=None,
                       on_stage=None):
    """Carry the donor's partition into the recipient (two GraphData, each with its own partition).

    Returns (partition, description_length): the partition of both graphs together, recipient vertices first, in
    the donor's cluster ids; and the description length of the two graphs together.

    on_stage(name, partition, description_length) is called, with the partition of both graphs, for:

        candidate_<start>   every candidate after reassignment
        aligned             the kept candidate's alignment
        reassigned          its reassignment
        search_<round>      every round of the vertex moves (no length: it is computed once at the end)
    """
    log, on_stage = callback(log), callback(on_stage)
    min_edge_weight = recipient.min_edge_weight
    recipient_partition = np.asarray(recipient_partition, dtype=np.int64)
    donor_partition = np.asarray(donor_partition, dtype=np.int64)
    n_recipient, n_donor = int(recipient_partition.max()) + 1, int(donor_partition.max()) + 1
    donor_means, _ = cluster_means(donor.profiles(donor_partition, n_donor), donor_partition, n_donor)
    recipient_clusters = cluster_graph(recipient.adjacency, recipient_partition, n_recipient)
    donor_clusters = cluster_graph(donor.adjacency, donor_partition, n_donor)
    union = sparse.block_diag([recipient.adjacency, donor.adjacency], format="csr")

    def joint(partition):
        """The partition of both graphs together, given the recipient's."""
        return np.r_[partition, donor_partition]

    best = None
    for start in range(-1, n_starts - 1):
        matching, overlap, outcome = match(recipient_clusters, donor_clusters, start)
        aligned = np.where(matching >= 0, matching, overlap.argmax(1))[recipient_partition]
        reassigned = reassign(recipient, aligned, donor_means)
        length = description_length(union, joint(reassigned), min_edge_weight)
        on_stage(f"candidate_{start}", joint(reassigned), length)
        log(f"   candidate {'barycentre' if start < 0 else f'start {start:2d}'}: matched "
            f"{int((matching >= 0).sum())} of {n_recipient} ({outcome}); joint L={length:,.0f}")
        if best is None or length < best.length:
            best = Candidate(length, start, matching, outcome, aligned, reassigned)
    log(f"   kept {'the barycentre' if best.start < 0 else f'start {best.start}'} (joint L={best.length:,.0f}): "
        f"{int((best.matching >= 0).sum())} of {n_recipient} recipient clusters matched to {n_donor} donor "
        f"clusters; {best.outcome}")
    on_stage("aligned", joint(best.aligned), None)
    on_stage("reassigned", joint(best.reassigned), best.length)
    search = MoveState(union, joint(best.reassigned), min_edge_weight=min_edge_weight)

    def round_done(round_number, moved, candidates):
        on_stage(f"search_{round_number}", search.partition.copy(), None)
        log(f"   search {round_number}: {candidates} with an improving move, {moved} moved")

    search.to_fixed_point(np.arange(recipient_partition.size, dtype=np.int64), on_round=round_done)
    length = search.description_length()
    log(f"   joint L={length:,.0f} after the moves")
    return search.partition.copy(), length
