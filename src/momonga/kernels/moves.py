"""The kernels of the exact single-vertex moves (momonga.transfer.moves.MoveState).  CPU only.

The state is dense and travels as one MoveArrays.  Per cluster c, row c of W_out / W_in holds the weight from / to
every vertex and row c of E_out / E_in the number of edges; C is the cluster matrix with its row and column totals
and supports (non-zero counts); shrink_* / grow_* are each cluster's change of the edges-term column if it lost /
gained one member; A_* and At_* are the adjacency and its transpose as CSR arrays.  The graph must have no
self-loops, and no cluster may empty.

Targets are pruned by lower bounds with a margin of 1e-8, so no candidate within float noise of the best is
discarded; a move is applied only when it lowers the description length by more than 1e-9.
"""
from collections import namedtuple

import numpy as np
from numba import njit, prange

from .code import clusters_row, edges_entry

MoveArrays = namedtuple("MoveArrays", [
    "partition", "size", "n_clusters", "C", "row_totals", "column_totals", "row_support", "column_support",
    "W_out", "W_in", "E_out", "E_in", "shrink_out", "grow_out", "shrink_in", "grow_in",
    "vertex_out_totals", "vertex_in_totals", "A_indptr", "A_indices", "A_weights", "At_indptr", "At_indices",
    "At_weights", "log_factorial", "min_edge_weight"])

# per cluster a linked list of its members (first[c], next[u], previous[u]), and two scratch arrays over the vertices
MemberLists = namedtuple("MemberLists", ["first", "next", "previous", "seen", "touched"])


@njit(cache=True)
def description_length(s):
    """The description length of the state s, summed in this order: the partition term, the clusters term, the
    profiles term (its cluster side, then its vertex side), the edges term."""
    G = s.log_factorial
    n, n_clusters = s.partition.size, s.n_clusters
    partition_term = G[n] + G[n - 1] - G[n_clusters - 1] - G[n - n_clusters]
    for c in range(n_clusters):
        partition_term -= G[s.size[c]]
    clusters_term = 0.0
    profiles_clusters = 0.0
    for a in range(n_clusters):
        row_total = 0
        row_support = 0
        column_total = 0
        column_support = 0
        for b in range(n_clusters):
            row_total += s.C[a, b]
            column_total += s.C[b, a]
            if s.C[a, b] > 0:
                row_support += 1
                profiles_clusters -= G[s.C[a, b]]
            if s.C[b, a] > 0:
                column_support += 1
        clusters_term += (0.5 * clusters_row(row_total, row_support, n_clusters, G)
                          + 0.5 * clusters_row(column_total, column_support, n_clusters, G))
        profiles_clusters += 0.5 * G[row_total] + 0.5 * G[column_total]
    profiles_vertices = 0.0
    for c in range(n_clusters):
        for v in range(n):
            profiles_vertices += 0.5 * G[s.W_out[c, v]] + 0.5 * G[s.W_in[c, v]]
    for v in range(n):
        profiles_vertices -= 0.5 * G[s.vertex_out_totals[v]] + 0.5 * G[s.vertex_in_totals[v]]
    edges_term = 0.0
    for c in range(n_clusters):
        for v in range(n):
            partners = s.size[c] - (1 if s.partition[v] == c else 0)
            if s.E_out[c, v] > 0:
                edges_term += 0.5 * edges_entry(partners, s.W_out[c, v], s.E_out[c, v], G, s.min_edge_weight)
            if s.E_in[c, v] > 0:
                edges_term += 0.5 * edges_entry(partners, s.W_in[c, v], s.E_in[c, v], G, s.min_edge_weight)
    return partition_term + clusters_term + profiles_clusters + profiles_vertices + edges_term


@njit(cache=True)
def edges_term_column_sums(c, partition, size, E, W, log_factorial, shrink, grow, min_edge_weight):
    """shrink[c], grow[c]: the change of cluster c's edges-term column if it lost / gained one member."""
    shrink_c = 0.0
    grow_c = 0.0
    for v in range(partition.size):
        n_edges = E[c, v]
        if n_edges > 0:
            partners = size[c] - (1 if partition[v] == c else 0)
            weight = W[c, v]
            now = edges_entry(partners, weight, n_edges, log_factorial, min_edge_weight)
            shrink_c += edges_entry(partners - 1, weight, n_edges, log_factorial, min_edge_weight) - now
            grow_c += edges_entry(partners + 1, weight, n_edges, log_factorial, min_edge_weight) - now
    shrink[c] = shrink_c
    grow[c] = grow_c


@njit(cache=True)
def _add_to_cluster_matrix(C, row_support, column_support, a, b, x):
    """C[a, b] += x, keeping the row and column supports."""
    old = C[a, b]
    new = old + x
    C[a, b] = new
    if old > 0 and new <= 0:
        row_support[a] -= 1
        column_support[b] -= 1
    elif old <= 0 and new > 0:
        row_support[a] += 1
        column_support[b] += 1


@njit(cache=True)
def log_count(x):
    """log x of a positive count."""
    return np.log(np.float64(x))


@njit(cache=True)
def source_terms(u, s, weight_to, weight_from, touched_to, touched_from):
    """The part of every move of u that does not depend on the target: the loss of u from its cluster a, in the
    partition term, the profiles term and the edges term of column a.  Also fills u's weight to and from every
    cluster (weight_to, weight_from), listing the clusters touched; returns (change, n_touched_to, n_touched_from).
    """
    G, min_edge_weight = s.log_factorial, s.min_edge_weight
    a = s.partition[u]
    n_touched_to = 0
    n_touched_from = 0
    change = 0.5 * s.shrink_out[a] + 0.5 * s.shrink_in[a] + log_count(s.size[a])
    for p in range(s.A_indptr[u], s.A_indptr[u + 1]):                  # the edges out of u
        v = s.A_indices[p]
        weight = s.A_weights[p]
        cluster_v = s.partition[v]
        if weight_to[cluster_v] == 0:
            touched_to[n_touched_to] = cluster_v
            n_touched_to += 1
        weight_to[cluster_v] += weight
        w_av = s.W_in[a, v]                                            # v's weight from cluster a
        n_edges = s.E_in[a, v]
        partners = s.size[a] - (1 if cluster_v == a else 0) - 1
        change += 0.5 * (G[w_av - weight] - G[w_av])
        change += 0.5 * (edges_entry(partners, w_av - weight, n_edges - 1, G, min_edge_weight)
                         - edges_entry(partners, w_av, n_edges, G, min_edge_weight))
    for p in range(s.At_indptr[u], s.At_indptr[u + 1]):                # the edges into u
        v = s.At_indices[p]
        weight = s.At_weights[p]
        cluster_v = s.partition[v]
        if weight_from[cluster_v] == 0:
            touched_from[n_touched_from] = cluster_v
            n_touched_from += 1
        weight_from[cluster_v] += weight
        w_av = s.W_out[a, v]                                           # v's weight to cluster a
        n_edges = s.E_out[a, v]
        partners = s.size[a] - (1 if cluster_v == a else 0) - 1
        change += 0.5 * (G[w_av - weight] - G[w_av])
        change += 0.5 * (edges_entry(partners, w_av - weight, n_edges - 1, G, min_edge_weight)
                         - edges_entry(partners, w_av, n_edges, G, min_edge_weight))
    for direction in range(2):                                         # u's own blocks in column a
        n_edges = s.E_out[a, u] if direction == 0 else s.E_in[a, u]
        weight = s.W_out[a, u] if direction == 0 else s.W_in[a, u]
        if n_edges > 0:
            partners = s.size[a] - 1
            change -= 0.5 * (edges_entry(partners - 1, weight, n_edges, G, min_edge_weight)
                             - edges_entry(partners, weight, n_edges, G, min_edge_weight))
    if weight_to[a] == 0:
        touched_to[n_touched_to] = a
        n_touched_to += 1
    return change, n_touched_to, n_touched_from


@njit(cache=True)
def target_terms(u, b, source_change, n_touched_to_0, n_touched_from, s, weight_to, weight_from, touched_to,
                 touched_from):
    """The change of every term but the target's profile term and the edges term, for moving u to cluster b:
    the partition term, the cluster matrix, and the row and column clusters terms, added to `source_change`."""
    G, n_clusters, C = s.log_factorial, s.n_clusters, s.C
    row_totals, column_totals, row_support, column_support = (s.row_totals, s.column_totals, s.row_support,
                                                              s.column_support)
    a = s.partition[u]
    change = source_change - log_count(s.size[b] + 1)
    n_touched_to = n_touched_to_0
    if weight_to[b] == 0:
        touched_to[n_touched_to] = b
        n_touched_to += 1
    profiles_change = 0.0
    clusters_change = 0.0
    column_clusters_change = 0.0
    total_change_a = 0
    total_change_b = 0
    support_a = row_support[a]
    support_b = row_support[b]
    column_support_a = column_support[a]
    column_support_b = column_support[b]
    for t in range(n_touched_to):                                      # rows a and b: the columns u touches
        j = touched_to[t]
        c_aj = C[a, j]
        c_bj = C[b, j]
        change_aj = -weight_to[j] + (-weight_from[a] if j == a else weight_from[a] if j == b else 0)
        change_bj = weight_to[j] + (-weight_from[b] if j == a else weight_from[b] if j == b else 0)
        new_aj = c_aj + change_aj
        new_bj = c_bj + change_bj
        profiles_change -= (G[new_aj] - G[c_aj] + G[new_bj] - G[c_bj])
        total_change_a += change_aj
        total_change_b += change_bj
        support_a += (1 if new_aj > 0 else 0) - (1 if c_aj > 0 else 0)
        support_b += (1 if new_bj > 0 else 0) - (1 if c_bj > 0 else 0)
        support_change = ((1 if new_aj > 0 else 0) - (1 if c_aj > 0 else 0)
                          + (1 if new_bj > 0 else 0) - (1 if c_bj > 0 else 0))
        if j == a:                                          # entries (a, a) and (b, a) belong to column a
            column_support_a += support_change
        elif j == b:                                        # entries (a, b) and (b, b) belong to column b
            column_support_b += support_change
        elif support_change != 0:                           # another column loses or gains a non-zero
            column_clusters_change += (clusters_row(column_totals[j], column_support[j] + support_change,
                                                    n_clusters, G)
                                       - clusters_row(column_totals[j], column_support[j], n_clusters, G))
    profiles_change += 0.5 * (G[row_totals[a] + total_change_a] - G[row_totals[a]]
                              + G[row_totals[b] + total_change_b] - G[row_totals[b]])
    clusters_change += 0.5 * (clusters_row(row_totals[a] + total_change_a, support_a, n_clusters, G)
                              - clusters_row(row_totals[a], row_support[a], n_clusters, G))
    clusters_change += 0.5 * (clusters_row(row_totals[b] + total_change_b, support_b, n_clusters, G)
                              - clusters_row(row_totals[b], row_support[b], n_clusters, G))
    column_change_a = -weight_from[a] - weight_from[b]
    column_change_b = weight_from[a] + weight_from[b]
    for t in range(n_touched_from):                                    # columns a and b: the rows u touches
        i = touched_from[t]
        if i == a or i == b:
            continue
        c_ia = C[i, a]
        c_ib = C[i, b]
        new_ia = c_ia - weight_from[i]
        new_ib = c_ib + weight_from[i]
        profiles_change -= (G[new_ia] - G[c_ia] + G[new_ib] - G[c_ib])
        column_change_a -= weight_from[i]
        column_change_b += weight_from[i]
        new_support_i = (row_support[i] + (1 if new_ia > 0 else 0) - (1 if c_ia > 0 else 0)
                         + (1 if new_ib > 0 else 0) - (1 if c_ib > 0 else 0))
        if new_support_i != row_support[i]:
            clusters_change += 0.5 * (clusters_row(row_totals[i], new_support_i, n_clusters, G)
                                      - clusters_row(row_totals[i], row_support[i], n_clusters, G))
        column_support_a += (1 if new_ia > 0 else 0) - (1 if c_ia > 0 else 0)
        column_support_b += (1 if new_ib > 0 else 0) - (1 if c_ib > 0 else 0)
    profiles_change += 0.5 * (G[column_totals[a] + column_change_a] - G[column_totals[a]]
                              + G[column_totals[b] + column_change_b] - G[column_totals[b]])
    column_clusters_change += (clusters_row(column_totals[a] + column_change_a, column_support_a, n_clusters, G)
                               - clusters_row(column_totals[a], column_support[a], n_clusters, G))
    column_clusters_change += (clusters_row(column_totals[b] + column_change_b, column_support_b, n_clusters, G)
                               - clusters_row(column_totals[b], column_support[b], n_clusters, G))
    clusters_change += 0.5 * column_clusters_change
    return change + clusters_change + profiles_change


@njit(cache=True)
def target_edges_term(u, b, s):
    """The exact change of the edges term in the target column b, for moving u there."""
    G, min_edge_weight = s.log_factorial, s.min_edge_weight
    change = 0.5 * s.grow_out[b] + 0.5 * s.grow_in[b]
    for p in range(s.A_indptr[u], s.A_indptr[u + 1]):                  # the edges out of u
        v = s.A_indices[p]
        weight = s.A_weights[p]
        w_bv = s.W_in[b, v]
        n_edges = s.E_in[b, v]
        partners = s.size[b] - (1 if s.partition[v] == b else 0) + 1
        change += 0.5 * (edges_entry(partners, w_bv + weight, n_edges + 1, G, min_edge_weight)
                         - edges_entry(partners, w_bv, n_edges, G, min_edge_weight))
    for p in range(s.At_indptr[u], s.At_indptr[u + 1]):                # the edges into u
        v = s.At_indices[p]
        weight = s.At_weights[p]
        w_bv = s.W_out[b, v]
        n_edges = s.E_out[b, v]
        partners = s.size[b] - (1 if s.partition[v] == b else 0) + 1
        change += 0.5 * (edges_entry(partners, w_bv + weight, n_edges + 1, G, min_edge_weight)
                         - edges_entry(partners, w_bv, n_edges, G, min_edge_weight))
    for direction in range(2):                                         # u's own blocks in column b
        n_edges = s.E_out[b, u] if direction == 0 else s.E_in[b, u]
        weight = s.W_out[b, u] if direction == 0 else s.W_in[b, u]
        if n_edges > 0:
            partners = s.size[b]
            change -= 0.5 * (edges_entry(partners + 1, weight, n_edges, G, min_edge_weight)
                             - edges_entry(partners, weight, n_edges, G, min_edge_weight))
    return change


@njit(cache=True)
def edges_entry_growth(partners, n_edges, weight, min_edge_weight):
    """The change of one edges-term block when the cluster grows by one, with its edges and weight fixed."""
    if n_edges <= 0:
        return 0.
    choices_now = min(partners, weight // min_edge_weight)
    choices_grown = min(partners + 1, weight // min_edge_weight)
    return (log_count(partners + 1) - log_count(partners + 1 - n_edges) + log_count(max(1, choices_grown))
            - log_count(max(1, choices_now)))


@njit(cache=True)
def target_bounds(u, b, s):
    """(profile_change, edges_bound): the exact change of the target's profile term, and a lower bound on the change
    of its edges term, for moving u to cluster b."""
    G, min_edge_weight = s.log_factorial, s.min_edge_weight
    profile_change = 0.
    edges_bound = 0.5 * s.grow_out[b] + 0.5 * s.grow_in[b]
    for direction in range(2):
        first = s.A_indptr[u] if direction == 0 else s.At_indptr[u]
        last = s.A_indptr[u + 1] if direction == 0 else s.At_indptr[u + 1]
        for p in range(first, last):
            v = s.A_indices[p] if direction == 0 else s.At_indices[p]
            weight = s.A_weights[p] if direction == 0 else s.At_weights[p]
            n_edges = s.E_in[b, v] if direction == 0 else s.E_out[b, v]
            w_bv = s.W_in[b, v] if direction == 0 else s.W_out[b, v]
            partners = s.size[b] - (1 if s.partition[v] == b else 0) + 1
            profile_change += 0.5 * (G[w_bv + weight] - G[w_bv])
            if n_edges == 0:
                edges_bound += 0.5 * (log_count(partners) + log_count(max(1, min(partners, weight // min_edge_weight))))
            else:
                # the exact support and header change after the cached size increase; for
                # slack = w_bv - min_edge_weight * n_edges >= 0, adding an edge of weight >= min_edge_weight
                # increases log C(slack + n_edges - 1, n_edges - 1) by at least log((slack + n_edges) / n_edges)
                slack = w_bv - min_edge_weight * n_edges
                increase = log_count(partners - n_edges) - log_count(n_edges + 1)
                increase += (log_count(max(1, min(partners, (w_bv + weight) // min_edge_weight)))
                             - log_count(max(1, min(partners, w_bv // min_edge_weight))))
                increase += log_count(slack + n_edges) - log_count(n_edges)
                edges_bound += 0.5 * increase
        n_edges = s.E_in[b, u] if direction == 0 else s.E_out[b, u]
        w_bu = s.W_in[b, u] if direction == 0 else s.W_out[b, u]
        edges_bound -= 0.5 * edges_entry_growth(s.size[b], n_edges, w_bu, min_edge_weight)
    return profile_change, edges_bound


@njit(cache=True)
def profile_term_bound(u, s):
    """A lower bound on the change of any target's profile term, for moving u."""
    G = s.log_factorial
    bound = 0.
    for p in range(s.A_indptr[u], s.A_indptr[u + 1]):
        bound += 0.5 * G[s.A_weights[p]]
    for p in range(s.At_indptr[u], s.At_indptr[u + 1]):
        bound += 0.5 * G[s.At_weights[p]]
    return bound


@njit(cache=True, parallel=True)
def best_improving(order, s, best_target, best_change, evaluated):
    """Every listed vertex's best move that lowers the description length by more than 1e-9, on the same state
    (read only): (-1, -1e-9) where there is none, including a vertex alone in its cluster.  evaluated[t] counts the
    targets whose edges-term change had to be computed."""
    n_clusters = s.n_clusters
    m = order.size
    n_chunks = 256
    for chunk in prange(n_chunks):
        weight_to = np.zeros(n_clusters, np.int64)
        weight_from = np.zeros(n_clusters, np.int64)
        touched_to = np.empty(n_clusters + 1, np.int64)
        touched_from = np.empty(n_clusters, np.int64)
        for t in range(chunk * m // n_chunks, (chunk + 1) * m // n_chunks):
            u = order[t]
            a = s.partition[u]
            best = -1e-9
            target = -1
            n_evaluated = 0
            if s.size[a] > 1:
                source_change, n_touched_to, n_touched_from = source_terms(u, s, weight_to, weight_from, touched_to,
                                                                           touched_from)
                profile_bound = profile_term_bound(u, s)
                for b in range(n_clusters):
                    if b != a:
                        without_edges = target_terms(u, b, source_change, n_touched_to, n_touched_from, s,
                                                     weight_to, weight_from, touched_to, touched_from)
                        if without_edges + profile_bound >= best + 1e-8:
                            continue
                        profile_change, edges_bound = target_bounds(u, b, s)
                        if without_edges + profile_change + edges_bound >= best + 1e-8:
                            continue
                        n_evaluated += 1
                        change = without_edges + profile_change + target_edges_term(u, b, s)
                        if change < best:
                            best = change
                            target = b
                for j in range(n_touched_to):
                    weight_to[touched_to[j]] = 0
                for j in range(n_touched_from):
                    weight_from[touched_from[j]] = 0
            best_target[t] = target
            best_change[t] = best
            evaluated[t] = n_evaluated


@njit(cache=True)
def refresh_column_sums(c, s, members):
    """Cluster c's edges-term column sums for a size change of -1 / +1, visiting only the partners of its members."""
    G, min_edge_weight = s.log_factorial, s.min_edge_weight
    for direction in range(2):
        n_touched = 0
        u = members.first[c]
        while u >= 0:                                       # every partner of every member, once
            first = s.At_indptr[u] if direction == 0 else s.A_indptr[u]
            last = s.At_indptr[u + 1] if direction == 0 else s.A_indptr[u + 1]
            for p in range(first, last):
                v = s.At_indices[p] if direction == 0 else s.A_indices[p]
                if not members.seen[v]:
                    members.seen[v] = True
                    members.touched[n_touched] = v
                    n_touched += 1
            u = members.next[u]
        shrink = 0.
        grow = 0.
        for t in range(n_touched):
            v = members.touched[t]
            members.seen[v] = False
            n_edges = s.E_out[c, v] if direction == 0 else s.E_in[c, v]
            weight = s.W_out[c, v] if direction == 0 else s.W_in[c, v]
            partners = s.size[c] - (1 if s.partition[v] == c else 0)
            now = edges_entry(partners, weight, n_edges, G, min_edge_weight)
            shrink += edges_entry(partners - 1, weight, n_edges, G, min_edge_weight) - now
            grow += edges_entry(partners + 1, weight, n_edges, G, min_edge_weight) - now
        if direction == 0:
            s.shrink_out[c] = shrink
            s.grow_out[c] = grow
        else:
            s.shrink_in[c] = shrink
            s.grow_in[c] = grow


@njit(cache=True)
def apply_move(u, b, s, members):
    """Move vertex u to cluster b, patching the state and the member lists."""
    C, row_support, column_support = s.C, s.row_support, s.column_support
    a = s.partition[u]
    for p in range(s.A_indptr[u], s.A_indptr[u + 1]):                  # the edges out of u
        v = s.A_indices[p]
        weight = s.A_weights[p]
        cluster_v = s.partition[v]
        _add_to_cluster_matrix(C, row_support, column_support, a, cluster_v, -weight)
        _add_to_cluster_matrix(C, row_support, column_support, b, cluster_v, weight)
        s.W_in[a, v] -= weight
        s.W_in[b, v] += weight
        s.E_in[a, v] -= 1
        s.E_in[b, v] += 1
    for p in range(s.At_indptr[u], s.At_indptr[u + 1]):                # the edges into u
        v = s.At_indices[p]
        weight = s.At_weights[p]
        cluster_v = s.partition[v]
        _add_to_cluster_matrix(C, row_support, column_support, cluster_v, a, -weight)
        _add_to_cluster_matrix(C, row_support, column_support, cluster_v, b, weight)
        s.W_out[a, v] -= weight
        s.W_out[b, v] += weight
        s.E_out[a, v] -= 1
        s.E_out[b, v] += 1
    s.row_totals[a] -= s.vertex_out_totals[u]
    s.row_totals[b] += s.vertex_out_totals[u]
    s.column_totals[a] -= s.vertex_in_totals[u]
    s.column_totals[b] += s.vertex_in_totals[u]
    s.size[a] -= 1
    s.size[b] += 1
    s.partition[u] = b
    previous = members.previous[u]                                     # unlink u from a's list
    following = members.next[u]
    if previous >= 0:
        members.next[previous] = following
    else:
        members.first[a] = following
    if following >= 0:
        members.previous[following] = previous
    old_first = members.first[b]                                       # and put it at the front of b's
    members.first[b] = u
    members.previous[u] = -1
    members.next[u] = old_first
    if old_first >= 0:
        members.previous[old_first] = u
    refresh_column_sums(a, s, members)
    refresh_column_sums(b, s, members)


@njit(cache=True)
def sweep(order, lo, hi, s, members):
    """Gauss-Seidel over order[lo:hi]: each vertex in turn moved to its best other cluster when that lowers the
    description length, applied at once.  Returns (moves, total change)."""
    n_clusters = s.n_clusters
    weight_to = np.zeros(n_clusters, np.int64)
    weight_from = np.zeros(n_clusters, np.int64)
    touched_to = np.empty(n_clusters + 1, np.int64)
    touched_from = np.empty(n_clusters, np.int64)
    moves = 0
    gain = 0.
    for t in range(lo, hi):
        u = order[t]
        a = s.partition[u]
        if s.size[a] <= 1:
            continue
        source_change, n_touched_to, n_touched_from = source_terms(u, s, weight_to, weight_from, touched_to,
                                                                   touched_from)
        profile_bound = profile_term_bound(u, s)
        best = -1e-9
        target = -1
        for b in range(n_clusters):
            if b != a:
                without_edges = target_terms(u, b, source_change, n_touched_to, n_touched_from, s, weight_to,
                                             weight_from, touched_to, touched_from)
                if without_edges + profile_bound >= best + 1e-8:
                    continue
                profile_change, edges_bound = target_bounds(u, b, s)
                if without_edges + profile_change + edges_bound >= best + 1e-8:
                    continue
                change = without_edges + profile_change + target_edges_term(u, b, s)
                if change < best:
                    best = change
                    target = b
        for j in range(n_touched_to):
            weight_to[touched_to[j]] = 0
        for j in range(n_touched_from):
            weight_from[touched_from[j]] = 0
        if target >= 0:
            apply_move(u, target, s, members)
            moves += 1
            gain += best
    return moves, gain
