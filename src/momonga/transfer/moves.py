"""Exact single-vertex moves with the number of clusters fixed: the last step of transfer (MoveState).

    description_length()     the description length of the state (momonga.objective, summed in this state's own order)
    best_improving(order)    every listed vertex's best move that shortens it
    move(u, b)               apply one move
    to_fixed_point(movable)  move vertices until none of the movable ones has a shortening move

Each round of to_fixed_point takes every movable vertex's best move on the same state, then visits the vertices
that have one in the given order, rescoring each on the current state and applying it if it still shortens the
description length.

The kernels are kernels.moves: per vertex the part of every move that does not depend on the target cluster
is computed once, and a target is skipped when a lower bound on its change exceeds the best move so far by more
than 1e-8, so they make the same moves as an exhaustive search over every target.
"""
from __future__ import annotations

import numpy as np

from ..kernels import moves as K
from ..kernels.moves import MemberLists, MoveArrays, edges_term_column_sums
from ..objective import PartitionCounts

TOLERANCE = 1e-9         # a move must lower the description length by more than this: float noise only


class MoveState(PartitionCounts):
    """A partition of the graph A into non-empty clusters, for single-vertex moves.

    A                 the adjacency matrix, integer weights, no self-loops
    partition         the cluster id of every vertex; every id up to the largest must be used
    min_edge_weight   the minimum edge weight the code assumes
    """

    def __init__(self, A, partition, *, min_edge_weight=5):
        super().__init__(A, partition, min_edge_weight=min_edge_weight)
        n, n_clusters = self.n, self.n_clusters
        # W (weights) and E (edge counts), out and in, dense with one row per cluster, column-major because the
        # kernels read one vertex's column across the clusters
        self.W_out, self.W_in = (np.asfortranarray(self.dense_by_cluster(X)) for X in (self.W_out, self.W_in))
        self.E_out, self.E_in = (np.asfortranarray(self.dense_by_cluster(X)) for X in (self.E_out, self.E_in))
        # per cluster, the change of its edges-term column if it lost (shrink) or gained (grow) one member
        self.shrink_out, self.grow_out, self.shrink_in, self.grow_in = (np.zeros(n_clusters) for _ in range(4))
        for c in range(n_clusters):
            edges_term_column_sums(c, self.partition, self.size, self.E_out, self.W_out, self.log_factorial,
                                   self.shrink_out, self.grow_out, self.min_edge_weight)
            edges_term_column_sums(c, self.partition, self.size, self.E_in, self.W_in, self.log_factorial,
                                   self.shrink_in, self.grow_in, self.min_edge_weight)
        A, A_t = self.A, self.A_t
        self.arrays = MoveArrays(
            self.partition, self.size, n_clusters, self.C, self.row_totals, self.column_totals, self.row_support,
            self.column_support, self.W_out, self.W_in, self.E_out, self.E_in, self.shrink_out, self.grow_out,
            self.shrink_in, self.grow_in, self.vertex_out_totals, self.vertex_in_totals,
            A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data.astype(np.int64),
            A_t.indptr.astype(np.int64), A_t.indices.astype(np.int64), A_t.data.astype(np.int64),
            self.log_factorial, self.min_edge_weight)
        # every cluster's members as a linked list, kept up to date by the moves; this build order sets the order
        # of the edges-term column sums
        first = np.full(n_clusters, -1, np.int64)
        next_member = np.full(n, -1, np.int64)
        previous = np.full(n, -1, np.int64)
        for u in range(n):
            cluster = self.partition[u]
            next_member[u] = first[cluster]
            if first[cluster] >= 0:
                previous[first[cluster]] = u
            first[cluster] = u
        self.members = MemberLists(first, next_member, previous, np.zeros(n, np.bool_), np.empty(n, np.int64))

    def description_length(self):
        """The description length of the current partition."""
        return K.description_length(self.arrays)

    def move(self, u, b):
        """Move vertex u to cluster b."""
        K.apply_move(u, b, self.arrays, self.members)

    def best_improving(self, order):
        """Every vertex of `order`'s best move that shortens the description length by more than TOLERANCE.

        Returns (target cluster, change) per vertex; (-1, -TOLERANCE) where there is none, including a vertex alone
        in its cluster.
        """
        best_target = np.full(order.size, -1, np.int64)
        best_change = np.full(order.size, np.inf)
        evaluated = np.empty(order.size, np.int64)          # targets whose edges-term change was computed
        K.best_improving(order, self.arrays, best_target, best_change, evaluated)
        return best_target, best_change

    def to_fixed_point(self, movable, on_round=None):
        """Move vertices of `movable`, in that order, until none has a move that shortens the description length.

        on_round   on_round(round, moved, candidates) after every round

        Returns the number of rounds.
        """
        movable = np.asarray(movable, dtype=np.int64)
        round_number = 0
        while True:
            round_number += 1
            _, best_change = self.best_improving(movable)
            candidates = movable[best_change < -TOLERANCE]
            moved = 0
            if candidates.size:
                moved, _ = K.sweep(candidates, 0, candidates.size, self.arrays, self.members)
            if on_round is not None:
                on_round(round_number, moved, int(candidates.size))
            if candidates.size == 0:
                return round_number
            if moved == 0:     # the first candidate sees the state best_improving scored, so this is a kernel bug
                raise RuntimeError(f"moves round {round_number}: {candidates.size} candidates with an improving "
                                   f"move, none moved")
