"""The merge state (MergeState): the description length of one partition, kept up to date under merges.

    description_length()  the description length (momonga.objective, summed in this state's own order)
    merge_change(a, b)    the exact change from merging clusters a and b
    merge(a, b)           apply that merge

The state starts from the PartitionCounts of the partition, made dense by cluster, and is kept over the original
cluster ids, 0 .. n_ids - 1: a merge empties one id and never renumbers.
"""
from __future__ import annotations

import numpy as np

from ..kernels import merge as K
from ..kernels.code import clusters_row, clusters_rows, log_binomial
from ..objective import PartitionCounts
from ..graph import membership

__all__ = ["MergeState"]


class MergeState(PartitionCounts):
    """The description length of the partition of the graph A, kept up to date under merges.

    A                 the adjacency matrix, integer weights (see kernels.code.count_matrix)
    partition         the cluster id of every vertex; every id up to the largest must be used
    min_edge_weight   the minimum edge weight the code assumes
    """

    def __init__(self, A, partition, *, min_edge_weight=5):
        super().__init__(A, partition, min_edge_weight=min_edge_weight)
        self.n_ids = self.n_clusters                                    # cluster ids; never renumbered
        # W (weights) and E (edge counts), out and in, dense with one row per cluster, so a merge adds two rows
        self.E_out, self.W_out = self.dense_by_cluster(self.E_out), self.dense_by_cluster(self.W_out)
        self.E_in, self.W_in = self.dense_by_cluster(self.E_in), self.dense_by_cluster(self.W_in)
        # cluster_partners[c]: the vertices with any edge into or out of cluster c, ascending
        both = (self.E_out > 0) | (self.E_in > 0)
        self.cluster_partners = [np.flatnonzero(both[c]).astype(np.int64) for c in range(self.n_ids)]
        self._union = np.empty(self.n, dtype=np.int64)                 # scratch for the union of two partner lists
        self.version = 0
        self.merges = []                                               # every merge applied, as (kept, removed)

    # -- the description length, in full --

    def _partition_term(self):
        """The partition term, without the constant log n."""
        G = self.log_factorial
        return (float(G[self.n] - G[self.size[self.size > 0]].sum())
                + float(log_binomial(self.n - 1, self.n_clusters - 1)))

    def _clusters_and_profiles_terms(self):
        """The clusters term plus the profiles term, both averaged over the two directions."""
        G = self.log_factorial
        live = np.flatnonzero(self.size > 0)
        M = membership(self.partition, self.n_ids)
        total = 0.0
        directions = ((self.C, self.vertex_out_totals, self.A), (self.C_t, self.vertex_in_totals, self.A_t))
        for C, vertex_totals, A in directions:
            total += 0.5 * self._clusters_term(C, live)
            C_live = C[np.ix_(live, live)]
            total += 0.5 * float(G[C_live.sum(1)].sum() - G[C_live[C_live > 0]].sum())
            vertex_part = float(G[vertex_totals].sum())
            W = A @ M                                                  # vertex-to-cluster weights, this direction
            vertex_part -= float(G[np.rint(W.data).astype(np.int64)].sum())
            total -= 0.5 * vertex_part
        return total

    def _clusters_term(self, C, live):
        """The clusters term of the cluster matrix C restricted to the live clusters, by rows."""
        C_live = C[np.ix_(live, live)]
        return float(clusters_rows(C_live.sum(1), (C_live > 0).sum(1).astype(np.int64), live.size,
                                   self.log_factorial).sum())

    def _edges_term(self):
        """The edges term, averaged over the two directions."""
        live = np.flatnonzero(self.size > 0)
        return sum(0.5 * sum(
            K.edges_term_of_cluster(int(c), self.cluster_partners[c], E, W, self.size, self.partition,
                                    self.log_factorial, self.min_edge_weight)
            for c in live)
            for E, W in ((self.E_out, self.W_out), (self.E_in, self.W_in)))

    def description_length(self):
        """The description length in nats."""
        return self._partition_term() + self._clusters_and_profiles_terms() + self._edges_term()

    # -- one merge's change --

    def _clusters_and_profiles_change(self, C, C_t, totals, support, live, kept, removed):
        """The change of the clusters and profiles terms in one direction when `removed` folds into `kept`.

        C, totals, support   the cluster matrix of this direction, its row totals and row supports
        C_t                  the other direction's cluster matrix, which is C transposed

        The scalar arithmetic is in NumPy's reduction order; only the array construction is in a kernel.
        """
        G = self.log_factorial
        n_live = live.size
        total_kept, total_removed = int(totals[kept]), int(totals[removed])
        # per other live cluster: the fold of its two columns and of its two row entries into one, and its
        # clusters term before and after the merge; and the merged row's support
        column_folds, row_folds, before, after, merged_support = K.fold_terms(C, C_t, live, kept, removed,
                                                                              support, totals, G)
        out = float(G[total_kept + total_removed] - G[total_kept] - G[total_removed])
        out -= float(column_folds.sum())
        out -= float(row_folds.sum())
        # the four entries that become one: kept-kept, kept-removed, removed-kept, removed-removed
        kk, kr, rk, rr = C[kept, kept], C[kept, removed], C[removed, kept], C[removed, removed]
        corner = int(kk) + int(kr) + int(rk) + int(rr)
        out -= float(G[corner] - G[kk] - G[kr] - G[rk] - G[rr])
        out = 0.5 * out
        out += 0.5 * float(after.sum() - before.sum())
        out += 0.5 * clusters_row(total_kept + total_removed, merged_support, n_live - 1, G)
        out -= 0.5 * (clusters_row(total_kept, int(support[kept]), n_live, G)
                      + clusters_row(total_removed, int(support[removed]), n_live, G))
        return out

    def merge_change(self, a, b):
        """The exact change of the description length from merging clusters a and b."""
        kept, removed = (a, b) if a < b else (b, a)
        G = self.log_factorial
        n_clusters = self.n_clusters
        size_kept, size_removed = int(self.size[kept]), int(self.size[removed])
        change = -(G[size_kept + size_removed] - G[size_kept] - G[size_removed])         # the partition term
        change += float(log_binomial(self.n - 1, n_clusters - 2)) - float(log_binomial(self.n - 1, n_clusters - 1))
        movers = self._union[:K.sorted_union(self.cluster_partners[kept], self.cluster_partners[removed],
                                             self._union)]                            # every vertex that can change
        live = np.flatnonzero(self.size > 0)
        directions = ((self.W_out, self.C, self.C_t, self.row_totals, self.row_support),
                      (self.W_in, self.C_t, self.C, self.column_totals, self.column_support))
        for W, C, C_t, totals, support in directions:                 # the clusters and profiles terms
            change += 0.5 * K.orderings_term(W, movers, kept, removed, G)
            change += self._clusters_and_profiles_change(C, C_t, totals, support, live, kept, removed)
        for E, W in ((self.E_out, self.W_out), (self.E_in, self.W_in)):   # the edges term
            change += 0.5 * K.edges_term_change(kept, removed, movers, E, W, self.size, self.partition, G,
                                                self.min_edge_weight)
        return change

    def merge(self, a, b):
        """Fold cluster `removed` = max(a, b) into `kept` = min(a, b)."""
        kept, removed = (a, b) if a < b else (b, a)
        live = np.flatnonzero(self.size > 0)
        # the columns kept and removed fold into one: every row with both loses one non-zero entry
        both = live[(self.C_t[kept, live] > 0) & (self.C_t[removed, live] > 0)]
        self.row_support[both[(both != kept) & (both != removed)]] -= 1
        # the rows kept and removed fold into one: every column with both likewise
        both = live[(self.C[kept, live] > 0) & (self.C[removed, live] > 0)]
        self.column_support[both[(both != kept) & (both != removed)]] -= 1
        self.version += 1
        for X in (self.E_out, self.W_out, self.E_in, self.W_in):
            X[kept] += X[removed]
            X[removed] = 0
        for C, totals in ((self.C, self.row_totals), (self.C_t, self.column_totals)):
            C[:, kept] += C[:, removed]
            C[:, removed] = 0
            C[kept] += C[removed]
            C[removed] = 0
            totals[kept] += totals[removed]
            totals[removed] = 0
        self.cluster_partners[kept] = np.union1d(self.cluster_partners[kept], self.cluster_partners[removed])
        self.cluster_partners[removed] = np.empty(0, dtype=np.int64)
        self.size[kept] += self.size[removed]
        self.size[removed] = 0
        self.partition[self.partition == removed] = kept
        self.n_clusters -= 1
        self.row_support[kept] = int((self.C[kept] > 0).sum())
        self.row_support[removed] = 0
        self.column_support[kept] = int((self.C_t[kept] > 0).sum())
        self.column_support[removed] = 0
        self.merges.append((kept, removed))
