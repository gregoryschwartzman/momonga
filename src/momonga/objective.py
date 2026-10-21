"""The description length of a graph given a partition, in nats: the quantity the method minimises.

It is the length of a code that sends the graph using the partition, and the sum of four terms:

    partition   which cluster each vertex is in
    clusters    the total weight between every pair of clusters
    profiles    each vertex's weight to and from every cluster
    edges       which vertices each vertex is connected to, and with what weights

The last three are averaged over the outgoing and incoming directions.  Every edge must weigh at least
min_edge_weight (GraphData removes lighter ones when the graph is made).  Divide by log 2 for bits.  A constant that
is the same for every partition is left out.

`clusters_row` and `edges_entry` (kernels.code) are the only place their formulas are written.  Every kernel and
`description_length` use them, so every total and every incremental change share one arithmetic.
"""
from __future__ import annotations

import numpy as np

from .kernels.code import clusters_rows, count_matrix, edges_sum, log_binomial, log_factorial_table
from .kernels.merge import row_supports, scatter_csr
from .graph import membership

__all__ = ["PartitionCounts", "description_length"]


def _integer_csr(X):
    """X as CSR with sorted indices, every value rounded to the integer it holds."""
    X = X.tocsr()
    X.sort_indices()
    X.data = np.rint(X.data)
    return X


def _dense(X):
    """The sparse integer matrix X as a dense int64 array."""
    X = _integer_csr(X)
    out = np.zeros(X.shape, dtype=np.int64)
    scatter_csr(X.indptr.astype(np.int64), X.indices.astype(np.int64), X.data.astype(np.int64), out)
    return out


class PartitionCounts:
    """The integer counts the code reads from the partition of the graph A.

    A                 the adjacency, as kernels.code.count_matrix validates it
    partition         the cluster id of every vertex; every id up to the largest must be used
    min_edge_weight   the minimum edge weight the code assumes

    Per vertex and cluster, W_out / W_in hold the weight from / to the vertex and E_out / E_in the number of edges,
    as sparse (n, n_clusters) matrices.  C is the cluster matrix, the weight from each cluster to each cluster, and
    C_t its transpose, dense; their row totals and supports (non-zero counts) are kept too.
    """

    def __init__(self, A, partition, *, min_edge_weight=5):
        self.min_edge_weight = int(min_edge_weight)
        self.A = count_matrix(A, self.min_edge_weight)
        self.A_t = self.A.T.tocsr()
        self.partition = np.asarray(partition, dtype=np.int64).copy()
        self.n = n = self.partition.size
        if n != self.A.shape[0]:
            raise ValueError(f"a partition of {n} vertices for a graph of {self.A.shape[0]}")
        self.n_clusters = n_clusters = int(self.partition.max()) + 1
        self.size = np.bincount(self.partition, minlength=n_clusters).astype(np.int64)
        if (self.size == 0).any():
            raise ValueError("every cluster id up to the largest must be used")
        M = membership(self.partition, n_clusters)
        B = self.A.copy()                                   # A with every weight 1
        B.data = np.ones_like(B.data)
        self.W_out = _integer_csr(self.A @ M)
        self.W_in = _integer_csr(self.A_t @ M)
        self.E_out = _integer_csr(B @ M)
        self.E_in = _integer_csr(B.T.tocsr() @ M)
        self.C = _dense(M.T @ self.W_out)
        self.C_t = _dense(M.T @ self.W_in)
        self.row_totals = self.C.sum(1)
        self.column_totals = self.C_t.sum(1)
        self.row_support = row_supports(self.C)
        self.column_support = row_supports(self.C_t)
        self.vertex_out_totals = np.asarray(self.W_out.sum(1)).ravel().astype(np.int64)
        self.vertex_in_totals = np.asarray(self.W_in.sum(1)).ravel().astype(np.int64)
        self.log_factorial = log_factorial_table(int(self.A.sum()) + n + n_clusters + 3)  # log_factorial[x] = log x!

    def dense_by_cluster(self, X):
        """A (n, n_clusters) count matrix transposed and dense: row c is cluster c."""
        return _dense(X.T)

    def description_length(self):
        """The description length of the partition, in nats."""
        G = self.log_factorial
        n, n_clusters, size = self.n, self.n_clusters, self.size
        partition_term = float(G[n] - G[size].sum()) + float(log_binomial(n - 1, n_clusters - 1))
        clusters = profiles = edges = 0.0
        directions = ((self.C, self.row_totals, self.row_support, self.W_out, self.E_out, self.vertex_out_totals),
                      (self.C_t, self.column_totals, self.column_support, self.W_in, self.E_in, self.vertex_in_totals))
        for C, totals, support, W, E, vertex_totals in directions:
            clusters += 0.5 * float(clusters_rows(totals, support, n_clusters, G).sum())
            weights = W.data.astype(np.int64)
            edge_counts = E.data.astype(np.int64)
            profiles += 0.5 * (float(G[totals].sum() - G[C[C > 0]].sum())
                               - float(G[vertex_totals].sum() - G[weights].sum()))
            edges += 0.5 * edges_sum(E.indptr.astype(np.int64), E.indices.astype(np.int64), edge_counts, weights,
                                     self.partition, size, G, self.min_edge_weight)
        return partition_term + clusters + profiles + edges


def description_length(A, partition, min_edge_weight=5):
    """The description length of the partition `partition` of the graph A, in nats.

    A                 the adjacency, sparse, integer weights, every edge at least min_edge_weight
    partition         the cluster id of every vertex; every id up to the largest must be used
    min_edge_weight   the minimum edge weight the code assumes
    """
    return PartitionCounts(A, partition, min_edge_weight=min_edge_weight).description_length()
