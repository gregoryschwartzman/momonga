"""The graph: GraphData holds the adjacency, the expert types for scoring, and builds the vertex profiles.

    profiles        the profile of every vertex against a partition: its weight to each cluster, then from each
                    cluster, 2 x n_clusters entries, kept sparse.  Every step of the method works in these
                    coordinates: the NTAC step, shattering, the split, the transfer.
    membership      the one-hot vertex x cluster matrix
    cluster_graph   the weight from each cluster to each cluster
    cluster_means   the mean profile of every cluster
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import sparse

from .kernels.confidence import own_and_nearest_other
from .kernels.profiles import Workspace, profiles
from .metrics import adjusted_rand_index, matched_partition_metrics
from .utils import relabel


def filter_adjacency(adjacency, min_edge_weight):
    """The adjacency with every edge lighter than `min_edge_weight` removed, as a canonical CSR.

    Duplicate entries are summed first: they represent one edge's total weight.  Vertices are kept, even those
    left without an edge.
    """
    A = sparse.csr_matrix(adjacency)
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    if not np.isfinite(A.data).all() or (A.data < 0).any():
        raise ValueError("edge weights must be finite and nonnegative")
    remove = A.data < min_edge_weight
    if remove.any():
        A = A.copy()
        A.data[remove] = 0
        A.eliminate_zeros()
    return A


def membership(partition, n_clusters, dtype=np.float64):
    """The (n, n_clusters) one-hot CSR matrix: row u has a 1 in the column of u's cluster."""
    n = partition.size
    return sparse.csr_matrix((np.ones(n, dtype), (np.arange(n), partition)), shape=(n, n_clusters))


def cluster_graph(A, partition, n_clusters):
    """The (n_clusters, n_clusters) matrix of the total weight from each cluster to each cluster, dense float64."""
    M = membership(partition, n_clusters)
    return (M.T @ A @ M).toarray().astype(np.float64)


def cluster_means(vertex_profiles, partition, n_clusters):
    """(means, sizes): the mean profile of every cluster, (n_clusters, 2 n_clusters) float64, and its size."""
    sizes = np.bincount(partition, minlength=n_clusters)
    sums = (membership(partition, n_clusters).T @ vertex_profiles).toarray().astype(np.float64)
    return sums / np.maximum(sizes, 1)[:, None], sizes


def _normalised_rows(X):
    """A float64 CSR copy of X with every row divided by its sum; an empty row stays zero."""
    X = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    X.sort_indices()
    counts = np.diff(X.indptr)
    sums = np.zeros(X.shape[0])
    nonempty = counts > 0
    sums[nonempty] = np.add.reduceat(X.data, X.indptr[:-1][nonempty])
    divisor = sums[np.repeat(np.arange(X.shape[0]), counts)]
    X.data = X.data / np.where(divisor > 0, divisor, 1.0)
    return X


class GraphData:
    """A directed, synapse-weighted graph and, per vertex, optionally its expert type, its id and its side.

    adjacency         a square sparse matrix of synapse counts, rows presynaptic
    labels            the expert type of each vertex, used for scoring only (NTAC's name for it)
    ids               each vertex's neuron id in the source dataset
    side              "left", "right", "center" or "unknown"
    min_edge_weight   edges lighter than this are removed here, once.  Everything downstream reads
                      `min_edge_weight` from the graph, as the code assumes every edge weighs at least that much.

    A vertex left without an edge stays in the graph and is left out of the scores (`has_edge`).
    """

    def __init__(self, adjacency, labels=None, *, ids=None, side=None, min_edge_weight=5):
        A = sparse.csr_matrix(adjacency).astype(np.float32)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square, got {A.shape}")
        self.n = A.shape[0]
        self.min_edge_weight = int(min_edge_weight)
        self.adjacency = filter_adjacency(A, self.min_edge_weight)      # the only place edges are filtered
        self.labels = self._per_vertex(labels, "labels")
        self.ids = self._per_vertex(ids, "ids")
        self.side = self._per_vertex(side, "side")
        self.has_edge = (np.asarray(self.adjacency.sum(1)).ravel() + np.asarray(self.adjacency.sum(0)).ravel()) > 0
        self._adjacency_t = None
        self._workspace = None

    def _per_vertex(self, values, name):
        if values is None:
            return None
        values = np.asarray(values)
        if values.size != self.n:
            raise ValueError(f"{values.size} {name} for {self.n} vertices")
        return values

    @property
    def adjacency_t(self):
        """The adjacency transposed, as CSR; built once."""
        if self._adjacency_t is None:
            self._adjacency_t = self.adjacency.T.tocsr()
        return self._adjacency_t

    def profiles(self, partition, n_clusters=None):
        """The (n, 2 n_clusters) CSR profile of every vertex against `partition`: its weight to each cluster, then
        from each cluster.  Float32, sorted indices.

        n_clusters   the number of cluster ids; by default one more than the largest id in `partition`
        """
        partition = np.asarray(partition)
        if n_clusters is None:
            n_clusters = int(partition.max()) + 1
        if self._workspace is None:
            self._workspace = Workspace(self.adjacency, self.adjacency_t)
        return profiles(self.adjacency, self.adjacency_t, partition, n_clusters, self._workspace)

    def without_labels(self):
        """A copy without the expert types: the graph, ids and sides only.

        Pass this to Momonga, so the method has no path to the types; score on the original.
        """
        result = copy.copy(self)
        result.labels = None
        return result

    def confidence(self, partition):
        """Per vertex, a confidence in [0, 1] that it belongs where `partition` put it: 1 - own / other, with
        `own` its weighted-Jaccard distance to its own cluster's profile and `other` the distance to the nearest
        other cluster's.  A cluster's profile is the sum of its members' profiles, normalised.

        0 is a tie with another cluster (including own = other = 0); 1 is a vertex far closer to its own cluster;
        with a single cluster every vertex gets 1.  It is a margin to the next cluster, not a position among cluster
        mates: a vertex can sit in the middle of a cluster that is itself wrong.  Uses no expert types.
        """
        partition = relabel(partition)
        n_clusters = int(partition.max()) + 1
        raw = self.profiles(partition, n_clusters)
        vertex_profiles = _normalised_rows(raw)
        cluster_profiles = _normalised_rows(membership(partition, n_clusters).T @ raw)
        cluster_profiles.eliminate_zeros()
        own, other = own_and_nearest_other(vertex_profiles, cluster_profiles, partition)
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = 1.0 - own / other
        return np.clip(np.nan_to_num(margin, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)   # nan: 0 / 0, a tie

    def get_metrics(self, partition):
        """The scores of `partition` against this graph's expert types, over the vertices with an edge:
        {"acc", "f1_macro", "f1", "ari", "purity", "n_clusters"} (see momonga.metrics)."""
        if self.labels is None:
            raise ValueError("this graph has no expert types to score against")
        partition = np.asarray(partition)
        if partition.size != self.n:
            raise ValueError(f"a partition of {partition.size} vertices for a graph of {self.n}")
        types, partition = self.labels[self.has_edge], partition[self.has_edge]
        scores = matched_partition_metrics(types, partition)
        return {"acc": scores["acc"], "f1_macro": scores["f1_macro"], "f1": scores["f1"],
                "ari": adjusted_rand_index(types, partition), "purity": scores["purity"],
                "n_clusters": scores["n_clusters"]}

    @classmethod
    def load(cls, name, min_edge_weight=5):
        """One of the paper's graphs by name, e.g. "fafb_ol_right", with its expert types, ids and sides
        (momonga.download_data fetches them).

        min_edge_weight   edges lighter than this are removed (the paper uses 5)
        """
        from .data import graph_files

        adjacency_file, vertex_file = graph_files(name)
        A = sparse.load_npz(adjacency_file).tocsr()
        vertices = np.load(vertex_file)
        # the stored files keep every neuron of the source; the paper's graphs leave out those without any synapse
        kept = (np.asarray(A.sum(1)).ravel() + np.asarray(A.sum(0)).ravel()) > 0
        return cls(A[kept][:, kept], labels=vertices["type"][kept].astype(str), ids=vertices["id"][kept],
                   side=vertices["side"][kept].astype(str), min_edge_weight=min_edge_weight)
