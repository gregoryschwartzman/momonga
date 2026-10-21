"""How well a partition matches the expert types.  Scoring only: nothing here is used to choose a partition.

Everything starts from the contingency table, counts[c, t] = the number of vertices in cluster c whose expert type
is t, and from one maximum-weight matching of clusters to types on it.

    matched_partition_metrics   accuracy, the fraction of vertices in the cluster matched to their type (vertices
                                in unmatched clusters count as errors); macro-F1, the mean over types of the matched
                                cluster's F1, zero for an unmatched type; the size-weighted F1; and purity, the
                                fraction of vertices in their cluster's majority type
    adjusted_rand_index         the adjusted Rand index of the two labellings
"""
from __future__ import annotations

import numpy as np

from .utils import max_weight_matching, relabel


def contingency(types, partition):
    """counts[c, t]: the number of vertices in cluster c with expert type t."""
    _, type_ids = np.unique(types, return_inverse=True)
    cluster_ids = relabel(partition)
    counts = np.zeros((cluster_ids.max() + 1, type_ids.max() + 1), dtype=np.int64)
    np.add.at(counts, (cluster_ids, type_ids), 1)
    return counts


def adjusted_rand_index(types, partition):
    """The adjusted Rand index of the two labellings (Hubert and Arabie), from the contingency table."""
    counts = contingency(types, partition)
    n = counts.sum()
    pairs = lambda x: (x * (x - 1) // 2).sum()
    index = pairs(counts)
    rows, cols = pairs(counts.sum(1)), pairs(counts.sum(0))
    expected = rows * cols / pairs(np.array([n]))
    maximum = (rows + cols) / 2
    return 1.0 if maximum == expected else float((index - expected) / (maximum - expected))


def matched_partition_metrics(types, partition):
    """{"acc", "f1_macro", "f1", "purity", "n_clusters", "n_types"} of `partition` against `types`, as fractions.

    One maximum-weight matching of clusters to types is shared by all scores.  A type's F1 is
    2 * matched_count / (type_size + matched_cluster_size); an unmatched type scores zero.  Only the types and
    clusters present in the given vertices count.
    """
    if np.asarray(types).size == 0:
        raise ValueError("cannot score an empty set of vertices")
    counts = contingency(types, partition)
    rows, cols = max_weight_matching(counts)
    matched = counts[rows, cols]
    type_size = counts.sum(0)
    f1 = np.zeros(counts.shape[1], dtype=float)
    f1[cols] = 2.0 * matched / (counts.sum(1)[rows] + type_size[cols])
    return dict(acc=float(matched.sum() / counts.sum()), f1_macro=float(f1.mean()),
                f1=float(f1 @ type_size / type_size.sum()), purity=float(counts.max(1).sum() / counts.sum()),
                n_clusters=counts.shape[0], n_types=counts.shape[1])
