"""The partition the NTAC steps carry: the cluster id of every vertex, with each cluster's members cached."""
from __future__ import annotations

import numpy as np

from ..utils import relabel


class Partition:
    """A partition of n vertices: `cluster_ids`, dense int32, and `members()`, the vertices of every cluster.

    The NTAC step reads both views on every iteration; keeping them together avoids rebuilding the member lists
    each time.
    """

    __slots__ = ("cluster_ids", "_members")

    def __init__(self, cluster_ids):
        self.cluster_ids = relabel(cluster_ids)
        self._members = None

    @classmethod
    def one_cluster(cls, n):
        return cls(np.zeros(n, dtype=np.int32))

    @property
    def n_clusters(self):
        return int(self.cluster_ids.max()) + 1 if self.cluster_ids.size else 0

    def members(self):
        """One index array per cluster id, each ascending."""
        if self._members is None:
            order = np.argsort(self.cluster_ids, kind="stable")
            bounds = np.searchsorted(self.cluster_ids[order], np.arange(self.n_clusters + 1))
            self._members = [order[bounds[c]:bounds[c + 1]] for c in range(self.n_clusters)]
        return self._members
