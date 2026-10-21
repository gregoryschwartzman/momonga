"""Unseeded NTAC on one graph, with its trajectory kept.

    ntac = Ntac(data)            # data: a GraphData, already filtered at the minimum edge weight
    ntac.run(250)                # one cluster per step; ntac.trajectory[j] is the partition with j clusters
    ntac.run(300)                # continues from 250
    ntac.best()                  # NTAC's own answer: the partition of lowest Jaccard cost

`Momonga` holds one of these as `model.ntac`.
"""
from __future__ import annotations

import time

import numpy as np

from ..utils import callback
from .cost import partition_cost
from .device import Context
from .partition import Partition
from .unseeded import step


class Ntac:
    """The unseeded NTAC run of one graph.

    trajectory      trajectory[j] is the partition with j clusters, for every j reached so far (row 0 unused)
    jaccard_costs   jaccard_costs[j] is NTAC's own cost of trajectory[j]
    backend         where the steps run: "auto" (cuda > metal > cpu), "cuda", "metal" or "cpu";
                    "cpu" reproduces the reference NTAC solver bit for bit
    """

    def __init__(self, data, backend="auto"):
        self.data = data
        self.backend = backend
        self._context = None
        self.trajectory = np.zeros((0, data.n), dtype=np.int32)
        self.jaccard_costs = np.zeros(0)

    @property
    def n_clusters(self):
        """The largest cluster count reached so far, 0 before the first run."""
        return max(self.trajectory.shape[0] - 1, 0)

    def run(self, n_clusters, log=None, *, n_representatives=5, n_seeded_iterations=12, candidate_fraction=1.0,
            batch_size=6000):
        """Add clusters, one per step, until there are `n_clusters`, continuing the trajectory already run.

        log                   log(line) gets a line every ten clusters
        n_representatives     members per cluster that form its trusted centre
        n_seeded_iterations   seeded NTAC iterations after each new seed
        candidate_fraction    the fraction of vertices tried as the next seed
        batch_size            how many candidates are scored at a time (memory only)

        The defaults are the reference solver's.  Stops early when no further seed exists.  Returns the last
        partition.
        """
        log = callback(log)
        if self._context is None:
            self._context = Context(self.data, backend=self.backend)
        context = self._context
        reached = self.n_clusters
        if reached >= n_clusters:
            return self.trajectory[n_clusters]
        trajectory = np.zeros((n_clusters + 1, self.data.n), dtype=np.int32)
        jaccard_costs = np.full(n_clusters + 1, np.nan)
        trajectory[:self.trajectory.shape[0]] = self.trajectory
        jaccard_costs[:self.jaccard_costs.shape[0]] = self.jaccard_costs
        if reached < 1:
            partition = Partition.one_cluster(self.data.n)
            trajectory[1] = partition.cluster_ids
            jaccard_costs[1] = partition_cost(context, partition)
        else:
            partition = Partition(self.trajectory[reached])
        started = time.perf_counter()
        while partition.n_clusters < n_clusters:
            grown = step(context, partition, n_representatives=n_representatives,
                         n_seeded_iterations=n_seeded_iterations, candidate_fraction=candidate_fraction,
                         batch_size=batch_size)
            if grown is None:
                log(f"  NTAC: no further seed at {partition.n_clusters} clusters; stopping")
                trajectory = trajectory[:partition.n_clusters + 1]
                jaccard_costs = jaccard_costs[:partition.n_clusters + 1]
                break
            partition = grown
            reached = partition.n_clusters
            trajectory[reached] = partition.cluster_ids
            jaccard_costs[reached] = partition_cost(context, partition)
            if reached % 10 == 0 or reached == n_clusters:
                log(f"  NTAC {reached}/{n_clusters} clusters  jaccard cost {jaccard_costs[reached]:.4f}  "
                    f"({time.perf_counter() - started:.0f}s, {context.backend})")
        self.trajectory, self.jaccard_costs = trajectory, jaccard_costs
        return self.trajectory[-1]

    def best(self):
        """NTAC's own answer: the partition of the trajectory with the lowest Jaccard cost."""
        if self.n_clusters < 1:
            raise RuntimeError("no trajectory yet: run NTAC first")
        return self.trajectory[int(np.nanargmin(self.jaccard_costs[1:])) + 1]
