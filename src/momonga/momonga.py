"""The method, as one object per graph: an NTAC start, the refinement, and transfer between two objects.

    refine(grid=(n1, n2, ...))   refine several NTAC partitions; the shortest description length is kept, and every
                                 refinement stays in `refinements`
    ntac                         the NTAC run (momonga.ntac.Ntac): `ntac.trajectory[j]` is its partition at j clusters
    ntac_partition()             NTAC's own answer: its partition of lowest Jaccard cost
    transfer(first, second)      carry the partition of one refined object into the other
"""
from __future__ import annotations

import numpy as np

from .graph import GraphData
from .ntac import Ntac
from .refine import refine
from .transfer import transfer_partition
from .utils import callback


class Momonga:
    """Unsupervised typing of one graph: an NTAC start, then the refinement that minimises the description length.

    data       a GraphData, or a sparse adjacency.  The method never reads expert types; pass
               `data.without_labels()` to make that explicit, and score on the original `data`.
    backend    where the NTAC steps run: "auto" (cuda > metal > cpu), "cuda", "metal" or "cpu"; "cpu" reproduces
               the reference NTAC solver bit for bit.  The refinement always runs on the CPU.
    log        log(line) gets progress lines from every step, e.g. log=print; None for silence
    verbose    also log every merge pass of the refinement

    After `refine`, `partition` is the kept refinement and `description_length` its length in nats; after
    `transfer` into this object, they are the transferred partition and the length of both graphs together.
    """

    def __init__(self, data, *, backend="auto", log=None, verbose=False):
        if not isinstance(data, GraphData):
            data = GraphData(data)
        self.data = data
        self.log = callback(log)
        self.verbose = verbose
        self.ntac = Ntac(self.data, backend=backend)
        self.refinements = {}                           # start cluster count -> (partition, description length)
        self.partition = None
        self.description_length = None
        self.start_n_clusters = None

    # -- the NTAC start --

    def unseeded_ntac(self, n_clusters, *, n_representatives=5, n_seeded_iterations=12, candidate_fraction=1.0,
                      batch_size=6000):
        """Run NTAC up to `n_clusters` clusters, continuing what was run before (momonga.ntac.Ntac.run).

        n_representatives     members per cluster that form its trusted centre
        n_seeded_iterations   seeded NTAC iterations after each new seed
        candidate_fraction    the fraction of vertices tried as the next seed
        batch_size            how many candidates are scored at a time (memory only)

        The defaults are the reference solver's.  Afterwards `ntac.trajectory[j]` is the partition with j clusters,
        for every j up to n_clusters.  Returns the last partition.
        """
        return self.ntac.run(n_clusters, log=self.log, n_representatives=n_representatives,
                             n_seeded_iterations=n_seeded_iterations, candidate_fraction=candidate_fraction,
                             batch_size=batch_size)

    def ntac_partition(self):
        """NTAC's own answer: the partition of its trajectory with the lowest Jaccard cost (`ntac.best()`)."""
        return self.ntac.best()

    # -- the refinement --

    def refine(self, grid=None, on_stage=None):
        """Refine partitions of the NTAC trajectory, and keep the result with the shortest description length.

        grid        the cluster counts of the NTAC trajectory to start from; default: its last partition
        on_stage    on_stage(name, partition, description_length) is called at every stage of every refinement

        Every result is kept in `refinements[start]` as (partition, description_length), `start` being the cluster
        count it started from.  The shortest sets `partition`, `description_length` and `start_n_clusters`.
        Returns the kept partition.
        """
        if self.ntac.n_clusters < 1:
            raise RuntimeError("nothing to refine: call unseeded_ntac(n_clusters) first")
        grid = [self.ntac.n_clusters] if grid is None else [int(start) for start in grid]
        for start in grid:
            if not 1 <= start <= self.ntac.n_clusters:
                raise ValueError(f"no NTAC partition with {start} clusters: the trajectory has 1.."
                                 f"{self.ntac.n_clusters}")
        for start in grid:
            self.log(f"refine from NTAC with {start} clusters")
            partition, length = refine(self.data, self.ntac.trajectory[start], on_stage=on_stage, log=self.log,
                                       verbose=self.verbose)
            self.refinements[start] = (partition, float(length))
        best_start = min(grid, key=lambda start: self.refinements[start][1])
        self.partition, self.description_length = self.refinements[best_start]
        self.start_n_clusters = best_start
        if len(grid) > 1:
            self.log(f"  kept the refinement from {best_start} (L={self.description_length:,.0f}, "
                     f"{self.partition.max() + 1} clusters) of "
                     + ", ".join(f"{start}: L={self.refinements[start][1]:,.0f}" for start in grid))
        return self.partition


def transfer(first, second, n_starts=11, on_stage=None):
    """Carry the partition of one refined Momonga object into the other.

    The donor is the object whose kept refinement has the shorter description length per edge; it is left
    unchanged.  Its partition is transferred, as it is, into the other object's kept refinement, which sets that
    object's `partition` (in the donor's cluster ids) and `description_length` (of both graphs together).
    Progress goes to the recipient's `log`.

    n_starts    how many cluster alignments the transfer tries
    on_stage    on_stage(name, partition, description_length) is called at every stage

    Returns the recipient.
    """
    for model in (first, second):
        if model.partition is None:
            raise RuntimeError("both objects must be refined before transfer")
    per_edge = [model.description_length / max(model.data.adjacency.nnz, 1) for model in (first, second)]
    donor, recipient = (first, second) if per_edge[0] <= per_edge[1] else (second, first)
    recipient.log(f"donor: the {'first' if donor is first else 'second'} graph (L per edge "
                  f"{min(per_edge):.4f} against {max(per_edge):.4f}); transferring into the other graph's "
                  f"refinement from {recipient.start_n_clusters}")
    partition, length = transfer_partition(recipient.data, recipient.partition, donor.data, donor.partition,
                                           n_starts=n_starts, log=recipient.log, on_stage=on_stage)
    recipient.partition = np.asarray(partition[:recipient.data.n], dtype=np.int32)
    recipient.description_length = float(length)
    return recipient
