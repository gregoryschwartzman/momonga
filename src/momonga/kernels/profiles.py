"""The vertex profiles, built in one pass over the edges: the kernel behind GraphData.profiles.

The profile of a vertex against a partition is its weight to each cluster, then its weight from each cluster:
2 x n_clusters entries per vertex, kept sparse.  The original NTAC implementation builds a dense n x 2k array that is a few
percent non-zero and converts it to CSR for every distance call; this writes the same numbers straight into CSR,
into buffers sized by vertex degree, so the workspace is allocated once per graph and does not depend on the number
of clusters.

Every entry is an integer, a sum of synapse counts, exact in float32 below 2**24, so the order of accumulation does
not matter; the kernel walks each row's edges in storage order and emits the columns ascending.
"""
from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange
from scipy.sparse import csr_matrix


def thread_chunks(n, per_thread=4):
    """Boundaries of (threads x per_thread) equal chunks of range(n), for a prange over chunks."""
    n_chunks = max(1, get_num_threads() * per_thread)
    return np.linspace(0, n, n_chunks + 1).astype(np.int64)


@njit(parallel=True, cache=True)
def _row_sums(indptr, data, out):
    for i in prange(out.shape[0]):
        total = data.dtype.type(0)
        for p in range(indptr[i], indptr[i + 1]):
            total += data[p]
        out[i] = total
    return out


def row_sums(F):
    """The sum of every row of the CSR F, in its dtype."""
    return _row_sums(F.indptr, F.data, np.zeros(F.shape[0], dtype=F.data.dtype))


@njit(parallel=True, cache=True)
def _accumulate(chunks, n_clusters, indptr, indices, data, in_indptr, in_indices, in_data, partition,
                slot_start, out_indices, out_data, counts):
    """Each vertex's profile into its slot of the workspace: columns ascending, values summed in edge order."""
    width = 2 * n_clusters
    for chunk in prange(chunks.shape[0] - 1):
        scratch = np.zeros(width, dtype=out_data.dtype)
        seen = np.zeros(width, dtype=np.uint8)
        touched = np.empty(width, dtype=np.int32)
        for u in range(chunks[chunk], chunks[chunk + 1]):
            n_touched = 0
            for p in range(indptr[u], indptr[u + 1]):                 # the weight to each cluster
                column = partition[indices[p]]
                if seen[column] == 0:
                    seen[column] = 1
                    touched[n_touched] = column
                    n_touched += 1
                scratch[column] += data[p]
            for p in range(in_indptr[u], in_indptr[u + 1]):           # the weight from each cluster
                column = n_clusters + partition[in_indices[p]]
                if seen[column] == 0:
                    seen[column] = 1
                    touched[n_touched] = column
                    n_touched += 1
                scratch[column] += in_data[p]
            # the touched columns ascending: insertion sort wins below about 48 of them, numpy's sort above
            if n_touched <= 48:
                for i in range(1, n_touched):
                    key = touched[i]
                    j = i - 1
                    while j >= 0 and touched[j] > key:
                        touched[j + 1] = touched[j]
                        j -= 1
                    touched[j + 1] = key
            else:
                touched[:n_touched].sort()
            slot = slot_start[u]
            for q in range(n_touched):
                column = touched[q]
                out_indices[slot + q] = column
                out_data[slot + q] = scratch[column]
                scratch[column] = 0.0
                seen[column] = 0
            counts[u] = n_touched


@njit(parallel=True, cache=True)
def _compact(slot_start, counts, slot_indices, slot_data, indptr, indices, data):
    for u in prange(counts.shape[0]):
        s = slot_start[u]
        d = indptr[u]
        for q in range(counts[u]):
            indices[d + q] = slot_indices[s + q]
            data[d + q] = slot_data[s + q]


class Workspace:
    """The buffers of the profile build for one graph: a slot per vertex, sized by its degree."""

    def __init__(self, A, A_in):
        n = A.shape[0]
        degree = np.diff(A.indptr).astype(np.int64) + np.diff(A_in.indptr).astype(np.int64)
        self.slot_start = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(degree, out=self.slot_start[1:])
        slots = int(self.slot_start[-1])
        self.indices = np.empty(slots, dtype=np.int32)
        self.data = np.empty(slots, dtype=A.data.dtype)
        self.counts = np.zeros(n, dtype=np.int64)
        self.chunks = thread_chunks(n)


def profiles(A, A_in, partition, n_clusters, workspace=None):
    """The (n, 2 n_clusters) CSR profile of every vertex: its weight to each cluster (from A), then from each cluster
    (from A_in, normally A transposed).  Sorted indices, A's dtype.

    workspace   a Workspace for (A, A_in), reused across calls; built here if not given
    """
    workspace = Workspace(A, A_in) if workspace is None else workspace
    n = A.shape[0]
    partition = np.asarray(partition, dtype=np.int32)
    _accumulate(workspace.chunks, n_clusters, A.indptr, A.indices, A.data, A_in.indptr, A_in.indices, A_in.data,
                partition, workspace.slot_start, workspace.indices, workspace.data, workspace.counts)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(workspace.counts, out=indptr[1:])
    nnz = int(indptr[-1])
    if nnz > np.iinfo(np.int32).max:
        raise OverflowError(f"{nnz} nonzeros do not fit an int32 indptr")
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=A.data.dtype)
    _compact(workspace.slot_start, workspace.counts, workspace.indices, workspace.data, indptr, indices, data)
    F = csr_matrix((data, indices, indptr.astype(np.int32)), shape=(n, 2 * n_clusters))
    F.has_sorted_indices = True
    return F
