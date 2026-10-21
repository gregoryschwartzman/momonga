"""The NVIDIA-GPU kernels of the NTAC steps, through numba.cuda: the same two as metal.py, for one profile matrix
resident on the device (CudaProfiles).

Unlike Metal, CUDA has float64, so smax is accumulated in double exactly as numba does on the CPU (`smax = 0.0`
is typed float64 even though the arrays are float32), and the distances are bit-identical to the CPU path.  The
gains are reduced per block of vertices and summed on the host, like Metal, so the seed choice can still differ
from the CPU at a near-tie.
"""
from __future__ import annotations

import math

import numpy as np

from ..kernels.profiles import row_sums
from .kernels import candidate_ranges

_UNAVAILABLE_REASON = None

try:
    from numba import cuda
    from numba import float32 as numba_f4, int32 as numba_i4
    _HAVE_CUDA = cuda.is_available()
    if not _HAVE_CUDA:
        _UNAVAILABLE_REASON = "numba.cuda imported but cuda.is_available() is False"
except Exception as exc:
    cuda = None
    _HAVE_CUDA = False
    _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def available():
    return _HAVE_CUDA


def why_unavailable():
    """Why this machine is not using CUDA, or None if it is.

    A broken CUDA install (a driver mismatch, the wrong toolkit) raises at import and is caught, which looks exactly
    like "no CUDA on this machine".  On a machine that should have CUDA, call this to find out which happened.
    """
    return _UNAVAILABLE_REASON


if _HAVE_CUDA:

    @cuda.jit(device=True)
    def _distance(indptr, indices, data, profile_sum, centres_t, col, centre_sum, u):
        """The weighted-Jaccard distance from vertex u (a CSR row with sum profile_sum) to the centre in column
        `col` of the transposed centres (sum centre_sum)."""
        smax = 0.0                      # float64, as numba types it on the CPU
        for p in range(indptr[u], indptr[u + 1]):
            difference = data[p] - centres_t[indices[p], col]
            if difference > 0.0:
                smax += difference
        denominator = centre_sum + smax
        return (((centre_sum - profile_sum) + 2.0 * smax) / denominator) if denominator != 0.0 else 1.0

    @cuda.jit(cache=True)
    def _nearest_sorted(indptr, indices, data, profile_sums, centres_t, centre_sums, order, range_start, range_end,
                        n, n_centres, distances):
        t, u = cuda.grid(2)
        if t >= n_centres or u >= n:
            return
        original = order[t]
        if t < range_start[u] or t >= range_end[u]:
            distances[u, original] = math.inf       # provably above the minimum
            return
        distances[u, original] = _distance(indptr, indices, data, profile_sums[u], centres_t, t, centre_sums[t], u)

    @cuda.jit(cache=True)
    def _argmin_rows(distances, n, n_centres, out):
        """The argmin of each row, on the device: one block per vertex, then a shared-memory reduction.  Ties
        resolve to the smallest index, matching numpy.  An all-inf row yields 0, also matching numpy (the caller
        guarantees a finite entry exists)."""
        u = cuda.blockIdx.x
        if u >= n:
            return
        thread = cuda.threadIdx.x
        n_threads = cuda.blockDim.x
        values = cuda.shared.array(256, dtype=numba_f4)
        indices = cuda.shared.array(256, dtype=numba_i4)
        best = np.float32(np.inf)
        best_index = 0
        t = thread
        while t < n_centres:
            value = distances[u, t]
            if value < best:
                best = value
                best_index = t
            t += n_threads
        values[thread] = best
        indices[thread] = best_index
        cuda.syncthreads()
        stride = n_threads // 2
        while stride > 0:
            if thread < stride:
                other = thread + stride
                if values[other] < values[thread] or (values[other] == values[thread]
                                                      and indices[other] < indices[thread]):
                    values[thread] = values[other]
                    indices[thread] = indices[other]
            cuda.syncthreads()
            stride //= 2
        if thread == 0:
            out[u] = indices[0]

    @cuda.jit(cache=True)
    def _gains(indptr, indices, data, profile_sums, centres_t, centre_sums, range_start, range_end, own_distances,
               offset, block_size, n, n_candidates, partials):
        i, block = cuda.grid(2)
        if i >= n_candidates or block >= partials.shape[0]:
            return
        u0 = block * block_size
        u1 = u0 + block_size            # two-arg min() is not accepted inside a kernel by every numba version
        if u1 > n:
            u1 = n
        sorted_position = i + offset
        centre_sum = centre_sums[i]
        gain = 0.0
        for u in range(u0, u1):
            if sorted_position < range_start[u] or sorted_position >= range_end[u]:
                continue                # provably a zero term
            improvement = own_distances[u] - _distance(indptr, indices, data, profile_sums[u], centres_t, i,
                                                       centre_sum, u)
            if improvement > 0.0:
                gain += improvement
        partials[block, i] = gain


def _grid(nx, ny, tx=32, ty=8):
    return (int(math.ceil(nx / tx)), int(math.ceil(ny / ty))), (tx, ty)


class CudaProfiles:
    """One profile matrix resident on the device.  The same surface as metal.MetalProfiles."""

    def __init__(self, F):
        if not _HAVE_CUDA:
            raise RuntimeError(f"CUDA kernels requested but unavailable: {_UNAVAILABLE_REASON}")
        self.n, self.width = F.shape
        self.indptr = cuda.to_device(F.indptr.astype(np.int32))
        self.indices = cuda.to_device(F.indices.astype(np.int32))
        self.data = cuda.to_device(F.data.astype(np.float32))
        self.row_sums = row_sums(F).astype(np.float32)
        self.profile_sums = cuda.to_device(self.row_sums)

    def candidate_ranges(self, sorted_sums, own_distances):
        """[start, end) per vertex of the candidates, sorted by row sum, that can contribute to its gain
        (metal.MetalProfiles.candidate_ranges)."""
        return candidate_ranges(sorted_sums, self.row_sums, own_distances)

    def nearest_centres_sorted(self, centres, centre_sums, bound):
        """The nearest centre of every vertex, with the centres sorted by row sum so that the candidates within
        `bound` of each vertex are contiguous."""
        n_centres = centres.shape[0]
        order = np.argsort(centre_sums, kind="stable").astype(np.int32)
        range_start, range_end = candidate_ranges(centre_sums[order].astype(np.float64), self.row_sums, bound)
        if not np.all(range_end > range_start):
            raise ValueError("a vertex has no surviving candidate; the bound must be at least each vertex's "
                             "distance to its own centre")
        distances = cuda.device_array((self.n, n_centres), dtype=np.float32)
        blocks, threads = _grid(n_centres, self.n)
        _nearest_sorted[blocks, threads](
            self.indptr, self.indices, self.data, self.profile_sums,
            cuda.to_device(np.ascontiguousarray(centres[order].T)), cuda.to_device(centre_sums[order]),
            cuda.to_device(order), cuda.to_device(range_start.astype(np.int32)),
            cuda.to_device(range_end.astype(np.int32)), self.n, n_centres, distances)
        nearest = cuda.device_array(self.n, dtype=np.int32)
        _argmin_rows[self.n, 256](distances, self.n, n_centres, nearest)
        return nearest.copy_to_host().astype(np.int64)

    def gains(self, candidates, own_distances, range_start, range_end, offset, block_size=64):
        """gain[c] = sum_u max(0, d_u - d(u, c)) for the dense candidate rows `candidates` (a chunk of the sorted
        candidates beginning at `offset`), the vertices' ranges [range_start, range_end) pruning the terms."""
        n_candidates = candidates.shape[0]
        n_blocks = int(math.ceil(self.n / block_size))
        partials = cuda.device_array((n_blocks, n_candidates), dtype=np.float32)
        blocks, threads = _grid(n_candidates, n_blocks)
        _gains[blocks, threads](
            self.indptr, self.indices, self.data, self.profile_sums,
            cuda.to_device(np.ascontiguousarray(candidates.T)),
            cuda.to_device(candidates.sum(axis=1).astype(np.float32)),
            cuda.to_device(range_start.astype(np.int32)), cuda.to_device(range_end.astype(np.int32)),
            cuda.to_device(np.asarray(own_distances, dtype=np.float32)),
            int(offset), int(block_size), self.n, n_candidates, partials)
        return partials.copy_to_host().astype(np.float64).sum(axis=0)
