"""The profile overlaps of the transfer's alignment (momonga.transfer.transfer.match), as a kernel.

out[c, j] is the weight the aligned outgoing and incoming profiles of recipient cluster c and donor cluster j
share, averaged over the two directions: (sum_l min(recipient_out[c, l], donor_out[j, l]) + the same for the
incoming profiles) / 2.  Each sum is taken in NumPy's pairwise order (`pairwise_sum`, the algorithm of numpy's
contiguous float64 reduction: blocks of eight partial sums up to 128 terms, halves above that), so the kernel gives
the same bits as `np.minimum(recipient_out[c][None, :], donor_out).sum(1)` did, and the matching it feeds is unchanged.
"""
import numpy as np
from numba import njit, prange

_BLOCK = 128
_DEPTH = 64          # the recursion depth numpy's halving needs for any length that fits in memory


@njit(cache=True, nogil=True, inline="always")
def _block_sum(x, y, start, n):
    """sum over i in [start, start + n) of min(x[i], y[i]) for n <= 128, in NumPy's unrolled order."""
    if n < 8:
        total = 0.0
        for i in range(start, start + n):
            total += min(x[i], y[i])
        return total
    sum_0 = min(x[start], y[start])
    sum_1 = min(x[start + 1], y[start + 1])
    sum_2 = min(x[start + 2], y[start + 2])
    sum_3 = min(x[start + 3], y[start + 3])
    sum_4 = min(x[start + 4], y[start + 4])
    sum_5 = min(x[start + 5], y[start + 5])
    sum_6 = min(x[start + 6], y[start + 6])
    sum_7 = min(x[start + 7], y[start + 7])
    i = 8
    while i < n - (n % 8):
        sum_0 += min(x[start + i], y[start + i])
        sum_1 += min(x[start + i + 1], y[start + i + 1])
        sum_2 += min(x[start + i + 2], y[start + i + 2])
        sum_3 += min(x[start + i + 3], y[start + i + 3])
        sum_4 += min(x[start + i + 4], y[start + i + 4])
        sum_5 += min(x[start + i + 5], y[start + i + 5])
        sum_6 += min(x[start + i + 6], y[start + i + 6])
        sum_7 += min(x[start + i + 7], y[start + i + 7])
        i += 8
    total = ((sum_0 + sum_1) + (sum_2 + sum_3)) + ((sum_4 + sum_5) + (sum_6 + sum_7))
    while i < n:
        total += min(x[start + i], y[start + i])
        i += 1
    return total


@njit(cache=True, nogil=True)
def pairwise_sum(x, y, start, n, starts, lengths, stages, left_sums):
    """sum over i in [start, start + n) of min(x[i], y[i]), in NumPy's pairwise order.  `starts`, `lengths`,
    `stages` (int64) and `left_sums` (float64) are scratch stacks of length _DEPTH: numpy's recursion, run with an
    explicit stack."""
    starts[0] = start
    lengths[0] = n
    stages[0] = 0
    depth = 1
    result = 0.0
    while depth > 0:
        top = depth - 1
        if stages[top] == 0:                        # first visit: a block, or push the left half
            if lengths[top] <= _BLOCK:
                result = _block_sum(x, y, starts[top], lengths[top])
                depth -= 1
            else:
                half = lengths[top] // 2
                half -= half % 8
                stages[top] = 1
                starts[depth] = starts[top]
                lengths[depth] = half
                stages[depth] = 0
                depth += 1
        elif stages[top] == 1:                      # the left half just returned in result: push the right half
            left_sums[top] = result
            half = lengths[top] // 2
            half -= half % 8
            stages[top] = 2
            starts[depth] = starts[top] + half
            lengths[depth] = lengths[top] - half
            stages[depth] = 0
            depth += 1
        else:                                       # the right half just returned in result
            result = left_sums[top] + result
            depth -= 1
    return result


@njit(cache=True, nogil=True, parallel=True)
def overlaps(recipient_out, recipient_in, donor_out, donor_in, out):
    """out[c, j] for every recipient cluster c (rows of recipient_out, recipient_in) and donor cluster j (rows of
    donor_out, donor_in), all over the recipient's cluster coordinates."""
    n_recipient, width = recipient_out.shape
    n_donor = donor_out.shape[0]
    for c in prange(n_recipient):
        starts = np.empty(_DEPTH, np.int64)
        lengths = np.empty(_DEPTH, np.int64)
        stages = np.empty(_DEPTH, np.int64)
        left_sums = np.empty(_DEPTH, np.float64)
        for j in range(n_donor):
            out[c, j] = (pairwise_sum(recipient_out[c], donor_out[j], 0, width, starts, lengths, stages, left_sums)
                         + pairwise_sum(recipient_in[c], donor_in[j], 0, width, starts, lengths, stages, left_sums)) / 2
