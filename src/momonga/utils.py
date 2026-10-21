"""Small helpers shared across the package.

    relabel               dense integer ids 0, 1, ... for any label array; equal values collapse to one id, so no
                          id is unused afterwards
    max_weight_matching   the one bipartite matching of the package, used by the scoring and by the transfer
    quiet, callback       a callback that does nothing, and the default it provides for every `log` and `on_stage`
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def quiet(*_args):
    """A callback that does nothing: the default for every `log` and `on_stage`."""


def callback(function):
    """`function`, or `quiet` when it is None, so callers can call it without checking."""
    return quiet if function is None else function


def max_weight_matching(weights):
    """(rows, cols) of the maximum-weight bipartite matching of a weight matrix."""
    return linear_sum_assignment(weights, maximize=True)


def relabel(labels):
    """Dense int32 ids 0, 1, ... for any label array, in sorted order of the original values."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int32, copy=False)
