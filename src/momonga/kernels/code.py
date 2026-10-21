"""The scalar primitives of the code, shared by every kernel: the log-factorial table and the two per-item terms.

The formulas are those of momonga.objective.  They are written here once and imported by the merge kernels, the
move kernels and the full description length, so every incremental quantity and every total use the same
arithmetic.  Each primitive adds its terms in one fixed order; changing the order changes the last bits, and the
merge decisions compare scores exactly.
"""
import numpy as np
from numba import njit
from scipy.special import gammaln

LOG2 = 0.6931471805599453             # log 2: the one bit saying whether a cluster profile is coded dense or sparse

_table = None                         # the log-factorial table, built on first use and grown on demand


def log_factorial_table(up_to):
    """One table T with T[x] = log x! for every 0 <= x <= up_to, shared by every caller."""
    global _table
    if _table is None or _table.size <= up_to:
        size = max(int(up_to) + 1, 1 << 20, 0 if _table is None else 2 * _table.size)
        _table = gammaln(np.arange(size + 1, dtype=np.float64) + 1.0)
    return _table


def log_binomial(a, b):
    """log C(a, b) for integers a and b; zero where b <= 0 or a < b."""
    aa, bb = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    table = log_factorial_table(int(np.max(aa, initial=0)) + 2)
    out = np.zeros(aa.shape, dtype=np.float64)
    ok = (bb > 0) & (aa >= bb)
    if ok.any():
        ai = np.rint(aa[ok]).astype(np.int64)
        bi = np.rint(bb[ok]).astype(np.int64)
        out[ok] = table[ai] - table[bi] - table[ai - bi]
    return out if out.ndim else float(out)


def count_matrix(A, min_edge_weight=1):
    """A as the code reads it: a square CSR of finite non-negative integer weights, duplicates summed, no stored
    zeros (a stored zero would count as an edge), indices sorted.  Copied only when a change is needed.

    Raises ValueError on anything the code cannot describe: a self-loop, or an edge lighter than `min_edge_weight`
    (GraphData removes those when the graph is made).
    """
    A = A.tocsr()
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"the adjacency must be square, got {A.shape}")
    if not A.has_canonical_format or (A.data == 0).any():
        A = A.copy()
        A.sum_duplicates()
        A.eliminate_zeros()
    weights = A.data
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights != np.rint(weights)).any():
        raise ValueError("the code needs finite non-negative integer edge weights")
    if A.diagonal().any():
        raise ValueError("the code does not handle self-loops")
    if (weights < min_edge_weight).any():
        raise ValueError(f"edges lighter than {min_edge_weight}: filter the graph first "
                         f"(GraphData(..., min_edge_weight={min_edge_weight}))")
    if not A.has_sorted_indices:
        A = A.copy()
        A.sort_indices()
    return A


@njit(cache=True, nogil=True, inline="always")
def clusters_row(total, support, n_clusters, log_factorial):
    """The clusters term of one cluster profile: weight `total` spread over `support` of the n_clusters clusters,
    plus the one bit saying whether the dense or the sparse description was used.  Zero for an empty profile."""
    if total <= 0:
        return 0.0
    if support < 1:
        support = 1
    if support > n_clusters:
        support = n_clusters
    G = log_factorial
    dense = G[total + n_clusters - 1] - G[n_clusters - 1] - G[total]
    sparse = (np.log(np.float64(n_clusters if n_clusters < total else total)) + G[n_clusters] - G[support]
              - G[n_clusters - support] + G[total - 1] - G[support - 1] - G[total - support])
    return LOG2 + (dense if dense < sparse else sparse)


@njit(cache=True, nogil=True)
def clusters_rows(totals, supports, n_clusters, log_factorial):
    """clusters_row of every profile in the arrays `totals`, `supports` (summed by the caller in its own order)."""
    out = np.empty(totals.size)
    for i in range(totals.size):
        out[i] = clusters_row(totals[i], supports[i], n_clusters, log_factorial)
    return out


@njit(cache=True, nogil=True, inline="always")
def edges_entry(partners, weight, n_edges, log_factorial, min_edge_weight):
    """The edges term of one (vertex, cluster) pair: n_edges edges of total weight `weight` into a cluster offering
    `partners` vertices to connect to, every edge weighing at least min_edge_weight.  Zero when there is no edge."""
    if n_edges <= 0:
        return 0.0
    choices = weight // min_edge_weight
    if partners < choices:
        choices = partners
    G = log_factorial
    return (np.log(np.float64(choices)) + G[partners] - G[n_edges] - G[partners - n_edges]
            + G[weight - (min_edge_weight - 1) * n_edges - 1] - G[n_edges - 1] - G[weight - min_edge_weight * n_edges])


@njit(cache=True, nogil=True)
def edges_sum(indptr, indices, edge_counts, weights, partition, size, log_factorial, min_edge_weight):
    """The edges term of one direction over a CSR of (vertex, cluster) edge counts and weights."""
    total = 0.0
    for u in range(indptr.size - 1):
        for p in range(indptr[u], indptr[u + 1]):
            c = indices[p]
            partners = size[c] - (1 if partition[u] == c else 0)
            total += edges_entry(partners, weights[p], edge_counts[p], log_factorial, min_edge_weight)
    return total
