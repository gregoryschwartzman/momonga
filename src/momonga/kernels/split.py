"""The kernel of the split (momonga.refine.refine.split): every cluster cut in two by a two-way Lloyd iteration
under the Kullback-Leibler divergence of its members' profiles.  CPU only.

A member u with profile x_u (synapse counts, total y_u) codes against a centre c at KL(x_u / y_u || c).  The seeds
are the member coding worst against the cluster's pooled profile, then the member coding worst against that one;
each iteration sends a member to half 1 when y_u KL(u || c1) < y_u KL(u || c0), then recomputes both centres as
pooled profiles.  At most MAX_ITERATIONS; no cut when the seeds coincide or a half empties.  Centres and pools are
integer sums, exact in float64; clusters are independent, so nothing depends on the thread count.
"""
import numpy as np
from numba import njit, prange

MAX_ITERATIONS = 50


@njit(cache=True)
def _kl(indptr, indices, values, u, total, centre):
    """KL(profile of u / total || centre)."""
    s = 0.0
    for p in range(indptr[u], indptr[u + 1]):
        x = values[p]
        if x > 0:
            P = x / total
            c = centre[indices[p]]
            if c < 1e-300:
                c = 1e-300
            s += P * np.log(P / c)
    return s


@njit(cache=True)
def _cut_one(indptr, indices, values, members, width, half_out, capped):
    """1 and half_out[:members.size] filled when the cluster is cut, else 0.  capped[0] = 1 when the Lloyd loop
    stopped at the iteration cap instead of converging."""
    m = members.size
    totals = np.zeros(m)
    for i in range(m):
        u = members[i]
        s = 0.0
        for p in range(indptr[u], indptr[u + 1]):
            s += values[p]
        totals[i] = s
    n_nonzero = 0
    for i in range(m):
        if totals[i] > 0:
            n_nonzero += 1
    if n_nonzero < 2:
        return 0
    pool = np.zeros(width)
    pool_total = 0.0
    for i in range(m):
        u = members[i]
        for p in range(indptr[u], indptr[u + 1]):
            pool[indices[p]] += values[p]
            pool_total += values[p]
    for j in range(width):
        pool[j] = pool[j] / (pool_total if pool_total > 1e-12 else 1e-12)
    a = -1
    best = -np.inf
    for i in range(m):
        if totals[i] > 0:
            v = _kl(indptr, indices, values, members[i], totals[i], pool)
            if v > best:
                best = v
                a = i
    centre_a = np.zeros(width)
    ua = members[a]
    for p in range(indptr[ua], indptr[ua + 1]):
        centre_a[indices[p]] = values[p] / totals[a]
    b = -1
    best = -np.inf
    for i in range(m):
        if totals[i] > 0:
            v = _kl(indptr, indices, values, members[i], totals[i], centre_a)
            if v > best:
                best = v
                b = i
    if a == b:
        return 0
    c0 = centre_a
    c1 = np.zeros(width)
    ub = members[b]
    for p in range(indptr[ub], indptr[ub + 1]):
        c1[indices[p]] = values[p] / totals[b]
    half = np.zeros(m, np.int64)
    new = np.zeros(m, np.int64)
    capped[0] = 1                                                # cleared when the loop converges or a half empties
    for it in range(MAX_ITERATIONS):
        for i in range(m):
            if totals[i] > 0:
                kl0 = _kl(indptr, indices, values, members[i], totals[i], c0)
                kl1 = _kl(indptr, indices, values, members[i], totals[i], c1)
                new[i] = 1 if totals[i] * kl1 < totals[i] * kl0 else 0
            else:
                new[i] = 0
        same = True
        for i in range(m):
            if new[i] != half[i]:
                same = False
                break
        if same:
            capped[0] = 0
            break
        for i in range(m):
            half[i] = new[i]
        for s in range(2):
            centre = c0 if s == 0 else c1
            for j in range(width):
                centre[j] = 0.0
            centre_total = 0.0
            count = 0
            for i in range(m):
                if half[i] == s:
                    count += 1
                    u = members[i]
                    for p in range(indptr[u], indptr[u + 1]):
                        centre[indices[p]] += values[p]
                        centre_total += values[p]
            if count == 0:
                capped[0] = 0
                return 0
            for j in range(width):
                centre[j] = centre[j] / (centre_total if centre_total > 1e-12 else 1e-12)
    n1 = 0
    for i in range(m):
        n1 += half[i]
    if n1 == 0 or n1 == m:
        return 0
    for i in range(m):
        half_out[i] = half[i]
    return 1


@njit(cache=True, parallel=True)
def cut_clusters(indptr, indices, values, order, cluster_start, width, half):
    """Cut every cluster with at least two members (members order[cluster_start[c]:cluster_start[c + 1]]) in
    parallel.  Fills `half` (0 or 1 per vertex); returns (cut per cluster, capped per cluster)."""
    n_clusters = cluster_start.size - 1
    cut = np.zeros(n_clusters, np.int64)
    capped = np.zeros(n_clusters, np.int64)
    for c in prange(n_clusters):
        members = order[cluster_start[c]:cluster_start[c + 1]]
        if members.size < 2:
            continue
        member_half = np.zeros(members.size, np.int64)
        flag = np.zeros(1, np.int64)
        if _cut_one(indptr, indices, values, members, width, member_half, flag):
            cut[c] = 1
            for i in range(members.size):
                half[members[i]] = member_half[i]
        capped[c] = flag[0]
    return cut, capped
