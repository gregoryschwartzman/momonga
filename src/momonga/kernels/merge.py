"""The merge kernels: the merge state's quantities (momonga.refine.merge_state) and the exact score of every
candidate merge (momonga.refine.merge).  CPU only.

The formulas are momonga.objective's; the two scalar primitives come from kernels.code.  Arrays that travel together
are passed as one named tuple:

    Csr            a sparse matrix by rows (indptr, indices, values), or by columns
    ColumnStore    per cluster: the vertices with an edge into it, their weights and edge counts
    VertexStore    per vertex: the clusters it has an edge into, and the weights
    ClusterTerms   per live cluster: its clusters term now, with one cluster fewer, the change when its support drops
                   by one, and its diagonal entry
    RowScratch, ColumnScratch   per-thread scratch rows of the two scoring kernels

Every parallel kernel but one writes each output from one thread.  The exception is `count_shift`, a parallel sum:
its last bits depend on the thread count, and through it, at an exact tie, a merge decision can.  Kernels that
compute the same quantity by different routes keep their own summation orders on purpose.
"""
from collections import namedtuple

import numpy as np
from numba import njit, prange

from .code import clusters_row, edges_entry

Csr = namedtuple("Csr", ["indptr", "indices", "values"])
ColumnStore = namedtuple("ColumnStore", ["indptr", "vertices", "weights", "edges"])
VertexStore = namedtuple("VertexStore", ["indptr", "clusters", "weights"])
ClusterTerms = namedtuple("ClusterTerms", ["now", "after", "support_drop", "diagonal"])
RowScratch = namedtuple("RowScratch", ["column_folds", "row_folds", "scores", "co_occurrences", "column_a"])
ColumnScratch = namedtuple("ColumnScratch", ["co_occurrences", "scores"])


# -- the clusters term --

@njit(cache=True, nogil=True, parallel=True)
def count_shift(totals, support, live, n_live, log_factorial):
    """The part of every merge's change of the clusters term that depends on the pair only through the cluster
    count: the sum over live clusters of their term at n_live - 1 clusters minus their term at n_live.  A parallel
    reduction: its last bits depend on the thread count."""
    shift = 0.0
    for i in prange(live.shape[0]):
        c = live[i]
        shift += (clusters_row(totals[c], support[c], n_live - 1, log_factorial)
                  - clusters_row(totals[c], support[c], n_live, log_factorial))
    return shift


@njit(cache=True, nogil=True, parallel=True)
def cluster_terms(totals, support, C, live, n_live, log_factorial):
    """Per live cluster: the clusters term at n_live and at n_live - 1 clusters, the change when its support drops
    by one, and its diagonal entry C[c, c] (a ClusterTerms, as a tuple)."""
    n_ids = totals.shape[0]
    now = np.zeros(n_ids)
    after = np.zeros(n_ids)
    support_drop = np.zeros(n_ids)
    diagonal = np.zeros(n_ids, np.int64)
    for i in prange(live.shape[0]):
        c = live[i]
        now[c] = clusters_row(totals[c], support[c], n_live, log_factorial)
        after[c] = clusters_row(totals[c], support[c], n_live - 1, log_factorial)
        support_drop[c] = (clusters_row(totals[c], support[c] - 1, n_live - 1, log_factorial)
                           - clusters_row(totals[c], support[c], n_live - 1, log_factorial))
        diagonal[c] = C[c, c]
    return now, after, support_drop, diagonal


# -- the edges term of one cluster, of one merge, and of every candidate pair --

@njit(cache=True)
def edges_term_of_cluster(c, vertices, E, W, size, partition, log_factorial, min_edge_weight):
    """The edges term of cluster c, over `vertices`, the vertices with edges into (or from) it."""
    total = 0.0
    for u in vertices:
        partners = size[c] - int(partition[u] == c)
        total += edges_entry(partners, W[c, u], E[c, u], log_factorial, min_edge_weight)
    return total


@njit(cache=True)
def edges_term_change(a, c, vertices, E, W, size, partition, log_factorial, min_edge_weight):
    """The change of the edges term when clusters a and c merge, over `vertices`, those with edges into either."""
    total = 0.0
    for u in vertices:
        partners_a = size[a] - int(partition[u] == a)
        partners_c = size[c] - int(partition[u] == c)
        edges_a, edges_c = E[a, u], E[c, u]
        weight_a, weight_c = W[a, u], W[c, u]
        total += (edges_entry(partners_a + partners_c, weight_a + weight_c, edges_a + edges_c, log_factorial,
                              min_edge_weight)
                  - edges_entry(partners_a, weight_a, edges_a, log_factorial, min_edge_weight)
                  - edges_entry(partners_c, weight_c, edges_c, log_factorial, min_edge_weight))
    return total


@njit(parallel=True, cache=True)
def edges_term_changes(first, second, store, size, partition, log_factorial, min_edge_weight, out):
    """The change of the edges term for every pair (first[t], second[t]), from the ColumnStore: the two columns
    are walked together, vertex by vertex."""
    n = partition.size
    for t in prange(first.size):
        a, c = first[t], second[t]
        p_a, p_c = store.indptr[a], store.indptr[c]
        end_a, end_c = store.indptr[a + 1], store.indptr[c + 1]
        total = 0.0
        while p_a < end_a or p_c < end_c:
            u = min(store.vertices[p_a] if p_a < end_a else n, store.vertices[p_c] if p_c < end_c else n)
            weight_a = weight_c = edges_a = edges_c = 0
            if p_a < end_a and store.vertices[p_a] == u:
                weight_a = store.weights[p_a]
                edges_a = store.edges[p_a]
                p_a += 1
            if p_c < end_c and store.vertices[p_c] == u:
                weight_c = store.weights[p_c]
                edges_c = store.edges[p_c]
                p_c += 1
            partners_a = size[a] - int(partition[u] == a)
            partners_c = size[c] - int(partition[u] == c)
            total += (edges_entry(partners_a + partners_c, weight_a + weight_c, edges_a + edges_c, log_factorial,
                                  min_edge_weight)
                      - edges_entry(partners_a, weight_a, edges_a, log_factorial, min_edge_weight)
                      - edges_entry(partners_c, weight_c, edges_c, log_factorial, min_edge_weight))
        out[t] = total


@njit(parallel=True, cache=True)
def edges_term_bounds(clusters, store, partition, size, min_edge_weight, out):
    """out[c] = sum over the vertices u with edges into c of log min(partners, weight // min_edge_weight).  A merge
    of a and c changes the edges term by at least -min(out[a], out[c]), so pairs whose bounded score cannot be
    negative skip the exact term."""
    for t in prange(clusters.size):
        c = clusters[t]
        bound = 0.0
        for p in range(store.indptr[c], store.indptr[c + 1]):
            u = store.vertices[p]
            partners = size[c] - int(partition[u] == c)
            bound += np.log(min(partners, store.weights[p] // min_edge_weight))
        out[c] = bound


# -- the orderings term of one merge --

@njit(cache=True, nogil=True)
def sorted_union(a, b, out):
    """Merge two sorted index arrays into out; returns the length written."""
    i = j = n = 0
    while i < a.size and j < b.size:
        if a[i] < b[j]:
            out[n] = a[i]
            i += 1
        elif a[i] > b[j]:
            out[n] = b[j]
            j += 1
        else:
            out[n] = a[i]
            i += 1
            j += 1
        n += 1
    while i < a.size:
        out[n] = a[i]
        i += 1
        n += 1
    while j < b.size:
        out[n] = b[j]
        j += 1
        n += 1
    return n


@njit(cache=True, nogil=True)
def orderings_term(W, vertices, kept, removed, log_factorial):
    """The sum over `vertices` of log C(w_kept + w_removed, w_kept): how a vertex's weight into the merged cluster
    splits between the two old ones (W stored cluster x vertex)."""
    term = 0.0
    for i in range(vertices.size):
        v = vertices[i]
        w_kept, w_removed = W[kept, v], W[removed, v]
        term += log_factorial[w_kept + w_removed] - log_factorial[w_kept] - log_factorial[w_removed]
    return term


# -- one merge's change of the clusters and profiles terms, by rows (the recheck) --

@njit(cache=True, fastmath=False)
def fold_terms(C, C_t, live, kept, removed, support, totals, log_factorial):
    """For the merge of `removed` into `kept`, per other live cluster u: the fold of its two column entries
    (column_folds), of its two row entries (row_folds), and its clusters term before and after the merge; plus the
    merged row's support."""
    G = log_factorial
    n_live = live.size
    column_folds = np.empty(n_live - 2)
    row_folds = np.empty(n_live - 2)
    before = np.empty(n_live - 2)
    after = np.empty(n_live - 2)
    q = 0
    merged_support = 0
    for u in live:
        if u == removed:
            continue
        merged = C[kept, u] + C[removed, u]
        if u == kept:
            merged += C[kept, removed] + C[removed, removed]
        merged_support += int(merged > 0)
        if u == kept:
            continue
        # the maintained transpose turns strided column reads into contiguous row reads: the same integer counts
        t_kept, t_removed = C_t[kept, u], C_t[removed, u]
        c_kept, c_removed = C[kept, u], C[removed, u]
        column_folds[q] = G[t_kept + t_removed] - G[t_kept] - G[t_removed]
        row_folds[q] = G[c_kept + c_removed] - G[c_kept] - G[c_removed]
        support_u = support[u]
        before[q] = clusters_row(totals[u], support_u, n_live, G)
        after[q] = clusters_row(totals[u], support_u - int(t_kept > 0 and t_removed > 0), n_live - 1, G)
        q += 1
    return column_folds, row_folds, before, after, merged_support


# -- the sparse views the pair scoring walks --

@njit(cache=True, nogil=True, inline="always")
def _first_above(indptr, indices, row, a):
    """The first position p of `row` with indices[p] > a (rows sorted ascending)."""
    low = indptr[row]
    high = indptr[row + 1]
    while low < high:
        middle = (low + high) >> 1
        if indices[middle] <= a:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True, nogil=True, parallel=True)
def live_rows(C, live):
    """C restricted to the live rows and columns, as a Csr (tuple) indexed by original id (dead rows empty)."""
    n_ids = C.shape[0]
    n_live = live.shape[0]
    counts = np.zeros(n_ids + 1, np.int64)
    for i in prange(n_live):
        r = live[i]
        count = 0
        for j in range(n_live):
            if C[r, live[j]] != 0:
                count += 1
        counts[r + 1] = count
    indptr = np.empty(n_ids + 1, np.int64)
    indptr[0] = 0
    for r in range(n_ids):
        indptr[r + 1] = indptr[r] + counts[r + 1]
    nnz = indptr[n_ids]
    indices = np.empty(nnz, np.int64)
    values = np.empty(nnz, np.int64)
    for i in prange(n_live):
        r = live[i]
        p = indptr[r]
        for j in range(n_live):
            value = C[r, live[j]]
            if value != 0:
                indices[p] = live[j]
                values[p] = value
                p += 1
    return indptr, indices, values


@njit(cache=True, nogil=True)
def transposed(csr):
    """The Csr by columns (a tuple): rows ascending within each column, by counting sort."""
    indptr, indices, values = csr.indptr, csr.indices, csr.values
    n_ids = indptr.shape[0] - 1
    nnz = indptr[n_ids]
    t_indptr = np.zeros(n_ids + 1, np.int64)
    for p in range(nnz):
        t_indptr[indices[p] + 1] += 1
    for c in range(n_ids):
        t_indptr[c + 1] += t_indptr[c]
    position = t_indptr[:n_ids].copy()
    t_indices = np.empty(nnz, np.int64)
    t_values = np.empty(nnz, np.int64)
    for r in range(n_ids):
        for p in range(indptr[r], indptr[r + 1]):
            c = indices[p]
            q = position[c]
            t_indices[q] = r
            t_values[q] = values[p]
            position[c] = q + 1
    return t_indptr, t_indices, t_values


@njit(cache=True, nogil=True, parallel=True)
def rebuild_column_store(store, changed, W, E):
    """The ColumnStore (as a tuple) with the clusters marked in `changed` rebuilt from the rows of W and E (cluster x
    vertex) and every other cluster's segment copied from `store`.  With every cluster changed and an empty store it
    is the full build."""
    n_ids = store.indptr.shape[0] - 1
    n = W.shape[1]
    lengths = np.empty(n_ids, np.int64)
    for c in prange(n_ids):
        if changed[c]:
            count = 0
            for v in range(n):
                if W[c, v] != 0:
                    count += 1
            lengths[c] = count
        else:
            lengths[c] = store.indptr[c + 1] - store.indptr[c]
    indptr = np.empty(n_ids + 1, np.int64)
    indptr[0] = 0
    for c in range(n_ids):
        indptr[c + 1] = indptr[c] + lengths[c]
    total = indptr[n_ids]
    vertices = np.empty(total, np.int64)
    weights = np.empty(total, np.int64)
    edges = np.empty(total, np.int64)
    for c in prange(n_ids):
        q = indptr[c]
        if changed[c]:
            for v in range(n):
                weight = W[c, v]
                if weight != 0:
                    vertices[q] = v
                    weights[q] = weight
                    edges[q] = E[c, v]
                    q += 1
        else:
            old_start = store.indptr[c]
            for p in range(store.indptr[c + 1] - old_start):
                vertices[q + p] = store.vertices[old_start + p]
                weights[q + p] = store.weights[old_start + p]
                edges[q + p] = store.edges[old_start + p]
    return indptr, vertices, weights, edges


@njit(cache=True, nogil=True)
def by_vertex(store, n):
    """The ColumnStore transposed, as a VertexStore (tuple): for each vertex, (cluster, weight), clusters ascending."""
    n_ids = store.indptr.shape[0] - 1
    nnz = store.indptr[n_ids]
    indptr = np.zeros(n + 1, np.int64)
    for p in range(nnz):
        indptr[store.vertices[p] + 1] += 1
    for v in range(n):
        indptr[v + 1] += indptr[v]
    position = indptr[:n].copy()
    clusters = np.empty(nnz, np.int64)
    weights = np.empty(nnz, np.int64)
    for c in range(n_ids):
        for p in range(store.indptr[c], store.indptr[c + 1]):
            v = store.vertices[p]
            q = position[v]
            clusters[q] = c
            weights[q] = store.weights[p]
            position[v] = q + 1
    return indptr, clusters, weights


# -- every live pair's exact change, organised by owner row --

@njit(cache=True, nogil=True, parallel=True)
def score_pairs(live, rows, columns, C, row_totals, column_totals, row_support, terms, shift, out_store,
                out_by_vertex, in_store, in_by_vertex, log_factorial, scratch, scores_without_edges):
    """scores_without_edges[a, b] = every term of the exact change of the description length for merging a and b,
    except the edges term, the partition's size term and the cluster-count term (`assemble_scores` adds those),
    and except the column-coded half of the clusters term (`score_clusters_term_by_columns`), for every live a < b.
    Work is organised by owner row a, so every co-occurrence is visited once.  The order of the additions, per pair:
        out direction   masses; + (column folds + row folds); - corner; + clusters_row(merged row); - clusters_row(a)
                        - clusters_row(b); + shift; - [after(a) - now(a)]; - [same for b];
                        + support_drop[u] for rows u in both columns
        in direction    masses (column totals); + (row folds + column folds); - corner with C[a, b] and C[b, a]
                        swapped
        orderings       + per vertex in both out-columns, ascending; then the same over in-columns
    A one-sided entry contributes G[x] - G[x] - G[0] = 0.0 exactly, so walking only intersections is exact."""
    G = log_factorial
    column_folds, row_folds, scores = scratch.column_folds, scratch.row_folds, scratch.scores
    co_occurrences, column_a = scratch.co_occurrences, scratch.column_a
    n_live = live.shape[0]
    n_after = n_live - 1
    n_threads = column_folds.shape[0]
    for thread in prange(n_threads):
        for i in range(thread, n_live - 1, n_threads):
            a = live[i]
            for j in range(i + 1, n_live):
                b = live[j]
                column_folds[thread, b] = 0.0
                row_folds[thread, b] = 0.0
                co_occurrences[thread, b] = 0
            # (1) column folds: rows u in both columns a and b, u not in {a, b}, ascending u
            for p in range(columns.indptr[a], columns.indptr[a + 1]):
                u = columns.indices[p]
                x = columns.values[p]
                column_a[thread, u] = x
                if u == a:
                    continue
                for q in range(_first_above(rows.indptr, rows.indices, u, a), rows.indptr[u + 1]):
                    b = rows.indices[q]
                    if b == u:
                        continue
                    y = rows.values[q]
                    column_folds[thread, b] -= G[x + y] - G[x] - G[y]
            # (2) row folds and the co-occurrence count: columns c in both rows a and b
            for p in range(rows.indptr[a], rows.indptr[a + 1]):
                c = rows.indices[p]
                if c == a:
                    continue
                x = rows.values[p]
                for q in range(_first_above(columns.indptr, columns.indices, c, a), columns.indptr[c + 1]):
                    b = columns.indices[q]
                    if b == c:
                        continue
                    y = columns.values[q]
                    row_folds[thread, b] -= G[x + y] - G[x] - G[y]
                    co_occurrences[thread, b] += 1
            # (3) the out direction: masses, folds, corner, the clusters term
            c_aa = terms.diagonal[a]
            total_a = row_totals[a]
            for j in range(i + 1, n_live):
                b = live[j]
                c_ab = C[a, b]
                c_ba = column_a[thread, b]
                c_bb = terms.diagonal[b]
                total_b = row_totals[b]
                score = 0.0
                score += 0.5 * (G[total_a + total_b] - G[total_a] - G[total_b])
                score += 0.5 * (column_folds[thread, b] + row_folds[thread, b])
                corner = c_aa + c_ab + c_ba + c_bb
                score -= 0.5 * (G[corner] - G[c_aa] - G[c_ab] - G[c_ba] - G[c_bb])
                support_without_corner = ((row_support[a] - (1 if c_aa > 0 else 0) - (1 if c_ab > 0 else 0))
                                          + (row_support[b] - (1 if c_ba > 0 else 0) - (1 if c_bb > 0 else 0))
                                          - co_occurrences[thread, b])
                merged_support = support_without_corner + (1 if corner > 0 else 0)
                score += 0.5 * clusters_row(total_a + total_b, merged_support, n_after, G)
                score -= 0.5 * (terms.now[a] + terms.now[b])
                score += 0.5 * shift
                score -= 0.5 * (terms.after[a] - terms.now[a])
                score -= 0.5 * (terms.after[b] - terms.now[b])
                scores[thread, b] = score
            # (4) every other row's support drops where both columns are non-zero, ascending u
            for p in range(columns.indptr[a], columns.indptr[a + 1]):
                u = columns.indices[p]
                if u == a:
                    continue
                drop_u = terms.support_drop[u]
                for q in range(_first_above(rows.indptr, rows.indices, u, a), rows.indptr[u + 1]):
                    b = rows.indices[q]
                    if b == u:
                        continue
                    scores[thread, b] += 0.5 * drop_u
            # (5) the in direction, the transpose: masses, the same folds, the corner with c_ab and c_ba swapped
            column_total_a = column_totals[a]
            for j in range(i + 1, n_live):
                b = live[j]
                c_ab = C[a, b]
                c_ba = column_a[thread, b]
                c_bb = terms.diagonal[b]
                column_total_b = column_totals[b]
                corner = c_aa + c_ab + c_ba + c_bb
                score = scores[thread, b]
                score += 0.5 * (G[column_total_a + column_total_b] - G[column_total_a] - G[column_total_b])
                score += 0.5 * (row_folds[thread, b] + column_folds[thread, b])
                score -= 0.5 * (G[corner] - G[c_aa] - G[c_ba] - G[c_ab] - G[c_bb])
                scores[thread, b] = score
            # (6) out-orderings: vertices in both out-columns, ascending
            for p in range(out_store.indptr[a], out_store.indptr[a + 1]):
                v = out_store.vertices[p]
                x = out_store.weights[p]
                for q in range(_first_above(out_by_vertex.indptr, out_by_vertex.clusters, v, a),
                               out_by_vertex.indptr[v + 1]):
                    b = out_by_vertex.clusters[q]
                    y = out_by_vertex.weights[q]
                    scores[thread, b] += 0.5 * (G[x + y] - G[x] - G[y])
            # (7) in-orderings
            for p in range(in_store.indptr[a], in_store.indptr[a + 1]):
                v = in_store.vertices[p]
                x = in_store.weights[p]
                for q in range(_first_above(in_by_vertex.indptr, in_by_vertex.clusters, v, a),
                               in_by_vertex.indptr[v + 1]):
                    b = in_by_vertex.clusters[q]
                    y = in_by_vertex.weights[q]
                    scores[thread, b] += 0.5 * (G[x + y] - G[x] - G[y])
            for j in range(i + 1, n_live):
                b = live[j]
                scores_without_edges[a, b] = scores[thread, b]
            for p in range(columns.indptr[a], columns.indptr[a + 1]):
                column_a[thread, columns.indices[p]] = 0


@njit(cache=True, nogil=True, parallel=True)
def score_clusters_term_by_columns(live, rows, columns, C, column_totals, column_support, diagonal, terms, shift,
                                   log_factorial, scratch, scores_without_edges):
    """scores_without_edges[a, b] += 0.5 * (the change of the column-coded clusters term when a and b merge), for
    every live a < b: the mirror of the clusters-term part of score_pairs with rows and columns exchanged.  `terms`
    and `shift` are those of the transposed cluster matrix."""
    G = log_factorial
    co_occurrences, scores = scratch.co_occurrences, scratch.scores
    n_live = live.shape[0]
    n_after = n_live - 1
    n_threads = co_occurrences.shape[0]
    for thread in prange(n_threads):
        for i in range(thread, n_live - 1, n_threads):
            a = live[i]
            for j in range(i + 1, n_live):
                b = live[j]
                co_occurrences[thread, b] = 0
                scores[thread, b] = 0.0
            # rows u in both columns a and b (u not a): the merged column's support loses one for each
            for p in range(columns.indptr[a], columns.indptr[a + 1]):
                u = columns.indices[p]
                if u == a:
                    continue
                for q in range(_first_above(rows.indptr, rows.indices, u, a), rows.indptr[u + 1]):
                    b = rows.indices[q]
                    if b == u:
                        continue
                    co_occurrences[thread, b] += 1
            # columns c in both rows a and b (c not a): column c's support drops by one
            for p in range(rows.indptr[a], rows.indptr[a + 1]):
                c = rows.indices[p]
                if c == a:
                    continue
                drop_c = terms.support_drop[c]
                for q in range(_first_above(columns.indptr, columns.indices, c, a), columns.indptr[c + 1]):
                    b = columns.indices[q]
                    if b == c:
                        continue
                    scores[thread, b] += drop_c
            c_aa = diagonal[a]
            column_total_a = column_totals[a]
            for j in range(i + 1, n_live):
                b = live[j]
                c_ab = C[a, b]
                c_ba = C[b, a]
                c_bb = diagonal[b]
                column_total_b = column_totals[b]
                corner = c_aa + c_ab + c_ba + c_bb
                support_without_corner = ((column_support[a] - (1 if c_aa > 0 else 0) - (1 if c_ba > 0 else 0))
                                          + (column_support[b] - (1 if c_ab > 0 else 0) - (1 if c_bb > 0 else 0))
                                          - co_occurrences[thread, b])
                merged_support = support_without_corner + (1 if corner > 0 else 0)
                score = scores[thread, b]
                score += clusters_row(column_total_a + column_total_b, merged_support, n_after, G)
                score -= terms.now[a] + terms.now[b]
                score += shift
                score -= terms.after[a] - terms.now[a]
                score -= terms.after[b] - terms.now[b]
                scores_without_edges[a, b] += 0.5 * score


# -- assembling the pair scores, and picking the mutual-best pairs --

@njit(cache=True, parallel=True)
def assemble_scores(live, scores_without_edges, edges_changes, edges_known, edges_lower_bound, size,
                    partition_change, log_factorial, scores, edges_needed):
    """scores = ((scores_without_edges + edges term) + size term) + partition_change for every live pair a < b,
    with the edges term exact where known and the bound -min(edges_lower_bound[a], edges_lower_bound[b])
    elsewhere; edges_needed[a, b] marks the pairs without an exact edges term whose bounded score is below a float
    margin of zero.  Returns, per owner row, the number of pairs marked."""
    G = log_factorial
    n_live = live.shape[0]
    counts = np.zeros(n_live, np.int64)
    for i in prange(n_live - 1):
        a = live[i]
        size_a = size[a]
        count = 0
        for j in range(i + 1, n_live):
            b = live[j]
            size_b = size[b]
            if edges_known[a, b]:
                edges_change = edges_changes[a, b]
            else:
                edges_change = -(edges_lower_bound[a] if edges_lower_bound[a] < edges_lower_bound[b]
                                 else edges_lower_bound[b])
            score = scores_without_edges[a, b]
            score = score + edges_change
            score = score + (-(G[size_a + size_b] - G[size_a] - G[size_b]))
            score = score + partition_change
            scores[a, b] = score
            needed = (not edges_known[a, b]) and score < 1e-9 * (abs(scores_without_edges[a, b]) + abs(edges_change)
                                                                 + 1.0)
            edges_needed[a, b] = needed
            if needed:
                count += 1
        counts[i] = count
    return counts


@njit(cache=True, parallel=True)
def needed_pairs(live, edges_needed, counts):
    """The pairs marked in edges_needed, row-major, given the per-row counts of assemble_scores."""
    n_live = live.shape[0]
    offset = np.zeros(n_live + 1, np.int64)
    for i in range(n_live):
        offset[i + 1] = offset[i] + counts[i]
    first = np.empty(offset[n_live], np.int64)
    second = np.empty(offset[n_live], np.int64)
    for i in prange(n_live - 1):
        a = live[i]
        q = offset[i]
        if counts[i] == 0:
            continue
        for j in range(i + 1, n_live):
            b = live[j]
            if edges_needed[a, b]:
                first[q] = a
                second[q] = b
                q += 1
    return first, second


@njit(cache=True, parallel=True)
def refresh_scores(first, second, scores_without_edges, edges_changes, size, partition_change, log_factorial,
                   scores, edges_needed):
    """scores for the pairs whose exact edges term just arrived, in assemble_scores's order of additions."""
    G = log_factorial
    for t in prange(first.size):
        a = first[t]
        b = second[t]
        size_a = size[a]
        size_b = size[b]
        score = scores_without_edges[a, b]
        score = score + edges_changes[a, b]
        score = score + (-(G[size_a + size_b] - G[size_a] - G[size_b]))
        score = score + partition_change
        scores[a, b] = score
        edges_needed[a, b] = False


@njit(cache=True, parallel=True)
def forget_edges_terms(clusters, live, edges_known):
    """Drop the exact edges term of every pair touching one of `clusters` (the clusters that merged)."""
    for t in prange(clusters.shape[0]):
        c = clusters[t]
        for j in range(live.shape[0]):
            other = live[j]
            edges_known[c, other] = False
            edges_known[other, c] = False


@njit(cache=True, nogil=True, parallel=True)
def negative_pairs(scores, live):
    """(first, second, score) for every live pair with a negative score, row-major."""
    n_live = live.shape[0]
    counts = np.zeros(n_live, np.int64)
    for i in prange(n_live - 1):
        a = live[i]
        count = 0
        for j in range(i + 1, n_live):
            if scores[a, live[j]] < 0:
                count += 1
        counts[i] = count
    offset = np.zeros(n_live + 1, np.int64)
    for i in range(n_live):
        offset[i + 1] = offset[i] + counts[i]
    total = offset[n_live]
    first = np.empty(total, np.int64)
    second = np.empty(total, np.int64)
    score = np.empty(total, np.float64)
    for i in prange(n_live - 1):
        a = live[i]
        q = offset[i]
        for j in range(i + 1, n_live):
            b = live[j]
            value = scores[a, b]
            if value < 0:
                first[q] = a
                second[q] = b
                score[q] = value
                q += 1
    return first, second, score


@njit(cache=True, nogil=True, parallel=True)
def scatter(M, first, second, values):
    """M[first[t], second[t]] = values[t] for every t."""
    for t in prange(first.shape[0]):
        M[first[t], second[t]] = values[t]


@njit(cache=True)
def mutual_best(first, second, score, n_ids):
    """The indices of the pairs that are each other's best partner (the lowest score; the first listed wins a tie),
    in the order listed."""
    best_partner = np.full(n_ids, -1, np.int64)
    best_score = np.full(n_ids, np.inf)
    for t in range(first.size):
        a = first[t]
        b = second[t]
        value = score[t]
        if value < best_score[a]:
            best_score[a] = value
            best_partner[a] = b
        if value < best_score[b]:
            best_score[b] = value
            best_partner[b] = a
    selected = np.empty(first.size, np.int64)
    count = 0
    for t in range(first.size):
        a = first[t]
        b = second[t]
        if best_partner[a] == b and best_partner[b] == a:
            selected[count] = t
            count += 1
    return selected[:count]


# -- building the dense state --

@njit(parallel=True, cache=True)
def scatter_csr(indptr, indices, data, out):
    """out[r, c] = X[r, c] for the CSR matrix X."""
    for r in prange(indptr.size - 1):
        for p in range(indptr[r], indptr[r + 1]):
            out[r, indices[p]] = data[p]


@njit(parallel=True, cache=True)
def row_supports(C):
    """The number of non-zero entries in every row of C."""
    out = np.empty(C.shape[0], np.int64)
    for r in prange(C.shape[0]):
        count = 0
        for c in range(C.shape[1]):
            count += int(C[r, c] > 0)
        out[r] = count
    return out
