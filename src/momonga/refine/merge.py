"""Merge mutual-best pairs of clusters until no merge shortens the description length (`merge`).

Each pass scores the exact change of the description length (merge_state.py) for every pair of live clusters.  It
then merges the pairs that are each other's most negative partner, each rechecked with MergeState.merge_change on
the current state before it is applied.  Passes repeat until one merges nothing.

The first pair of the sorted list is mutual-best even under ties, so a pass that merges nothing while negative
pairs are listed means the pair scorer and the recheck disagree on that pair's sign (float rounding).  That stop
is not certified, the pairs below it never having been rechecked, and a warning says so.

Scoring every pair is made cheap without approximating any score:

  * the terms that depend on the cluster matrix or on the number of clusters are recomputed each pass by
    kernels.merge.score_pairs, organised by row so every co-occurrence is visited once;
  * the edges term is the expensive one.  For a merge of a and b it is at least
    -min(edges_lower_bound[a], edges_lower_bound[b]) (kernels.merge.edges_term_bounds), so a pair gets its exact
    edges term only when that bound leaves a possible negative, and keeps it until one of its two clusters merges.

The negative pairs and their scores are therefore exactly those a full recomputation gives.
"""
from __future__ import annotations

import time
import warnings

import numba
import numpy as np

from ..kernels import merge as K
from ..kernels.code import log_binomial
from ..kernels.merge import ClusterTerms, ColumnScratch, ColumnStore, Csr, RowScratch, VertexStore
from ..utils import callback, relabel
from .merge_state import MergeState

PAIRS_PER_CALL = 4_000_000          # pairs per call of the exact edges-term kernel: bounds memory only


class PairScorer:
    """Every live pair's exact change of the description length for one MergeState, kept across passes.

    Dense (n_ids x n_ids) arrays over the original cluster ids, read at [a, b] with a < b:

        scores                 the change of the description length if a and b merge
        scores_without_edges   the same without the edges term and the two scalars added last
        edges_changes          the exact edges-term changes computed so far
        edges_known            which entries of edges_changes are exact and current
        edges_needed           the pairs whose bounded score may be negative, so need the exact edges term

    and per cluster, edges_lower_bound: a lower bound on the edges-term change of any merge with it.
    """

    def __init__(self, state):
        self.state = state
        n_ids = state.n_ids
        self.scores = np.zeros((n_ids, n_ids))
        self.scores_without_edges = np.zeros((n_ids, n_ids))
        self.edges_changes = np.zeros((n_ids, n_ids))
        self.edges_known = np.zeros((n_ids, n_ids), dtype=np.bool_)
        self.edges_needed = np.zeros((n_ids, n_ids), dtype=np.bool_)
        self.edges_lower_bound = np.zeros(n_ids)
        self._bound_out, self._bound_in = np.zeros(n_ids), np.zeros(n_ids)     # the bound in each direction
        threads = max(1, int(numba.get_num_threads()))
        self._row_scratch = RowScratch(np.zeros((threads, n_ids)), np.zeros((threads, n_ids)),
                                       np.zeros((threads, n_ids)), np.zeros((threads, n_ids), np.int64),
                                       np.zeros((threads, n_ids), np.int64))
        self._column_scratch = ColumnScratch(np.zeros((threads, n_ids), np.int64), np.zeros((threads, n_ids)))
        self._merges_seen = 0              # how many of state.merges the stores and bounds already reflect
        self._bounds_ready = False
        # the column stores, out and in: per cluster, the vertices with edges, their weights and edge counts
        self._out_store = self._in_store = None
        self._cached = None                # the last result of negative_pairs, valid while state.version is unchanged
        self._cached_version = None

    def _update_stores(self, touched):
        """Rebuild the column stores of the clusters that merged since the last pass (all of them the first time)."""
        state = self.state
        if self._out_store is None:
            changed = np.ones(state.n_ids, dtype=bool)
            empty = np.empty(0, np.int64)
            self._out_store = self._in_store = ColumnStore(np.zeros(state.n_ids + 1, np.int64), empty, empty, empty)
        else:
            changed = np.zeros(state.n_ids, dtype=bool)
            changed[touched] = True
        self._out_store = ColumnStore(*K.rebuild_column_store(self._out_store, changed, state.W_out, state.E_out))
        self._in_store = ColumnStore(*K.rebuild_column_store(self._in_store, changed, state.W_in, state.E_in))

    def _update_bounds(self, clusters):
        """edges_lower_bound of `clusters`: the average of the two directions' bounds.

        This is a valid lower bound, since min(x_a + y_a, x_b + y_b) >= min(x_a, x_b) + min(y_a, y_b).
        """
        state = self.state
        for store, bound in ((self._out_store, self._bound_out), (self._in_store, self._bound_in)):
            K.edges_term_bounds(clusters, store, state.partition, state.size, state.min_edge_weight, bound)
        self.edges_lower_bound[:] = 0.5 * self._bound_out + 0.5 * self._bound_in

    def _exact_edges_changes(self, first, second):
        """The exact edges-term change of every pair (first[t], second[t]), both directions averaged, into
        edges_changes."""
        state = self.state
        done = 0
        while done < first.size:
            end = min(first.size, done + PAIRS_PER_CALL)
            change = np.zeros(end - done)
            for store in (self._out_store, self._in_store):
                part = np.zeros(end - done)
                K.edges_term_changes(first[done:end], second[done:end], store, state.size, state.partition,
                                     state.log_factorial, state.min_edge_weight, part)
                change += 0.5 * part
            K.scatter(self.edges_changes, first[done:end], second[done:end], change)
            done = end

    def _score_without_edges(self, live):
        """Fill scores_without_edges for every live pair: every term but the edges term and the two scalars."""
        state = self.state
        G = state.log_factorial
        n_live = live.size
        rows = Csr(*K.live_rows(state.C, live))                      # the live part of the cluster matrix, by rows
        columns = Csr(*K.transposed(rows))                           # and by columns
        out_by_vertex = VertexStore(*K.by_vertex(self._out_store, state.n))
        in_by_vertex = VertexStore(*K.by_vertex(self._in_store, state.n))
        # the cluster matrix coded by rows: every merge's clusters-term change that depends on the pair only
        # through the cluster count, then per live cluster its terms (kernels.merge.ClusterTerms)
        shift = float(K.count_shift(state.row_totals, state.row_support, live, n_live, G))
        terms = ClusterTerms(*K.cluster_terms(state.row_totals, state.row_support, state.C, live, n_live, G))
        K.score_pairs(live, rows, columns, state.C, state.row_totals, state.column_totals, state.row_support, terms,
                      shift, self._out_store, out_by_vertex, self._in_store, in_by_vertex, G, self._row_scratch,
                      self.scores_without_edges)
        # the cluster matrix coded by columns: the same quantities from the transpose
        shift_t = float(K.count_shift(state.column_totals, state.column_support, live, n_live, G))
        terms_t = ClusterTerms(*K.cluster_terms(state.column_totals, state.column_support, state.C_t, live, n_live,
                                                G))
        K.score_clusters_term_by_columns(live, rows, columns, state.C, state.column_totals, state.column_support,
                                         terms.diagonal, terms_t, shift_t, G, self._column_scratch,
                                         self.scores_without_edges)

    def negative_pairs(self):
        """(first, second, change) for every live pair first < second whose merge lowers the description length,
        most negative first."""
        state = self.state
        if self._cached is not None and state.version == self._cached_version:
            return self._cached
        live = np.flatnonzero(state.size > 0).astype(np.int64)
        new_merges = state.merges[self._merges_seen:]
        self._merges_seen = len(state.merges)
        touched = sorted({cluster for merge in new_merges for cluster in merge})
        survivors = np.array([cluster for cluster in touched if state.size[cluster] > 0], dtype=np.int64)
        if live.size < 2:
            empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0))
            self._cached, self._cached_version = empty, state.version
            return empty
        if self._out_store is None or touched:
            self._update_stores(touched)
        if not self._bounds_ready:
            self._update_bounds(live)
            self._bounds_ready = True
        elif survivors.size:
            K.forget_edges_terms(survivors, live, self.edges_known)          # their exact edges terms are stale
            self._update_bounds(survivors)
        self._score_without_edges(live)
        # the change of the partition's cluster-count term, the same for every merge
        partition_change = float(log_binomial(state.n - 1, state.n_clusters - 2)
                                 - log_binomial(state.n - 1, state.n_clusters - 1))
        # scores with the bound in place of every unknown edges term; then the exact term only where the bound
        # leaves a possible negative
        counts = K.assemble_scores(live, self.scores_without_edges, self.edges_changes, self.edges_known,
                                   self.edges_lower_bound, state.size, partition_change, state.log_factorial,
                                   self.scores, self.edges_needed)
        first, second = K.needed_pairs(live, self.edges_needed, counts)
        if first.size:
            self._exact_edges_changes(first, second)
            self.edges_known[first, second] = True
            K.refresh_scores(first, second, self.scores_without_edges, self.edges_changes, state.size,
                             partition_change, state.log_factorial, self.scores, self.edges_needed)
        first, second, change = K.negative_pairs(self.scores, live)
        order = np.argsort(change, kind="stable")                      # ties keep the scan order: by first, then second
        result = (first[order], second[order], change[order])
        self._cached, self._cached_version = result, state.version
        return result


def merge(A, partition, *, min_edge_weight=5, log=None):
    """Merge mutual-best pairs of clusters of `partition` of the graph A until no merge shortens the description
    length.  Returns (partition, description_length).

    log   log(line) gets one line per pass
    """
    log = callback(log)
    state = MergeState(A, partition, min_edge_weight=min_edge_weight)
    scorer = PairScorer(state)
    passes = 0
    stalled = None
    while True:
        started = time.perf_counter()
        first, second, change = scorer.negative_pairs()
        if first.size == 0:
            break
        selected = K.mutual_best(first, second, change, state.n_ids)
        applied = 0
        rejected = 0                                                    # mutual-best pairs the recheck turned down
        for index in selected:
            a, b = int(first[index]), int(second[index])
            if state.size[a] == 0 or state.size[b] == 0:
                continue
            if state.merge_change(a, b) < 0:                            # rechecked at the current state
                state.merge(a, b)
                applied += 1
            else:
                rejected += 1
        passes += 1
        log(f"    merge pass {passes}: {first.size} negative pairs, {applied} mutual-best merged, {rejected} "
            f"rejected; {state.n_clusters} clusters  ({time.perf_counter() - started:.1f}s)")
        if applied == 0:
            a, b = int(first[0]), int(second[0])
            stalled = (f"merge stopped at {state.n_clusters} clusters without a certified fixed point: the pair "
                       f"scorer lists {first.size} negative pairs and puts the most negative, ({a}, {b}), at "
                       f"{change[0]:.6g}, but the recheck puts it at {state.merge_change(a, b):.6g}; the other "
                       f"listed pairs were not rechecked")
            break
    if stalled:
        log(f"    warning: {stalled}")
        warnings.warn(stalled, RuntimeWarning, stacklevel=2)
    return relabel(state.partition), state.description_length()
