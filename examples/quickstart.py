"""MOMONGA on a synthetic graph: NTAC, the refinement, and the scores.  Runs in seconds, needs no data.

    python examples/quickstart.py
"""
import numpy as np
from scipy import sparse

from momonga import GraphData, Momonga


def block_model(n=1500, n_types=6, density=0.3, seed=0):
    """A directed stochastic block model: n_types equal blocks, an edge from block i to block j with its own
    probability, integer weights 5..12 so that nothing is cut by the weight floor.  Every block gets its own
    connection profile, which is what makes the types recoverable from connectivity."""
    rng = np.random.default_rng(seed)
    blocks = np.arange(n) % n_types
    edge_probability = rng.random((n_types, n_types)) * density
    edges = rng.random((n, n)) < edge_probability[blocks[:, None], blocks[None, :]]
    np.fill_diagonal(edges, False)
    adjacency = sparse.csr_matrix(edges.astype(np.float32))
    adjacency.data = rng.integers(5, 13, size=adjacency.nnz).astype(np.float32)
    return GraphData(adjacency, labels=np.array([f"T{block}" for block in blocks]))


def report(name, scores):
    print(f"{name}: {scores['n_clusters']} clusters, accuracy {scores['acc']:.3f}, macro-F1 {scores['f1_macro']:.3f}")


data = block_model()
model = Momonga(data.without_labels(), backend="cpu")       # the method is given a graph without the types
model.unseeded_ntac(12)                                    # NTAC up to twice the true number of types
model.refine(grid=(6, 12))                                 # refine two NTAC starts; the shorter result is kept
report("NTAC   ", data.get_metrics(model.ntac_partition()))     # NTAC's own answer: its lowest-cost partition
report("MOMONGA", data.get_metrics(model.partition))
print(f"description length {model.description_length:,.0f} nats, from the NTAC start with "
      f"{model.start_n_clusters} clusters")
print("confidence quartiles:", np.round(np.percentile(data.confidence(model.partition), [25, 50, 75]), 3))
