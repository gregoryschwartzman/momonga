"""Where the NTAC steps run, and what a run shares between steps.

Backends are different machines, not synonyms:

    "cpu"    numba kernels    anywhere
    "metal"  MLX              Apple silicon
    "cuda"   numba.cuda       an NVIDIA machine

`select()` picks cuda > metal > cpu; `Context.backend` says which one a run is using.
"""
from __future__ import annotations

BACKENDS = ("cpu", "metal", "cuda")


def select():
    """cuda > metal > cpu, by what is usable on this machine.

    If this returns "cpu" on a machine expected to have CUDA, `momonga.ntac.cuda.why_unavailable()` says why: a
    broken install and an absent one are both caught here and look identical.
    """
    try:
        from .cuda import available

        if available():
            return "cuda"
    except Exception:
        pass
    try:
        import mlx.core as mx

        mx.default_device()      # mlx can import with no usable device; ask for one
        return "metal"
    except Exception:
        return "cpu"


def resolve(backend):
    """A validated backend name; "auto" becomes select()."""
    if backend == "auto":
        backend = select()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS} or 'auto', got {backend!r}")
    return backend


class Context:
    """What every step of one run shares: the graph (a GraphData, which builds the profiles) and the kernels.

    On backend "cpu" every step is bit-identical to the reference solver.  On an accelerator the distances are
    bit-identical, but the seed-selection gains are reduced through float32 block partials, so the argmax can differ
    from the CPU at a near-tie.
    """

    def __init__(self, data, backend="auto"):
        self.data = data
        self.backend = resolve(backend)
        self.n = data.n

    def profiles(self, partition, n_clusters):
        """The (n, 2 n_clusters) CSR profile of every vertex against the partition (GraphData.profiles)."""
        return self.data.profiles(partition, n_clusters)

    def accelerator(self, vertex_profiles):
        """The profiles uploaded to the accelerator, with its distance kernels (a fresh upload per call)."""
        if self.backend == "cuda":
            from .cuda import CudaProfiles

            return CudaProfiles(vertex_profiles)
        if self.backend == "metal":
            from .metal import MetalProfiles

            return MetalProfiles(vertex_profiles)
        raise ValueError(f"no accelerator for backend {self.backend!r}")

    @property
    def accelerated(self):
        return self.backend != "cpu"
