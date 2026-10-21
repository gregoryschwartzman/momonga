# The paper's graphs

`momonga.download_data()` puts the ten graphs under `data/graphs/<organism>/<rest>.{adj.npz,npz}`; nothing
here is committed. Every graph is two files:

- `<rest>.adj.npz` — the directed adjacency: scipy CSR, float32 synapse counts, rows = presynaptic neurons.
- `<rest>.npz` — three arrays, one entry per adjacency row: `id` (int64, the neuron's id in its dataset), `type`
  (text, the expert type; `"?"` if untyped) and `side` (`"left"`, `"right"`, `"center"` or `"unknown"`).

`GraphData.load(name)` reads a graph by name (`fafb_ol_left` is `data/graphs/fafb/ol_left.*`), leaves out the
neurons without any synapse, and exposes the three arrays as `labels`, `ids` and `side`. The adjacency is
unthresholded; the method drops edges below `min_edge_weight` itself.

| graph | neurons | what it is |
|---|---|---|
| `fafb_ol_left`, `fafb_ol_right` | 43,430 / 43,175 | female optic lobe (FAFB), one side each |
| `fafb_visual_left`, `fafb_visual_right` | 47,600 / 47,404 | female visual system: optic lobe plus the connecting neurons |
| `fafb_central` | 41,714 | female central brain, both sides |
| `mcns_ol_left`, `mcns_ol_right` | 44,567 / 44,768 | male optic lobe (MaleCNS) |
| `mcns_visual_left`, `mcns_visual_right` | 51,777 / 53,277 | male visual system |
| `manc_vnc` | 23,641 | male ventral nerve cord (MANC), both sides |

## Where each graph comes from

| graphs | synapses | `type` | which neurons, and their `side` |
|---|---|---|---|
| FAFB (`fafb_*`) | the Codex FlyWire v783 download, `connections_princeton_no_threshold.csv.gz`, summed over neuropils | Codex `visual_neuron_types.type`, else `consolidated_cell_types.primary_type`, else `"?"` | `left_and_right_clusters.csv` of the NTAC v0.1.0 release: its `side` and `compartment` columns only (optic per side; visual = optic + connecting; central = compartment central, all sides) |
| MaleCNS (`mcns_ol_*`, `mcns_visual_*`) | the per-side flywire.ai exports (`*_edges.csv`) | the exports' vertex → type tables (`cluster`) | the export itself, one per side; neurons with at least one edge |
| MANC (`manc_vnc`) | `connections_princeton.csv.gz` of the MANC Codex export, summed over neuropils, self-connections dropped | `Primary Cell Type` (placeholder names such as `SNxxxx` kept), else `"?"` | every neuron of `neurons.csv.gz`; `Soma side` |

Nothing from NTAC is used beyond the FAFB node sets and sides. The raw downloads are public: FAFB at
https://codex.flywire.ai, MaleCNS at https://male-cns.janelia.org and flywire.ai, MANC at
https://www.janelia.org/project-team/flyem/manc-connectome.
