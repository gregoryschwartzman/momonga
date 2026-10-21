"""The two female optic lobes (FAFB), from scratch: NTAC and MOMONGA on each lobe, then a transfer between them.
Minutes on a laptop; downloads the graphs on first use.

    python examples/optic_lobe.py
"""
import momonga
from momonga import GraphData, Momonga, transfer

N_CLUSTERS = 250                                          # NTAC clusters to run to; the lobes have about 230 types


def report(name, data, model):
    ntac, result = data.get_metrics(model.ntac_partition()), data.get_metrics(model.partition)
    print(f"{name}: NTAC {ntac['n_clusters']} clusters, accuracy {ntac['acc']:.3f}, macro-F1 {ntac['f1_macro']:.3f}"
          f" | MOMONGA {result['n_clusters']} clusters, accuracy {result['acc']:.3f}, macro-F1 "
          f"{result['f1_macro']:.3f} (from the NTAC start with {model.start_n_clusters} clusters, "
          f"L={model.description_length:,.0f})")


def momonga_vs_ntac(name):
    """NTAC and MOMONGA on one graph: the graph with its expert types, and the fitted Momonga."""
    data = GraphData.load(name)
    model = Momonga(data.without_labels(), log=print)     # the method is given a graph without the types
    model.unseeded_ntac(N_CLUSTERS)                       # model.ntac.trajectory[j]: the NTAC partition at j clusters
    model.refine(grid=range(50, N_CLUSTERS + 1, 50))      # one refinement per start; the shortest is kept
    report(name, data, model)
    return data, model


momonga.download_data()
right_data, right = momonga_vs_ntac("fafb_ol_right")
left_data, left = momonga_vs_ntac("fafb_ol_left")
recipient = transfer(left, right)                         # the lobe refined shorter per edge donates its partition
print("after the transfer:")
if recipient is left:
    report("fafb_ol_left", left_data, left)
else:
    report("fafb_ol_right", right_data, right)
