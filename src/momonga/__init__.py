"""MOMONGA: unsupervised neuronal cell typing from connectivity by minimum description length.

    from momonga import Momonga, GraphData, transfer

    data = GraphData(A_csr, labels=types)       # the expert types are for scoring only
    model = Momonga(data.without_labels())      # the method is given a graph without them
    model.unseeded_ntac(250)                    # the NTAC start; model.ntac.trajectory[j]: its partition at j clusters
    model.refine()                              # model.partition and model.description_length
    print(data.get_metrics(model.partition))
"""

__version__ = "1.0.0"

from .objective import description_length
from .data import download_data
from .graph import GraphData
from .momonga import Momonga, transfer

__all__ = ["Momonga", "GraphData", "transfer", "description_length", "download_data"]
