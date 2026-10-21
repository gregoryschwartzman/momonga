"""The paper's data in one call: the ten graphs, from a GitHub release of this repo.

    import momonga
    momonga.download_data()          # ~100 MB into data/graphs, once

The archive holds graphs/<organism>/<rest>.{adj.npz,npz}: the adjacency (scipy CSR, synapse counts, rows
presynaptic) and the neurons' `id`, `type` (the expert type, "?" if untyped) and `side`.  The graphs are derived
from the FlyWire (Codex v783), MaleCNS and MANC connectomes.  `GraphData.load(name)` reads them from here.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # the repository
GRAPHS = ROOT / "data" / "graphs"              # where the graphs live
ORGANISMS = ("fafb", "mcns", "manc")           # each owns a folder of data/graphs
DATA_URL = "https://github.com/gregoryschwartzman/momonga/releases/download/v1.0/momonga_data.zip"
GRAPH_NAMES = ("fafb_ol_left", "fafb_ol_right", "fafb_visual_left", "fafb_visual_right", "fafb_central",
               "mcns_ol_left", "mcns_ol_right", "mcns_visual_left", "mcns_visual_right", "manc_vnc")


def graph_files(name):
    """(adjacency file, vertex file) of a graph: "fafb_ol_left" is data/graphs/fafb/ol_left.{adj.npz,npz}."""
    organism, _, rest = name.partition("_")
    if organism not in ORGANISMS or not rest:
        raise ValueError(f"graph name {name!r} must start with one of {ORGANISMS} and an underscore")
    return GRAPHS / organism / f"{rest}.adj.npz", GRAPHS / organism / f"{rest}.npz"


def have_data(names=GRAPH_NAMES):
    """True when every graph of `names` is in place."""
    return all(path.exists() for name in names for path in graph_files(name))


def download_data(url=DATA_URL, verbose=True):
    """Download and unpack the graphs unless they are already in place.  Returns the graphs directory."""
    if have_data():
        if verbose:
            print(f"data present under {GRAPHS}")
        return GRAPHS
    archive = ROOT / "momonga_data.zip"
    if not archive.exists():
        if verbose:
            print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, archive)
    if verbose:
        print(f"unpacking into {GRAPHS} ...")
    GRAPHS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.namelist():
            folder, _, rest = member.partition("/")             # only files under graphs/ are unpacked
            if folder != "graphs" or not rest or member.endswith("/"):
                continue
            target = GRAPHS / rest
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
    archive.unlink()
    if not have_data():
        raise RuntimeError("the archive did not contain every graph")
    if verbose:
        print("done")
    return GRAPHS

