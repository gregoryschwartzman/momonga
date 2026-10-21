# MOMONGA

Unsupervised neuronal cell typing from connectivity alone, by minimum description length.
MOMONGA (MDL Optimization for Mapping Of Neuronal Groups from Adjacency) is the method of the paper
*Unsupervised neuronal cell typing from connectivity by minimum description length*.

```bash
git clone https://github.com/gregoryschwartzman/momonga && cd momonga
python -m venv .venv && .venv/bin/pip install -e .      # add ".[metal]" on Apple silicon
.venv/bin/python examples/quickstart.py                   # synthetic graph, seconds
.venv/bin/python examples/optic_lobe.py                   # the paper's female optic lobes, minutes; downloads the data
```

See the [examples](examples) folder for a quickstart; the paper's graphs are described in
[data/README.md](data/README.md).

MIT license, see [LICENSE](LICENSE).
