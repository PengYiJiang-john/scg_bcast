# Propagation graph measurements

This module reads 120 H22 traces and 20 SciFact traces. It measures two step graphs: prescribed workflow dependencies and the subgraph induced by saved regex-active labels. Prefix-intensity scores are included as annotations.

From the repository root:

```sh
python analysis/propagation_graphs/analyze.py --self-test
python analysis/propagation_graphs/analyze.py --output-dir reproduced/local/propagation_graphs
python analysis/propagation_graphs/plot_examples.py --input-dir reproduced/local/propagation_graphs --output-dir reproduced/local/propagation_graphs/figures
```

Outputs include per-run metrics, step annotations, grouped summaries, graph records, and selected example figures. H22 inputs come from `archive/source_access/raw/`; additional inputs are in `data/`.

Metrics include degrees, branching and merging nodes, connected components, and longest-path coverage. A strict chain is a connected DAG with at least two nodes, `n-1` edges, and maximum indegree and outdegree of one. Chain fractions use graphs with at least two nodes as the denominator. Empty graphs and singletons have separate categories.
