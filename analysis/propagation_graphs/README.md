# Post hoc propagation-graph audit

This module analyzes saved graph structures from **120 H22 runs** and **20 older SciFact runs**. It makes no API/model calls and adds no LLM experiment results. The graph measures and diagnostic selections are post hoc and exploratory.

**Finding:** the archives contain a mixture of prescribed structures and unreliable legacy text labels. They do not establish that naturally recovered semantic propagation graphs are approximately chains. The strict-chain counts below are not an estimate of an undefined “approximately chain-like” rate.

## What the two graphs mean

1. **Full workflow graph:** one node per saved workflow step, with directed edges from `depends_on`. These edges were prescribed by the experiment. They are not extracted SCG semantic-source dependencies. The original task prompt was also supplied at every step; that common external input is not included as a graph node.
2. **Legacy active-step proxy:** a node is included only if its saved regex status is `introduced_false`, `preserved_false`, `transformed_false`, `amplified_false`, `laundered_false`, or `finalized_false`. An edge is included only if it is a workflow dependency and **both endpoints are active**. This induced graph describes the archived labels; it is not a validated semantic propagation graph.

The old stored `links` include edges from active sources to correction, challenge, suppression, and other nonactive targets. Using all of those links as error-propagation edges would be incorrect. The script removes such edges and confirms that the filtered links equal the active-induced workflow edges for every analyzed record.

The graphs use **steps**, not individual semantic nodes inside complete responses, and do not instantiate the full SCG provenance hypergraph. One step may include both a claim quotation and its rejection.

## Run offline

From the experiment repository root:

```sh
python analysis/propagation_graphs/analyze.py --self-test
python analysis/propagation_graphs/analyze.py
python analysis/propagation_graphs/plot_examples.py
```

To keep the published snapshots intact, send a new run to an output directory:

```sh
python analysis/propagation_graphs/analyze.py --output-dir reproduced/local/propagation_graphs
python analysis/propagation_graphs/plot_examples.py --input-dir reproduced/local/propagation_graphs --output-dir reproduced/local/propagation_graphs/figures
```

The plot command reads `graph_records.json` from `--input-dir`, writes `selected_examples.json` back to that input directory, and writes PNGs to `--output-dir`. Image paths in the selection JSON are relative to its own directory.

The analyzer needs only the Python standard library. The figure generator needs the repository's Matplotlib dependency. Ten independent graph fixtures check empty/singleton policies, chains, a fork, a diamond, the two-source six-node tree, disconnected paths, a shortcut DAG, correction-edge filtering, and cycle rejection. The diamond has path coverage `3/4`; the six-node tree has `4/6`; a shortcut DAG can have coverage `1` without being a chain.

Source hashes are checked before analysis. The archived H22 records are reused from `../../archive/source_access/raw/`; additional selected files and their hashes are in `data/source_manifest.json`. Source snapshots are retained as found, not certified historical code versions. Original saved event labels are analyzed without rerunning or repairing the old classifier.

## Measures and denominators

For both graphs the script reports node/edge count, maximum indegree/outdegree, number of branching nodes (`outdegree > 1`), merging nodes (`indegree > 1`), weakly connected components, longest directed path in nodes, and longest-path coverage (`path nodes / all graph nodes`). Empty-graph coverage is undefined. Groups also report path coverage separately among graphs with at least two nodes, avoiding singleton inflation.

A **strict nontrivial chain** is a connected DAG with at least two nodes, `n-1` edges, maximum indegree at most one, and maximum outdegree at most one. Empty graphs and singletons are separate categories and never positive chain evidence. The reported strict-chain fraction uses graphs with **at least two nodes** as its denominator. Disconnected active sets remain in that denominator even if they contain no edge.

No approximate-chain threshold was predeclared or estimated. A high longest-path coverage alone is insufficient: a graph may contain a path through all nodes plus many shortcut, branch, or merge edges.

## Observed legacy-proxy counts

| Archive | Runs | Empty | Singleton | Strict nontrivial chain | Disconnected | Connected branching/merging | Strict chains / graphs with ≥2 nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| H22 | 120 | 51 | 18 | 6 | 29 | 16 | 6/51 = 11.8% |
| Earlier SciFact | 20 | 11 | 8 | 0 | 0 | 1 | 0/1 |

These mutually exclusive structure categories describe saved regex labels. They are neither semantic extraction accuracy nor a direct test of approximate-chain structure. Some disconnected graphs also have a branch or merge within a component; the independent branch/merge counts in JSON preserve this distinction.

H22 has 225 stored legacy links. Filtering removes 87 with nonactive targets, including 9 into correction/challenge and 78 into suppression/other statuses, leaving 138 active-active edges. The older SciFact archive has 23 stored links; 20 are removed, leaving three edges in one triangle-shaped proxy.

| H22 condition | Empty | Singleton | Strict chain | Disconnected | Connected branching/merging |
|---|---:|---:|---:|---:|---:|
| Absent | 5 | 4 | 2 | 11 | 8 |
| De-emphasized | 14 | 5 | 2 | 7 | 2 |
| Pointer only | 8 | 4 | 2 | 10 | 6 |
| Visible | 24 | 5 | 0 | 1 | 0 |

Each condition has 30 saved case/workflow cells (five base cases times six prescribed workflows), not 30 independent tasks or seeds. All groupings by workflow, declared topology, condition, and condition/workflow are in `results/summary.json`. Prescribed H22 workflow path coverage ranges from `4/6` to `6/6`; maximum indegree ranges from one to four. These are design properties, not learned semantic structure.

The old `gatekeeper_laundering` metadata says `linear`, yet its saved dependency graph has four nodes, five edges, and maximum indegree/outdegree two. The script trusts the actual edges for metrics; it preserves `topology` only as a supplied grouping label.

## Cross-checks and limitations

The existing H22 Pro endpoint scores identify 75 records with intensity at least `0.8` (an exploratory descriptive threshold). Of these, 17 have an empty regex graph and 31 have a nonactive final regex status. Sixteen records have Pro intensity exactly `1.0` despite an empty regex graph. These disagreements are not a calibrated false-negative rate: Pro intensity is a different, system/prefix-level measurement.

For annotations, the module also joins 720 archived Flash step scores to the exact saved step texts. The score is **current SCG prefix/system intensity**, not the local node's truth or adoption label. It is never thresholded to create graph nodes or edges. The saved raw responses contain 718 complete JSON objects and two truncated responses with a recoverable numerical `intensity` field. Both recovered numbers match the stored scores; the CSV flags them individually. No parse-failed response was converted to zero by this audit.

Six deliberately selected raw-text checks are documented in `qualitative_checks.md`. They reveal both Chinese/paraphrase misses and false positives when a rejected claim is quoted. They are qualitative diagnostics selected after inspecting the archive, not a representative annotation study or an estimate of overall semantic accuracy. Four of them are illustrated in `figures/`.

Reliable natural-language structural claims would require a declared semantic-node/link extraction procedure and independent validation. Neither a prescribed workflow graph, these regex-induced proxies, nor system-level intensity scores alone supplies that evidence.

## Outputs

- `results/per_run_graph_metrics.csv`: 140 run/claim records, with all full/proxy measures and endpoint comparisons.
- `results/per_step_annotations.csv`: 810 saved step annotations; Flash prefix intensity is available for the 720 H22 steps only.
- `results/by_workflow_compact.csv`: compact counts and measures for ten dataset/workflow groups.
- `results/summary.json`: complete grouped summaries, definitions, parsing counts, and scope restrictions.
- `results/graph_records.json`: exact analyzed nodes, edges, saved statuses, score annotations, and texts.
- `results/selected_examples.json`: explicit selection reasons, metrics, original texts, and filtered-out links for the four plotted cases.
- `figures/01_regex_empty_high_adoption.png` through `04_path_coverage_is_not_chain.png`: diagnostic graph comparisons; the example selection is purposeful, not random.
- `results/self_test.txt`: independent graph-fixture check output.
- `data/source_manifest.json`: original project-relative paths, byte hashes, and bounded credential-scan record.

Selected extra data are curated SciFact snippets and archived generated responses, not patient records. No environment files, keys, or manuscript files were copied. Historical generator configuration and exact code revision remain incompletely recorded, as described by the source-access archive.
