# Natural Multi-Agent Workflows

This archive contains all **400 frozen task instances** from eight datasets, with 50 instances per dataset. It provides the task inputs, six-agent outputs, recovered semantic graphs, and saved counterfactual endpoint comparisons. Cases are retained in their original frozen order, including incorrect answers and zero or negative loss differences.

## Data

| Dataset | Task | Cases | Saved replay pairs | Files |
|---|---|---:|---:|---|
| HotpotQA | Multi-hop question answering | 50 | 137 | [Data](data/hotpotqa/) |
| TAT-QA | Financial table-and-text reasoning | 50 | 134 | [Data](data/tatqa/) |
| LegalBench, hearsay | Legal rule application | 50 | 162 | [Data](data/legal_hearsay/) |
| SciFact | Scientific claim verification | 50 | 183 | [Data](data/scifact/) |
| DROP | Discrete reasoning over passages | 50 | 132 | [Data](data/drop/) |
| GSM8K | Mathematical word problems | 50 | 158 | [Data](data/gsm8k/) |
| MuSiQue | Compositional multi-hop question answering | 50 | 118 | [Data](data/musique/) |
| TabFact | Table-based fact verification | 50 | 159 | [Data](data/tabfact/) |
| **Total** | | **400** | **1,183** | [Case index](case_index.csv) |

Each dataset directory contains:

- `inputs.jsonl`: case identifiers, questions, source passages or tables, reference answers, and dataset metadata. Reference answers are used only for scoring.
- `factual.jsonl`: complete A-F output records, final answers, and endpoint scores.
- `semantic_graphs.jsonl.gz`: complete saved parser records, including semantic nodes, owned relations, equivalent-node mappings, extraction stages, and merge checks.
- `replay_pairs.jsonl.gz`: matched target-absent and target-present replays, including the background subset, identity check, inputs and outputs at every stage, final answers, and losses.

The [case index](case_index.csv) lists every instance and its record counts. [Summary](summary.json) and [source checksums](source_manifest.json) accompany the data. Compressed files use standard gzip; no Git LFS is required. Dataset material remains subject to its upstream terms. Dataset URLs, frozen selection procedures, and scoring functions are recorded in [the data module](src/natural_bcast/data.py).

## Workflow

The workflow has two reasoning branches and a final reconciliation step:

```text
A + B -> D
A + C -> E
D + E -> F
```

| Agent | Function | Upstream records | Source text |
|---|---|---|---|
| A | Decompose the question into decision-relevant subquestions | None | No |
| B | Extract and organize source evidence | None | Yes |
| C | Solve the task independently | None | Yes |
| D | Compose a derivation from the plan and evidence | A, B | No |
| E | Test the independent solution and resolve concrete conflicts | A, C | Yes |
| F | Reconcile the two reasoning paths and supply the final answer | D, E | No |

All roles receive the task question. Only B, C, and E receive source text; the others receive source identifiers. A and B cannot set the final-answer field. Each role also has dataset-specific instructions, preserved in [the workflow module](src/natural_bcast/workflow.py).

The workflow model is Gemini 2.5 Flash-Lite (temperature 0.35). Semantic extraction uses Gemini 2.5 Flash (temperature 0). The replay editor uses Gemini 2.5 Flash with Gemini 2.5 Pro as its configured fallback. Seeds, generation limits, and model identifiers are in [config.json](config.json); extraction and replay prompts are included in [src/natural_bcast/](src/natural_bcast/).

NEW and TRANSFORMED records identify candidate semantic operations within each task instance. PRESERVED records describe semantic continuity and are retained as background metadata, not target players. Each released replay pair fixes all PRESERVED identifiers in its requested background and changes one NEW or TRANSFORMED target. Dependency-inactive records and the actual downstream text remain available for inspection.

For each saved pair, `marginal_loss = with_loss - without_loss`; a positive value means that retaining the target increases endpoint loss in that background. These are individual background-specific comparisons, not complete Behavioral Shapley vectors. [Replay availability](replay_availability.jsonl.gz) records all 18,802 checked comparisons and their cache/identity/execution status, including comparisons without a complete saved pair. No missing comparison is assigned a zero loss difference.

## Offline Inspection

From the repository root, inspect one case:

```sh
python natural_workflows/inspect_case.py \
  --case-id 'hotpot:5a738fe855429908901be2fb' \
  --section factual
```

Omit `--section` to read its input, factual trajectory, semantic graph, and available replay pairs together. Add `--output reproduced/local/case.json` to export readable JSON.

Verify all 400 cases and rescore all 2,766 saved endpoints:

```sh
python natural_workflows/verify_archive.py \
  --output-dir reproduced/local/natural_workflows
```

These commands use only saved files and Python's standard library. Verification checks case coverage, graph references, workflow routing, fixed-preservation backgrounds, target identity, endpoint binding, and every recorded loss difference. It makes no model API calls. The repository-wide `python reproduce.py` includes this verification.
