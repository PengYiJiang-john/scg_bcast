# SCG / BCAST Experiments

Code, saved experimental records, numerical results, and figures for semantic behavior accountability and incentives.

## Experiments

| Experiment | Materials |
|---|---|
| Numerical verification | [Coalition-span bounds and potential checks](verification/), [tests](tests/test_verification.py) |
| Behavioral allocation example | [Three-behavior calculation and counterfactual figure](illustrative_example/) |
| Controlled LLM incentives | [Protocol, 648 replay records, exact Behavioral Shapley values, equilibria, and English figure](llm_incentive/) |
| Natural multi-agent workflows | [400 cases across eight datasets, full trajectories and semantic graphs, and 1,183 saved replay pairs](natural_workflows/) |
| Source-access observations | [120 saved workflows, judge responses, and scores](archive/source_access/) |
| Propagation structure | [Workflow and regex-active graph measurements](analysis/propagation_graphs/) |

The controlled incentive experiment uses three causal transformation positions: evidence extraction, reasoning, and decision. Each position chooses among three prompted modes of Gemini 2.5 Pro. Cancellation passes the incoming state through unchanged, and downstream generation uses the resulting state. All 27 mode profiles and eight retained-behavior subsets are evaluated with three repetitions.

## Reproduce

Use Python 3.14 and install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The offline run recomputes tables and figures from saved records and writes results to `reproduced/local/`. It does not call model APIs. The [incentive experiment guide](llm_incentive/README.md) describes the workflow, scoring, data fields, and optional generation command. The [natural-workflow archive](natural_workflows/README.md) provides all 400 task inputs and trajectories, a case index, and offline semantic-graph and replay inspection.

`manifest.json` contains checksums for the released files:

```sh
python refresh_manifest.py --check
```
