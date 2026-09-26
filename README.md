# SCG / BCAST Experiments

Code and saved outputs for numerical checks, behavioral allocation examples, and controlled incentive experiments.

| Directory | Contents |
|---|---|
| `verification/` | Coalition-span linear programs and numerical checks |
| `illustrative_example/` | Exact three-behavior example and counterfactual figure |
| [`llm_incentive/`](llm_incentive/) | Saved responses, fixed-aggregator incentive games, tables, and figures |
| [`archive/source_access/`](archive/source_access/) | Saved workflow traces and terminal judge scores |
| [`analysis/propagation_graphs/`](analysis/propagation_graphs/) | Workflow and regex-active graph measurements |
| `tests/` | Numerical and archive consistency tests |

## Run

Use Python 3.14 and install the pinned dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The offline run recomputes tables and figures, checks saved results, and writes outputs to `reproduced/local/`. Set another destination with `--output-dir`.

To run only the incentive experiment:

```sh
python llm_incentive/verify_archive.py --output-dir reproduced/local/llm_incentive
```

Module READMEs describe their inputs and outputs.
