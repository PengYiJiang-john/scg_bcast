# Controlled incentive experiment

The input contains 810 saved structured responses from 30 constructed tasks, three participants, three modes, and three repetitions. The modes are careful, shortcut, and deliberately distorted.

The analysis reuses these responses under modular, majority-vote, and priority-override aggregation. Cancellation removes a participant's entire saved response. Loss is the fraction of incorrect fields. The code enumerates mode profiles and coalitions, then computes first-layer Shapley, leave-one-out, and team-loss incentives.

## Run

From the repository root, after installing `requirements.txt`:

```sh
python llm_incentive/verify_archive.py --output-dir reproduced/local/llm_incentive
```

This rebuilds the result tables and figure offline and compares CSV/JSON values with the saved results. Outputs go to `results/`, `paper/`, and `verification.json` within the selected directory.

Individual steps:

```sh
python llm_incentive/run_experiment.py --output-dir reproduced/local/llm_incentive/results
python llm_incentive/analyze_paper_results.py --results-dir reproduced/local/llm_incentive/results --output-dir reproduced/local/llm_incentive/paper
```

## Generate responses

New responses use an external Python backend with credentials configured there. It must provide `CachedLLM(cache_path, retries, schema_retries)`, its `generate_json(system, user, request, schema)` method, and `Request(model, temperature, max_output_tokens, purpose, seed)`.

For all 30 tasks with three repetitions, replace `my_backend` with the backend's importable module name:

```sh
python llm_incentive/run_experiment.py --generate --backend-module my_backend --cache reproduced/local/new_generation/responses.cache --full --repeats 3 --output-dir reproduced/local/new_generation/results
```

## Files

- `run_experiment.py`: task definitions, experimental prompts, aggregation, and scoring.
- `results_full/llm_outputs.csv`: saved responses.
- `results_full/`: coalition losses, payoffs, equilibria, and best-response paths.
- `paper/`: summary tables, task-level bootstrap intervals, and the figure.
- `source_manifest.json`: source file hashes.
