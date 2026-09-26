# Controlled LLM mode-selection experiment

This directory releases the author-supplied structured responses and code from
`SLIC_LLM_Incentive_Experiment_20260926.zip`. The source archive hash and file
hashes are recorded in `source_manifest.json`. The 810 saved responses were
provided with the experiment; this release reconstructs their scores and game
calculations offline. It does not independently repeat their model generation.

## Design and scope

The experiment uses 30 constructed, objectively scored tasks: 10 procurement,
10 eligibility, and 10 logistics cases. Each case has nine output fields. Three
roles (A, B, C) each produce an output in three assigned modes, with three saved
samples per case/role/mode: 30 × 3 × 3 × 3 = 810 outputs.

The archived model mapping is `gemini-2.5-flash` for `careful` and `distorter`,
and `gemini-2.5-flash-lite` for `shortcut`. Modes therefore combine model choice
and prompting. `distorter` explicitly requests incorrect transformations as a
stress condition; its behavior is not an estimate of naturally occurring errors.
Task definitions, answer keys, prompts, schema, and the generation request
parameters declared by the original code are in `run_experiment.py`.

Stored outputs are reused under three fixed aggregation rules:

- **Modular:** A supplies fields s1–s3, B supplies s4–s6, and C supplies s7–s9.
- **Majority:** each field uses the unique modal answer among active roles;
  ties yield `UNKNOWN`.
- **Gatekeeper:** the highest-priority active role supplies all fields, with
  priority C > B > A.

Cancelling a role removes its stored output without resampling any remaining
role. All cancelled fields use the same `UNKNOWN` reference where applicable.
The code enumerates 27 mode profiles and eight coalitions in all three
aggregation rules, yielding 58,320 case/repetition/coalition losses. It evaluates
first-layer behavior Shapley penalties, leave-one-out penalties, and team loss.
Best responses are calculated over these finite tables; they are not additional
LLM calls or observed learning by agents. These tasks test the controlled
incentive construction, not natural multi-agent communication or second-layer
SSV allocation.

Loss is the fraction of incorrect fields. Numeric answers admit a 0.5% relative
tolerance; categorical answers use the declared canonicalization. Bootstrap
intervals use 10,000 resamples at the task level after averaging the three
samples, with seeds saved in the analysis code. They condition on the constructed
task set and the stored outputs.

## Offline reproduction

From the repository root, install `requirements.txt`, then run:

```sh
python llm_incentive/verify_archive.py --output-dir reproduced/local/llm_incentive
```

This starts from `results_full/llm_outputs.csv`, recomputes all seven result CSVs
and `summary.json`, regenerates the six analysis tables and the figure, and
checks all released CSV/JSON values against the archive. Floating-point
comparisons use an absolute tolerance of 1e-12. No network credentials or private
backend are needed. Figure bytes can vary with fonts/render metadata, so the
numeric source tables, not binary image equality, are checked. The same workflow
runs in CI and in the root `reproduce.py`.

Individual steps can also be run with separate output directories:

```sh
python llm_incentive/run_experiment.py --output-dir reproduced/local/llm_incentive/results
python llm_incentive/analyze_paper_results.py --results-dir reproduced/local/llm_incentive/results --output-dir reproduced/local/llm_incentive/paper
python -m unittest discover -s tests -v
```

## Optional generation interface

Offline reconstruction is the default. New model calls require an explicit
`--generate`, an importable `--backend-module`, and a separate `--cache` path.
No backend, credentials, or original private cache is shipped. The external
module must supply `CachedLLM(cache_path, retries=4, schema_retries=2)` and
`Request(model, temperature, max_output_tokens, purpose, seed)`. Its client must
implement `generate_json(system, user, request=..., schema=...)` and return a
parsed dictionary. Generation can incur API charges and is never invoked by CI,
unit tests, or `verify_archive.py`.

The original script declared temperature 0.35, maximum output tokens 700, and a
deterministic request-seed formula. The archive does not contain provider-level
request/response metadata or the original transport implementation, so whether
those parameters were all honored cannot be verified from this release. The
optional interface preserves the original call contract; it does not claim
identical fresh generations across providers or dates.

## Files

- `results_full/llm_outputs.csv`: the 810 original structured outputs and rationales.
- `results_full/coalition_case_losses.csv`: all 58,320 deterministic rescored losses.
- Other `results_full/*.csv` and `summary.json`: mode quality, profile payoffs,
  equilibria, strict best-response paths, and potential/stability calculations.
- `paper/table_*.csv` and `paper/figure_incentive_experiment.{pdf,png}`: archived
  numeric tables and scientific figure; manuscript prose is not included.
- `verify_archive.py`: full offline reconstruction and archive comparison.
- `../tests/test_llm_incentive.py`: independent permutation-based Shapley checks,
  equilibrium enumeration, cancellation-reference checks, and archive validation.
