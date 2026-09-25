# Recovered H22 source-access observations

This archive contains **historical output records** recovered on 2026-09-26 from the author's local `scg_rumor_propagation` project. It is separate from the unexecuted BCAST E1–E3 protocols. No new model call was made during recovery.

## What was recovered and checked

- 120 saved workflow traces, each with six generated-text steps (720 saved step outputs).
- 120 final-prefix records and 120 direct judge responses, including the returned JSON text and reasons.
- 120 corresponding terminal scores; all join to the same case/workflow/final response.
- Case construction, generation, scoring, and rubric source snapshots; generation and judge stdout logs.
- All 120 raw judge JSON responses parse without recovery and equal the saved numerical scores.
- The recovered `intensity_scores.json` has SHA-256 `d6d30e9fdbbed4259a487ab698f2153a6811f55ec10f35105deb761ecf5bbbe8`, matching the original release's evidence manifest.

Recompute without dependencies, API keys, or model access:

```sh
python archive/source_access/recompute.py
```

The script verifies source-file hashes, joins raw records, reparses every raw judge response, verifies terminal texts and targets, checks complete factorial coverage, and writes `recomputed/per_run_scores.csv`, `recomputed/condition_summary.csv`, and `recomputed/summary.json`.

| Source condition | Saved case/workflow records | Recomputed mean |
|---|---:|---:|
| Visible | 30 | 0.100000 |
| De-emphasized | 30 | 0.460000 |
| Pointer only | 30 | 0.993333 |
| Absent | 30 | 0.980000 |

**Thirty is five base cases times six workflow structures, not thirty independent cases or random seeds.** There is one saved trace per base-case/condition/workflow cell. The five bases cover SciFact case 105 and four authored toy scenarios (finance cash flow, legal deadline, privacy retention, product recall). These are curated prompts and generated answers; the selected subset contains no patient case records. Medical role wording was nevertheless reused in some nonmedical workflows, which limits interpretation.

## Measurement and interpretation

All terminal judgment records identify `models/gemini-2.5-pro`. The rubric measures adoption and operative use of the declared false semantic family on a continuous `[0,1]` scale with anchors at `0, .2, .4, .6, .8, 1`. It is an LLM-judge score, not a calibrated probability, human consensus, counterfactual effect, or Shapley allocation. The judge sees the target, correction, latest answer, and prefix.

The **de-emphasized** prompt still includes the full corrective source text in the actual model input; it merely says that the archive is not currently opened. The pointer-only condition presents a textual statement that a source exists, without a retrieval tool. These are controlled prompt presentations, not verified external source-access events. Every agent also receives the original case prompt at each step. Workflow dependence is prescribed; semantic-source graph recovery is not tested by these records.

The local traces, command logs, and raw judge responses support an archived LLM-output analysis and allow exact reaggregation. They do not permit a fresh independent reproduction of the historical provider execution. In particular:

- The generation records do not persist an exact model identifier, dates, generation settings, seeds, provider request IDs, or token usage. The local project describes Gemini generation, but the exact generator is not recoverable from these saved records.
- Judge model identifiers are recorded, but historical judge runtime settings and execution dates are not. Source-code defaults are not evidence of the actual historical settings.
- The source files are snapshots as found during recovery; their exact historical revision is unrecorded.
- Empty terminal error logs do not prove zero transient retries or zero excluded attempts; billing and total call counts remain unknown.
- No human calibration, repeated-judge uncertainty, or independent replication is supplied. One base case per domain does not establish domain generalization.
- `ci_low` and `ci_high` in the original score JSON are point-score placeholders (`ci_low == ci_high == intensity`), not confidence intervals; `bootstrap_samples` is zero. No confidence interval is claimed or computed here.
- Keyword-derived `events`, `links`, and `summaries` in `runs.jsonl` are legacy ancillary fields. They can miss Chinese/paraphrased content and are not used to recover the means or validate semantic graph extraction.

## Provenance and safety of the offline package

`source_manifest.json` gives original project-relative locations and byte hashes. Raw source files are preserved; corrected interpretation is in this README and the recomputed outputs. Source snapshots use `.py.txt` so the archive is not presented as a complete executable historical API runner. The included entry point only reads local files.

Only the H22 evidence subset was copied. Environment files, credentials, other project outputs, and manuscript files were excluded. Selected files were scanned for common API-token/private-key patterns and email identifiers; the scan found none. This is a bounded check of this selected subset, not a certification of the original project.
