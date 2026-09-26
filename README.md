# SCG / BCAST: numerical checks and controlled LLM incentive experiments

**本仓库提供有限状态数值核验、历史来源访问记录，以及新增的三参与者、三模式 LLM 受控激励实验。新增实验保存了 810 份输出，可离线复算联盟损失与均衡。此前未执行的 E1–E3 方案已从当前版本移除。**

This repository accompanies a draft on semantic behavioral accountability. It contains executable finite-table checks, an exact deterministic example, a recovered H22 historical output archive, and a controlled LLM mode-selection experiment. The latter uses saved outputs from three participants with three candidate modes and evaluates first-layer behavioral Shapley incentives under three fixed aggregation rules. It is separate from the eight-task semantic behavior audit in the manuscript; that audit’s raw replay data are not included here.

## Evidence status

| Material | What is available | Status and permitted interpretation |
|---|---|---|
| Fixed-profile coalition-span LP | Original code/output, audited code, complete optimizer-generated witness tables | Locally reproduced mathematical constructions; no LLM data |
| Three-behavior example | Explicit state transitions, eight losses, exact Shapley/SSV and missing-domain calculations, figure | Deterministic illustration; no LLM data |
| Controlled LLM mode selection | 30 constructed tasks, 810 archived structured outputs, generation prompts, 58,320 derived coalition losses and equilibrium analysis | Offline reproduction from saved outputs; empirical finite game with fixed aggregation, whole-output cancellation and first-layer penalties |
| Historical source-access means | 120 traces and 120 raw direct-judge responses, verified joins and reaggregation | Recovered descriptive output archive; exact generation runtime metadata incomplete |
| Historical propagation graph audit | 120 H22 and 20 earlier SciFact traces, workflow and legacy rule graph metrics, examples | New post hoc analysis of archived outputs; legacy extraction has documented semantic errors |

The 2026-09-26 audit reran the supplied numerical code after source inspection. The initial ZIPs supplied only four historical means; the subsequent local recovery found their underlying H22 records and reproduced all four means. Each condition has five base cases times six workflow structures, not thirty independent samples. See `archive/source_access/README.md` for the records and remaining limits; exact generation model/settings/dates are not recoverable from the saved traces.

## Controlled LLM mode selection

The [`llm_incentive/`](llm_incentive/) module contains three constructed task families (30 tasks in total), nine scored fields per task, and three repeated outputs for each participant and mode. `careful` and `distorter` use the recorded `gemini-2.5-flash` setting; `shortcut` uses `gemini-2.5-flash-lite`. Modes combine model and prompt changes, and `distorter` is an intentionally degraded stress condition rather than a naturally occurring agent type.

The same saved outputs are reused under modular, majority-vote, and priority-override aggregation. Cancellation removes one participant’s whole output. The offline analysis enumerates 27 mode profiles and eight coalitions, computes Shapley penalties, and searches every pure profile. Each resulting empirical game has a unique Shapley equilibrium at `careful|careful|careful`, with full-participation losses 0.117284, 0.102469 and 0.106173. Each equals the best candidate loss; the priority-override environment has nine task-loss minimizers, so its optimum is not unique.

Under priority override, LOO and team-loss incentives each admit nine equilibria. Their losses with participant C absent range from 0.116049 to 0.766667, compared with 0.116049 at the Shapley equilibrium. These are properties of the saved finite games, not evidence of convergence or optimality in arbitrary LLM workflows. In particular, the modular stability span is zero by construction, while the other two spans give uninformative worst-case loss bounds. The experiment does not implement local semantic cancellation, downstream LLM regeneration, or second-layer SSV incentives.

See the module README for offline reproduction, archived-data provenance, and the explicitly separate optional interface for generating new outputs. The full manuscript and the supplied draft experiment narrative are not published in this repository.

## What the old propagation graphs establish

The old propagation experiments were located. `analysis/propagation_graphs/` recomputes structural metrics separately for the prescribed workflow DAG and the archived rule-detected active graph. Only links whose two endpoints are active are counted as continuing-error links. Empty and singleton graphs are reported separately from nontrivial chains.

Of the 120 H22 rule graphs, 51 are empty, 18 are singletons, 6 are nontrivial single chains, 29 are disconnected, and 16 are connected with branching or merging. Thus 6 of the 51 graphs with at least two active nodes are **strict** single chains. This is not a near-chain rate: no approximation threshold was preregistered, and the legacy labels are not verified semantic ground truth. The 20 earlier SciFact rule graphs contain 11 empty graphs, 8 singletons, and one connected merging graph.

Text inspection found a six-step Chinese trajectory that retains the target error while every step is labelled absent, and an English denial labelled as active propagation. A separate six-step tree example retains the erroneous claim on both branches before merging, with longest-path coverage 4/6. Those input-dependency edges do not establish that both branches are causally necessary. Accordingly, these archives cannot establish that real error propagation is generally near-chain. Prefix-intensity scores describe the cumulative system state and are not local node or edge truth labels. See the module README, metric definitions, per-run results, and selected text checks for the precise scope.

## Reproduce the available numerical checks

The local audit used CPython 3.14.7, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, and pandas 3.0.6 on macOS arm64. Use Python 3.14 and the pinned requirements:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` writes to `reproduced/local/`, leaving the archived and dated audit outputs intact. It runs the unmodified legacy LP, the audited LP, the exact deterministic calculation, eleven unit tests for numerical calculations, stored-output rescoring, cancellation, and empirical equilibria, graph metric fixtures, and figure generators. It saves raw stdout, both complete LP witness tables, numerical summaries, test output, environment information, source hashes, and installed package versions. It also recomputes the recovered H22 archive and post hoc graph metrics entirely offline and checks archived versus fresh legacy LP output numerically (tolerance `1e-8`). The suite also reconstructs the controlled incentive game from its 810 stored outputs and checks every released result and statistical table. It makes no new model calls; all model outputs are read as archived data.

Individual commands:

```sh
python verification/verify_span_bound.py --output-dir reproduced/local
python illustrative_example/deterministic_example.py --output reproduced/local/deterministic_results.json
python -m unittest discover -s tests -v
python illustrative_example/make_counterfactual_overview.py --output-dir reproduced/local/figures
python analysis/propagation_graphs/analyze.py --self-test
python analysis/propagation_graphs/analyze.py --output-dir reproduced/local/propagation_graphs
```

GitHub Actions is configured in `.github/workflows/verify.yml` to execute the same scientific checks on Linux. The checked-in local audit does not certify a future remote Actions run.

## Reproduced numerical results

The LP uses a value for every key in `{-1, 0, 1}^m`, where `-1` is cancellation and `0, 1` are alternative modes. Each value is constrained to `[0, 1]`; the all-reference value is fixed at `0.8`. The constraints limit differences between coalition marginals **within each fixed mode profile**. The all-zero profile is constrained to be a pure equilibrium of the first-layer Shapley penalty game.

| Positions `m` | Within-profile span `eta` | Variables | Maximum designated-profile loss gap | `m * eta` |
|---:|---:|---:|---:|---:|
| 2 | 0.02 | 9 | 0.04 | 0.04 |
| 3 | 0.02 | 27 | 0.06 | 0.06 |
| 4 | 0.02 | 81 | 0.08 | 0.08 |
| 4 | 0 | 81 | 0 | 0 |

The objective is the all-zero equilibrium's loss minus the designated all-one profile's loss. The audited code additionally enumerates all full profiles in each returned witness: the all-one profile is a loss minimizer in each of these four witnesses. This enumeration is a property of those witnesses; it is not a proof of a general worst-equilibrium bound. The separate LP for a selected same-mode penalty change returns `0.02, 0.02, 0.02, 0` to floating-point tolerance.

The exact-potential residual is zero for these witnesses; the efficiency residual is at most `3.4e-17`. Tests independently recompute Shapley values from all permutations, inspect every span constraint, verify equilibrium inequalities, and check the potential identity's coefficient vectors for all modes/profiles at `m=2,3,4`. Solver tolerance is `1e-8`; these floating-point checks do not replace a proof.

**Scope:** this LP implements the older within-profile coalition-span condition. It does not verify the draft's stronger stability condition across other participants' candidate modes, the new `2 * sum(epsilon_i)` bound, natural-language cancellation quality, or final two-layer SSV incentives. A finite state machine can realize these tables by recording active modes and using the table as a terminal Bernoulli parameter; no LLM behavior is thereby established.

The exact three-behavior state machine produces the loss table `v({H,B})=1` and zero for the other seven coalitions. It gives:

- Behavioral Shapley: `(1/6, -1/3, 1/6)`.
- Factual leave-one-out effect: `(0, -1, 0)`.
- SSV under the explicitly declared chain support functions: `(1/18, -1/9, 1/18)`.
- With `{H,B}` missing: coverage `(5/6, 2/3, 5/6)` and joint allocation set `(t/6, -t/3, t/6)`, `0 <= t <= 1`.

The support functions are illustrative assumptions, not inferred causal provenance. The figure is generated from the deterministic loss function; its messages are authored explanatory text, not model outputs.

## Files and provenance

- `llm_incentive/`: controlled mode-selection code, saved structured outputs, analysis tables, and offline verification.
- `verification/verify_span_bound.py`: audited LP with precise result labels and saved witness tables.
- `provenance/legacy_verify_span_bound.py`: byte-preserved original numerical script.
- `verification/archived_results.txt`: supplied legacy stdout.
- `provenance/legacy_make_counterfactual_overview.py`: byte-preserved original figure script.
- `illustrative_example/deterministic_example.py`: added executable toy transitions and exact arithmetic.
- `reproduced/audit_2026-09-26/`: results actually generated in this audit, including tests and source hashes.
- `reproduced/final_integration_2026-09-26/`: successful final offline integration logs, including graph checks; freshly computed graph tables match the published analysis tables byte-for-byte.
- `reproduced/numerical_results.txt` and `reproduced/environment.json`: prior reproduction files supplied in the archive, not this audit's environment.
- `provenance/original_manifest.json`: original manifest, retained as a historical claim. Its source-archive name and library ID were supplied metadata, not independently authenticated.
- `manifest.json`: current package hashes and evidence classifications. It excludes itself, virtual environments, caches, and local reruns.
- `archived_observations/`: original four-row historical CSV and a link to recovered provenance.
- `archive/source_access/`: recovered raw H22 traces/judgments, source hashes, offline reaggregation, and limitations.
- `analysis/propagation_graphs/`: post hoc graph metrics on archived H22 and earlier SciFact outputs, diagnostic examples, additional source hashes, and explicit extraction limits.
