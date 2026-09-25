# SCG / BCAST: numerical checks and planned LLM protocols

**有限状态 LP 和确定性示例已复跑；历史来源访问实验已恢复 120 条轨迹与原始评判记录，四个均值可离线重算。历史生成模型元数据仍不完整。E1–E3 的真实 LLM 比较实验尚未执行。**

This repository accompanies a draft on semantic behavioral accountability. It contains executable finite-table checks, an exact deterministic example, and proposed LLM experiment protocols. It includes a recovered, separately identified H22 historical output archive. It does **not** contain completed FEVER/HoVer model evaluations or an implemented E1–E3 evaluation runner.

## Evidence status

| Material | What is available | Status and permitted interpretation |
|---|---|---|
| Fixed-profile coalition-span LP | Original code/output, audited code, complete optimizer-generated witness tables | Locally reproduced mathematical constructions; no LLM data |
| Three-behavior example | Explicit state transitions, eight losses, exact Shapley/SSV and missing-domain calculations, figure | Deterministic illustration; no LLM data |
| E1 semantic recovery and local interventions | Draft protocol and planned configuration | Not executed; no measured scores |
| E2 misleading behavior and remediation | Draft protocol | Not executed; eligible sample count and replay budget remain undetermined |
| E3 mode selection | Draft protocol and planned configuration | Not executed; first-layer behavioral penalties only, no final SSV incentive validation |
| Historical source-access means | 120 traces and 120 raw direct-judge responses, verified joins and reaggregation | Recovered descriptive output archive; exact generation runtime metadata incomplete |

The 2026-09-26 audit reran the supplied numerical code after source inspection. The initial ZIPs supplied only four historical means; the subsequent local recovery found their underlying H22 records and reproduced all four means. Each condition has five base cases times six workflow structures, not thirty independent samples. See `archive/source_access/README.md` for the records and remaining limits; exact generation model/settings/dates are not recoverable from the saved traces.

## Reproduce the available numerical checks

The local audit used CPython 3.14.7, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.2 on macOS arm64. Use Python 3.14 and the pinned requirements:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` writes to `reproduced/local/`, leaving the archived and dated audit outputs intact. It runs the unmodified legacy LP, the audited LP, the exact deterministic calculation, seven independent test cases, and the figure generator. It saves raw stdout, both complete LP witness tables, numerical summaries, test output, environment information, source hashes, and installed package versions. It also recomputes the recovered H22 archive entirely offline and checks archived versus fresh legacy output numerically (tolerance `1e-8`). There are no fresh random simulations or model calls in this suite; historical model outputs are read as archived data.

Individual commands:

```sh
python verification/verify_span_bound.py --output-dir reproduced/local
python illustrative_example/deterministic_example.py --output reproduced/local/deterministic_results.json
python -m unittest discover -s tests -v
python illustrative_example/make_counterfactual_overview.py --output-dir reproduced/local/figures
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

- `verification/verify_span_bound.py`: audited LP with precise result labels and saved witness tables.
- `provenance/legacy_verify_span_bound.py`: byte-preserved original numerical script.
- `verification/archived_results.txt`: supplied legacy stdout.
- `provenance/legacy_make_counterfactual_overview.py`: byte-preserved original figure script.
- `illustrative_example/deterministic_example.py`: added executable toy transitions and exact arithmetic.
- `reproduced/audit_2026-09-26/`: results actually generated in this audit, including tests and source hashes.
- `reproduced/numerical_results.txt` and `reproduced/environment.json`: prior reproduction files supplied in the archive, not this audit's environment.
- `provenance/original_manifest.json`: original manifest, retained as a historical claim. Its source-archive name and library ID were supplied metadata, not independently authenticated.
- `manifest.json`: current package hashes and evidence classifications. It excludes itself, virtual environments, caches, and local reruns.
- `archived_observations/`: original four-row historical CSV and a link to recovered provenance.
- `archive/source_access/`: recovered raw H22 traces/judgments, source hashes, offline reaggregation, and limitations.
- `protocols/LLM_EXPERIMENT_PROTOCOL.md` and `configs/planned_experiments.json`: draft study plans, not implemented experiments or frozen runtime assets. Manuscript LaTeX excerpts are retained locally but excluded from the GitHub experiment release.

## LLM plan and remaining implementation work

The draft plan proposes FEVER and HoVer, Qwen3-8B for generation, and Mistral Small 3.1 24B Instruct for semantic analysis and a separate model extension. Those model/dataset names are planned choices; no associated evaluation was executed here. Exact revisions, task manifests, frozen retrieval index, full prompts, matching/scoring rules, annotator records, runner, logs, and results are still missing.

The listed budgets are arithmetic plans: E3 development `19,440` chain replays, E3 test `12,800`, and the listed E1 interventions `5,120`, totaling `37,360`. At four generated steps per replay, the **first-attempt** ceiling is `149,440` calls; allowing one retry per step raises that ceiling to `298,880`. Neither figure includes additional E2 coalition replays, initial natural-workflow trajectories, semantic analysis, reference editing, input-deletion diagnostics, visible-text ablations, or the second-model extension. The original `149,440` must not be described as the overall run ceiling.

The protocol files mark unrun results explicitly. Their prescriptive language describes a proposal and is not a record that the proposed assets exist. No outcome numbers should be inserted without execution records and independent evaluation.
