"""Offline audit of historical H22 traces and judge responses; no model calls."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(value, description):
    if not value:
        raise ValueError(description)


def audit():
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    for row in manifest["files"]:
        require(hashlib.sha256((HERE/row["archive_path"]).read_bytes()).hexdigest() == row["sha256"],
                "Changed archived source: "+row["archive_path"])
    runs = load_lines(HERE/"raw/runs.jsonl")
    prefixes = load_lines(HERE/"raw/prefixes.jsonl")
    judgments = load_lines(HERE/"raw/direct_judgments.jsonl")
    scores_file = json.loads((HERE/"raw/intensity_scores.json").read_text())
    scores = scores_file["scores"]
    require(len(runs) == len(prefixes) == len(judgments) == len(scores) == 120, "Expected 120 matched records")
    by_prefix = {row["prefix_id"]: row for row in prefixes}
    by_score = {row["prefix_id"]: row for row in scores}
    by_judge = {row["prefix_id"]: row for row in judgments}
    require(len(by_prefix) == len(by_score) == len(by_judge) == 120, "Duplicate prefix ID")
    require(set(by_prefix) == set(by_score) == set(by_judge), "Unequal score/judge/prefix coverage")
    indexed_runs = {(row["case"]["case_id"], row["workflow"]["workflow_id"]): row for row in runs}
    require(len(indexed_runs) == 120, "Duplicate case-workflow run")
    groups = defaultdict(list)
    joined = []
    cells = set()
    saved_steps = 0
    for run in runs:
        require(not run["case"].get("mock_outputs"), "A source case contains mock outputs")
        require(len(run["trace"]["steps"]) == 6, "Unexpected workflow length")
        saved_steps += len(run["trace"]["steps"])
        step_ids = {step["step_id"] for step in run["workflow"]["steps"]}
        require(step_ids == {step["step_id"] for step in run["trace"]["steps"]}, "Trace/workflow step mismatch")
    for prefix_id, prefix in by_prefix.items():
        score, judgment = by_score[prefix_id], by_judge[prefix_id]
        run = indexed_runs[(prefix["case_id"], prefix["workflow_id"])]
        final = [step for step in run["trace"]["steps"] if step["is_final"]]
        require(len(final) == 1, "Missing/ambiguous final step")
        require(prefix["latest_content"] == score["latest_content"] == final[0]["content"], "Final text mismatch")
        require(prefix["is_final"] and score["is_final"], "Nonfinal score entered aggregation")
        raw = judgment["raw_response"]
        match = re.search(r"\{.*\}", raw, re.S)
        require(match is not None, "Judge response has no JSON")
        parsed = json.loads(match.group(0))
        intensity = float(parsed["intensity"])
        require(0 <= intensity <= 1, "Out-of-range raw intensity")
        require(intensity == float(judgment["intensity"]) == float(score["intensity"]), "Judge/score mismatch")
        require(parsed["stage"] != "parse_failed", "Failed parse was scored")
        require(prefix["prompt"] == run["case"]["prompt"], "Original prompt mismatch")
        claim = next(row for row in run["case"]["false_claims"] if row["claim_id"] == prefix["claim_id"])
        require(prefix["truth"] == claim["truth"] and prefix["false_claim"] == claim["label"], "Target mismatch")
        base, condition = prefix["case_id"].removeprefix("h22_").split("__")
        cell = (base, condition, prefix["workflow_id"])
        require(cell not in cells, "Duplicate design cell")
        cells.add(cell)
        groups[condition].append(intensity)
        joined.append(dict(prefix_id=prefix_id, base_case=base, condition=condition,
                           workflow_id=prefix["workflow_id"], intensity=intensity,
                           judge_model=judgment["judge_model"]))
    expected = {"source_visible": .100, "source_deemphasized": .460,
                "source_pointer_only": 29.8/30, "source_absent": .980}
    summary = []
    for condition, mean in expected.items():
        values = groups[condition]
        require(len(values) == 30, "Condition does not have 30 case-workflow cells")
        actual = sum(values)/len(values)
        require(abs(actual-mean) < 1e-12, "Historical mean does not reproduce")
        summary.append(dict(condition=condition, case_workflow_count=len(values),
                            base_case_count=5, workflows_per_case=6,
                            mean_false_semantic_intensity=actual))
    bases = sorted({row["base_case"] for row in joined})
    workflows = sorted({row["workflow_id"] for row in joined})
    require(len(bases)==5 and len(workflows)==6 and len(cells)==5*4*6, "Incomplete factorial coverage")
    # The generator omitted configuration and transport metadata from its saved traces.
    result = dict(
        audit_kind="offline_reaggregation_of_archived_outputs_not_independent_LLM_replication",
        observed_runs=len(runs), saved_generation_step_outputs=saved_steps,
        matched_raw_judge_responses=len(judgments), matched_terminal_scores=len(scores),
        judge_models=dict(Counter(row["judge_model"] for row in judgments)),
        base_cases=bases, workflows=workflows, condition_summary=summary,
        successful_raw_json_parses=len(judgments), terminal_judge_errors=len(load_lines(HERE/"raw/judge_errors.jsonl")),
        uncertainty_status="No confidence intervals were computed. Source ci_low/ci_high are point-score placeholders; bootstrap_samples=0.",
        generation_model=None, generation_configuration=None, execution_dates=None,
        missing_metadata=["exact generation model", "generation/judge runtime settings", "historical source commit",
                          "run dates", "provider request IDs", "token usage", "complete retry counts", "random seeds"],
        limitations=["Five base cases, not 30 independent cases per condition; one saved trace per design cell.",
                     "Conditions are prompt manipulations: deemphasized includes corrective text in the actual input.",
                     "Workflow dependency graphs are prescribed; this does not validate recovered semantic-source graphs.",
                     "One LLM judge per final output; no human calibration or repeated-judge uncertainty.",
                     "Medical role templates were reused in nonmedical domains.",
                     "No new model execution, causal responsibility evaluation, or incentive experiment was performed."],
    )
    return result, joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "recomputed")
    args = parser.parse_args()
    result, joined = audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (args.output_dir/"per_run_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    with (args.output_dir/"condition_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["condition_summary"][0]))
        writer.writeheader()
        writer.writerows(result["condition_summary"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
