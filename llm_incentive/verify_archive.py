"""Recompute the replacement experiment from saved traces without model calls."""

import argparse
import ast
import gzip
import hashlib
import subprocess
import sys
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
POSITIONS = ("A", "B", "C")
MODES = ("careful", "shortcut", "distorter")
TARGETS = dict(zip(POSITIONS, ("facts", "analysis", "final")))
COALITIONS = [frozenset(c) for k in range(4) for c in itertools.combinations(POSITIONS, k)]
PROFILES = list(itertools.product(MODES, repeat=3))

# Read literal scoring constants without importing the API runner.
constants = {}
for node in ast.parse((ROOT / "protocol.py").read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {
            "SOURCE", "GOLD_FACTS", "GOLD_ANALYSIS", "GOLD_FINAL"
        }:
            constants[target.id] = ast.literal_eval(node.value)


def canonical(value):
    text = "_".join(str(value).strip().upper().replace("$", "").replace(",", "").split())
    return {
        "SUPPLIER_A": "A", "SUPPLIER_B": "B", "WITHIN_BUDGET": "WITHIN",
        "TRUE": "YES", "FALSE": "NO", "YES.": "YES", "NO.": "NO",
    }.get(text, text)


def correct(value, gold):
    value, gold = canonical(value), canonical(gold)
    if value == gold:
        return True
    try:
        value, gold = float(value), float(gold)
    except ValueError:
        return False
    return abs(value - gold) <= max(1e-8, abs(gold) * 0.005)


def verify_saved(data_dir=ROOT / "data"):
    with gzip.open(data_dir / "profile_game_traces.json.gz", "rt", encoding="utf-8") as handle:
        traces = json.load(handle)
    assert len(traces) == 648
    seen = set()
    prefixes = {}
    losses = defaultdict(list)
    for row in traces:
        profile = tuple(row["profile"].split("|"))
        coalition = frozenset() if row["coalition"] == "EMPTY" else frozenset(row["coalition"])
        key = profile, coalition, row["repeat"]
        assert key not in seen
        seen.add(key)
        state = {
            "task_id": "procurement-smoke-01", "raw_source": constants["SOURCE"],
            "facts": {}, "analysis": {}, "final": {},
        }
        assert len(row["events"]) == 3
        for index, event in enumerate(row["events"]):
            position = POSITIONS[index]
            assert event["position"] == position and event["mode"] == profile[index]
            keys = {
                "A": ("task_id", "raw_source"), "B": ("task_id", "facts"),
                "C": ("task_id", "facts", "analysis"),
            }[position]
            observed = {k: state[k] for k in keys}
            assert event["observed_input"] == observed
            if position not in coalition:
                assert event["action"] == "identity"
                assert event["generated_node"] is None and event["state_after"] == state
            else:
                assert event["action"] == "transform"
                expected = dict(state)
                expected[TARGETS[position]] = event["generated_node"]
                assert event["state_after"] == expected
                prefix = position, profile[index], row["repeat"], json.dumps(observed, sort_keys=True)
                output = json.dumps(event["generated_node"], sort_keys=True)
                assert prefixes.setdefault(prefix, output) == output
            state = event["state_after"]
        scores = {
            f"{field}.{name}": correct(state[field].get(name, "UNKNOWN"), gold)
            for field, gold_name in (("facts", "GOLD_FACTS"), ("analysis", "GOLD_ANALYSIS"), ("final", "GOLD_FINAL"))
            for name, gold in constants[gold_name].items()
        }
        assert scores == row["slot_correctness"] and len(scores) == 20
        loss = 1 - sum(scores.values()) / 20
        assert abs(loss - row["loss"]) < 1e-12
        losses[profile, coalition].append(loss)

    assert seen == set(itertools.product(PROFILES, COALITIONS, range(3)))
    values = {key: sum(samples) / len(samples) for key, samples in losses.items()}
    penalties, potentials = {}, {}
    for profile in PROFILES:
        for position in POSITIONS:
            penalties[profile, position] = sum(
                math.factorial(len(u)) * math.factorial(2 - len(u)) / 6
                * (values[profile, u | {position}] - values[profile, u])
                for u in COALITIONS if position not in u
            )
        potentials[profile] = sum(
            math.factorial(len(u) - 1) * math.factorial(3 - len(u)) / 6 * values[profile, u]
            for u in COALITIONS if u
        )

    full = frozenset(POSITIONS)
    efficiency_error = max(abs(sum(penalties[p, i] for i in POSITIONS) - values[p, full] + values[p, frozenset()]) for p in PROFILES)
    potential_error = 0.0
    equilibria = []
    for profile in PROFILES:
        stable = True
        for index, position in enumerate(POSITIONS):
            for mode in MODES:
                alternative = profile[:index] + (mode,) + profile[index + 1:]
                delta = penalties[alternative, position] - penalties[profile, position]
                potential_error = max(potential_error, abs(delta - potentials[alternative] + potentials[profile]))
                stable = stable and delta >= -1e-10
        if stable:
            equilibria.append(profile)

    global_loss = min(values[p, full] for p in PROFILES)
    optima = [p for p in PROFILES if abs(values[p, full] - global_loss) < 1e-10]
    endpoints = {}
    for start in PROFILES:
        profile = start
        for _ in range(20):
            changed = False
            for index, position in enumerate(POSITIONS):
                options = [profile[:index] + (mode,) + profile[index + 1:] for mode in MODES]
                best = min(options, key=lambda p: penalties[p, position])
                if penalties[best, position] < penalties[profile, position] - 1e-10:
                    profile, changed = best, True
            if not changed:
                break
        else:
            raise AssertionError("Best response did not terminate")
        endpoints["|".join(start)] = "|".join(profile)

    with (data_dir / "profile_game.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 27
    for row in rows:
        p = tuple(row[f"mode_{i}"] for i in POSITIONS)
        assert (row["is_equilibrium"] == "True") == (p in equilibria)
        assert (row["is_global_optimum"] == "True") == (p in optima)
        assert abs(float(row["full_loss"]) - values[p, full]) < 1e-12
        assert abs(float(row["potential"]) - potentials[p]) < 1e-12
        for i in POSITIONS:
            assert abs(float(row[f"psi_{i}"]) - penalties[p, i]) < 1e-12

    saved = json.loads((data_dir / "profile_game_summary.json").read_text())
    assert saved["equilibria"] == ["|".join(p) for p in equilibria]
    assert saved["global_optimum_profiles"] == ["|".join(p) for p in optima]
    assert saved["best_response_endpoints"] == endpoints
    assert equilibria == optima == [("careful",) * 3]
    assert efficiency_error < 1e-14 and potential_error < 1e-14

    report = {
        "saved_replays": len(traces), "profiles": len(PROFILES), "subsets": len(COALITIONS),
        "repeats": 3, "scored_nodes": 20, "model_calls_made_by_verification": 0,
        "identity_violations": 0, "ownership_violations": 0, "prefix_consistency_violations": 0,
        "score_mismatches": 0, "csv_mismatches": 0,
        "max_shapley_efficiency_error": efficiency_error,
        "max_potential_deviation_error": potential_error,
        "equilibria": ["|".join(p) for p in equilibria], "global_optimum_loss": global_loss,
        "convergent_initial_profiles": sum(p == "careful|careful|careful" for p in endpoints.values()),
        "homogeneous_losses": {m: values[(m,) * 3, full] for m in MODES},
    }
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "reproduced/local/llm_incentive")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for item in manifest["files"]:
        content = (ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"], item["path"]
    report = verify_saved()
    subprocess.run([sys.executable, str(ROOT / "run_experiment.py"), "--output-dir", str(out / "results")],
                   check=True, stdout=subprocess.PIPE, text=True)
    subprocess.run([sys.executable, str(ROOT / "analyze_paper_results.py"), "--results-dir",
                    str(out / "results"), "--output-dir", str(out / "paper")], check=True)
    with gzip.open(ROOT / "data/profile_game_traces.json.gz", "rt", encoding="utf-8") as handle:
        expected = json.load(handle)
    with gzip.open(out / "results/profile_game_traces.json.gz", "rt", encoding="utf-8") as handle:
        actual = json.load(handle)
    assert actual == expected, "Offline replay differs from the recorded state transitions"
    import pandas as pd
    for saved in [ROOT / "data/profile_game.csv", *sorted((ROOT / "paper").glob("*.csv"))]:
        produced = out / ("results" if saved.parent.name == "data" else "paper") / saved.name
        pd.testing.assert_frame_equal(pd.read_csv(saved), pd.read_csv(produced),
                                      check_dtype=False, rtol=1e-10, atol=1e-12)
    report["offline_replay_matches_archive"] = True
    report["summary_tables_match_archive"] = True
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
