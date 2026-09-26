"""Rebuild all released incentive tables from the 810 stored outputs, offline.

New model generation is never requested. Numeric CSV/JSON fields are compared
within 1e-12; labels, row order, shape and discrete values must match exactly.
Rendered image/PDF bytes are not compared because font/render metadata can vary.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-12


def compare_csv(expected: Path, actual: Path) -> float:
    old = pd.read_csv(expected, keep_default_na=False, dtype=str)
    new = pd.read_csv(actual, keep_default_na=False, dtype=str)
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        raise AssertionError(f"CSV shape/columns differ: {expected.name}")
    maximum = 0.0
    for column in old:
        a, b = old[column], new[column]
        if a.equals(b):
            continue
        an = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
        bn = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(an).all() and np.isfinite(bn).all()):
            raise AssertionError(f"Non-numeric fields differ: {expected.name}:{column}")
        difference = float(np.max(np.abs(an - bn), initial=0))
        if difference > TOLERANCE:
            raise AssertionError(f"Numeric mismatch: {expected.name}:{column} ({difference:g})")
        maximum = max(maximum, difference)
    return maximum


def compare_json(old, new, location="summary") -> float:
    if isinstance(old, bool) or isinstance(old, str) or old is None:
        if old != new or type(old) is not type(new):
            raise AssertionError(f"Metadata differs: {location}")
        return 0.0
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if not math.isfinite(old) or not math.isfinite(new):
            raise AssertionError(f"Nonfinite numeric value: {location}")
        difference = abs(old-new)
        if difference > TOLERANCE:
            raise AssertionError(f"Numeric metadata differs: {location}")
        return difference
    if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
        return max((compare_json(old[k], new[k], f"{location}.{k}") for k in old), default=0.0)
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return max((compare_json(a, b, f"{location}[{i}]") for i, (a, b) in enumerate(zip(old, new))), default=0.0)
    raise AssertionError(f"Metadata structure differs: {location}")


def verify(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results, paper = output_dir / "results", output_dir / "paper"
    for log, command in (
        ("recompute_stdout.json", ["run_experiment.py", "--output-dir", str(results)]),
        ("tables_stdout.txt", ["analyze_paper_results.py", "--results-dir", str(results), "--output-dir", str(paper)]),
    ):
        process = subprocess.run([sys.executable, *command], cwd=ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output_dir / log).write_text(process.stdout, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"Offline computation failed; see {log}")
    comparisons = {}
    for directory, rebuilt in ((ROOT / "results_full", results), (ROOT / "paper", paper)):
        for expected in sorted(directory.glob("*.csv")):
            comparisons[str(expected.relative_to(ROOT))] = compare_csv(expected, rebuilt / expected.name)
    comparisons["results_full/summary.json"] = compare_json(
        json.loads((ROOT / "results_full/summary.json").read_text()),
        json.loads((results / "summary.json").read_text()),
    )
    result = {
        "status": "all_archived_tables_match",
        "stored_structured_outputs": 810,
        "recomputed_coalition_losses": 58320,
        "comparison_tolerance": TOLERANCE,
        "maximum_absolute_difference": max(comparisons.values()),
        "comparisons": comparisons,
        "scope": "Offline reconstruction from author-supplied saved responses; no new model calls or independent generation replication",
    }
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "reproduced/local/llm_incentive")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
