"""Reproduce numerical checks locally, with stdout, environment, and source hashes.

No network, API, GPU, model download, or paid model call is used by this script.
Install requirements before running. Saved loss tables are LP constructions.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced" / "local")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_arg = Path(os.path.relpath(out, ROOT))
    jobs = [
        ("legacy_stdout.txt", ["provenance/legacy_verify_span_bound.py"]),
        ("numerical_stdout.txt", ["verification/verify_span_bound.py", "--output-dir", str(output_arg)]),
        ("deterministic_stdout.txt", ["illustrative_example/deterministic_example.py", "--output", str(output_arg / "deterministic_results.json")]),
        ("tests.txt", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("source_access_stdout.json", ["archive/source_access/recompute.py", "--output-dir", str(output_arg / "source_access")]),
        ("graph_tests.txt", ["analysis/propagation_graphs/analyze.py", "--self-test"]),
        ("graph_stdout.json", ["analysis/propagation_graphs/analyze.py", "--output-dir", str(output_arg / "propagation_graphs")]),
        ("graph_figures.txt", ["analysis/propagation_graphs/plot_examples.py", "--input-dir", str(output_arg / "propagation_graphs"), "--output-dir", str(output_arg / "propagation_graphs/figures")]),
        ("figure_stdout.txt", ["illustrative_example/make_counterfactual_overview.py", "--output-dir", str(output_arg / "figures")]),
    ]
    statuses = []
    for logfile, command in jobs:
        process = subprocess.run([sys.executable, *command], cwd=ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / logfile).write_text(process.stdout)
        # Avoid recording the user's absolute local directory in a public log.
        statuses.append({"log": logfile, "returncode": process.returncode})
        print(f"{logfile}: return code {process.returncode}")
        if process.returncode:
            raise RuntimeError(f"Verification failed; inspect {out / logfile}")
    archived = [ast.literal_eval(line) for line in (ROOT / "verification/archived_results.txt").read_text().splitlines() if line.strip()]
    rerun = [ast.literal_eval(line) for line in (out / "legacy_stdout.txt").read_text().splitlines() if line.strip()]
    if len(archived) != len(rerun):
        raise RuntimeError("Archive comparison: unequal number of runs")
    maximum_difference = 0.0
    for old, new in zip(archived, rerun):
        if set(old) != set(new):
            raise RuntimeError("Archive comparison: unequal fields")
        for field in old:
            maximum_difference = max(maximum_difference, abs(old[field]-new[field]))
    if maximum_difference > 1e-8:
        raise RuntimeError("Archive comparison: discrepancy exceeds 1e-8")
    source_paths = ["reproduce.py", "verification/verify_span_bound.py",
                    "provenance/legacy_verify_span_bound.py",
                    "illustrative_example/deterministic_example.py",
                    "illustrative_example/make_counterfactual_overview.py",
                    "tests/test_verification.py", "configs/planned_experiments.json", "requirements.txt",
                    "archive/source_access/recompute.py", "archive/source_access/source_manifest.json",
                    "analysis/propagation_graphs/analyze.py", "analysis/propagation_graphs/plot_examples.py",
                    "analysis/propagation_graphs/data/source_manifest.json"]
    environment = {
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.system(), "machine": platform.machine(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "matplotlib")},
        "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source_paths},
        "jobs": statuses,
        "archive_comparison_max_absolute_difference": maximum_difference,
        "comparison_tolerance": 1e-8,
        "result_scope": "finite numerical constructions, exact toy arithmetic, offline H22 reaggregation and post hoc archived graph metrics; no new LLM runs",
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True)
    (out / "installed_packages.txt").write_text(freeze.stdout)
    print("Local numerical checks and historical-output reaggregation passed. No new LLM execution or independent empirical replication was performed.")


if __name__ == "__main__":
    main()
