"""Finite cancellation-table LP checks; no empirical LLM evaluation.

This reproduces the archived fixed-profile coalition-span construction, not
verification of the manuscript's stronger cross-mode stability theorem.
The optimizer constructs loss tables. They are not observed task data.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

TOLERANCE = 1e-8


class FiniteTable:
    def __init__(self, n):
        if not isinstance(n, int) or n < 2 or n > 4:
            raise ValueError("This bounded verification supports n = 2, 3, 4.")
        self.n = n
        self.keys = list(itertools.product((-1, 0, 1), repeat=n))
        self.index = {key: j for j, key in enumerate(self.keys)}
        self.profiles = list(itertools.product((0, 1), repeat=n))
        self.nv = len(self.keys)
        self.full = (1 << n) - 1

    def value(self, a, mask):
        vec = np.zeros(self.nv)
        key = tuple(a[i] if mask & (1 << i) else -1 for i in range(self.n))
        vec[self.index[key]] = 1
        return vec

    def marginal(self, a, i, mask):
        return self.value(a, mask | (1 << i)) - self.value(a, mask)

    def penalty(self, a, i):
        vec = np.zeros(self.nv)
        for mask in range(1 << self.n):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(self.n-k-1) / math.factorial(self.n)
            vec += weight * self.marginal(a, i, mask)
        return vec

    def potential(self, a):
        vec = np.zeros(self.nv)
        for mask in range(1, 1 << self.n):
            k = mask.bit_count()
            weight = math.factorial(k-1) * math.factorial(self.n-k) / math.factorial(self.n)
            vec += weight * (self.value(a, mask) - self.value(a, 0))
        return vec


def require(condition, message):
    # Unlike assert, these release checks are not disabled by python -O.
    if not condition:
        raise RuntimeError(message)


def solve(n, eta):
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    table = FiniteTable(n)
    rows = []
    for a in table.profiles:
        for i in range(n):
            ds = [table.marginal(a, i, s) for s in range(1 << n) if not s & (1 << i)]
            rows.extend(x-y for j, x in enumerate(ds) for k, y in enumerate(ds) if j != k)
    constraints = csr_matrix(np.array(rows))
    rhs = np.full(len(rows), eta)
    bounds = [(0, 1)] * table.nv
    bounds[table.index[(-1,) * n]] = (0.8, 0.8)
    zero, one = (0,) * n, (1,) * n
    deviations = []
    for i in range(n):
        alt = tuple(1 if k == i else 0 for k in range(n))
        deviations.append(table.penalty(zero, i) - table.penalty(alt, i))
    ne_constraints = vstack([constraints, csr_matrix(deviations)])
    ne_rhs = np.concatenate([rhs, np.zeros(n)])
    gap_obj = table.value(zero, table.full) - table.value(one, table.full)
    res = linprog(-gap_obj, A_ub=ne_constraints, b_ub=ne_rhs, bounds=bounds, method="highs")
    require(res.success, f"Gap LP failed: {res.message}")

    # This second optimization constrains coalition spans only; no NE constraints.
    other = (0,) + (1,) * (n-1)
    env_obj = table.penalty(other, 0) - table.penalty(zero, 0)
    env = linprog(-env_obj, A_ub=constraints, b_ub=rhs, bounds=bounds, method="highs")
    require(env.success, f"Penalty-change LP failed: {env.message}")

    potential_residual = efficiency_residual = 0.0
    for a in table.profiles:
        efficiency_residual = max(efficiency_residual, abs(float(
            (sum((table.penalty(a, i) for i in range(n)), start=np.zeros(table.nv))
             - table.value(a, table.full) + table.value(a, 0)) @ res.x)))
        for i in range(n):
            alt = tuple(1-a[k] if k == i else a[k] for k in range(n))
            residual = (table.potential(alt)-table.potential(a)
                        -table.penalty(alt, i)+table.penalty(a, i)) @ res.x
            potential_residual = max(potential_residual, abs(float(residual)))

    def feasibility(result, matrix, bound):
        violations = [0.0, float(np.max(matrix @ result.x - bound)),
                      float(-np.min(result.x)), float(np.max(result.x)-1),
                      abs(float(result.x[table.index[(-1,) * n]])-0.8)]
        return max(violations)

    max_gap = float(gap_obj @ res.x)
    penalty_change = float(env_obj @ env.x)
    losses = [float(table.value(a, table.full) @ res.x) for a in table.profiles]
    ne_loss = float(table.value(zero, table.full) @ res.x)
    min_loss = min(losses)
    summary = dict(
        n=n, eta=eta, variables=table.nv,
        span_constraint_rows=len(rows), equilibrium_constraint_rows=n,
        max_designated_profile_gap=max_gap,
        legacy_comparison_n_eta=n*eta,
        max_selected_same_mode_penalty_change=penalty_change,
        designated_equilibrium_loss=ne_loss,
        designated_comparator_loss=float(table.value(one, table.full) @ res.x),
        witness_min_profile_loss=min_loss,
        witness_gap_to_best_profile=ne_loss-min_loss,
        witness_best_profiles=[list(a) for a, loss in zip(table.profiles, losses)
                               if abs(loss-min_loss) <= TOLERANCE],
        gap_lp_max_feasibility_violation=feasibility(res, ne_constraints, ne_rhs),
        penalty_lp_max_feasibility_violation=feasibility(env, constraints, rhs),
        potential_residual=potential_residual,
        efficiency_residual=efficiency_residual,
        gap_solver_status=int(res.status), penalty_solver_status=int(env.status),
        scope="constructed finite tables; fixed-profile coalition spans; first-layer penalties",
    )
    require(max_gap <= n*eta+TOLERANCE, "Legacy designated-gap check failed")
    require(penalty_change <= eta+TOLERANCE, "Selected penalty-change check failed")
    for key in ("gap_lp_max_feasibility_violation", "penalty_lp_max_feasibility_violation",
                "potential_residual", "efficiency_residual"):
        require(summary[key] < TOLERANCE, f"Failed {key}: {summary[key]}")
    witnesses = [dict(key=list(key), gap_lp_loss=float(res.x[j]),
                      penalty_change_lp_loss=float(env.x[j])) for j, key in enumerate(table.keys)]
    return dict(summary=summary, witness_tables=witnesses)


def run(n, eta):
    """Compatibility entry point, with accurate output labels."""
    result = solve(n, eta)
    print(json.dumps(result["summary"], sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Save summaries and both complete optimizer-generated witness tables")
    args = parser.parse_args()
    results = [run(n, 0.02) for n in (2, 3, 4)] + [run(4, 0.0)]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "numerical_results.json").write_text(
            json.dumps({"result_kind": "constructed_LP_tables_not_LLM_results",
                        "tolerance": TOLERANCE, "runs": results}, indent=2)+"\n")
        with (args.output_dir / "numerical_summary.csv").open("w", newline="") as handle:
            fields = ["n", "eta", "variables", "max_designated_profile_gap",
                      "legacy_comparison_n_eta", "max_selected_same_mode_penalty_change",
                      "witness_gap_to_best_profile", "potential_residual", "efficiency_residual"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for result in results:
                writer.writerow({field: result["summary"][field] for field in fields})


if __name__ == "__main__":
    main()
