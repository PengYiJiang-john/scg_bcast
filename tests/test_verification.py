"""Independent arithmetic/feasibility checks for the released numerical artifacts."""
from fractions import Fraction
from itertools import permutations, product
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verification.verify_span_bound import FiniteTable, solve
from illustrative_example.deterministic_example import PLAYERS, State, loss, powerset, replay, results


def permutation_penalty(a, i, table):
    """Compute Shapley directly from all orders and raw table lookups."""
    values = []
    for order in permutations(range(len(a))):
        key = [-1] * len(a)
        for player in order:
            before = table[tuple(key)]
            key[player] = a[player]
            after = table[tuple(key)]
            if player == i:
                values.append(after-before)
                break
    return sum(values)/len(values)


class LPWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [solve(n, eta) for n, eta in ((2, .02), (3, .02), (4, .02), (4, 0.0))]

    def test_archived_objectives_and_complete_feasible_witnesses(self):
        for result in self.runs:
            summary = result["summary"]
            n, eta = summary["n"], summary["eta"]
            with self.subTest(n=n, eta=eta):
                self.assertAlmostEqual(summary["max_designated_profile_gap"], n*eta, places=8)
                self.assertAlmostEqual(summary["max_selected_same_mode_penalty_change"], eta, places=8)
                for field in ("gap_lp_loss", "penalty_change_lp_loss"):
                    values = {tuple(row["key"]): row[field] for row in result["witness_tables"]}
                    self.assertEqual(len(values), 3**n)
                    self.assertAlmostEqual(values[(-1,)*n], .8)
                    self.assertTrue(all(-1e-8 <= value <= 1+1e-8 for value in values.values()))
                    # Independent direct-table form of every fixed-profile span constraint.
                    for a in product((0, 1), repeat=n):
                        for i in range(n):
                            margins = []
                            for mask in range(1 << n):
                                if mask & (1 << i):
                                    continue
                                key = tuple(a[k] if mask & (1 << k) else -1 for k in range(n))
                                retained = tuple(a[k] if k == i else key[k] for k in range(n))
                                margins.append(values[retained]-values[key])
                            self.assertLessEqual(max(margins)-min(margins), eta+1e-8)

    def test_equilibrium_and_shapley_against_permutation_definition(self):
        for result in self.runs:
            n = result["summary"]["n"]
            table = FiniteTable(n)
            values = {tuple(row["key"]): row["gap_lp_loss"] for row in result["witness_tables"]}
            vector = np.array([values[key] for key in table.keys])
            for a in table.profiles:
                direct = [permutation_penalty(a, i, values) for i in range(n)]
                for i in range(n):
                    self.assertAlmostEqual(direct[i], float(table.penalty(a, i) @ vector), places=10)
                self.assertAlmostEqual(sum(direct), values[a]-values[(-1,)*n], places=10)
            zero = (0,)*n
            for i in range(n):
                alt = tuple(1 if k == i else 0 for k in range(n))
                self.assertLessEqual(permutation_penalty(zero, i, values),
                                     permutation_penalty(alt, i, values)+1e-8)

    def test_potential_identity_coefficients_on_complete_domain(self):
        # Check coefficients, independent of a favorable witness table.
        for n in (2, 3, 4):
            table = FiniteTable(n)
            for a in table.profiles:
                for i in range(n):
                    alt = tuple(1-a[k] if k == i else a[k] for k in range(n))
                    residual = (table.potential(alt)-table.potential(a)
                                -table.penalty(alt, i)+table.penalty(a, i))
                    self.assertLess(float(np.max(np.abs(residual))), 1e-12)


class DeterministicExampleTests(unittest.TestCase):
    def test_transitions_preserve_sibling_semantics_and_realize_table(self):
        for retained in powerset(PLAYERS):
            terminal = replay(retained)
            self.assertEqual(terminal.evidence, State().evidence)
            self.assertEqual(terminal.source, State().source)
            self.assertEqual(loss(retained), int(retained == frozenset(("H", "B"))))
        self.assertEqual(replay(set()), State())

    def test_exact_two_layer_and_missing_domain_values(self):
        output = results()
        self.assertEqual(output["behavioral_shapley"], {"H": "1/6", "R": "-1/3", "B": "1/6"})
        self.assertEqual(output["leave_one_out"], {"H": 0, "R": -1, "B": 0})
        self.assertEqual(output["agent_ssv"], {"A": "1/18", "B": "-1/9", "C": "1/18"})
        self.assertEqual(output["coverage"], {"H": "5/6", "R": "2/3", "B": "5/6"})
        self.assertEqual(output["conditional_means"], {"H": "0", "R": "0", "B": "0"})
        self.assertEqual(output["missing_table_coordinate_intervals"],
                         {"H": ["0", "1/6"], "R": ["-1/3", "0"], "B": ["0", "1/6"]})


class HistoricalSourceAccessTests(unittest.TestCase):
    def test_raw_judge_responses_join_and_reproduce_all_means(self):
        from archive.source_access.recompute import audit
        result, rows = audit()
        self.assertEqual(len(rows), 120)
        self.assertEqual(result["saved_generation_step_outputs"], 720)
        self.assertEqual(result["judge_models"], {"models/gemini-2.5-pro": 120})
        self.assertIsNone(result["generation_model"])


if __name__ == "__main__":
    unittest.main()
