"""Checks independent of the released equilibrium-summary calculations."""
from itertools import permutations, product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_incentive.run_experiment import (
    AGENTS, MODES, SLOTS, aggregate, all_cases, loss, mode_quality,
    shapley_penalties, validate_outputs,
)


class ArchivedIncentiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = ROOT / "llm_incentive/results_full"
        cls.outputs = pd.read_csv(cls.base / "llm_outputs.csv", keep_default_na=False, dtype={s: str for s in SLOTS})
        cls.coalitions = pd.read_csv(cls.base / "coalition_case_losses.csv")
        cls.payoffs = pd.read_csv(cls.base / "profile_payoffs.csv")

    def test_complete_archive_and_rescoring_without_network(self):
        validate_outputs(self.outputs, all_cases(), 3)
        self.assertEqual(len(self.outputs), 810)
        with patch("socket.socket", side_effect=AssertionError("Network use forbidden")):
            recomputed = mode_quality(self.outputs, all_cases())
        archived = pd.read_csv(self.base / "mode_quality.csv")
        self.assertEqual(len(recomputed), 810)
        for a, b in zip(recomputed.standalone_loss, archived.standalone_loss):
            self.assertAlmostEqual(a, b, places=12)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_outputs(pd.concat([self.outputs, self.outputs.iloc[:1]]), all_cases(), 3)
        with self.assertRaisesRegex(ValueError, "complete declared design"):
            validate_outputs(self.outputs.iloc[1:], all_cases(), 3)

    def test_cancellation_reference_independent_of_cancelled_mode(self):
        self.assertEqual(len(self.coalitions), 58320)
        full_groups = self.coalitions.groupby(["environment", "profile", "coalition"])
        self.assertEqual(len(full_groups), 3 * 27 * 8)
        self.assertTrue((full_groups.size() == 90).all())
        # Compare all case/repeat values when only an absent agent's mode changes.
        for agent_index, agent in enumerate(AGENTS):
            subset = self.coalitions[~self.coalitions.coalition.str.contains(agent)]
            subset = subset.copy()
            subset["other_modes"] = subset.profile.map(lambda p: "|".join(m for i, m in enumerate(p.split("|")) if i != agent_index))
            groups = subset.groupby(["environment", "other_modes", "coalition", "case_id", "repeat"])
            self.assertTrue((groups.loss.nunique() == 1).all())

    def test_shapley_and_equilibria_from_permutations(self):
        means = self.coalitions.groupby(["environment", "profile", "coalition"]).loss.mean().to_dict()
        for environment, records in self.payoffs.groupby("environment"):
            penalties = {}
            for row in records.itertuples(index=False):
                profile = (row.mode_A, row.mode_B, row.mode_C)
                profile_name = "|".join(profile)
                sums = dict.fromkeys(AGENTS, 0.0)
                for order in permutations(AGENTS):
                    active = set()
                    previous = means[(environment, profile_name, "EMPTY")]
                    for agent in order:
                        active.add(agent)
                        current = means[(environment, profile_name, "".join(sorted(active)))]
                        sums[agent] += (current - previous) / 6
                        previous = current
                penalties[profile] = sums
                self.assertAlmostEqual(sum(sums.values()), row.full_loss - 1, places=12)
                for agent in AGENTS:
                    self.assertAlmostEqual(sums[agent], getattr(row, "psi_" + agent), places=12)
            equilibria = []
            for profile in product(MODES, repeat=3):
                stable = True
                for index, agent in enumerate(AGENTS):
                    for mode in MODES:
                        alternative = tuple(mode if i == index else m for i, m in enumerate(profile))
                        if penalties[alternative][agent] < penalties[profile][agent] - 1e-12:
                            stable = False
                if stable:
                    equilibria.append(profile)
            self.assertEqual(equilibria, [("careful",) * 3])


class ControlledRuleTests(unittest.TestCase):
    def test_aggregation_tie_and_gatekeeper_removal(self):
        outputs = {a: {slot: value for slot in SLOTS} for a, value in zip(AGENTS, ("YES", "NO", "NO"))}
        self.assertEqual(set(aggregate(outputs, frozenset(("A", "B")), "majority").values()), {"UNKNOWN"})
        self.assertEqual(set(aggregate(outputs, frozenset(AGENTS), "majority").values()), {"NO"})
        self.assertEqual(set(aggregate(outputs, frozenset(("A",)), "gatekeeper").values()), {"YES"})
        self.assertEqual(set(aggregate(outputs, frozenset(), "gatekeeper").values()), {"UNKNOWN"})

    def test_missing_slots_and_numeric_tolerance(self):
        gold = dict.fromkeys(SLOTS, "100")
        self.assertEqual(loss({}, gold), 1.0)
        self.assertEqual(loss(dict.fromkeys(SLOTS, "100.5"), gold), 0.0)
        self.assertEqual(loss(dict.fromkeys(SLOTS, "100.51"), gold), 1.0)


if __name__ == "__main__":
    unittest.main()
