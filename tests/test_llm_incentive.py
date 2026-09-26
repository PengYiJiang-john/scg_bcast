"""Offline checks for the controlled semantic-transformation game."""
import gzip
import json
from itertools import permutations, product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_incentive.protocol import (
    POSITIONS, MODES, GOLD_FACTS, GOLD_ANALYSIS, GOLD_FINAL,
    initial_state, visible_state, terminal_loss, slot_correct, shapley_loss,
)
from llm_incentive.verify_archive import verify_saved


class ArchivedIncentiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ROOT / "llm_incentive/data/profile_game_traces.json.gz", "rt") as handle:
            cls.traces = json.load(handle)

    def test_complete_archive_rescoring_and_equilibria_without_network(self):
        with patch("socket.socket", side_effect=AssertionError("Network use forbidden")):
            report = verify_saved()
        self.assertEqual(report["saved_replays"], 648)
        self.assertEqual(report["equilibria"], ["careful|careful|careful"])
        self.assertEqual(report["convergent_initial_profiles"], 27)
        self.assertEqual(report["identity_violations"], 0)
        self.assertEqual(report["prefix_consistency_violations"], 0)

    def test_cancelled_position_mode_does_not_change_replay(self):
        for index, position in enumerate(POSITIONS):
            grouped = {}
            for row in self.traces:
                subset = row["coalition"]
                if subset != "EMPTY" and position in subset:
                    continue
                profile = row["profile"].split("|")
                key = tuple(m for j, m in enumerate(profile) if j != index), subset, row["repeat"]
                final_state = row["events"][-1]["state_after"]
                if key in grouped:
                    self.assertEqual(grouped[key], final_state)
                grouped[key] = final_state

    def test_shapley_matches_permutation_formula(self):
        values = {}
        for row in self.traces:
            profile = tuple(row["profile"].split("|"))
            subset = frozenset() if row["coalition"] == "EMPTY" else frozenset(row["coalition"])
            values.setdefault(profile, {}).setdefault(subset, []).append(row["loss"])
        for profile in product(MODES, repeat=3):
            losses = {u: sum(samples) / len(samples) for u, samples in values[profile].items()}
            exact = shapley_loss(losses)
            permuted = dict.fromkeys(POSITIONS, 0.0)
            for order in permutations(POSITIONS):
                active = frozenset()
                for position in order:
                    extended = active | {position}
                    permuted[position] += (losses[extended] - losses[active]) / 6
                    active = extended
            for position in POSITIONS:
                self.assertAlmostEqual(exact[position], permuted[position], places=12)
            self.assertAlmostEqual(sum(exact.values()), losses[frozenset(POSITIONS)] - losses[frozenset()], places=12)

    def test_all_careful_subset_losses(self):
        expected = {"EMPTY": 1.0, "A": .6, "B": 1.0, "C": 1.0, "AB": .05, "AC": .6, "BC": 1.0, "ABC": 0.0}
        for row in self.traces:
            if row["profile"] == "careful|careful|careful":
                self.assertAlmostEqual(row["loss"], expected[row["coalition"]], places=12)


class ProtocolTests(unittest.TestCase):
    def test_visibility_and_field_ownership(self):
        state = initial_state()
        self.assertEqual(set(visible_state(state, "A")), {"task_id", "raw_source"})
        self.assertEqual(set(visible_state(state, "B")), {"task_id", "facts"})
        self.assertEqual(set(visible_state(state, "C")), {"task_id", "facts", "analysis"})
        with self.assertRaises(ValueError):
            visible_state(state, "D")

    def test_scoring_unique_nodes_and_tolerance(self):
        state = initial_state()
        self.assertEqual(terminal_loss(state)[0], 1.0)
        state["facts"] = dict(GOLD_FACTS)
        state["analysis"] = dict(GOLD_ANALYSIS)
        state["final"] = dict(GOLD_FINAL)
        loss, scores = terminal_loss(state)
        self.assertEqual(len(scores), 20)
        self.assertEqual(loss, 0.0)
        self.assertTrue(slot_correct("100.5", "100"))
        self.assertFalse(slot_correct("100.51", "100"))
        state["final"]["budget"] = "wrong copied value"
        self.assertEqual(terminal_loss(state)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
