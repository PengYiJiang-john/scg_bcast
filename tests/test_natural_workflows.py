"""Offline archive validation and rejection of inconsistent replay records."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'natural_workflows'
sys.path.insert(0, str(ROOT))
from verify_archive import check_pair, verify
from inspect_case import load_case
from natural_bcast.data import TaskCase


class NaturalWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_case('hotpot:5a738fe855429908901be2fb')
        cls.case = TaskCase.from_dict(cls.record['input'])
        cls.graph = cls.record['semantic_graph']
        cls.pair = cls.record['replay_pairs'][0]

    def test_full_archive_without_network(self):
        with patch('socket.socket', side_effect=AssertionError('Network access is forbidden')):
            report = verify()
        self.assertEqual(report['cases'], 400)
        self.assertEqual(len(report['tasks']), 8)
        self.assertEqual(report['saved_replay_pairs'], 1183)
        self.assertEqual(report['endpoint_scores_checked'], 2766)

    def test_loss_tampering_rejected(self):
        row = copy.deepcopy(self.pair)
        row['marginal_loss'] += 0.25
        with self.assertRaisesRegex(ValueError, 'Marginal loss mismatch'):
            check_pair(self.case, self.graph, row)

    def test_preserved_target_rejected(self):
        row = copy.deepcopy(self.pair)
        row['target'] = row['fixed_preserved_ids'][0]
        with self.assertRaisesRegex(ValueError, 'NEW or TRANSFORMED'):
            check_pair(self.case, self.graph, row)

    def test_unknown_case_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown case'):
            load_case('missing:case')


if __name__ == '__main__':
    unittest.main()
