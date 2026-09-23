"""Contract for the isolated T-Social 10-wide GAT-v2 smoke model."""
import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_tsocial_smoke import TSocialGADBenchGATV2  # noqa: E402
from run_tsocial_smoke import is_cuda_oom, protected_tsocial_history_roots  # noqa: E402


class TSocialHidden10ContractTest(unittest.TestCase):
    def test_two_heads_of_five_preserve_total_width_ten(self):
        model = TSocialGADBenchGATV2(input_dim=10, hidden_total_dim=10, num_heads=2)
        self.assertEqual(model.hidden_total_dim, 10)
        self.assertEqual(model.num_heads, 2)
        self.assertEqual(model.per_head_dim, 5)
        self.assertEqual(model.output_linear.in_features, 10)
        self.assertEqual(model.output_linear.out_features, 2)

    def test_rejects_any_width_or_head_count_other_than_frozen_tsocial_contract(self):
        with self.assertRaises(ValueError):
            TSocialGADBenchGATV2(input_dim=10, hidden_total_dim=64, num_heads=4)

    def test_protection_manifest_covers_all_preexisting_gat_v2_dataset_roots(self):
        roots = protected_tsocial_history_roots()
        for dataset in ("weibo", "amazon", "yelp", "tolokers", "tfinance"):
            self.assertIn(f"results/experiments/gat_v2_gadbench/{dataset}", roots)

    def test_dgl_cuda_oom_is_classified_as_oom_not_generic_error(self):
        self.assertTrue(is_cuda_oom(RuntimeError("DGL CUDA: out of memory")))
        self.assertFalse(is_cuda_oom(RuntimeError("invalid graph type")))


if __name__ == "__main__":
    unittest.main()
