"""Contract for the isolated GCN-v3 class-weight probe."""
import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_kipf_v3_class_weight_probe import train_ratio_class_weight  # noqa: E402


class ClassWeightContractTest(unittest.TestCase):
    def test_weight_uses_only_frozen_training_labels(self):
        labels = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0])
        self.assertEqual(train_ratio_class_weight(labels), [1.0, 3.0])

    def test_empty_anomaly_training_split_is_rejected(self):
        with self.assertRaises(ValueError):
            train_ratio_class_weight(torch.tensor([0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
