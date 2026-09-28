import os
import sys
import unittest

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from model import FeatureMLP


class FeatureMLPTest(unittest.TestCase):
    def test_feature_only_model_produces_two_class_logits(self):
        model = FeatureMLP(input_dim=3, hidden_dim=64, dropout=0.5)
        logits = model(torch.zeros(4, 3))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 386)


if __name__ == "__main__":
    unittest.main()
