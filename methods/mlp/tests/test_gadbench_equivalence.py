"""Numerical equivalence check against the frozen GADBench MLP reference."""

import importlib.util
import os
import sys
import unittest

import torch


MLP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(MLP_ROOT, "src")
REPO_ROOT = os.path.normpath(os.path.join(MLP_ROOT, "..", ".."))
REFERENCE_FILE = os.path.join(REPO_ROOT, "audit", "mlp_reference", "GADBench", "models", "gnn.py")
sys.path.insert(0, SRC_DIR)

from model import FeatureMLP


def load_reference_mlp():
    spec = importlib.util.spec_from_file_location("gadbench_gnn_reference", REFERENCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLP


class GADBenchEquivalenceTest(unittest.TestCase):
    def test_two_layer_feature_mode_matches_reference_logits_in_eval_mode(self):
        torch.manual_seed(20260813)
        reference_class = load_reference_mlp()
        reference = reference_class(
            in_feats=7, h_feats=64, num_classes=2, num_layers=2,
            dropout_rate=0.5, activation="ReLU",
        ).eval()
        current = FeatureMLP(input_dim=7, hidden_dim=64, dropout=0.5).eval()
        with torch.no_grad():
            current.network[0].weight.copy_(reference.layers[0].weight)
            current.network[0].bias.copy_(reference.layers[0].bias)
            current.network[3].weight.copy_(reference.layers[1].weight)
            current.network[3].bias.copy_(reference.layers[1].bias)
        feature = torch.randn(13, 7)
        torch.testing.assert_close(current(feature), reference(feature, is_graph=False), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
