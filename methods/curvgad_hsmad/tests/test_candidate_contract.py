import inspect
import unittest

import dgl
import torch


class CurvGADCandidateContract(unittest.TestCase):
    def test_balanced_mixed_manifold_config_totals_hsmad_hidden64(self):
        from methods.curvgad_hsmad.src.model import balanced_mixed_manifold_config

        config = balanced_mixed_manifold_config(64)
        self.assertEqual(config, "H22S21E21")
        self.assertEqual(sum(int(value) for value in ("22", "21", "21")), 64)

    def test_official_model_is_reused_and_forward_accepts_graph_only(self):
        from methods.curvgad_hsmad.src.model import build_official_model

        model = build_official_model(in_feats=3, hidden_dim=64, num_classes=2, k=3, dropout=0.0)
        signature = inspect.signature(model.forward)
        self.assertEqual(list(signature.parameters), ["graph"])
        graph = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])), num_nodes=3)
        graph.ndata["feature"] = torch.randn(3, 3)
        logits = model(graph)
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(model.total_dim, 64)
        self.assertEqual(model.K, 3)

    def test_masked_protocol_separates_train_validation_and_test(self):
        from methods.curvgad_hsmad.src.protocol import masked_train_loss, test_values, validation_values

        logits = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        train = torch.tensor([True, True, False, False, False, False])
        val = torch.tensor([False, False, True, True, False, False])
        test = torch.tensor([False, False, False, False, True, True])
        loss = masked_train_loss(logits, labels, train)
        expected = torch.nn.functional.cross_entropy(logits[train], labels[train])
        self.assertTrue(torch.equal(loss, expected))
        probabilities = logits.softmax(1)[:, 1].detach()
        validation = validation_values(labels, probabilities, val, thresholds=[0.05, 0.50, 0.95])
        final = test_values(labels, probabilities, test, threshold=validation["threshold"])
        self.assertEqual(validation["count"], 2)
        self.assertEqual(final["count"], 2)
        self.assertEqual(final["threshold"], validation["threshold"])


if __name__ == "__main__":
    unittest.main()
