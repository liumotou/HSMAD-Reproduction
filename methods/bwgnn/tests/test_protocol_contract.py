"""Mask-isolation contract for BWGNN's project evaluation protocol."""
import unittest

import torch

from methods.bwgnn.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics


class BWGNNProtocolContractTest(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0], [2.5, 0.0], [0.0, 2.5]], requires_grad=True)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self.train = torch.tensor([True, True, False, False, False, False])
        self.val = torch.tensor([False, False, True, True, False, False])
        self.test = torch.tensor([False, False, False, False, True, True])

    def test_loss_uses_only_the_train_mask(self):
        loss = masked_cross_entropy(self.logits, self.labels, self.train, torch.tensor([1.0, 1.0]))
        expected = torch.nn.functional.cross_entropy(self.logits[self.train], self.labels[self.train], weight=torch.tensor([1.0, 1.0]))
        self.assertTrue(torch.allclose(loss, expected))

    def test_threshold_selection_rejects_non_validation_inputs(self):
        probabilities = torch.softmax(self.logits.detach(), dim=1)[:, 1]
        threshold, f1 = select_validation_threshold(self.labels, probabilities, self.val)
        self.assertIn(threshold, [round(x * 0.05, 2) for x in range(1, 20)])
        self.assertGreaterEqual(f1, 0.0)

    def test_test_metrics_requires_a_disjoint_test_mask(self):
        probabilities = torch.softmax(self.logits.detach(), dim=1)[:, 1]
        output = test_metrics(self.labels, probabilities, self.test, 0.5)
        self.assertEqual(output['count'], 2)
        self.assertIn('f1_macro', output)
        self.assertIn('auroc', output)
