import unittest

import torch

from methods.caregnn.src.model import SingleRelationCareGNN


class CareGNNModelContractTest(unittest.TestCase):
    def test_single_relation_forward_and_backward(self):
        torch.manual_seed(0)
        features = torch.randn(6, 4)
        adjacency = [
            [0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5],
        ]
        model = SingleRelationCareGNN(4, 8, 2, lambda_1=2.0, step_size=0.02)
        nodes = torch.tensor([1, 3, 4])
        labels = torch.tensor([1, 0, 1])
        logits, label_logits, relation_scores = model(
            features, adjacency, nodes
        )
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(label_logits.shape), (3, 2))
        self.assertEqual(len(relation_scores), 3)
        loss = model.loss(logits, label_logits, labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_official_initial_threshold_and_single_relation(self):
        model = SingleRelationCareGNN(4, 8, 2, lambda_1=2.0, step_size=0.02)
        self.assertEqual(model.num_relations, 1)
        self.assertEqual(model.thresholds, [0.5])

    def test_rl_threshold_update_uses_positive_training_rows(self):
        model = SingleRelationCareGNN(4, 8, 2, lambda_1=2.0, step_size=0.02)
        labels = torch.tensor([1, 0])
        model.update_threshold([[1.0], [99.0]], labels, batch_num=1)
        model.update_threshold([[0.5], [99.0]], labels, batch_num=1)
        model.update_threshold([[0.25], [99.0]], labels, batch_num=1)
        self.assertAlmostEqual(model.thresholds[0], 0.52)


if __name__ == "__main__":
    unittest.main()
