import unittest

import torch

from methods.pcgnn.src.model import SingleRelationPCGNN


class PCGNNModelContractTests(unittest.TestCase):
    def test_single_relation_forward_shape_and_backward(self):
        torch.manual_seed(0)
        features = torch.randn(6, 4)
        adjacency = [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5]]
        model = SingleRelationPCGNN(4, 8, 2, train_positive_nodes=torch.tensor([1, 4]), rho=0.5, alpha=2.0)
        nodes = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 0])
        logits, label_logits = model(features, adjacency, nodes, labels, train_flag=True)
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(label_logits.shape), (3, 2))
        model.loss(logits, label_logits, labels).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_fixed_top_p_and_minority_oversampling_contract(self):
        model = SingleRelationPCGNN(3, 4, 2, train_positive_nodes=torch.tensor([2, 3]), rho=0.5, alpha=2.0)
        self.assertEqual(model.num_relations, 1)
        self.assertEqual(model.relation_threshold, 0.5)
        self.assertEqual(model.rho, 0.5)


if __name__ == "__main__":
    unittest.main()
