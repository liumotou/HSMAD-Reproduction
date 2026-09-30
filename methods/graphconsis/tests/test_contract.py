import inspect
import unittest

import torch


class GraphConsisContractTest(unittest.TestCase):
    def test_official_shape_semantics_sampling_and_backward(self):
        from methods.graphconsis.src.model import GraphConsisSingleRelationCandidate

        signature = inspect.signature(GraphConsisSingleRelationCandidate.forward)
        self.assertEqual(list(signature.parameters), ["self", "features", "adjacency", "seeds", "generator"])
        model = GraphConsisSingleRelationCandidate(input_dim=4, hidden_dim=64, num_classes=2)
        self.assertEqual(model.fanouts, (25, 10))
        self.assertEqual(model.relation_augmented_dim, 256)
        features = torch.randn(40, 4)
        adjacency = torch.randint(0, 40, (40, 128))
        generator = torch.Generator().manual_seed(0)
        logits = model(features, adjacency, torch.tensor([0, 1, 2]), generator)
        self.assertEqual(tuple(logits.shape), (3, 2))
        logits.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_masked_protocol_isolates_splits(self):
        from methods.graphconsis.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics

        logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.], [0., 1.], [2., 0.], [0., 2.]], requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        train = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
        val = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)
        test = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
        loss = masked_cross_entropy(logits, labels, train)
        changed = labels.clone(); changed[~train] = 1 - changed[~train]
        self.assertEqual(float(loss), float(masked_cross_entropy(logits, changed, train)))
        probabilities = torch.softmax(logits.detach(), 1)[:, 1]
        threshold, _ = select_validation_threshold(labels, probabilities, val)
        changed = labels.clone(); changed[test] = 1 - changed[test]
        self.assertEqual(threshold, select_validation_threshold(changed, probabilities, val)[0])
        original = test_metrics(labels, probabilities, test, threshold)
        changed = labels.clone(); changed[~test] = 1 - changed[~test]
        self.assertEqual(original, test_metrics(changed, probabilities, test, threshold))


if __name__ == "__main__":
    unittest.main()
