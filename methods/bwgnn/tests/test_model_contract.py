"""Contract for the isolated GADBench-style BWGNN model."""
import inspect
import unittest

import dgl
import torch

from methods.bwgnn.src.model import GADBenchBWGNN


class BWGNNModelContractTest(unittest.TestCase):
    def test_model_returns_node_logits_and_supports_backward(self):
        graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]), num_nodes=4)
        graph = dgl.add_self_loop(graph)
        features = torch.randn(4, 3, requires_grad=True)
        graph.ndata['feature'] = features
        model = GADBenchBWGNN(in_feats=3, h_feats=64, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0.0)

        logits = model(graph)

        self.assertEqual(tuple(logits.shape), (4, 2))
        logits.sum().backward()
        self.assertIsNotNone(features.grad)

    def test_forward_accepts_only_graph_not_labels_or_masks(self):
        parameters = list(inspect.signature(GADBenchBWGNN.forward).parameters)
        self.assertEqual(parameters, ['self', 'graph'])

