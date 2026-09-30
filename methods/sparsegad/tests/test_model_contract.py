import unittest

import dgl
import torch


class SparseGADModelContractTest(unittest.TestCase):
    def test_official_style_model_produces_logits_and_gradients(self):
        from methods.sparsegad.src.model import SparseGADModel

        graph = dgl.add_self_loop(dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3))
        model = SparseGADModel(input_dim=4, hidden_dim=64, output_dim=2, num_layers=2,
                               dropout=0.2, dropout_adj=0.1)
        logits = model(graph, torch.randn(3, 4))
        self.assertEqual(tuple(logits.shape), (3, 2))
        logits.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_forward_accepts_only_graph_and_features(self):
        from inspect import signature
        from methods.sparsegad.src.model import SparseGADModel

        self.assertEqual(list(signature(SparseGADModel.forward).parameters), ['self', 'graph', 'features'])
