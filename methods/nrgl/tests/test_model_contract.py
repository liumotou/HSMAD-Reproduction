import unittest

import dgl
import torch


class NRGLModelContractTest(unittest.TestCase):
    def test_official_core_model_produces_two_class_logits_without_labels_or_masks(self):
        from src.model import NRGLCore

        graph = dgl.add_self_loop(dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3))
        feature = torch.randn(3, 4)
        model = NRGLCore(num_nodes=3, input_dim=4, hidden_dim=64, order=2, alpha=0.1)
        logits = model(graph, feature)

        self.assertEqual(tuple(logits.shape), (3, 2))
        logits.sum().backward()

    def test_saturated_edge_gate_remains_normalizable_with_self_loops(self):
        from src.model import NRGLCore

        graph = dgl.add_self_loop(dgl.graph(([0, 1], [1, 0]), num_nodes=2))
        feature = torch.ones(2, 3)
        model = NRGLCore(num_nodes=2, input_dim=3, hidden_dim=4, order=2, alpha=1.0)
        with torch.no_grad():
            model.edge_mlp.weight.fill_(100.0)
            model.edge_mlp.bias.fill_(100.0)
        low_graph, high_graph = model._weighted_graphs(graph, feature)
        source, destination = graph.edges(order="eid")
        self_loop = source == destination
        self.assertTrue(torch.isfinite(low_graph.edata["w"]).all())
        self.assertTrue(torch.isfinite(high_graph.edata["w"]).all())
        self.assertTrue(torch.equal(low_graph.edata["w"][self_loop], torch.ones(int(self_loop.sum()))))
        self.assertTrue(torch.equal(high_graph.edata["w"][self_loop], torch.ones(int(self_loop.sum()))))


if __name__ == "__main__":
    unittest.main()
