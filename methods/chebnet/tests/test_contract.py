import site
import unittest

site.addsitedir('/root/miniconda3/lib/python3.10/site-packages')
import dgl
import torch

from methods.chebnet.src.adapter import dgl_to_pyg_frozen
from methods.chebnet.src.model import ChebNetCandidate
from methods.chebnet.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics


class ChebNetContractTest(unittest.TestCase):
    def setUp(self):
        self.graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
        self.graph.ndata['feature'] = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        self.graph.ndata['label'] = torch.tensor([0, 1, 0])
        self.graph.ndata['train_mask'] = torch.tensor([True, False, False])
        self.graph.ndata['val_mask'] = torch.tensor([False, True, False])
        self.graph.ndata['test_mask'] = torch.tensor([False, False, True])

    def test_adapter_preserves_frozen_graph_semantics(self):
        data = dgl_to_pyg_frozen(self.graph)
        self.assertEqual(data.num_nodes, 3)
        self.assertTrue(torch.equal(data.edge_index, torch.tensor([[0, 1, 2], [1, 2, 0]])))
        self.assertTrue(torch.equal(data.x, self.graph.ndata['feature']))
        self.assertTrue(torch.equal(data.y, self.graph.ndata['label']))
        self.assertTrue(torch.equal(data.train_mask, self.graph.ndata['train_mask']))
        self.assertTrue(torch.equal(data.val_mask, self.graph.ndata['val_mask']))
        self.assertTrue(torch.equal(data.test_mask, self.graph.ndata['test_mask']))

    def test_model_returns_two_logits_and_backpropagates(self):
        data = dgl_to_pyg_frozen(self.graph)
        model = ChebNetCandidate(input_dim=2, hidden_dim=64, output_dim=2, order=2, dropout=0.0)
        logits = model(data.x, data.edge_index)
        self.assertEqual(tuple(logits.shape), (3, 2))
        logits.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_protocol_keeps_train_val_and_test_uses_separate(self):
        labels = torch.tensor([0, 1, 0, 1])
        logits = torch.tensor([[3., 0.], [0., 3.], [2., 0.], [0., 2.]], requires_grad=True)
        train = torch.tensor([True, False, False, False])
        val = torch.tensor([False, True, False, False])
        test = torch.tensor([False, False, True, True])
        loss = masked_cross_entropy(logits, labels, train, None)
        loss.backward()
        self.assertEqual(float(logits.grad[~train].abs().sum()), 0.0)
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        threshold, _ = select_validation_threshold(labels, probabilities, val)
        result = test_metrics(labels, probabilities, test, threshold)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['actual_anomaly_count'], 1)


if __name__ == '__main__':
    unittest.main()
