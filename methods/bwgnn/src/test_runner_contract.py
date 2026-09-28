"""Contract for the non-formal BWGNN smoke runner."""
import unittest

import dgl
import torch

from methods.bwgnn.src.run_smoke import prepare_training_graph, smoke_contract


class BWGNNRunnerContractTest(unittest.TestCase):
    def test_smoke_contract_is_fixed_to_weibo_seed_zero_and_five_epochs(self):
        contract = smoke_contract()
        self.assertEqual(contract['dataset'], 'weibo')
        self.assertEqual(contract['seed'], 0)
        self.assertEqual(contract['max_epoch'], 5)
        self.assertEqual(contract['run_type'], 'smoke')
        self.assertEqual(contract['edge_access'], 'graph_edges_required')
        self.assertEqual(contract['hidden_dim'], 64)
        self.assertEqual(contract['order'], 2)
        self.assertEqual(contract['checkpoint_protocol'], 'none_smoke_last_epoch')

    def test_preprocessing_retains_feature_for_gadbench_model_input(self):
        raw = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        raw.ndata['feature'] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        graph = prepare_training_graph(raw)
        self.assertTrue(torch.equal(graph.ndata['feature'], raw.ndata['feature']))
        self.assertEqual(graph.num_edges(), 4)
