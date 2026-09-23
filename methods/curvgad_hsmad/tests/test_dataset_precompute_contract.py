import tempfile
import unittest
from pathlib import Path

import dgl
import torch


class DatasetPrecomputeContract(unittest.TestCase):
    def test_load_applies_frozen_graph_path_once_and_preserves_node_data(self):
        from methods.curvgad_hsmad.src.dataset import load_preprocessed_frozen_graph

        graph = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 0, 1])), num_nodes=2)
        graph.ndata["feature"] = torch.tensor([[1.0], [2.0]])
        graph.ndata["label"] = torch.tensor([0, 1])
        graph.ndata["train_mask"] = torch.tensor([True, False])
        graph.ndata["val_mask"] = torch.tensor([False, True])
        graph.ndata["test_mask"] = torch.tensor([False, True])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.bin"
            dgl.save_graphs(str(path), [graph])
            loaded = load_preprocessed_frozen_graph(path)
        self.assertEqual(loaded.num_nodes(), 2)
        self.assertEqual(loaded.num_edges(), 4)
        self.assertTrue(torch.equal(loaded.ndata["feature"], graph.ndata["feature"]))
        self.assertTrue(torch.equal(loaded.ndata["label"], graph.ndata["label"]))
        self.assertTrue(torch.equal(loaded.ndata["train_mask"], graph.ndata["train_mask"]))

    def test_runner_declares_exact_method_and_never_accepts_approximation(self):
        from methods.curvgad_hsmad.src.precompute_dataset import build_parser

        parser = build_parser()
        args = parser.parse_args(["--dataset", "weibo", "--dataset-path", "datasets/weibo", "--cache-dir", "cache"])
        self.assertEqual(args.method, "exact_orc")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dataset", "weibo", "--dataset-path", "datasets/weibo", "--cache-dir", "cache", "--method", "accelerated"])


if __name__ == "__main__":
    unittest.main()
