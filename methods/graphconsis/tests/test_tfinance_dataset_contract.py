import unittest
from pathlib import Path


class GraphConsisTFinanceContractTest(unittest.TestCase):
    def test_frozen_tfinance_config_masks_and_scalable_builder(self):
        from methods.graphconsis.src.run import (
            ROOT,
            load_config,
            prepare_training_graph,
            select_adjacency_builder,
            sha256_file,
            sha256_tensor,
        )
        import dgl

        config = load_config(
            Path("methods/graphconsis/configs/tfinance_single_relation_smoke.json")
        )
        self.assertEqual(config["dataset"], "tfinance")
        self.assertEqual(
            config["expected"], {"nodes": 39357, "training_edges": 42484443}
        )
        self.assertEqual(select_adjacency_builder(config)[0], "stable_csr")
        raw_path = ROOT / config["dataset_file"]
        graph = prepare_training_graph(dgl.load_graphs(str(raw_path))[0][0])
        self.assertEqual(
            sha256_file(raw_path),
            "051b27bd8d32086a2f39b629d84a1e900d47e75d5726c7dc2d6a8dd54de4db98",
        )
        expected = {
            "train_mask": (
                15742,
                "88727debf7b9f9e9dba73095a8706dc175e830012220fb38b2ff39a2e21fe0f1",
            ),
            "val_mask": (
                7792,
                "654bddb549a4f0a72950c796b5feccd712656ba89d866484d0c585c932b5c7f2",
            ),
            "test_mask": (
                15823,
                "34234f45acfca50e465087b99634b5ef8bb2143f9edb8d7478ed837f2dfe1364",
            ),
        }
        for name, (count, digest) in expected.items():
            mask = graph.ndata[name].bool()
            self.assertEqual(int(mask.sum()), count)
            self.assertEqual(sha256_tensor(mask), digest)


if __name__ == "__main__":
    unittest.main()
