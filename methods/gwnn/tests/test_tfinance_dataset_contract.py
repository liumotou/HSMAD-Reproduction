import unittest
from pathlib import Path


class GWNNTFinanceDatasetContractTest(unittest.TestCase):
    def test_frozen_tfinance_input_and_dense_basis_preflight(self):
        from methods.gwnn.src.run import ROOT, load_config, prepare_training_graph, sha256_file, sha256_tensor
        import dgl

        config_path = Path("methods/gwnn/configs/tfinance_gwnn_paper_formula_h64_smoke.json")
        config = load_config(config_path)
        self.assertEqual(config["dataset"], "tfinance")
        self.assertEqual(config["expected"], {"nodes": 39357, "training_edges": 42484443})
        raw_path = ROOT / config["dataset_file"]
        graph = prepare_training_graph(dgl.load_graphs(str(raw_path))[0][0])
        self.assertEqual(sha256_file(raw_path), "051b27bd8d32086a2f39b629d84a1e900d47e75d5726c7dc2d6a8dd54de4db98")
        self.assertEqual((graph.num_nodes(), graph.num_edges()), (39357, 42484443))
        expected = {
            "train_mask": (15742, "88727debf7b9f9e9dba73095a8706dc175e830012220fb38b2ff39a2e21fe0f1"),
            "val_mask": (7792, "654bddb549a4f0a72950c796b5feccd712656ba89d866484d0c585c932b5c7f2"),
            "test_mask": (15823, "34234f45acfca50e465087b99634b5ef8bb2143f9edb8d7478ed837f2dfe1364"),
        }
        for name, (count, digest) in expected.items():
            mask = graph.ndata[name].bool()
            self.assertEqual(int(mask.sum()), count)
            self.assertEqual(sha256_tensor(mask), digest)
        one_dense_gib = graph.num_nodes() ** 2 * 4 / 1024 ** 3
        self.assertGreater(one_dense_gib, 5.7)
        self.assertGreater(one_dense_gib * 4, 22.8)


if __name__ == "__main__":
    unittest.main()
