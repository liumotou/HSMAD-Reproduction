import unittest
from pathlib import Path


class GWNNTolokersDatasetContractTest(unittest.TestCase):
    def test_frozen_tolokers_input_and_smoke_config(self):
        from methods.gwnn.src.run import (
            ROOT,
            load_config,
            prepare_training_graph,
            sha256_file,
            sha256_tensor,
        )
        import dgl

        config_path = Path("methods/gwnn/configs/tolokers_gwnn_paper_formula_h64_smoke.json")
        config = load_config(config_path)
        self.assertEqual(config["dataset"], "tolokers")
        self.assertEqual(config["dataset_file"], "datasets/tolokers")
        self.assertEqual(config["expected"], {"nodes": 11758, "training_edges": 1049758})
        self.assertEqual(config["hidden_dim"], 64)
        self.assertEqual(config["run_type"], "smoke")
        self.assertEqual(config["max_epoch"], 5)

        dataset_path = ROOT / config["dataset_file"]
        raw = dgl.load_graphs(str(dataset_path))[0][0]
        graph = prepare_training_graph(raw)
        self.assertEqual(sha256_file(dataset_path), "1d400b4cdced54b25c394cc3d258921f443f04b2ae2e09bbb70fbc7073ed0b65")
        self.assertEqual((graph.num_nodes(), graph.num_edges()), (11758, 1049758))
        expected_masks = {
            "train_mask": (4703, "4f298168b1a4d72eedf0f4594d1deb196ddb6e1c80ee95058dd8c05da1bb10af"),
            "val_mask": (2328, "bf4c7797ce28dc6a47a51c2389ad56f86538c5fa852113de76a17eaaaba9bb36"),
            "test_mask": (4727, "5fc331d7846b8f113c954295f8c9503209157d8a6393e5520de4ff029a6c1662"),
        }
        for name, (count, digest) in expected_masks.items():
            mask = graph.ndata[name].bool()
            self.assertEqual(int(mask.sum()), count)
            self.assertEqual(sha256_tensor(mask), digest)


if __name__ == "__main__":
    unittest.main()
