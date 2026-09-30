import json
import tempfile
import unittest
from pathlib import Path

import torch


class GraphConsisRunnerContractTest(unittest.TestCase):
    def test_weibo_smoke_config_and_refuse_overwrite(self):
        from methods.graphconsis.src.run import build_run_spec, ensure_new_output, load_config

        config = load_config(Path("methods/graphconsis/configs/weibo_single_relation_smoke.json"))
        spec = build_run_spec(config)
        self.assertEqual((spec.dataset, spec.seed, spec.run_type, spec.max_epoch), ("weibo", 0, "smoke", 5))
        self.assertEqual(config["source_commit"], "22b72d75f81dd057762f0c7225a4558a25095b8f")
        self.assertEqual(config["relation_policy"], "single_flattened_relation_project_adaptation")
        self.assertEqual(config["fanouts"], [25, 10])
        self.assertEqual(config["max_degree"], 128)
        self.assertTrue(config["candidate_protocol_not_author_exact"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                ensure_new_output(Path(directory))

    def test_padded_adjacency_has_official_max_degree(self):
        from methods.graphconsis.src.run import build_padded_adjacency

        source = torch.tensor([0, 0, 1, 2, 2, 2])
        destination = torch.tensor([0, 1, 1, 0, 1, 2])
        adjacency = build_padded_adjacency(source, destination, 3, 128, seed=0)
        self.assertEqual(tuple(adjacency.shape), (3, 128))
        self.assertTrue(bool(((adjacency >= 0) & (adjacency < 3)).all()))


if __name__ == "__main__":
    unittest.main()
