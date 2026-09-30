import json
import tempfile
import unittest
from pathlib import Path

import torch

from methods.pcgnn.src.run import (
    assert_uncovered_prefix_excluded,
    build_run_spec,
    build_single_relation_adjacency,
    ensure_new_output,
    load_config,
    pick_training_nodes,
)


class PCGNNRunnerContractTests(unittest.TestCase):
    def test_amazon_uncovered_prefix_is_excluded_from_all_masks(self):
        masks = {
            "train_mask": torch.tensor([False, False, True]),
            "val_mask": torch.tensor([False, False, False]),
            "test_mask": torch.tensor([False, False, False]),
        }
        assert_uncovered_prefix_excluded(masks, 2)
        masks["test_mask"][0] = True
        with self.assertRaises(RuntimeError):
            assert_uncovered_prefix_excluded(masks, 2)

    def test_config_requires_official_validation_auroc_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "dataset": "weibo", "seed": 0, "run_type": "smoke",
                "result_dir": "results/pcgnn/smoke/seed_0",
                "max_epoch": 5, "validation_interval": 5,
                "candidate_protocol_not_author_exact": True,
                "relation_policy": "single_flattened_relation_project_adaptation",
                "checkpoint_protocol": "validation_auroc_official_pcgnn",
                "threshold_protocol": "validation_f1_macro_grid_0.05_to_0.95",
            }), encoding="utf-8")
            spec = build_run_spec(load_config(path))
            self.assertEqual(spec.dataset, "weibo")
            self.assertEqual(spec.validation_interval, 5)

    def test_adjacency_is_one_relation_and_preserves_edge_count(self):
        source = torch.tensor([0, 0, 1, 2])
        destination = torch.tensor([1, 2, 2, 0])
        adjacency = build_single_relation_adjacency(source, destination, 3)
        self.assertEqual(adjacency, [[1, 2], [2], [0]])
        self.assertEqual(sum(map(len, adjacency)), 4)

    def test_pick_step_uses_only_frozen_train_nodes_and_is_reproducible(self):
        labels = torch.tensor([0, 0, 1, 1, 0, 1])
        train_mask = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool)
        adjacency = [[0], [0, 1], [0, 1, 2, 3], [3], [0, 4], [5]]
        first = pick_training_nodes(labels, train_mask, adjacency, seed=3)
        second = pick_training_nodes(labels, train_mask, adjacency, seed=3)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), 2)
        self.assertTrue(bool(train_mask[first].all()))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                ensure_new_output(Path(directory))


if __name__ == "__main__":
    unittest.main()
