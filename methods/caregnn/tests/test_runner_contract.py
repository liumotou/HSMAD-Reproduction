import json
import tempfile
import unittest
from pathlib import Path

import torch

from methods.caregnn.src.run import (
    assert_uncovered_prefix,
    balanced_training_nodes,
    build_run_spec,
    build_single_relation_adjacency,
    ensure_new_output,
    load_config,
)


class CareGNNRunnerContractTests(unittest.TestCase):
    def test_config_contract_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "weibo",
                        "seed": 0,
                        "run_type": "smoke",
                        "result_dir": "results/caregnn/smoke/seed_0",
                        "max_epoch": 5,
                        "patience": 10,
                        "candidate_protocol_not_author_exact": True,
                        "relation_policy": "single_flattened_relation_project_adaptation",
                        "checkpoint_protocol": "validation_auprc_project_choice",
                        "threshold_protocol": "validation_f1_macro_grid_0.05_to_0.95",
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            spec = build_run_spec(config)
            self.assertEqual(spec.dataset, "weibo")
            self.assertEqual(spec.max_epoch, 5)
            self.assertEqual(len(config["_config_sha256"]), 64)

    def test_single_relation_adjacency_preserves_edges_without_triplication(self):
        source = torch.tensor([0, 0, 1, 2, 2])
        destination = torch.tensor([1, 2, 2, 0, 2])
        adjacency = build_single_relation_adjacency(source, destination, num_nodes=3)
        self.assertEqual(adjacency, [[1, 2], [2], [0, 2]])
        self.assertEqual(sum(map(len, adjacency)), source.numel())

    def test_balanced_nodes_are_train_only_equal_class_and_reproducible(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
        train_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
        first = balanced_training_nodes(labels, train_mask, seed=7)
        second = balanced_training_nodes(labels, train_mask, seed=7)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(train_mask[first].all()))
        chosen = labels[first]
        self.assertEqual(int((chosen == 0).sum()), int((chosen == 1).sum()))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                ensure_new_output(Path(directory))

    def test_amazon_uncovered_prefix_is_excluded_from_all_masks(self):
        masks = {
            "train_mask": torch.tensor([0, 0, 1, 0], dtype=torch.bool),
            "val_mask": torch.tensor([0, 0, 0, 1], dtype=torch.bool),
            "test_mask": torch.tensor([0, 0, 0, 0], dtype=torch.bool),
        }
        assert_uncovered_prefix(masks, 2)
        masks["test_mask"][1] = True
        with self.assertRaises(RuntimeError):
            assert_uncovered_prefix(masks, 2)


if __name__ == "__main__":
    unittest.main()
