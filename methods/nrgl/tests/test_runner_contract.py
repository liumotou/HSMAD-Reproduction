import unittest
import tempfile
from pathlib import Path

import torch


class NRGLRunnerContractTest(unittest.TestCase):
    def test_smoke_and_formal_controls_are_isolated(self):
        from src.runner import execution_contract

        smoke = execution_contract("smoke")
        repaired_smoke = execution_contract("smoke_edge_weight_fix")
        formal = execution_contract("formal")
        self.assertEqual(smoke["max_epoch"], 5)
        self.assertEqual(smoke["run_type"], "smoke")
        self.assertEqual(repaired_smoke["max_epoch"], 5)
        self.assertEqual(repaired_smoke["run_type"], "smoke")
        self.assertEqual(formal["max_epoch"], 200)
        self.assertEqual(formal["patience"], 50)
        self.assertEqual(formal["checkpoint_metric"], "validation_auprc")

    def test_loss_is_insensitive_to_non_training_labels(self):
        from src.runner import masked_cross_entropy

        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1, 0, 1])
        train = torch.tensor([True, True, False, False])
        baseline = masked_cross_entropy(logits, labels, train, torch.tensor([1.0, 2.0]))
        changed = labels.clone()
        changed[~train] = 1 - changed[~train]
        self.assertEqual(float(baseline), float(masked_cross_entropy(logits, changed, train, torch.tensor([1.0, 2.0]))))

    def test_output_path_contains_dataset_run_type_and_seed(self):
        from src.runner import output_directory

        result = output_directory("tolokers", "diagnostic", 0)
        self.assertTrue(str(result).endswith("results/experiments/nrgl/tolokers/nrgl_hsmad_candidate/diagnostic/seed_0"))

    def test_run_records_append_without_replacing_previous_seed(self):
        from src.runner import append_run_record

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs.csv"
            append_run_record(path, {"seed": 0, "status": "OK", "f1_macro": 0.1})
            append_run_record(path, {"seed": 1, "status": "OK", "f1_macro": 0.2})
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 3)
        self.assertIn("0", rows[1])
        self.assertIn("1", rows[2])


if __name__ == "__main__":
    unittest.main()
