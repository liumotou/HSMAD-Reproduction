import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class RunContractTest(unittest.TestCase):
    def test_test_prediction_counts_are_explicit(self):
        from run_kipf_v2 import test_prediction_counts

        self.assertEqual(
            test_prediction_counts([0, 1, 1, 0], [0, 1, 0, 1]),
            {"actual_anomaly_count": 2, "predicted_anomaly_count": 2},
        )

    def test_formal_metadata_contains_checkpoint_protocols(self):
        from run_kipf_v2 import protocol_metadata

        metadata = protocol_metadata(
            {
                "checkpoint_protocol": "GADBench_style_AUPRC_best",
                "early_stop_protocol": "validation_F1_macro",
                "threshold_protocol": "validation_F1_macro_grid_0.05_to_0.95",
            }
        )

        self.assertEqual(metadata["checkpoint_protocol"], "GADBench_style_AUPRC_best")
        self.assertEqual(metadata["early_stop_protocol"], "validation_F1_macro")
        self.assertEqual(
            metadata["threshold_protocol"], "validation_F1_macro_grid_0.05_to_0.95"
        )

    def test_contract_uses_configured_seed_and_protocol(self):
        from run_kipf_v2 import run_contract

        seed, suffix = run_contract(
            {
                "protocol_version": "protocol_v2_kipf_two_layer_hidden64",
                "dataset": "weibo",
                "run_type": "formal",
                "seed": 7,
            }
        )

        self.assertEqual(seed, 7)
        self.assertEqual(
            suffix.parts[-4:],
            (
                "weibo",
                "protocol_v2_kipf_two_layer_hidden64",
                "formal",
                "seed_7",
            )[-4:],
        )
