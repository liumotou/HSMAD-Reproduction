import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class DatasetSummaryTest(unittest.TestCase):
    def test_summary_uses_dataset_specific_paper_values_and_sample_std(self):
        from summarize_kipf_v2_dataset import summarize_rows

        result = summarize_rows(
            [{"f1_macro": "0.6", "auroc": "0.7"}, {"f1_macro": "0.8", "auroc": "0.9"}],
            "tolokers",
            "protocol_v2_kipf_two_layer_hidden64",
            {"f1_macro": 0.0, "auroc": 0.0},
        )

        self.assertEqual(result["dataset"], "tolokers")
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["auroc_std_sample"], 0.14142135623730953)
        self.assertEqual(result["checkpoint_protocol"], "GADBench_style_AUPRC_best")
