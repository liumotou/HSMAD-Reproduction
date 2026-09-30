import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class AmazonSummaryTest(unittest.TestCase):
    def test_summary_carries_protocol_metadata(self):
        from summarize_amazon_kipf_v2 import summarize_rows

        result = summarize_rows(
            [{"f1_macro": "0.6", "auroc": "0.8"}, {"f1_macro": "0.8", "auroc": "0.9"}],
            "protocol_v2_kipf_two_layer_hidden64",
        )

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["checkpoint_protocol"], "GADBench_style_AUPRC_best")
        self.assertAlmostEqual(result["f1_macro_std_sample"], 0.14142135623730953)
