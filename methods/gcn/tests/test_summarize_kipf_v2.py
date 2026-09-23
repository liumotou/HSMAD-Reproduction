import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class SummarizeKipfV2Test(unittest.TestCase):
    def test_uses_sample_standard_deviation(self):
        from summarize_kipf_v2 import summarize_rows

        summary = summarize_rows(
            [
                {"f1_macro": "0.90", "auroc": "0.80"},
                {"f1_macro": "0.92", "auroc": "0.84"},
            ],
            {"f1_macro": 0.9502, "auroc": 0.9791},
        )

        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["f1_macro_mean"], 0.91)
        self.assertAlmostEqual(summary["f1_macro_std_sample"], 0.014142135623730963)
