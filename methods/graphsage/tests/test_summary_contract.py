import unittest


class GraphSAGESummaryContractTest(unittest.TestCase):
    def test_summary_uses_only_formal_ok_and_sample_standard_deviation(self):
        from src.summary import summarize_rows

        summary = summarize_rows([
            {"run_type": "formal", "status": "OK", "f1_macro": "0.4", "auroc": "0.8"},
            {"run_type": "formal", "status": "OK", "f1_macro": "0.6", "auroc": "1.0"},
            {"run_type": "smoke", "status": "smoke", "f1_macro": "1.0", "auroc": "1.0"},
        ])

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["f1_macro_mean"], 0.5)
        self.assertAlmostEqual(summary["f1_macro_std_sample"], 0.1 * 2 ** 0.5)
        self.assertEqual(summary["auroc_mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
