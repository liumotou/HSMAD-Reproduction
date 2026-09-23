import unittest


class NRGLSummaryContractTest(unittest.TestCase):
    def test_summary_uses_only_formal_ok_and_sample_std(self):
        from audit.summarize_formal import summarize_records

        rows = [
            {"run_type": "formal", "status": "OK", "f1_macro": "0.7", "auroc": "0.8"},
            {"run_type": "formal", "status": "OK", "f1_macro": "0.9", "auroc": "1.0"},
            {"run_type": "smoke", "status": "smoke", "f1_macro": "0.0", "auroc": "0.0"},
        ]
        summary = summarize_records(rows)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["f1_macro_mean"], 0.8)
        self.assertAlmostEqual(summary["f1_macro_std_sample"], 0.14142135623730953)
        self.assertAlmostEqual(summary["auroc_std_sample"], 0.14142135623730948)


if __name__ == "__main__":
    unittest.main()
