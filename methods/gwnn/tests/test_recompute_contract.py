import unittest


class GWNNRecomputeContractTests(unittest.TestCase):
    def test_exact_comparison_accepts_only_identical_saved_metrics(self):
        from methods.gwnn.audit.recompute import compare_metrics

        original = {"f1_macro": 0.8, "auroc": 0.9, "threshold": 0.5, "best_epoch": 7, "predicted_anomaly_count": 12}
        self.assertEqual(compare_metrics(original, dict(original))["status"], "recompute_match")
        changed = dict(original); changed["auroc"] += 1e-12
        result = compare_metrics(original, changed)
        self.assertEqual(result["status"], "recompute_mismatch")
        self.assertIn("auroc", result["differences"])


if __name__ == "__main__":
    unittest.main()
