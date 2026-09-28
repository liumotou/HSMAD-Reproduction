import unittest


class RecomputeAuditContractTest(unittest.TestCase):
    def test_comparison_requires_exact_metrics_and_checkpoint_hash(self):
        from methods.nrgl.audit.recompute_checkpoint import compare_original_and_recomputed

        original = {"f1_macro": 0.8, "auroc": 0.9, "validation_auprc": 0.7, "threshold": 0.5, "predicted_anomaly_count": 10}
        recomputed = dict(original)
        self.assertEqual(compare_original_and_recomputed(original, recomputed, "abc", "abc")["status"], "RECOMPUTE_MATCH_WITHIN_TOLERANCE")
        recomputed["auroc"] += 5e-7
        self.assertEqual(compare_original_and_recomputed(original, recomputed, "abc", "abc")["status"], "RECOMPUTE_MATCH_WITHIN_TOLERANCE")
        recomputed["auroc"] = 0.91
        self.assertEqual(compare_original_and_recomputed(original, recomputed, "abc", "abc")["status"], "recompute_mismatch")


if __name__ == "__main__":
    unittest.main()
