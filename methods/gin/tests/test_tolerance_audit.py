import unittest

from methods.gin.audit.tolerance_audit import classify_secondary_audit


class GINToleranceAuditTests(unittest.TestCase):
    def test_accepts_only_documented_cuda_auroc_variance(self):
        strict = {
            "status": "recompute_mismatch",
            "differences": {"auroc": {"original": 0.6, "recomputed": 0.6000002}},
            "original": {"f1_macro": 0.7, "threshold": 0.95, "predicted_anomaly_count": 10, "best_epoch": 3},
            "recomputed": {"f1_macro": 0.7, "threshold": 0.95, "predicted_anomaly_count": 10, "best_epoch": 3},
        }
        result = classify_secondary_audit(strict)
        self.assertEqual(result["status"], "recompute_match_with_documented_cuda_auroc_tolerance")

    def test_rejects_non_auroc_or_large_difference(self):
        strict = {
            "status": "recompute_mismatch",
            "differences": {"f1_macro": {"original": 0.7, "recomputed": 0.71}},
            "original": {"f1_macro": 0.7, "threshold": 0.95, "predicted_anomaly_count": 10, "best_epoch": 3},
            "recomputed": {"f1_macro": 0.71, "threshold": 0.95, "predicted_anomaly_count": 10, "best_epoch": 3},
        }
        self.assertEqual(classify_secondary_audit(strict)["status"], "recompute_mismatch")


if __name__ == "__main__":
    unittest.main()
