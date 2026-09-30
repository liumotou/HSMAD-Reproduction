import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))


class NumericalStabilityPolicyContract(unittest.TestCase):
    def test_accepts_only_small_auroc_drift_with_exact_discrete_metrics(self):
        from numerical_stability_policy import classify_comparison

        comparison = {
            "f1_macro": {"kind": "exact", "original": 0.8, "recomputed": 0.8},
            "threshold": {"kind": "exact", "original": 0.5, "recomputed": 0.5},
            "predicted_anomaly_count": {"kind": "exact", "original": 10, "recomputed": 10},
            "actual_anomaly_count": {"kind": "exact", "original": 8, "recomputed": 8},
            "validation_threshold": {"kind": "exact", "original": 0.5, "recomputed": 0.5},
            "auroc": {"kind": "mismatch", "original": 0.9, "recomputed": 0.9000004},
        }
        result = classify_comparison(comparison)
        self.assertEqual(result["status"], "numerical_stability_match")
        self.assertEqual(result["auroc_absolute_tolerance"], 1e-6)
        self.assertTrue(result["strict_audit_preserved"])

    def test_rejects_large_auroc_drift(self):
        from numerical_stability_policy import classify_comparison

        comparison = {
            "f1_macro": {"kind": "exact", "original": 0.8, "recomputed": 0.8},
            "threshold": {"kind": "exact", "original": 0.5, "recomputed": 0.5},
            "predicted_anomaly_count": {"kind": "exact", "original": 10, "recomputed": 10},
            "actual_anomaly_count": {"kind": "exact", "original": 8, "recomputed": 8},
            "validation_threshold": {"kind": "exact", "original": 0.5, "recomputed": 0.5},
            "auroc": {"kind": "mismatch", "original": 0.9, "recomputed": 0.90001},
        }
        self.assertEqual(classify_comparison(comparison)["status"], "numerical_stability_mismatch")

    def test_rejects_any_discrete_metric_difference(self):
        from numerical_stability_policy import classify_comparison

        comparison = {
            "f1_macro": {"kind": "mismatch", "original": 0.8, "recomputed": 0.79},
            "threshold": {"kind": "exact", "original": 0.5, "recomputed": 0.5},
            "predicted_anomaly_count": {"kind": "exact", "original": 10, "recomputed": 10},
            "actual_anomaly_count": {"kind": "exact", "original": 8, "recomputed": 8},
            "validation_threshold": {"kind": "exact", "original": 0.5, "recomputed": 0.5},
            "auroc": {"kind": "mismatch", "original": 0.9, "recomputed": 0.9000001},
        }
        self.assertEqual(classify_comparison(comparison)["status"], "numerical_stability_mismatch")


if __name__ == "__main__":
    unittest.main()
