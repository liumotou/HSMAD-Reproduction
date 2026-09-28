import unittest

from methods.gin.audit.recompute_tolerance import compare_metrics_with_numerical_tolerance


class GINRecomputeNumericalToleranceContractTest(unittest.TestCase):
    def test_cuda_scatter_auroc_roundoff_below_one_e_minus_six_is_match(self):
        original = {
            'f1_macro': 0.47172149661948937,
            'auroc': 0.645496611805184,
            'threshold': 0.95,
            'predicted_anomaly_count': 3134,
            'best_epoch': 3,
        }
        recomputed = dict(original, auroc=0.6454968740493648)
        self.assertEqual(
            compare_metrics_with_numerical_tolerance(original, recomputed)['status'],
            'recompute_match',
        )

    def test_auroc_difference_above_one_e_minus_six_is_mismatch(self):
        original = {
            'f1_macro': 0.5,
            'auroc': 0.7,
            'threshold': 0.5,
            'predicted_anomaly_count': 3,
            'best_epoch': 2,
        }
        recomputed = dict(original, auroc=0.700002)
        self.assertEqual(
            compare_metrics_with_numerical_tolerance(original, recomputed)['status'],
            'recompute_mismatch',
        )


if __name__ == '__main__':
    unittest.main()
