import unittest

from methods.gin.audit.recompute import compare_metrics


class GINRecomputeContractTest(unittest.TestCase):
    def test_exact_metric_match_is_reported(self):
        original = {
            'f1_macro': 0.5,
            'auroc': 0.7,
            'threshold': 0.4,
            'predicted_anomaly_count': 3,
            'best_epoch': 9,
        }
        recomputed = dict(original)
        self.assertEqual(compare_metrics(original, recomputed)['status'], 'recompute_match')

    def test_metric_difference_is_reported(self):
        original = {
            'f1_macro': 0.5,
            'auroc': 0.7,
            'threshold': 0.4,
            'predicted_anomaly_count': 3,
            'best_epoch': 9,
        }
        recomputed = dict(original, auroc=0.6)
        result = compare_metrics(original, recomputed)
        self.assertEqual(result['status'], 'recompute_mismatch')
        self.assertIn('auroc', result['differences'])


if __name__ == '__main__':
    unittest.main()
