import unittest

from methods.chebnet.audit.recompute import compare_metrics


class ChebNetRecomputeContractTest(unittest.TestCase):
    def test_exact_metric_match_is_reported(self):
        self.assertEqual(compare_metrics({'f1_macro': 0.5, 'auroc': 0.7, 'threshold': 0.4, 'predicted_anomaly_count': 3}, {'f1_macro': 0.5, 'auroc': 0.7, 'threshold': 0.4, 'predicted_anomaly_count': 3})['status'], 'recompute_match')


if __name__ == '__main__':
    unittest.main()
