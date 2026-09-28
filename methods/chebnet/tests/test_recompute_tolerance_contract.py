import unittest
from methods.chebnet.audit.recompute import compare_metrics

class ChebNetRecomputeToleranceContractTest(unittest.TestCase):
    def test_float_roundoff_below_one_e12_is_a_match(self):
        original={'f1_macro':0.5,'auroc':0.8116912389464078,'threshold':0.35,'predicted_anomaly_count':3}
        recomputed={'f1_macro':0.5,'auroc':0.8116912389464077,'threshold':0.35,'predicted_anomaly_count':3}
        self.assertEqual(compare_metrics(original,recomputed)['status'],'recompute_match')

if __name__=='__main__': unittest.main()
