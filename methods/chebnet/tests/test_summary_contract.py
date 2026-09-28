import unittest

from methods.chebnet.audit.summarize import sample_summary


class ChebNetSummaryContractTest(unittest.TestCase):
    def test_sample_standard_deviation_uses_ddof_one(self):
        result = sample_summary([{'f1_macro': 1.0, 'auroc': 0.2}, {'f1_macro': 3.0, 'auroc': 0.4}])
        self.assertEqual(result['n'], 2)
        self.assertAlmostEqual(result['f1_macro_std_sample'], 2 ** 0.5)


if __name__ == '__main__':
    unittest.main()
