import unittest


class SparseGADSummaryContractTest(unittest.TestCase):
    def test_summary_uses_only_formal_ok_and_sample_standard_deviation(self):
        from methods.sparsegad.audit.summarize_formal import summarize_rows
        summary = summarize_rows([
            {'run_type': 'formal', 'status': 'OK', 'f1_macro': '0.8', 'auroc': '0.9'},
            {'run_type': 'formal', 'status': 'OK', 'f1_macro': '1.0', 'auroc': '1.0'},
            {'run_type': 'diagnostic', 'status': 'OK', 'f1_macro': '0.1', 'auroc': '0.1'},
        ])
        self.assertEqual(summary['n'], 2)
        self.assertAlmostEqual(summary['f1_macro_std_sample'], 0.1414213562)
