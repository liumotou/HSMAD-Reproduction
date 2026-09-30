import unittest


class SparseGADRecomputeContractTest(unittest.TestCase):
    def test_recompute_script_accepts_an_independent_run_directory(self):
        from methods.sparsegad.audit.recompute_run import recompute_contract
        self.assertEqual(recompute_contract()['checkpoint_name'], 'checkpoint_auprc_best.pt')
        self.assertEqual(recompute_contract()['selection_mask'], 'val_mask')
        self.assertEqual(recompute_contract()['evaluation_mask'], 'test_mask')
