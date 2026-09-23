import unittest


class BWGNNFormalRunnerContractTest(unittest.TestCase):
    def test_formal_contract_uses_validation_only_selection(self):
        from methods.bwgnn.src.run_formal import formal_contract
        contract = formal_contract()
        self.assertEqual(contract['max_epoch'], 200)
        self.assertEqual(contract['patience'], 50)
        self.assertEqual(contract['early_stop_metric'], 'validation_auprc')
        self.assertEqual(contract['checkpoint_metric'], 'validation_auprc')
        self.assertEqual(contract['threshold_protocol'], 'validation_F1_macro_grid_0.05_to_0.95')
