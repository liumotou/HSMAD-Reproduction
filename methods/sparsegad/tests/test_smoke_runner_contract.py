import unittest


class SparseGADSmokeRunnerContractTest(unittest.TestCase):
    def test_tfinance_smoke_contract_accepts_only_frozen_smoke_controls(self):
        from methods.sparsegad.src.run_smoke import validate_smoke_config

        config = {
            'dataset': 'tfinance',
            'seed': 0,
            'max_epoch': 5,
            'run_type': 'smoke',
        }
        self.assertIsNone(validate_smoke_config(config))

    def test_smoke_contract_rejects_nonzero_seed(self):
        from methods.sparsegad.src.run_smoke import validate_smoke_config

        config = {
            'dataset': 'tfinance',
            'seed': 1,
            'max_epoch': 5,
            'run_type': 'smoke',
        }
        with self.assertRaises(RuntimeError):
            validate_smoke_config(config)
