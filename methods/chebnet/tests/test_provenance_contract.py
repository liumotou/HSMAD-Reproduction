import unittest

from methods.chebnet.src.run import build_run_spec


class ChebNetProvenanceContractTest(unittest.TestCase):
    def test_config_fingerprint_is_required_for_training_provenance(self):
        with self.assertRaises(ValueError):
            build_run_spec({'dataset': 'weibo', 'seed': 0, 'run_type': 'smoke', 'result_dir': 'x', 'max_epoch': 5, 'patience': 50})


if __name__ == '__main__':
    unittest.main()
