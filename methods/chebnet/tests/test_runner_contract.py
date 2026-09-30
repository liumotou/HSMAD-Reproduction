import unittest
from pathlib import Path

from methods.chebnet.src.run import build_run_spec


class ChebNetRunnerContractTest(unittest.TestCase):
    def test_run_spec_isolated_and_candidate_labeled(self):
        spec = build_run_spec({
            'dataset': 'weibo', 'seed': 0, 'run_type': 'smoke',
            'result_dir': 'results/experiments/chebnet/weibo/chebnet_h64_candidate/smoke/seed_0',
            'max_epoch': 5, 'patience': 50, '_config_sha256': 'test',
        })
        self.assertEqual(spec.dataset, 'weibo')
        self.assertEqual(spec.seed, 0)
        self.assertEqual(spec.run_type, 'smoke')
        self.assertTrue(spec.result_dir.as_posix().endswith('/smoke/seed_0'))
        self.assertEqual(spec.result_label, 'candidate_protocol_not_author_exact')

    def test_runner_rejects_nonpositive_epoch_limit(self):
        with self.assertRaises(ValueError):
            build_run_spec({'dataset': 'weibo', 'seed': 0, 'run_type': 'smoke', 'result_dir': 'x', 'max_epoch': 0, 'patience': 50, '_config_sha256': 'test'})


if __name__ == '__main__':
    unittest.main()
