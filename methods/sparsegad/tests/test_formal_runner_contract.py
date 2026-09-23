import unittest
from tempfile import TemporaryDirectory
from pathlib import Path


class SparseGADFormalRunnerContractTest(unittest.TestCase):
    def test_formal_contract_uses_validation_only_selection(self):
        from methods.sparsegad.src.run_formal import formal_contract
        contract = formal_contract()
        self.assertEqual(contract['max_epoch'], 200)
        self.assertEqual(contract['patience'], 50)
        self.assertEqual(contract['early_stop_metric'], 'validation_auprc')
        self.assertEqual(contract['checkpoint_metric'], 'validation_auprc')
        self.assertEqual(contract['threshold_protocol'], 'validation_F1_macro_grid_0.05_to_0.95')

    def test_amazon_prefix_is_verified_as_uncovered(self):
        from methods.sparsegad.src.run_formal import amazon_prefix_uncovered
        self.assertTrue(amazon_prefix_uncovered([False] * 3305, [False] * 3305, [False] * 3305))
        self.assertFalse(amazon_prefix_uncovered([True] + [False] * 3304, [False] * 3305, [False] * 3305))

    def test_oom_record_is_explicitly_marked(self):
        from methods.sparsegad.src.run_formal import failure_record
        record = failure_record('amazon', 0, 'formal', 'CUDA out of memory', 12.5, 20480.0)
        self.assertEqual(record['status'], 'OOM')
        self.assertEqual(record['dataset'], 'amazon')
        self.assertIn('CUDA out of memory', record['error'])

    def test_archive_existing_failure_preserves_preexisting_artifacts(self):
        from methods.sparsegad.src.run_formal import archive_existing_failure
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'preflight.json').write_text('{}', encoding='utf-8')
            result = archive_existing_failure(folder, 'amazon', 0, 'diagnostic', 'CUDA out of memory', 2.0, 3.0)
            self.assertEqual(result['status'], 'OOM')
            self.assertTrue((folder / 'failure.json').exists())
            self.assertEqual((folder / 'preflight.json').read_text(encoding='utf-8'), '{}')
