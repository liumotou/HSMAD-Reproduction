import unittest

import torch


class ValidationOnlySelectionContractTest(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [3.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 3.0],
            [0.0, 1.0], [1.0, 0.0],
        ])
        self.labels = torch.tensor([0, 0, 1, 1, 1, 0])
        self.train = torch.tensor([True, True, False, False, False, False])
        self.val = torch.tensor([False, False, True, True, False, False])
        self.test = torch.tensor([False, False, False, False, True, True])

    def test_validation_selection_never_reads_test_labels(self):
        from methods.nrgl.src.selection import select_validation_checkpoint_metrics

        baseline = select_validation_checkpoint_metrics(
            self.logits, self.labels, self.val, threshold_candidates=[0.05, 0.5, 0.95]
        )
        altered = self.labels.clone()
        altered[self.test] = 1 - altered[self.test]
        after_test_label_change = select_validation_checkpoint_metrics(
            self.logits, altered, self.val, threshold_candidates=[0.05, 0.5, 0.95]
        )
        self.assertEqual(baseline, after_test_label_change)

    def test_test_metrics_never_reads_validation_labels(self):
        from methods.nrgl.src.selection import compute_test_metrics

        baseline = compute_test_metrics(self.logits, self.labels, self.test, threshold=0.5)
        altered = self.labels.clone()
        altered[self.val] = 1 - altered[self.val]
        after_val_label_change = compute_test_metrics(self.logits, altered, self.test, threshold=0.5)
        self.assertEqual(baseline, after_val_label_change)

    def test_checkpoint_key_is_validation_auprc(self):
        from methods.nrgl.src.selection import should_replace_checkpoint

        self.assertTrue(should_replace_checkpoint(candidate_auprc=0.8, best_auprc=0.7))
        self.assertFalse(should_replace_checkpoint(candidate_auprc=0.7, best_auprc=0.8))


if __name__ == "__main__":
    unittest.main()
