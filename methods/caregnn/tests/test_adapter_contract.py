import unittest

import torch

from methods.caregnn.src.protocol import (
    build_train_positive_nodes,
    labels_for_model_forward,
    validate_single_relation,
)


class CareGNNAdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 1, 1, 0, 1, 0])
        self.train_mask = torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.bool)

    def test_train_positive_pool_uses_only_train_mask(self):
        positives = build_train_positive_nodes(self.labels, self.train_mask)
        self.assertEqual(positives.tolist(), [1])

    def test_training_forward_rejects_nodes_outside_train_mask(self):
        with self.assertRaises(ValueError):
            labels_for_model_forward(
                self.labels, torch.tensor([1, 2]), self.train_mask, train_flag=True
            )

    def test_evaluation_forward_never_receives_ground_truth(self):
        result = labels_for_model_forward(
            self.labels, torch.tensor([2, 4]), self.train_mask, train_flag=False
        )
        self.assertTrue(torch.equal(result, torch.zeros(2, dtype=torch.long)))

    def test_relation_must_be_one_flattened_relation_not_triplicated(self):
        validate_single_relation([object()])
        with self.assertRaises(ValueError):
            validate_single_relation([object(), object(), object()])


if __name__ == "__main__":
    unittest.main()
