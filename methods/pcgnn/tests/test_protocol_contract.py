import unittest

import torch

from methods.pcgnn.src.protocol import (
    build_train_positive_nodes,
    labels_for_forward,
    validate_single_relation,
)


class PCGNNProtocolContractTests(unittest.TestCase):
    def test_train_positive_nodes_never_escape_frozen_train_mask(self):
        labels = torch.tensor([0, 1, 1, 0, 1])
        mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
        self.assertTrue(torch.equal(build_train_positive_nodes(labels, mask), torch.tensor([1])))

    def test_training_labels_reject_nodes_outside_train_mask(self):
        labels = torch.tensor([0, 1, 1])
        mask = torch.tensor([1, 1, 0], dtype=torch.bool)
        with self.assertRaises(ValueError):
            labels_for_forward(labels, torch.tensor([1, 2]), mask, train_flag=True)

    def test_evaluation_forward_receives_placeholder_not_val_or_test_truth(self):
        labels = torch.tensor([0, 1, 1])
        mask = torch.tensor([1, 0, 0], dtype=torch.bool)
        value = labels_for_forward(labels, torch.tensor([1, 2]), mask, train_flag=False)
        self.assertTrue(torch.equal(value, torch.zeros(2, dtype=torch.long)))

    def test_relation_triplication_is_rejected(self):
        validate_single_relation([object()])
        with self.assertRaises(ValueError):
            validate_single_relation([object(), object(), object()])


if __name__ == "__main__":
    unittest.main()
