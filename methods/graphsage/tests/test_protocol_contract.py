import unittest

import torch


class GraphSAGEProtocolContractTest(unittest.TestCase):
    def test_weight_uses_only_frozen_training_labels(self):
        from src.protocol import class_weight_from_train_labels

        weight, normal_count, anomaly_count = class_weight_from_train_labels(
            torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        )

        self.assertEqual((normal_count, anomaly_count), (3, 2))
        self.assertEqual(weight, [1.0, 1.5])

    def test_threshold_grid_is_closed_over_point_zero_five_to_point_nine_five(self):
        from src.protocol import threshold_grid

        self.assertEqual(threshold_grid()[0], 0.05)
        self.assertEqual(threshold_grid()[-1], 0.95)
        self.assertEqual(len(threshold_grid()), 19)


if __name__ == "__main__":
    unittest.main()
