import unittest


class NRGLFrozenProtocolContractTest(unittest.TestCase):
    def test_candidate_contract_disables_synthetic_label_noise_and_uses_frozen_masks(self):
        from methods.nrgl.src.protocol import candidate_protocol_contract

        contract = candidate_protocol_contract()
        self.assertTrue(contract["frozen_masks_required"])
        self.assertFalse(contract["synthetic_label_noise_injection"])
        self.assertEqual(contract["loss_mask"], "train_mask")
        self.assertEqual(contract["checkpoint_mask"], "val_mask")
        self.assertEqual(contract["threshold_mask"], "val_mask")
        self.assertEqual(contract["test_metric_mask"], "test_mask")


if __name__ == "__main__":
    unittest.main()
