import unittest


class NRGLFrozenDataAdapterContractTest(unittest.TestCase):
    def test_weibo_contract_uses_existing_frozen_file_without_split_generation(self):
        from src.data import frozen_dataset_contract

        contract = frozen_dataset_contract("weibo")
        self.assertEqual(contract["dataset_file"], "datasets/weibo")
        self.assertEqual(contract["nodes"], 8405)
        self.assertEqual(contract["feature_dim"], 400)
        self.assertTrue(contract["reuse_frozen_masks"])
        self.assertFalse(contract["regenerate_masks"])
        self.assertFalse(contract["inject_label_noise"])


if __name__ == "__main__":
    unittest.main()
