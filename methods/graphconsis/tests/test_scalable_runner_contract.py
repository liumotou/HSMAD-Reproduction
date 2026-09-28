import unittest

from methods.graphconsis.src.run import select_adjacency_builder


class ScalableRunnerContractTest(unittest.TestCase):
    def test_legacy_is_default_for_existing_configs(self):
        name, _ = select_adjacency_builder({})
        self.assertEqual(name, "legacy_python_lists")

    def test_tfinance_can_explicitly_select_stable_csr(self):
        name, _ = select_adjacency_builder({"adjacency_builder": "stable_csr"})
        self.assertEqual(name, "stable_csr")

    def test_unknown_builder_is_rejected(self):
        with self.assertRaises(ValueError):
            select_adjacency_builder({"adjacency_builder": "unknown"})


if __name__ == "__main__":
    unittest.main()
