import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


class StrictForwardProbeContractTest(unittest.TestCase):
    def test_contract_is_checkpoint_only_and_requires_strict_runtime(self):
        from strict_forward_probe import strict_probe_contract

        contract = strict_probe_contract()
        self.assertEqual(contract["required_cublas_workspace_config"], ":4096:8")
        self.assertEqual(contract["required_pythonhashseed"], "0")
        self.assertTrue(contract["torch_deterministic_algorithms"])
        self.assertFalse(contract["warn_only"])
        self.assertFalse(contract["backward"])
        self.assertFalse(contract["optimizer_step"])
        self.assertFalse(contract["checkpoint_save"])

    def test_seed_can_be_recovered_from_a_legacy_seed_directory(self):
        from strict_forward_probe import seed_from_run_dir

        self.assertEqual(seed_from_run_dir("/tmp/formal/seed_1", {}), 1)


if __name__ == "__main__":
    unittest.main()
