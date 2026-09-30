"""Contract for the strict one-step Amazon GraphSAGE determinism probe."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


class StrictDeterminismProbeContractTest(unittest.TestCase):
    def test_strict_contract_requires_one_step_and_error_on_nondeterministic_ops(self):
        from strict_determinism_probe import strict_probe_contract

        contract = strict_probe_contract()

        self.assertEqual(contract["optimizer_steps"], 1)
        self.assertFalse(contract["save_checkpoint"])
        self.assertFalse(contract["compute_validation_or_test_metrics"])
        self.assertTrue(contract["deterministic_algorithms"])
        self.assertFalse(contract["deterministic_warn_only"])
        self.assertEqual(contract["required_cublas_workspace_config"], ":4096:8")
        self.assertEqual(contract["required_pythonhashseed"], "0")


if __name__ == "__main__":
    unittest.main()
