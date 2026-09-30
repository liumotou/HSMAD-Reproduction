import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class StrictFormalRunnerContractTest(unittest.TestCase):
    def test_strict_formal_contract_requires_prelaunch_environment(self):
        from run_formal import strict_determinism_contract

        contract = strict_determinism_contract()

        self.assertEqual(contract["required_cublas_workspace_config"], ":4096:8")
        self.assertEqual(contract["required_pythonhashseed"], "0")
        self.assertTrue(contract["dgl_seed"])
        self.assertTrue(contract["torch_deterministic_algorithms"])
        self.assertFalse(contract["warn_only"])

    def test_strict_formal_setup_rejects_missing_prelaunch_contract(self):
        from run_formal import configure_strict_determinism

        previous_cublas = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        previous_hash = os.environ.pop("PYTHONHASHSEED", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "before Python starts"):
                configure_strict_determinism(0)
        finally:
            if previous_cublas is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas
            if previous_hash is not None:
                os.environ["PYTHONHASHSEED"] = previous_hash


if __name__ == "__main__":
    unittest.main()
