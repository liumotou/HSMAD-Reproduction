import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


class RecomputeTFinanceStrictDiagnosticContractTest(unittest.TestCase):
    def test_contract_is_read_only_and_uses_saved_threshold(self):
        from recompute_tfinance_strict_diagnostic import recompute_contract

        contract = recompute_contract()
        self.assertTrue(contract["checkpoint_read_only"])
        self.assertFalse(contract["backward"])
        self.assertFalse(contract["optimizer_step"])
        self.assertFalse(contract["threshold_search"])
        self.assertTrue(contract["test_mask_only"])


if __name__ == "__main__":
    unittest.main()
