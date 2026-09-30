import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


class RecomputeFormalCheckpointContractTest(unittest.TestCase):
    def test_contract_recomputes_only_saved_checkpoint_on_test_mask(self):
        from recompute_formal_checkpoint import recompute_contract

        contract = recompute_contract()
        self.assertTrue(contract["checkpoint_read_only"])
        self.assertFalse(contract["backward"])
        self.assertFalse(contract["optimizer_step"])
        self.assertFalse(contract["threshold_search"])
        self.assertTrue(contract["test_mask_only"])


if __name__ == "__main__":
    unittest.main()
