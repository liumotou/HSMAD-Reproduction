"""Contract for the isolated one-step GraphSAGE determinism probe."""

import sys
import unittest
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT_DIR))


class DeterminismProbeContractTest(unittest.TestCase):
    def test_probe_contract_has_exactly_one_optimizer_step_and_no_evaluation_outputs(self):
        from determinism_probe import probe_contract

        contract = probe_contract()

        self.assertEqual(contract["optimizer_steps"], 1)
        self.assertFalse(contract["save_checkpoint"])
        self.assertFalse(contract["compute_test_metrics"])
        self.assertEqual(contract["run_type"], "determinism_probe")
        self.assertEqual(contract["seed"], 0)


if __name__ == "__main__":
    unittest.main()
