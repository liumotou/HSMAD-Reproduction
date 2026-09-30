import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class GADBenchGATV2ContractTest(unittest.TestCase):
    def test_hidden_width_is_64_with_four_16_dimensional_heads(self):
        from contracts import architecture_contract

        contract = architecture_contract()
        self.assertEqual(contract["hidden_total_dim"], 64)
        self.assertEqual(contract["num_heads"], 4)
        self.assertEqual(contract["per_head_dim"], 16)
        self.assertEqual(contract["num_gat_blocks"], 2)
        self.assertEqual(contract["activation"], "GELU")
        self.assertEqual(contract["output_dim"], 2)

    def test_formal_execution_contract_keeps_diagnostic_hyperparameters(self):
        from contracts import formal_execution_contract

        contract = formal_execution_contract()
        self.assertEqual(contract["max_epoch"], 200)
        self.assertEqual(contract["patience"], 50)
        self.assertEqual(contract["early_stop_metric"], "validation_AUPRC")
        self.assertEqual(contract["checkpoint_metric"], "validation_AUPRC_best")


if __name__ == "__main__":
    unittest.main()
