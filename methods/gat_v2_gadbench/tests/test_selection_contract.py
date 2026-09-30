import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class SelectionContractTest(unittest.TestCase):
    def test_auprc_controls_patience_and_checkpoint_while_f1_is_diagnostic(self):
        from selection import update_selection

        state = {"best_auprc": -1.0, "auprc_best_epoch": None, "best_f1": -1.0,
                 "f1_best_epoch": None, "patience_counter": 0}
        state, auprc_improved, f1_improved = update_selection(state, 1, 0.70, 0.50)
        self.assertTrue(auprc_improved)
        self.assertTrue(f1_improved)
        state, auprc_improved, f1_improved = update_selection(state, 2, 0.60, 0.60)
        self.assertFalse(auprc_improved)
        self.assertTrue(f1_improved)
        self.assertEqual(state["auprc_best_epoch"], 1)
        self.assertEqual(state["f1_best_epoch"], 2)
        self.assertEqual(state["patience_counter"], 1)


if __name__ == "__main__":
    unittest.main()
