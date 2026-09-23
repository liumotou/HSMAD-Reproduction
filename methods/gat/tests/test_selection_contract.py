import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class SelectionContractTest(unittest.TestCase):
    def test_f1_controls_patience_while_auprc_controls_checkpoint(self):
        from selection import update_selection

        state = {"best_f1": -1.0, "f1_best_epoch": None, "best_auprc": -1.0,
                 "checkpoint_epoch": None, "patience_counter": 0}
        state, f1_improved, auprc_improved = update_selection(state, 1, 0.70, 0.30)
        self.assertTrue(f1_improved)
        self.assertTrue(auprc_improved)
        state, f1_improved, auprc_improved = update_selection(state, 2, 0.69, 0.40)
        self.assertFalse(f1_improved)
        self.assertTrue(auprc_improved)
        self.assertEqual(state["f1_best_epoch"], 1)
        self.assertEqual(state["checkpoint_epoch"], 2)
        self.assertEqual(state["patience_counter"], 1)


if __name__ == "__main__":
    unittest.main()
