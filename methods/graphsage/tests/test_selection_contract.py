import unittest


class GraphSAGESelectionContractTest(unittest.TestCase):
    def test_validation_auprc_improvement_resets_patience_and_selects_checkpoint(self):
        from src.selection import update_auprc_selection

        state = {"best_auprc": -1.0, "best_epoch": None, "patience_counter": 4}
        state, improved, should_stop = update_auprc_selection(state, epoch=7, validation_auprc=0.6, patience=50)

        self.assertTrue(improved)
        self.assertFalse(should_stop)
        self.assertEqual(state, {"best_auprc": 0.6, "best_epoch": 7, "patience_counter": 0})

    def test_gadbench_style_stops_only_after_more_than_patience_non_improvements(self):
        from src.selection import update_auprc_selection

        state = {"best_auprc": 0.6, "best_epoch": 7, "patience_counter": 50}
        state, improved, should_stop = update_auprc_selection(state, epoch=58, validation_auprc=0.6, patience=50)

        self.assertFalse(improved)
        self.assertTrue(should_stop)
        self.assertEqual(state["patience_counter"], 51)


if __name__ == "__main__":
    unittest.main()
