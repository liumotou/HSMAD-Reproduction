import os
import unittest


SOURCE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src", "train.py"))


class FeatureOnlyIsolationTest(unittest.TestCase):
    def test_runner_has_no_edge_or_graph_preprocessing_api(self):
        with open(SOURCE, encoding="utf-8") as handle:
            source = handle.read()
        prohibited = (".edges(", "to_bidirected", "add_self_loop", "remove_self_loop", "update_all", "message_passing")
        for token in prohibited:
            self.assertNotIn(token, source)
        self.assertIn('ndata["feature"]', source)
        self.assertIn('ndata["label"]', source)
        self.assertIn('("feature", "label", "train_mask", "val_mask", "test_mask")', source)
        self.assertIn('ndata[name]', source)


if __name__ == "__main__":
    unittest.main()
