import os
import sys
import unittest


SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from protocol_v3_dropout_zero_probe import allowed_v1_v3_diff


class ProtocolV3ConfigDiffTest(unittest.TestCase):
    def test_only_dropout_is_an_allowed_semantic_difference(self):
        v1 = {"dropout": 0.5, "class_weight": None, "learning_rate": 0.01}
        v3 = {"dropout": 0.0, "class_weight": None, "learning_rate": 0.01}
        self.assertEqual(allowed_v1_v3_diff(v1, v3), {"dropout": {"v1": 0.5, "v3": 0.0}})


if __name__ == "__main__":
    unittest.main()
