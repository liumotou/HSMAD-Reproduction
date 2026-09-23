import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))


class RecomputeComparisonContract(unittest.TestCase):
    def test_machine_roundoff_is_distinguished_from_semantic_difference(self):
        from recompute_kipf_v2 import compare_scalar

        self.assertEqual(compare_scalar(0.5, 0.5000000000000001)["kind"], "roundoff")
        self.assertEqual(compare_scalar(0.5, 0.50001)["kind"], "mismatch")
        self.assertEqual(compare_scalar(4, 4)["kind"], "exact")


if __name__ == "__main__":
    unittest.main()
