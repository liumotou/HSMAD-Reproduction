import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class RunTypeContractTest(unittest.TestCase):
    def test_missing_run_type_keeps_legacy_formal_default(self):
        from run_formal import validated_run_type

        self.assertEqual(validated_run_type({}), "formal")

    def test_diagnostic_run_type_is_explicitly_supported_and_isolated(self):
        from run_formal import validated_run_type

        self.assertEqual(validated_run_type({"run_type": "diagnostic"}), "diagnostic")


if __name__ == "__main__":
    unittest.main()
