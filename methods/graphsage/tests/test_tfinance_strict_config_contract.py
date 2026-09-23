import json
import unittest
from pathlib import Path


class TFinanceStrictConfigContractTest(unittest.TestCase):
    def test_tfinance_formal_config_enables_strict_determinism_without_protocol_changes(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "tfinance_graphsage_gadbench_h64_strict_formal.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["dataset"], "tfinance")
        self.assertTrue(config["strict_determinism"])
        self.assertEqual(config["h_feats"], 64)
        self.assertEqual(config["aggregation"], "pool")
        self.assertEqual(config["learning_rate"], 0.01)
        self.assertEqual(config["max_epoch"], 200)
        self.assertEqual(config["patience"], 50)
        self.assertEqual(config["expected_training_graph_edges"], 42484443)


if __name__ == "__main__":
    unittest.main()
