import sys
import unittest
import json
from pathlib import Path
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class AmazonContractTest(unittest.TestCase):
    def test_frozen_amazon_contract_preserves_uncovered_nodes(self):
        from contracts import amazon_contract

        contract = amazon_contract()
        self.assertEqual(contract["input_dim"], 25)
        self.assertEqual(contract["uncovered_prefix_nodes"], 3305)
        self.assertEqual(contract["training_graph_edges"], 8808728)
        self.assertEqual(contract["train_mask_count"], 3455)
        self.assertEqual(contract["val_mask_count"], 1710)
        self.assertEqual(contract["test_mask_count"], 3474)

    def test_frozen_amazon_train_ratio_weight(self):
        config_path = Path(__file__).parents[1] / "configs" / "amazon_protocol_v2_gadbench_hidden64_diagnostic_full.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["expected_anomaly_weight"], 3127 / 328)

    def test_formal_config_is_locked_to_diagnostic_protocol(self):
        config_path = Path(__file__).parents[1] / "configs" / "amazon_protocol_v2_gadbench_hidden64_formal.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["run_type"], "formal")
        self.assertEqual(config["result_dir_template"], "results/experiments/gat_v2_gadbench/amazon/protocol_v2_gadbench_hidden64/formal/seed_{seed}")
        self.assertEqual(config["early_stop_protocol"], "validation_AUPRC_GADBench")
        self.assertEqual(config["checkpoint_protocol"], "validation_AUPRC_best_GADBench")


if __name__ == "__main__":
    unittest.main()
