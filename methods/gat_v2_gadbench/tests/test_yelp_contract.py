import sys
import unittest
import json
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class YelpContractTest(unittest.TestCase):
    def test_frozen_yelp_contract(self):
        from contracts import yelp_contract

        contract = yelp_contract()
        self.assertEqual(contract["input_dim"], 32)
        self.assertEqual(contract["training_graph_edges"], 7739912)
        self.assertEqual(contract["train_mask_count"], 18381)
        self.assertEqual(contract["val_mask_count"], 9099)
        self.assertEqual(contract["test_mask_count"], 18474)

    def test_formal_config_is_locked_to_diagnostic_protocol(self):
        path = Path(__file__).parents[1] / "configs" / "yelp_protocol_v2_gadbench_hidden64_formal.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["run_type"], "formal")
        self.assertEqual(config["result_dir_template"], "results/experiments/gat_v2_gadbench/yelp/protocol_v2_gadbench_hidden64/formal/seed_{seed}")


if __name__ == "__main__":
    unittest.main()
