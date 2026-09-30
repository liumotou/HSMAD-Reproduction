import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/gcn/src"))

from run_kipf_v3_f1_checkpoint import should_replace_f1_checkpoint

PROTOCOL = "protocol_v3_kipf_two_layer_hidden64_f1_checkpoint"

def test_f1_checkpoint_selection_ignores_higher_auprc():
    current = {"val_f1_macro": 0.6706471495, "val_auprc": 0.2648216980}
    previous = {"val_f1_macro": 0.5041313438, "val_auprc": 0.3639122508}
    assert should_replace_f1_checkpoint(current, previous) is True
    assert should_replace_f1_checkpoint(previous, current) is False
    assert should_replace_f1_checkpoint(current, current) is False

def test_v3_config_contracts():
    expected = [
        ("protocol_v3_weibo_smoke_full.json", "weibo", "smoke_full", 400),
        ("protocol_v3_amazon_candidate_full.json", "amazon", "candidate_full", 25),
    ]
    for name, dataset, run_type, input_dim in expected:
        config = json.loads((ROOT / "methods/gcn/configs" / name).read_text())
        assert config["protocol_version"] == PROTOCOL
        assert config["dataset"] == dataset
        assert config["run_type"] == run_type
        assert config["input_dim"] == input_dim
        assert config["hidden_dim"] == 64 and config["output_dim"] == 2 and config["layers"] == 2
        assert config["dropout"] == 0.0 and config["weight_decay"] == 0.0
        assert config["optimizer"] == "Adam" and config["learning_rate"] == 0.01
        assert config["class_weight"] == "none"
        assert config["early_stop_metric"] == "validation F1-Macro"
        assert config["checkpoint_selection_metric"] == "validation F1-Macro"
    amazon = json.loads((ROOT / "methods/gcn/configs/protocol_v3_amazon_candidate_full.json").read_text())
    assert amazon["uncovered_prefix_nodes"] == 3305

if __name__ == "__main__":
    test_f1_checkpoint_selection_ignores_higher_auprc()
    test_v3_config_contracts()
    print("PASS: protocol v3 F1-checkpoint contract")
