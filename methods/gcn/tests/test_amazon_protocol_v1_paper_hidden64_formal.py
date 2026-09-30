import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'methods/gcn/configs/amazon_protocol_v1_paper_hidden64_formal.json'
RUNNER = ROOT / 'methods/gcn/src/run_amazon_protocol_v1_paper_hidden64.py'


def test_amazon_formal_contract():
    config = json.loads(CONFIG.read_text())
    assert config['run_type'] == 'formal'
    assert config['training_seeds'] == list(range(10))
    assert config['hidden_dim'] == 64
    assert config['max_epoch'] == 200 and config['patience'] == 50
    assert config['class_weight'] == 'none'
    assert config['early_stop_metric'] == 'validation F1-Macro'
    assert config['checkpoint_selection_metric'] == 'validation AUPRC'
    assert RUNNER.is_file()


if __name__ == '__main__':
    test_amazon_formal_contract()
    print('PASS: Amazon formal contract')
