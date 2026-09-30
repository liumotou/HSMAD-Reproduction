import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'methods/gcn/configs/weibo_protocol_v1_paper_hidden64_formal.json'
RUNNER = ROOT / 'methods/gcn/src/run_weibo_protocol_v1_paper_hidden64.py'


def test_formal_config_contract():
    config = json.loads(CONFIG.read_text())
    assert config['protocol_version'] == 'v1_paper_hidden64'
    assert config['run_type'] == 'formal'
    assert config['max_epoch'] == 200
    assert config['patience'] == 50
    assert config['hidden_dim'] == 64
    assert config['dropout'] == 0.0
    assert config['weight_decay'] == 0.0
    assert config['early_stop_metric'] == 'validation F1-Macro'
    assert config['checkpoint_selection_metric'] == 'validation AUPRC'
    assert config['edge_access'] == 'graph_edges_required'
    assert RUNNER.is_file()


if __name__ == '__main__':
    test_formal_config_contract()
    print('PASS: formal protocol contract')
