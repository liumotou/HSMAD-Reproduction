import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'methods/gcn/configs/amazon_protocol_v1_paper_hidden64_diagnostic.json'


def test_amazon_diagnostic_contract():
    config = json.loads(CONFIG.read_text())
    assert config['dataset'] == 'amazon'
    assert config['seed'] == 0
    assert config['max_epoch'] == 200
    assert config['patience'] == 50
    assert config['hidden_dim'] == 64
    assert config['dropout'] == 0.0
    assert config['weight_decay'] == 0.0
    assert config['early_stop_metric'] == 'validation F1-Macro'
    assert config['checkpoint_selection_metric'] == 'validation AUPRC'
    assert config['graph_policy'] == 'full_transductive_graph'
    assert config['exclude_prefix_nodes_from_masks_and_metrics'] == 3305


if __name__ == '__main__':
    test_amazon_diagnostic_contract()
    print('PASS: Amazon diagnostic contract')
