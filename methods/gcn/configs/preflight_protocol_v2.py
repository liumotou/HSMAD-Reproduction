import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = 'protocol_v2_kipf_two_layer_hidden64'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    assert config['protocol_version'] == EXPECTED
    assert config['run_type'] in ('smoke', 'diagnostic') and config['seed'] == 0
    output = ROOT / 'results/experiments/gcn' / config['dataset'] / EXPECTED / config['run_type'] / 'seed_0'
    assert EXPECTED in str(output)
    assert output.name == 'seed_0'
    assert output / 'terminal.log' != output / 'checkpoint_val_auprc_best.pt'
    assert config['layers'] == 2 and config['hidden_dim'] == 64 and config['output_dim'] == 2
    assert config['dropout'] == 0.0 and config['weight_decay'] == 0.0
    print(json.dumps({'preflight': 'PASS', 'protocol_version': EXPECTED, 'output_dir': str(output),
                      'terminal_log': str(output / 'terminal.log'),
                      'checkpoint': str(output / 'checkpoint_val_auprc_best.pt'),
                      'metrics': str(output / 'metrics.json'), 'runs_csv': str(output / 'runs.csv')}))


if __name__ == '__main__':
    main()
