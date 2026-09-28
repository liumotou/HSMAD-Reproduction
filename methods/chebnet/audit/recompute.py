"""Read-only checkpoint recomputation for isolated ChebNet artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
from pathlib import Path

PROJECT_ROOT = Path('/root/autodl-tmp/HSMAD')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
site.addsitedir('/root/miniconda3/lib/python3.10/site-packages')
import dgl
import torch

from methods.chebnet.src.adapter import dgl_to_pyg_frozen
from methods.chebnet.src.model import ChebNetCandidate
from methods.chebnet.src.protocol import test_metrics
from methods.chebnet.src.run import prepare_training_graph


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_metrics(original: dict[str, object], recomputed: dict[str, object]) -> dict[str, object]:
    fields = ('f1_macro', 'auroc', 'threshold', 'predicted_anomaly_count')
    differences = {}
    for field in fields:
        left, right = original.get(field), recomputed.get(field)
        equal = abs(float(left) - float(right)) <= 1e-12 if field in ('f1_macro', 'auroc', 'threshold') else left == right
        if not equal:
            differences[field] = {'original': left, 'recomputed': right}
    return {'status': 'recompute_match' if not differences else 'recompute_mismatch', 'differences': differences}


def recompute(output: Path) -> dict[str, object]:
    config, original = json.loads((output / 'config_snapshot.json').read_text()), json.loads((output / 'metrics.json').read_text())
    raw = dgl.load_graphs(str(PROJECT_ROOT / config['dataset_file']))[0][0]
    data = dgl_to_pyg_frozen(prepare_training_graph(raw))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); data = data.to(device)
    checkpoint_path = output / 'checkpoint_auprc_best.pt'; checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ChebNetCandidate(data.x.shape[1], int(config['hidden_dim']), 2, int(config['order']), float(config['dropout'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    with torch.no_grad(): probability = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    recomputed = test_metrics(data.y, probability, data.test_mask, float(original['threshold']))
    recomputed.update({'threshold': float(original['threshold']), 'best_epoch': int(checkpoint['epoch']), 'checkpoint_sha256': sha256_file(checkpoint_path)})
    result = compare_metrics(original, recomputed)
    result.update({'original': {key: original[key] for key in ('f1_macro','auroc','threshold','predicted_anomaly_count','best_epoch')}, 'recomputed': recomputed})
    audit = output / 'audit_recompute'; audit.mkdir(exist_ok=True)
    (audit / 'recompute_result.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(recompute(Path(args.output)), sort_keys=True))

if __name__ == '__main__':
    main()
