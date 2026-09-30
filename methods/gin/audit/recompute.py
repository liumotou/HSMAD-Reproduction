"""Read-only checkpoint recomputation for isolated GIN artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from methods.project_paths import project_root

PROJECT_ROOT = project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import dgl
import torch

from methods.gin.src.adapter import dgl_to_pyg_frozen
from methods.gin.src.model import GINCandidate
from methods.gin.src.protocol import test_metrics
from methods.gin.src.run import prepare_training_graph


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_metrics(
    original: dict[str, object], recomputed: dict[str, object]
) -> dict[str, object]:
    fields = ('f1_macro', 'auroc', 'threshold', 'predicted_anomaly_count', 'best_epoch')
    differences: dict[str, dict[str, object]] = {}
    for field in fields:
        left, right = original.get(field), recomputed.get(field)
        if field in ('f1_macro', 'auroc', 'threshold'):
            equal = abs(float(left) - float(right)) <= 1e-12
        else:
            equal = left == right
        if not equal:
            differences[field] = {'original': left, 'recomputed': right}
    return {
        'status': 'recompute_match' if not differences else 'recompute_mismatch',
        'differences': differences,
    }


def recompute(output: Path) -> dict[str, object]:
    config = json.loads((output / 'config_snapshot.json').read_text(encoding='utf-8'))
    original = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    raw = dgl.load_graphs(str(PROJECT_ROOT / config['dataset_file']))[0][0]
    data = dgl_to_pyg_frozen(prepare_training_graph(raw))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    checkpoint_path = output / 'checkpoint_auprc_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GINCandidate(
        data.x.shape[1], int(config['hidden_dim']), 2, float(config['dropout'])
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    recomputed = test_metrics(
        data.y, probability, data.test_mask, float(original['threshold'])
    )
    recomputed.update(
        {
            'threshold': float(original['threshold']),
            'best_epoch': int(checkpoint['epoch']),
            'checkpoint_sha256': sha256_file(checkpoint_path),
        }
    )
    result = compare_metrics(original, recomputed)
    result.update(
        {
            'original': {
                key: original[key]
                for key in (
                    'f1_macro',
                    'auroc',
                    'threshold',
                    'predicted_anomaly_count',
                    'best_epoch',
                )
            },
            'recomputed': recomputed,
        }
    )
    audit = output / 'audit_recompute'
    audit.mkdir(exist_ok=True)
    (audit / 'recompute_result.json').write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding='utf-8'
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(recompute(Path(args.output)), sort_keys=True))


if __name__ == '__main__':
    main()
