"""Checkpoint-only, frozen-mask GHRN evaluator; never trains or saves models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from methods.ghrn.src.model import GHRNModel
from methods.ghrn.src.protocol import final_test_values, validation_values
from methods.ghrn.src.runner import (
    edge_reduction_probability,
    load_frozen_hsmad_data,
    official_style_random_walk_update,
    prepare_training_graph,
)


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def compare_metrics(original: dict[str, object], recomputed: dict[str, object]) -> dict[str, dict[str, object]]:
    fields = ('f1_macro', 'auroc', 'predicted_anomaly_count', 'actual_anomaly_count', 'confusion_matrix')
    def equivalent(left: object, right: object) -> bool:
        if isinstance(left, float) and isinstance(right, float):
            return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        return left == right
    comparison = {
        field: {'original': original.get(field), 'recomputed': recomputed.get(field), 'match': equivalent(original.get(field), recomputed.get(field))}
        for field in fields
    }
    comparison['threshold'] = {
        'original': original.get('threshold'),
        'recomputed': recomputed.get('checkpoint_threshold'),
        'match': original.get('threshold') == recomputed.get('checkpoint_threshold'),
    }
    comparison['validation_threshold'] = {
        'original': original.get('threshold'),
        'recomputed': recomputed.get('validation_selected_threshold'),
        'match': original.get('threshold') == recomputed.get('validation_selected_threshold'),
    }
    return comparison


def recompute_run(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    original = json.loads((run_dir / 'metrics.json').read_text())
    config = json.loads((run_dir / 'config_snapshot.json').read_text())
    raw, features, labels, masks, _ = load_frozen_hsmad_data(config['dataset'])
    device = torch.device('cuda')
    graph = prepare_training_graph(raw).to(device)
    features, labels = features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    with torch.no_grad():
        stage1 = GHRNModel(features.shape[1], int(config['hidden_dim']), int(config['order'])).to(device)
        stage1.load_state_dict(torch.load(run_dir / 'checkpoint_bootstrap_auprc_best.pt', map_location=device)['model_state_dict'])
        bootstrap_probability = edge_reduction_probability(stage1(graph, features))
        reduced = official_style_random_walk_update(graph, bootstrap_probability, float(config['delete_ratio']))
        stage2 = GHRNModel(features.shape[1], int(config['hidden_dim']), int(config['order'])).to(device)
        stage2.load_state_dict(torch.load(run_dir / 'checkpoint_auprc_best.pt', map_location=device)['model_state_dict'])
        probability = edge_reduction_probability(stage2(reduced, features))[:, 1]
    selected_threshold, validation_f1, validation_auprc = validation_values(labels, probability, masks['val_mask'])
    metrics = final_test_values(labels, probability, masks['test_mask'], float(original['threshold']))
    metrics.update({
        'checkpoint_threshold': float(original['threshold']),
        'validation_selected_threshold': selected_threshold,
        'validation_f1_macro': validation_f1,
        'validation_auprc': validation_auprc,
        'recomputed_stage2_training_edges': reduced.num_edges(),
    })
    comparison = compare_metrics(original, metrics)
    status = 'recompute_match' if all(item['match'] for item in comparison.values()) else 'recompute_mismatch'
    return {'status': status, 'metrics': metrics, 'comparison': comparison}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--audit-name', default='audit_recompute')
    args = parser.parse_args()
    result = recompute_run(args.run_dir)
    out = Path(args.run_dir) / args.audit_name
    _dump(out / 'recompute_metrics.json', result['metrics'])
    _dump(out / 'compare_to_original.json', {'status': result['status'], 'comparison': result['comparison']})
    (out / 'audit.md').write_text('# GHRN checkpoint-only recompute audit\n\n' + json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
