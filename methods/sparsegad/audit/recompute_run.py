"""Read-only checkpoint recomputation for SparseGAD candidate artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from methods.sparsegad.src.model import SparseGADModel
from methods.sparsegad.src.run_smoke import select

ROOT = Path('/root/autodl-tmp/HSMAD')


def recompute_contract() -> dict:
    return {
        'checkpoint_name': 'checkpoint_auprc_best.pt',
        'selection_mask': 'val_mask',
        'evaluation_mask': 'test_mask',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--audit-name', default='audit_recompute')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = run_dir / args.audit_name
    output.mkdir(exist_ok=False)
    original = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    state = torch.load(run_dir / recompute_contract()['checkpoint_name'], map_location='cpu')
    config = state['config']
    raw = dgl.load_graphs(str(ROOT / config['dataset_file']))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    features = raw.ndata['feature'].float()
    labels = raw.ndata['label'].long().reshape(-1)
    masks = {key: raw.ndata[key].bool() for key in ('train_mask', 'val_mask', 'test_mask')}
    model = SparseGADModel(features.shape[1], config['hidden_dim'], 2, config['num_layers'], config['dropout'], config['dropout_adj'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph, features), 1)[:, 1]
    threshold, val_f1 = select(labels, probability, masks['val_mask'])
    truth = labels[masks['test_mask']].numpy()
    score = probability[masks['test_mask']].numpy()
    prediction = (score >= threshold).astype(int)
    result = {
        'f1_macro': float(f1_score(truth, prediction, average='macro')),
        'auroc': float(roc_auc_score(truth, score)),
        'auprc': float(average_precision_score(truth, score)),
        'threshold': threshold,
        'validation_f1_macro': val_f1,
        'predicted_anomaly_count': int(prediction.sum()),
        'actual_anomaly_count': int(truth.sum()),
    }
    comparison = {
        key: {'original': original.get(key), 'recomputed': value,
              'match': abs(original[key] - value) <= 1e-6 if isinstance(value, float) else original.get(key) == value}
        for key, value in result.items()
    }
    status = 'recompute_match' if all(item['match'] for item in comparison.values()) else 'recompute_mismatch'
    (output / 'recompute_metrics.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    (output / 'compare_to_original.json').write_text(json.dumps({'status': status, 'comparison': comparison}, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'status': status, 'metrics': result}, sort_keys=True))


if __name__ == '__main__':
    main()
