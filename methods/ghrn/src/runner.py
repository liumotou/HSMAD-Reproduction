"""No-test-leakage GHRN candidate runner.

The official repository's model topology and random-walk edge score are kept,
but split generation and per-epoch test evaluation are intentionally excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn.pytorch.conv import EdgeWeightNorm

from methods.ghrn.src.model import GHRNModel
from methods.ghrn.src.protocol import (
    ROOT,
    class_weight,
    ensure_new_result_directory,
    final_test_values,
    masked_loss,
    result_directory,
    setup_seed,
    validation_values,
)

DATASETS = {
    'weibo': {'file': 'datasets/weibo', 'nodes': 8405, 'feature_dim': 400},
    'tolokers': {'file': 'datasets/tolokers', 'nodes': 11758, 'feature_dim': 10},
    'amazon': {'file': 'datasets/amazon', 'nodes': 11944, 'feature_dim': 25},
    'tfinance': {'file': 'datasets/tfinance', 'nodes': 39357, 'feature_dim': 10},
}


def dataset_contract(dataset: str) -> dict[str, object]:
    if dataset not in DATASETS:
        raise ValueError(f'unsupported dataset: {dataset}')
    return dict(DATASETS[dataset])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def prepare_training_graph(raw):
    return dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw, copy_ndata=True)))


def edge_reduction_probability(logits: torch.Tensor) -> torch.Tensor:
    """Return the official GHRN edge-score input: one two-class vector/node."""
    return torch.softmax(logits, dim=1)


def load_frozen_hsmad_data(dataset: str):
    contract = dataset_contract(dataset)
    raw_path = ROOT / str(contract['file'])
    raw = dgl.load_graphs(str(raw_path))[0][0]
    if raw.num_nodes() != contract['nodes']:
        raise RuntimeError(f'{dataset} node count mismatch: {raw.num_nodes()} != {contract["nodes"]}')
    features = raw.ndata['feature'].float()
    labels = raw.ndata['label'].long().reshape(-1)
    if features.shape[1] != contract['feature_dim']:
        raise RuntimeError(f'{dataset} feature dimension mismatch')
    masks = {name: raw.ndata[name].bool().clone() for name in ('train_mask', 'val_mask', 'test_mask')}
    meta = {
        'dataset_file_sha256': sha256_file(raw_path),
        'feature_sha256': sha256_tensor(features),
        'label_sha256': sha256_tensor(labels),
        'mask_counts': {name: int(mask.sum()) for name, mask in masks.items()},
        'mask_sha256': {name: sha256_tensor(mask) for name, mask in masks.items()},
        'raw_nodes': raw.num_nodes(),
        'raw_edges': raw.num_edges(),
    }
    return raw, features, labels, masks, meta


def candidate_config(dataset: str, run_type: str = 'formal') -> dict[str, object]:
    dataset_contract(dataset)
    if run_type not in {'smoke', 'diagnostic', 'formal'}:
        raise ValueError(f'unsupported run type: {run_type}')
    return {
        'dataset': dataset,
        'run_type': run_type,
        'positioning': 'candidate_protocol_not_author_exact',
        'protocol_version': 'ghrn_official_h64_frozen_mask_candidate',
        'hidden_dim': 64,
        'order': 2,
        'delete_ratio': 0.015,
        'optimizer': 'Adam',
        'learning_rate': 0.01,
        'weight_decay': 0.0,
        'class_weight': 'train_normal_over_train_anomaly',
        'max_epoch': 5 if run_type == 'smoke' else 200,
        'patience': 5 if run_type == 'smoke' else 50,
        'early_stop_metric': 'validation_auprc',
        'checkpoint_metric': 'validation_auprc',
        'threshold_protocol': 'validation_F1_macro_grid_0.05_to_0.95',
        'graph_preprocess': 'to_bidirected->remove_self_loop->add_self_loop',
        'edge_access': 'graph_edges_required',
        'official_core_source': 'blacksingular/GHRN@d47e047b76df429c0c0a8444f5d9a5dea57f6801',
    }


def official_style_random_walk_update(graph, probabilities: torch.Tensor, delete_ratio: float):
    """Official dataset.py:124-149 semantics with only a device placement fix."""
    if not 0.0 <= delete_ratio < 1.0:
        raise ValueError('delete_ratio must be in [0, 1)')
    with graph.local_scope():
        weights = torch.ones(graph.num_edges(), device=graph.device)
        graph.edata['w'] = EdgeWeightNorm(norm='both')(graph, weights)
        graph.ndata['h'] = probabilities
        graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'ay'))
        graph.ndata['ly'] = probabilities - graph.ndata['ay']
        graph.apply_edges(lambda edges: {'inner_black': (edges.src['ly'] * edges.dst['ly']).sum(axis=1)})
        remove_count = int(delete_ratio * graph.num_edges())
        edge_ids = graph.edata['inner_black'].sort()[1][:remove_count]
        return dgl.remove_edges(graph, edge_ids)


def _train_validation_selected_stage(graph, features, labels, masks, config, device, stage: str):
    model = GHRNModel(features.shape[1], int(config['hidden_dim']), int(config['order'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    weights = class_weight(labels, masks['train_mask'])
    best_auprc = float('-inf')
    best_state = None
    best_epoch = 0
    remaining = 0
    history = []
    for epoch in range(1, int(config['max_epoch']) + 1):
        model.train()
        logits = model(graph, features)
        loss = masked_loss(logits, labels, masks['train_mask'], weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = edge_reduction_probability(model(graph, features))
        threshold, validation_f1, validation_auprc = validation_values(labels, probability[:, 1], masks['val_mask'])
        history.append({
            'stage': stage,
            'epoch': epoch,
            'train_loss': float(loss.detach().cpu()),
            'validation_auprc': validation_auprc,
            'validation_f1_macro': validation_f1,
            'validation_threshold': threshold,
        })
        if validation_auprc > best_auprc:
            best_auprc = validation_auprc
            best_epoch = epoch
            remaining = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            remaining += 1
        if remaining >= int(config['patience']):
            break
    if best_state is None:
        raise RuntimeError('no validation checkpoint was selected')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = edge_reduction_probability(model(graph, features))
    threshold, validation_f1, validation_auprc = validation_values(labels, probability[:, 1], masks['val_mask'])
    return model, probability.detach(), {
        'best_epoch': best_epoch,
        'actual_epochs': epoch,
        'validation_auprc': validation_auprc,
        'validation_f1_macro': validation_f1,
        'validation_threshold': threshold,
        'history': history,
        'class_weight': [float(weights[0]), float(weights[1])],
        'train_normal_count': int((labels[masks['train_mask']] == 0).sum()),
        'train_anomaly_count': int((labels[masks['train_mask']] == 1).sum()),
    }


def run_one(dataset: str, seed: int, run_type: str = 'diagnostic', retry_id: str | None = None) -> dict[str, object]:
    config = candidate_config(dataset, run_type)
    output = result_directory(dataset, run_type, seed, retry_id=retry_id)
    ensure_new_result_directory(output)
    setup_seed(seed)
    raw, features, labels, masks, meta = load_frozen_hsmad_data(dataset)
    graph = prepare_training_graph(raw)
    meta['training_nodes'] = graph.num_nodes()
    meta['training_edges'] = graph.num_edges()
    meta['model_sha256'] = sha256_file(Path(__file__).with_name('model.py'))
    meta['protocol_sha256'] = sha256_file(Path(__file__).with_name('protocol.py'))
    meta['runner_sha256'] = sha256_file(Path(__file__))
    dump(output / 'config_snapshot.json', config)
    dump(output / 'preflight.json', {'passed': True, 'seed': seed, 'meta': meta})
    device = torch.device('cuda')
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    stage1_model, stage1_probability, stage1 = _train_validation_selected_stage(graph, features, labels, masks, config, device, 'bootstrap')
    reduced = official_style_random_walk_update(graph, stage1_probability, float(config['delete_ratio']))
    stage2_model, _, stage2 = _train_validation_selected_stage(reduced, features, labels, masks, config, device, 'reduced_graph')
    stage2_model.eval()
    with torch.no_grad():
        final_probability = edge_reduction_probability(stage2_model(reduced, features))[:, 1]
    threshold, validation_f1, validation_auprc = validation_values(labels, final_probability, masks['val_mask'])
    metrics = final_test_values(labels, final_probability, masks['test_mask'], threshold)
    metrics.update({
        'method': 'GHRN-official-core-h64',
        'dataset': dataset,
        'seed': seed,
        'run_type': run_type,
        'retry_id': retry_id,
        'status': 'smoke' if run_type == 'smoke' else 'OK',
        'positioning': config['positioning'],
        'protocol_version': config['protocol_version'],
        'best_epoch': stage2['best_epoch'],
        'actual_epochs': stage2['actual_epochs'],
        'validation_auprc': validation_auprc,
        'validation_f1_macro': validation_f1,
        'threshold': threshold,
        'stage1': {key: value for key, value in stage1.items() if key != 'history'},
        'stage1_training_edges': graph.num_edges(),
        'stage2_training_edges': reduced.num_edges(),
        'wall_time_sec': time.monotonic() - start,
        'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024 ** 2),
        'hashes': meta,
        'edge_access': 'graph_edges_required',
    })
    torch.save({'model_state_dict': stage2_model.state_dict(), 'config': config, 'best_epoch': stage2['best_epoch']}, output / 'checkpoint_auprc_best.pt')
    torch.save({'model_state_dict': stage1_model.state_dict(), 'config': config, 'best_epoch': stage1['best_epoch']}, output / 'checkpoint_bootstrap_auprc_best.pt')
    dump(output / 'validation_history.json', stage1['history'] + stage2['history'])
    dump(output / 'metrics.json', metrics)
    dump(output / 'artifact_sha256s.json', {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASETS))
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--run-type', default='diagnostic', choices=('smoke', 'diagnostic', 'formal'))
    parser.add_argument('--retry-id', default=None)
    args = parser.parse_args()
    base = result_directory(args.dataset, args.run_type, 0, retry_id=args.retry_id).parent
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_seed in args.seeds.split(','):
        seed = int(raw_seed)
        try:
            rows.append(run_one(args.dataset, seed, args.run_type, retry_id=args.retry_id))
        except Exception as error:
            rows.append({'dataset': args.dataset, 'seed': seed, 'run_type': args.run_type, 'status': 'ERROR', 'error': repr(error)})
        fields = sorted({key for row in rows for key in row})
        with (base / 'runs.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, sort_keys=True))


if __name__ == '__main__':
    main()
