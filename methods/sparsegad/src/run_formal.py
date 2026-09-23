"""Bounded SparseGAD candidate runner with validation-only model selection.

This is deliberately separate from ``run_smoke.py``: smoke artifacts remain
immutable, while diagnostic/formal runs use the same frozen graph and mask
protocol with a 200-epoch AUPRC-selected checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from methods.sparsegad.src.model import SparseGADModel
from methods.sparsegad.src.run_smoke import THRESHOLDS, select

ROOT = Path('/root/autodl-tmp/HSMAD')


def formal_contract() -> dict:
    return {
        'max_epoch': 200,
        'patience': 50,
        'early_stop_metric': 'validation_auprc',
        'checkpoint_metric': 'validation_auprc',
        'threshold_protocol': 'validation_F1_macro_grid_0.05_to_0.95',
    }


def amazon_prefix_uncovered(train_mask, val_mask, test_mask) -> bool:
    """Amazon's official HSMAD split excludes its first 3305 nodes."""
    return not bool(torch.as_tensor(train_mask)[:3305].any() or torch.as_tensor(val_mask)[:3305].any() or torch.as_tensor(test_mask)[:3305].any())


def failure_record(dataset: str, seed_value: int, run_type: str, error: str,
                   wall_time_sec: float, peak_gpu_mb: float) -> dict:
    """Structured terminal failure record; does not alter a checkpoint or rerun."""
    return {
        'dataset': dataset, 'seed': seed_value, 'run_type': run_type,
        'status': 'OOM' if 'out of memory' in error.lower() else 'ERROR',
        'error': error, 'wall_time_sec': wall_time_sec, 'peak_gpu_mb': peak_gpu_mb,
    }


def archive_existing_failure(run_dir: Path, dataset: str, seed_value: int, run_type: str,
                             error: str, wall_time_sec: float, peak_gpu_mb: float) -> dict:
    """Record a captured failure alongside, never in place of, existing artifacts."""
    record = failure_record(dataset, seed_value, run_type, error, wall_time_sec, peak_gpu_mb)
    _write(run_dir / 'failure.json', record)
    return record


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    dgl.seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _evaluate(model, graph, features, labels, masks):
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph, features), 1)[:, 1]
    threshold, val_f1 = select(labels, probability, masks['val_mask'])
    val_truth = labels[masks['val_mask']].detach().cpu().numpy()
    val_score = probability[masks['val_mask']].detach().cpu().numpy()
    return probability, threshold, val_f1, float(roc_auc_score(val_truth, val_score)), float(average_precision_score(val_truth, val_score))


def _append_run(path: Path, record: dict) -> None:
    fields = ['method', 'protocol_version', 'positioning', 'dataset', 'seed', 'run_type', 'status', 'f1_macro', 'auroc', 'auprc', 'best_epoch', 'threshold', 'actual_epochs', 'predicted_anomaly_count', 'wall_time_sec', 'peak_gpu_mb', 'edge_access']
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in fields})


def _run(config: dict, config_path: Path, seed_value: int, run_type: str) -> dict:
    contract = formal_contract()
    base = ROOT / config['result_root'] / run_type / f'seed_{seed_value}'
    if base.exists():
        raise FileExistsError(base)
    _seed(seed_value)
    raw_path = ROOT / config['dataset_file']
    raw = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    features = raw.ndata['feature'].float()
    labels = raw.ndata['label'].long().reshape(-1)
    masks = {key: raw.ndata[key].bool() for key in ('train_mask', 'val_mask', 'test_mask')}
    if graph.num_nodes() != config['expected']['nodes'] or graph.num_edges() != config['expected']['training_edges']:
        raise RuntimeError('frozen graph mismatch')
    if config['dataset'] == 'amazon' and not amazon_prefix_uncovered(masks['train_mask'], masks['val_mask'], masks['test_mask']):
        raise RuntimeError('Amazon frozen mask must exclude nodes 0..3304')
    hashes = {
        'dataset_file_sha256': _sha(raw_path), 'feature_sha256': _tensor_sha(features),
        'label_sha256': _tensor_sha(labels), **{f'{key}_sha256': _tensor_sha(value) for key, value in masks.items()},
        'model_py_sha256': _sha(ROOT / 'methods/sparsegad/src/model.py'),
        'runner_py_sha256': _sha(Path(__file__)), 'config_sha256': _sha(config_path),
    }
    snapshot = dict(config, seed=seed_value, run_type=run_type, **contract)
    base.mkdir(parents=True)
    _write(base / 'config_snapshot.json', snapshot)
    _write(base / 'preflight.json', {
        'passed': True, 'raw_graph': {'nodes': raw.num_nodes(), 'edges': raw.num_edges()},
        'training_graph': {'nodes': graph.num_nodes(), 'edges': graph.num_edges()},
        'feature_shape': list(features.shape), 'mask_counts': {key: int(value.sum()) for key, value in masks.items()},
        'edge_access': config['edge_access'], 'hashes': hashes, 'contract': contract,
        'amazon_prefix_3305_uncovered': config['dataset'] != 'amazon' or amazon_prefix_uncovered(masks['train_mask'], masks['val_mask'], masks['test_mask']),
    })
    device = torch.device('cuda')
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {key: value.to(device) for key, value in masks.items()}
    model = SparseGADModel(features.shape[1], config['hidden_dim'], 2, config['num_layers'], config['dropout'], config['dropout_adj']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    torch.cuda.reset_peak_memory_stats(device)
    best_auprc, best_epoch, stale, checkpoint = -float('inf'), 0, 0, None
    history, start = [], time.monotonic()
    for epoch in range(1, contract['max_epoch'] + 1):
        model.train()
        logits = model(graph, features)
        loss = torch.nn.functional.cross_entropy(logits[masks['train_mask']], labels[masks['train_mask']])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        probability, threshold, val_f1, val_auroc, val_auprc = _evaluate(model, graph, features, labels, masks)
        improved = val_auprc > best_auprc
        if improved:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            checkpoint = {'epoch': epoch, 'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'config': snapshot}
        else:
            stale += 1
        record = {'epoch': epoch, 'train_loss': float(loss), 'validation_f1_macro': val_f1, 'validation_auroc': val_auroc, 'validation_auprc': val_auprc, 'validation_threshold': threshold, 'checkpoint_selected': improved, 'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if stale >= contract['patience']:
            break
    assert checkpoint is not None
    checkpoint_path = base / 'checkpoint_auprc_best.pt'
    torch.save(checkpoint, checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    probability, threshold, val_f1, val_auroc, val_auprc = _evaluate(model, graph, features, labels, masks)
    truth = labels[masks['test_mask']].detach().cpu().numpy()
    score = probability[masks['test_mask']].detach().cpu().numpy()
    prediction = (score >= threshold).astype(np.int64)
    result = {
        'method': config['method'], 'protocol_version': config['protocol_version'], 'positioning': config['positioning'],
        'dataset': config['dataset'], 'seed': seed_value, 'run_type': run_type, 'status': 'OK',
        'f1_macro': float(f1_score(truth, prediction, average='macro')), 'auroc': float(roc_auc_score(truth, score)),
        'auprc': float(average_precision_score(truth, score)), 'best_epoch': best_epoch, 'threshold': threshold,
        'validation_f1_macro': val_f1, 'validation_auroc': val_auroc, 'validation_auprc': val_auprc,
        'actual_epochs': len(history), 'predicted_anomaly_count': int(prediction.sum()), 'actual_anomaly_count': int(truth.sum()),
        'wall_time_sec': time.monotonic() - start, 'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024 ** 2),
        'edge_access': config['edge_access'], 'hashes': hashes, 'checkpoint_sha256': _sha(checkpoint_path),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    _write(base / 'validation_history.json', history)
    _write(base / 'metrics.json', result)
    _write(base / 'artifact_sha256s.json', {path.name: _sha(path) for path in base.iterdir() if path.is_file()})
    _append_run(ROOT / config['result_root'] / run_type / 'runs.csv', result)
    print('RUN_COMPLETE ' + json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-type', choices=('diagnostic', 'formal'), required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for seed_value in args.seeds:
        _run(config, config_path, seed_value, args.run_type)


if __name__ == '__main__':
    main()
