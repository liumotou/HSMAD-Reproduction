"""One isolated, non-formal BWGNN Weibo smoke run."""
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
from sklearn.metrics import average_precision_score, roc_auc_score

from methods.bwgnn.src.model import GADBenchBWGNN
from methods.bwgnn.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics
from methods.project_paths import project_root

ROOT = project_root()


def smoke_contract() -> dict[str, object]:
    return {
        'dataset': 'weibo', 'seed': 0, 'max_epoch': 5, 'run_type': 'smoke',
        'edge_access': 'graph_edges_required', 'hidden_dim': 64, 'order': 2,
        'checkpoint_protocol': 'none_smoke_last_epoch',
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha256_edges(graph) -> str:
    source, destination = graph.edges(order='eid')
    return hashlib.sha256(source.cpu().numpy().tobytes() + destination.cpu().numpy().tobytes()).hexdigest()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def prepare_training_graph(raw):
    """Apply the frozen graph transform while preserving the model's feature input."""
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    graph.ndata['feature'] = raw.ndata['feature'].float()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    options = parser.parse_args()
    config_path = Path(options.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if {name: config[name] for name in smoke_contract()} != smoke_contract():
        raise RuntimeError('smoke configuration violates fixed BWGNN smoke contract')
    output = ROOT / config['result_dir']
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing smoke output: {output}')
    setup_seed(config['seed'])
    raw_path = ROOT / config['dataset_file']
    raw = dgl.load_graphs(str(raw_path))[0][0]
    graph = prepare_training_graph(raw)
    features = raw.ndata['feature'].float()
    labels = raw.ndata['label'].long().reshape(-1)
    masks = {name: raw.ndata[name].bool() for name in ('train_mask', 'val_mask', 'test_mask')}
    hashes = {
        'dataset_file_sha256': sha256_file(raw_path), 'feature_sha256': sha256_tensor(features),
        'label_sha256': sha256_tensor(labels), 'train_mask_sha256': sha256_tensor(masks['train_mask']),
        'val_mask_sha256': sha256_tensor(masks['val_mask']), 'test_mask_sha256': sha256_tensor(masks['test_mask']),
        'raw_graph_edges_sha256': sha256_edges(raw), 'training_graph_edges_sha256': sha256_edges(graph),
        'model_py_sha256': sha256_file(ROOT / 'methods/bwgnn/src/model.py'),
        'protocol_py_sha256': sha256_file(ROOT / 'methods/bwgnn/src/protocol.py'),
        'runner_py_sha256': sha256_file(Path(__file__)), 'config_sha256': sha256_file(config_path),
    }
    for field, expected in config['expected'].items():
        actual = graph.num_nodes() if field == 'nodes' else graph.num_edges() if field == 'training_edges' else hashes[field]
        if actual != expected:
            raise RuntimeError(f'frozen input mismatch for {field}: {actual!r} != {expected!r}')
    if any(torch.logical_and(masks[left], masks[right]).any() for left, right in (('train_mask','val_mask'), ('train_mask','test_mask'), ('val_mask','test_mask'))):
        raise RuntimeError('frozen masks overlap')
    train_labels = labels[masks['train_mask']]
    anomaly_count, normal_count = int(train_labels.sum()), int((train_labels == 0).sum())
    if not anomaly_count:
        raise RuntimeError('train mask contains no anomalies')
    class_weight = [1.0, normal_count / anomaly_count]
    output.mkdir(parents=True)
    write_json(output / 'config_snapshot.json', config)
    write_json(output / 'preflight.json', {
        'passed': True, 'raw_graph': {'nodes': raw.num_nodes(), 'edges': raw.num_edges()},
        'training_graph': {'nodes': graph.num_nodes(), 'edges': graph.num_edges()},
        'feature_shape': list(features.shape), 'mask_counts': {name: int(mask.sum()) for name, mask in masks.items()},
        'train_normal_count': normal_count, 'train_anomaly_count': anomaly_count, 'class_weight': class_weight,
        'hashes': hashes, 'edge_access': config['edge_access'], 'model': {'h_feats': 64, 'order': 2, 'mlp_layers': 2},
    })
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = GADBenchBWGNN(features.shape[1], 64, 2, 2, 2, 0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    history = []
    for epoch in range(1, config['max_epoch'] + 1):
        model.train()
        logits = model(graph)
        loss = masked_cross_entropy(logits, labels, masks['train_mask'], weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(graph), dim=1)[:, 1]
        threshold, val_f1 = select_validation_threshold(labels, probabilities, masks['val_mask'])
        val_truth = labels[masks['val_mask']].detach().cpu().numpy()
        val_probability = probabilities[masks['val_mask']].detach().cpu().numpy()
        record = {
            'epoch': epoch, 'train_loss': float(loss.item()), 'validation_f1_macro': val_f1,
            'validation_auroc': float(roc_auc_score(val_truth, val_probability)),
            'validation_auprc': float(average_precision_score(val_truth, val_probability)),
            'validation_threshold': threshold,
            'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    torch.save({'epoch': config['max_epoch'], 'model_state_dict': model.state_dict(), 'config': config}, output / 'checkpoint_last_epoch.pt')
    with torch.no_grad():
        final_probabilities = torch.softmax(model(graph), dim=1)[:, 1]
    threshold, validation_f1 = select_validation_threshold(labels, final_probabilities, masks['val_mask'])
    test = test_metrics(labels, final_probabilities, masks['test_mask'], threshold)
    metrics = {
        'method': config['method'], 'protocol_version': config['protocol_version'], 'dataset': config['dataset'],
        'seed': config['seed'], 'run_type': 'smoke', 'status': 'smoke', 'actual_epochs': config['max_epoch'],
        'f1_macro': test['f1_macro'], 'auroc': test['auroc'], 'auprc': test['auprc'], 'threshold': threshold,
        'validation_f1_macro': validation_f1, 'predicted_anomaly_count': test['predicted_anomaly_count'],
        'actual_anomaly_count': test['actual_anomaly_count'], 'confusion_matrix': test['confusion_matrix'],
        'wall_time_sec': time.monotonic() - started,
        'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0,
        'edge_access': config['edge_access'], 'hashes': hashes,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / 'validation_history.json', history)
    write_json(output / 'metrics.json', metrics)
    runs = output.parent / 'runs.csv'
    fields = ['method','protocol_version','dataset','seed','run_type','status','actual_epochs','f1_macro','auroc','threshold','wall_time_sec','peak_gpu_mb','edge_access']
    with runs.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics[field] for field in fields})
    artifacts = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    write_json(output / 'artifact_sha256s.json', artifacts)
    print('SMOKE_COMPLETE ' + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
