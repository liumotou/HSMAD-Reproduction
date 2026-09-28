"""Isolated ChebNet candidate runner using HSMAD frozen data/masks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import site
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path('/root/autodl-tmp/HSMAD')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
site.addsitedir('/root/miniconda3/lib/python3.10/site-packages')
import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from methods.chebnet.src.adapter import dgl_to_pyg_frozen
from methods.chebnet.src.model import ChebNetCandidate
from methods.chebnet.src.protocol import select_validation_threshold, test_metrics

ROOT = Path('/root/autodl-tmp/HSMAD')


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    run_type: str
    result_dir: Path
    max_epoch: int
    patience: int
    result_label: str = 'candidate_protocol_not_author_exact'


def build_run_spec(config: dict[str, object]) -> RunSpec:
    if not config.get('_config_sha256'):
        raise ValueError('missing config SHA256 provenance')
    max_epoch, patience = int(config['max_epoch']), int(config['patience'])
    if max_epoch < 1 or patience < 1:
        raise ValueError('max_epoch and patience must be positive')
    if config['run_type'] not in {'smoke', 'diagnostic', 'formal'}:
        raise ValueError('unsupported run_type')
    return RunSpec(str(config['dataset']), int(config['seed']), str(config['run_type']), Path(str(config['result_dir'])), max_epoch, patience)


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


def append_run_row(path: Path, fields: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fields})


def prepare_training_graph(raw):
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    for name in ('feature', 'label', 'train_mask', 'val_mask', 'test_mask'):
        graph.ndata[name] = raw.ndata[name]
    return graph


def train(config: dict[str, object]) -> dict[str, object]:
    spec = build_run_spec(config)
    output = ROOT / spec.result_dir
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {output}')
    setup_seed(spec.seed)
    raw_path = ROOT / str(config['dataset_file'])
    raw = dgl.load_graphs(str(raw_path))[0][0]
    graph = prepare_training_graph(raw)
    data = dgl_to_pyg_frozen(graph)
    masks = {name: getattr(data, name) for name in ('train_mask', 'val_mask', 'test_mask')}
    if any(torch.logical_and(masks[a], masks[b]).any() for a, b in (('train_mask','val_mask'), ('train_mask','test_mask'), ('val_mask','test_mask'))):
        raise RuntimeError('frozen masks overlap')
    expected = config.get('expected', {})
    observed = {'nodes': int(graph.num_nodes()), 'training_edges': int(graph.num_edges())}
    for field, value in expected.items():
        if observed.get(field) != value:
            raise RuntimeError(f'frozen input mismatch for {field}: {observed.get(field)!r} != {value!r}')
    hashes = {
        'dataset_file_sha256': sha256_file(raw_path), 'feature_sha256': sha256_tensor(data.x),
        'label_sha256': sha256_tensor(data.y), 'train_mask_sha256': sha256_tensor(data.train_mask),
        'val_mask_sha256': sha256_tensor(data.val_mask), 'test_mask_sha256': sha256_tensor(data.test_mask),
        'raw_graph_edges_sha256': sha256_edges(raw), 'training_graph_edges_sha256': sha256_edges(graph),
        'adapter_py_sha256': sha256_file(ROOT / 'methods/chebnet/src/adapter.py'),
        'model_py_sha256': sha256_file(ROOT / 'methods/chebnet/src/model.py'),
        'protocol_py_sha256': sha256_file(ROOT / 'methods/chebnet/src/protocol.py'),
        'runner_py_sha256': sha256_file(Path(__file__)),
        'config_sha256': str(config['_config_sha256']),
    }
    output.mkdir(parents=True)
    write_json(output / 'config_snapshot.json', config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = ChebNetCandidate(data.x.shape[1], int(config['hidden_dim']), 2, int(config['order']), float(config['dropout'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    write_json(output / 'preflight.json', {
        'passed': True, 'candidate_protocol_not_author_exact': True, 'dataset': spec.dataset,
        'raw_graph': {'nodes': raw.num_nodes(), 'edges': raw.num_edges()},
        'training_graph': observed, 'feature_shape': list(data.x.shape),
        'mask_counts': {name: int(value.sum()) for name, value in masks.items()}, 'hashes': hashes,
        'edge_access': 'graph_edges_required', 'model': {'operator': 'torch_geometric.nn.ChebConv', 'layers': 2, 'hidden_dim': int(config['hidden_dim']), 'order': int(config['order'])},
        'selection': {'early_stop': 'validation_auprc', 'checkpoint': 'validation_auprc', 'threshold': 'validation_f1_macro_grid_0.05_to_0.95'},
    })
    started, history, best_epoch, best_auprc = time.monotonic(), [], 0, float('-inf')
    checkpoint_path = output / 'checkpoint_auprc_best.pt'
    for epoch in range(1, spec.max_epoch + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): probabilities = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
        val_y, val_p = data.y[data.val_mask].detach().cpu().numpy(), probabilities[data.val_mask].detach().cpu().numpy()
        threshold, val_f1 = select_validation_threshold(data.y, probabilities, data.val_mask)
        val_auprc, val_auroc = float(average_precision_score(val_y, val_p)), float(roc_auc_score(val_y, val_p))
        record = {'epoch': epoch, 'train_loss': float(loss.item()), 'validation_f1_macro': val_f1, 'validation_auroc': val_auroc, 'validation_auprc': val_auprc, 'validation_threshold': threshold, 'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        if val_auprc > best_auprc:
            best_epoch, best_auprc = epoch, val_auprc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config}, checkpoint_path)
        if epoch - best_epoch >= spec.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    with torch.no_grad(): probabilities = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    threshold, validation_f1 = select_validation_threshold(data.y, probabilities, data.val_mask)
    result = test_metrics(data.y, probabilities, data.test_mask, threshold)
    metrics = {'method': 'ChebNet', 'protocol_version': str(config['protocol_version']), 'candidate_protocol_not_author_exact': True, 'dataset': spec.dataset, 'seed': spec.seed, 'run_type': spec.run_type, 'status': 'smoke' if spec.run_type == 'smoke' else 'OK', 'actual_epochs': len(history), 'best_epoch': checkpoint['epoch'], 'f1_macro': result['f1_macro'], 'auroc': result['auroc'], 'auprc': result['auprc'], 'threshold': threshold, 'validation_f1_macro': validation_f1, 'validation_auprc': best_auprc, 'predicted_anomaly_count': result['predicted_anomaly_count'], 'actual_anomaly_count': result['actual_anomaly_count'], 'confusion_matrix': result['confusion_matrix'], 'wall_time_sec': time.monotonic() - started, 'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0, 'edge_access': 'graph_edges_required', 'hashes': hashes, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'validation_history.json', history); write_json(output / 'metrics.json', metrics)
    fields = ['method','protocol_version','dataset','seed','run_type','status','actual_epochs','best_epoch','f1_macro','auroc','threshold','wall_time_sec','peak_gpu_mb','edge_access']
    append_run_row(output.parent / 'runs.csv', fields, metrics)
    write_json(output / 'artifact_sha256s.json', {item.name: sha256_file(item) for item in output.iterdir() if item.is_file()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['_config_sha256'] = sha256_file(config_path)
    print('RUN_COMPLETE ' + json.dumps(train(config), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
