import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'methods/mlp/src'))
from utils import file_sha256, setup_seed, tensor_sha256

sys.path.append(str(ROOT / 'audit/mlp_reference/GADBench'))
from models.gnn import GCN

CONFIG_PATH = ROOT / 'methods/gcn/configs/amazon_protocol_v1_paper_hidden64_diagnostic.json'
OUT = ROOT / 'results/experiments/gcn/amazon/protocol_v1_paper_hidden64/diagnostic/seed_0'
REF = ROOT / 'audit/mlp_reference/GADBench/models/gnn.py'
COMMIT = 'f9aa021ce9b6c6580427fb633b596843be76ddc6'


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def threshold(labels, probabilities, candidates):
    answer = (-1.0, candidates[0])
    for candidate in candidates:
        score = f1_score(labels, (probabilities > candidate).astype(np.int64), average='macro')
        if score > answer[0]:
            answer = (float(score), float(candidate))
    return answer


def state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def write_csv(path, row):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    config = json.loads(CONFIG_PATH.read_text())
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'environment').mkdir()
    result = {'method': 'GCN', 'protocol_version': config['protocol_version'], 'dataset': 'amazon', 'seed': 0,
              'run_type': 'diagnostic', 'status': 'ERROR', 'edge_access': config['edge_access'],
              'config_sha256': sha_json(config), 'code_sha256': file_sha256(str(Path(__file__)))}
    try:
        setup_seed(0)
        raw = dgl.load_graphs(str(ROOT / 'datasets/amazon'))[0][0]
        required = ['feature', 'label', 'train_mask', 'val_mask', 'test_mask']
        if any(name not in raw.ndata for name in required):
            raise RuntimeError('Amazon DGL graph lacks a required frozen ndata field')
        feature = raw.ndata['feature'].float().contiguous()
        label = raw.ndata['label'].long().reshape(-1).contiguous()
        masks = {name: raw.ndata[name].bool().reshape(-1).contiguous() for name in required[2:]}
        prefix = config['exclude_prefix_nodes_from_masks_and_metrics']
        if any(int(mask[:prefix].sum()) != 0 for mask in masks.values()):
            raise RuntimeError('Amazon frozen mask unexpectedly covers one of the first 3305 nodes')
        covered = masks['train_mask'] | masks['val_mask'] | masks['test_mask']
        if int(covered[:prefix].sum()) != 0 or int((masks['train_mask'].int() + masks['val_mask'].int() + masks['test_mask'].int() > 1).sum()) != 0:
            raise RuntimeError('Amazon mask coverage or overlap invariant failed')
        fingerprints = {'dataset_file_sha256': file_sha256(str(ROOT / 'datasets/amazon')),
                        'feature_sha256': tensor_sha256(feature), 'label_sha256': tensor_sha256(label),
                        **{name + '_sha256': tensor_sha256(mask) for name, mask in masks.items()}}
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)));
        graph.ndata['feature'], graph.ndata['label'] = feature, label
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        graph, label = graph.to(device), label.to(device)
        masks = {name: mask.to(device) for name, mask in masks.items()}
        mask_stats = {name: {'nodes': int(mask.sum().cpu()), 'anomalies': int(label[mask].sum().cpu())} for name, mask in masks.items()}
        preflight = {'reference_repo': 'https://github.com/squareRoot3/GADBench.git', 'reference_commit': COMMIT,
                     'reference_file_sha256': file_sha256(str(REF)), 'runner_code_sha256': result['code_sha256'],
                     'input_sha256': fingerprints, 'raw_graph': {'nodes': raw.num_nodes(), 'edges': raw.num_edges()},
                     'training_graph': {'nodes': graph.num_nodes(), 'edges': graph.num_edges(), 'preprocess': config['graph_preprocess']},
                     'frozen_mask_stats': mask_stats, 'uncovered_prefix': {'node_count': prefix, 'covered_node_count': 0},
                     'graph_policy': config['graph_policy'], 'edge_access': config['edge_access'], 'config': config}
        (OUT / 'preflight.json').write_text(json.dumps(preflight, indent=2))
        (OUT / 'config_snapshot.json').write_text(json.dumps(config, indent=2))
        (OUT / 'environment/framework_versions.json').write_text(json.dumps({'python': sys.version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'dgl': dgl.__version__, 'gpu': torch.cuda.get_device_name(device)}, indent=2))
        model = GCN(feature.shape[1], 64, 2, 2, 1, 0.0, 'ReLU').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        f1_best, auprc_best, stalled, history = {'value': -1.0, 'state': None, 'epoch': 0}, {'value': -1.0, 'state': None, 'epoch': 0}, 0, []
        torch.cuda.reset_peak_memory_stats(device); start = time.perf_counter()
        for epoch in range(1, config['max_epoch'] + 1):
            model.train(); optimizer.zero_grad(); logits = model(graph)
            loss = F.cross_entropy(logits[masks['train_mask']], label[masks['train_mask']]); loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad(): probability = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
            vm, vy = masks['val_mask'].cpu().numpy(), label[masks['val_mask']].cpu().numpy()
            vf1, selected_threshold = threshold(vy, probability[vm], config['threshold_candidates'])
            vauroc, vauprc = float(roc_auc_score(vy, probability[vm])), float(average_precision_score(vy, probability[vm]))
            history.append({'epoch': epoch, 'loss': float(loss.detach().cpu()), 'val_f1_macro': vf1, 'val_auroc': vauroc, 'val_auprc': vauprc, 'threshold': selected_threshold})
            if vf1 > f1_best['value']: f1_best, stalled = {'value': vf1, 'state': state(model), 'epoch': epoch}, 0
            else: stalled += 1
            if vauprc > auprc_best['value']: auprc_best = {'value': vauprc, 'state': state(model), 'epoch': epoch}
            print(json.dumps({'epoch': epoch, 'val_f1_macro': vf1, 'val_auroc': vauroc, 'val_auprc': vauprc, 'f1_patience_count': stalled}), flush=True)
            if stalled > config['patience']: break
        f1_ckpt, auprc_ckpt = OUT / 'checkpoint_val_f1_earlystop_best.pt', OUT / 'checkpoint_val_auprc_best.pt'
        torch.save({'model_state_dict': f1_best['state'], 'epoch': f1_best['epoch']}, f1_ckpt); torch.save({'model_state_dict': auprc_best['state'], 'epoch': auprc_best['epoch']}, auprc_ckpt)
        model.load_state_dict(auprc_best['state']); model.eval()
        with torch.no_grad(): probability = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
        vm, vy = masks['val_mask'].cpu().numpy(), label[masks['val_mask']].cpu().numpy(); _, selected_threshold = threshold(vy, probability[vm], config['threshold_candidates'])
        tm, ty = masks['test_mask'].cpu().numpy(), label[masks['test_mask']].cpu().numpy(); tp = probability[tm]
        result.update({'status': 'OK', 'f1_macro': float(f1_score(ty, (tp > selected_threshold).astype(np.int64), average='macro')),
          'auroc': float(roc_auc_score(ty, tp)), 'threshold': selected_threshold, 'best_epoch': auprc_best['epoch'], 'validation_auprc': auprc_best['value'],
          'early_stop_best_epoch': f1_best['epoch'], 'epochs_executed': len(history), 'wall_time_sec': time.perf_counter()-start,
          'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device)/1024**2), 'checkpoint_sha256': file_sha256(str(auprc_ckpt)),
          'f1_checkpoint_sha256': file_sha256(str(f1_ckpt)), **fingerprints, 'raw_nodes': raw.num_nodes(), 'raw_edges': raw.num_edges(),
          'training_nodes': graph.num_nodes(), 'training_edges': graph.num_edges(), 'uncovered_prefix_nodes': prefix})
        (OUT / 'validation_history.json').write_text(json.dumps(history, indent=2))
    except Exception:
        result['error'] = traceback.format_exc()
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2)); write_csv(OUT / 'runs.csv', result); print(json.dumps(result), flush=True)
    if result['status'] != 'OK': raise SystemExit(1)


if __name__ == '__main__':
    main()
