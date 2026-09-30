import hashlib
import json
import shutil
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
from train import EXPECTED_WEIBO_MASKS, load_feature_data
from utils import file_sha256, setup_seed, tensor_sha256

sys.path.append(str(ROOT / 'audit/mlp_reference/GADBench'))
from models.gnn import GCN

OUT = ROOT / 'results/experiments/gcn/weibo/protocol_v1_paper_hidden64/smoke/seed_0'
CONFIG = ROOT / 'methods/gcn/configs/weibo_protocol_v1_paper_hidden64_smoke_seed0.json'
PRIOR_PREFLIGHT = ROOT / 'results/experiments/gcn/weibo/smoke/seed_0/preflight.json'
REF = ROOT / 'audit/mlp_reference/GADBench/models/gnn.py'
COMMIT = 'f9aa021ce9b6c6580427fb633b596843be76ddc6'


def config_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def best_threshold(labels, anomaly_probabilities):
    best = (-1.0, 0.05)
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(labels, (anomaly_probabilities > threshold).astype(np.int64), average='macro')
        if score > best[0]:
            best = (score, float(threshold))
    return best


def main():
    config = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'environment').mkdir()
    result = {
        'method': 'GCN', 'protocol_version': config['protocol_version'], 'dataset': 'weibo',
        'seed': 0, 'run_type': 'smoke', 'status': 'ERROR',
        'edge_access': 'graph_edges_required', 'config_sha256': config_sha256(config),
    }
    try:
        setup_seed(0)
        raw_graph = dgl.load_graphs(str(ROOT / 'datasets/weibo'))[0][0]
        feature, label, masks = load_feature_data(ROOT / 'datasets/weibo')
        fingerprints = {
            'dataset_file_sha256': file_sha256(str(ROOT / 'datasets/weibo')),
            'feature_sha256': tensor_sha256(feature),
            'label_sha256': tensor_sha256(label),
            **{name + '_sha256': tensor_sha256(mask) for name, mask in masks.items()},
        }
        if any(fingerprints[name + '_sha256'] != digest for name, digest in EXPECTED_WEIBO_MASKS.items()):
            raise RuntimeError('frozen Weibo mask mismatch')

        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
        graph.ndata['feature'] = feature
        graph.ndata['label'] = label
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        graph, label = graph.to(device), label.to(device)
        masks = {name: mask.to(device) for name, mask in masks.items()}

        prior = json.loads(PRIOR_PREFLIGHT.read_text())
        comparison = {
            'input_sha256_equal': prior['input_sha256'] == fingerprints,
            'raw_graph_equal': prior['raw_graph'] == {'nodes': raw_graph.num_nodes(), 'edges': raw_graph.num_edges()},
            'training_graph_equal': prior['training_graph']['nodes'] == graph.num_nodes()
                and prior['training_graph']['edges'] == graph.num_edges()
                and prior['training_graph']['preprocess'] == config['graph_preprocess'],
        }
        if not all(comparison.values()):
            raise RuntimeError('hidden64 smoke input or graph differs from the prior hidden32 smoke: ' + json.dumps(comparison))

        preflight = {
            'reference_repo': 'https://github.com/squareRoot3/GADBench.git',
            'reference_commit': COMMIT,
            'reference_file_sha256': file_sha256(str(REF)),
            'input_sha256': fingerprints,
            'raw_graph': {'nodes': raw_graph.num_nodes(), 'edges': raw_graph.num_edges()},
            'training_graph': {'nodes': graph.num_nodes(), 'edges': graph.num_edges(), 'preprocess': config['graph_preprocess']},
            'edge_access': 'graph_edges_required',
            'comparison_to_prior_hidden32_smoke': comparison,
            'config': config,
        }
        (OUT / 'preflight.json').write_text(json.dumps(preflight, indent=2))
        shutil.copy2(CONFIG, OUT / 'config_snapshot.json')
        (OUT / 'environment/framework_versions.json').write_text(json.dumps({
            'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'dgl': dgl.__version__, 'gpu': torch.cuda.get_device_name(device),
        }, indent=2))

        model = GCN(feature.shape[1], 64, 2, 2, 1, 0.0, 'ReLU').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        history, best = [], {'value': -1.0, 'state': None, 'epoch': 0}
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for epoch in range(1, config['max_epoch'] + 1):
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(graph)[masks['train_mask']], label[masks['train_mask']])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probability = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
            validation_mask = masks['val_mask'].cpu().numpy()
            validation_label = label[masks['val_mask']].cpu().numpy()
            validation_f1, threshold = best_threshold(validation_label, probability[validation_mask])
            validation_auprc = float(average_precision_score(validation_label, probability[validation_mask]))
            history.append({'epoch': epoch, 'loss': float(loss.detach().cpu()), 'val_f1_macro': validation_f1,
                            'val_auprc': validation_auprc, 'threshold': threshold})
            if validation_auprc > best['value']:
                best = {'value': validation_auprc,
                        'state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                        'epoch': epoch}
        checkpoint = OUT / 'checkpoint_val_auprc_best.pt'
        torch.save({'model_state_dict': best['state'], 'epoch': best['epoch']}, checkpoint)
        model.load_state_dict(best['state'])
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
        validation_mask = masks['val_mask'].cpu().numpy()
        validation_label = label[masks['val_mask']].cpu().numpy()
        _, threshold = best_threshold(validation_label, probability[validation_mask])
        test_mask = masks['test_mask'].cpu().numpy()
        test_label = label[masks['test_mask']].cpu().numpy()
        test_probability = probability[test_mask]
        result.update({
            'status': 'smoke',
            'f1_macro': float(f1_score(test_label, (test_probability > threshold).astype(np.int64), average='macro')),
            'auroc': float(roc_auc_score(test_label, test_probability)), 'threshold': threshold,
            'best_epoch': best['epoch'], 'validation_auprc': best['value'],
            'wall_time_sec': time.perf_counter() - start,
            'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024 ** 2),
            'checkpoint_sha256': file_sha256(str(checkpoint)), **fingerprints,
            'raw_nodes': raw_graph.num_nodes(), 'raw_edges': raw_graph.num_edges(),
            'training_nodes': graph.num_nodes(), 'training_edges': graph.num_edges(),
        })
        (OUT / 'validation_history.json').write_text(json.dumps(history, indent=2))
    except Exception:
        result['error'] = traceback.format_exc()
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2))
    (OUT / 'runs.csv').write_text(','.join(result) + '\n' + ','.join(json.dumps(value) if isinstance(value, (dict, list)) else str(value) for value in result.values()) + '\n')
    print(json.dumps(result))
    if result['status'] != 'smoke':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
