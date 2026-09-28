import argparse
import csv
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

CONFIG_PATH = ROOT / 'methods/gcn/configs/weibo_protocol_v1_paper_hidden64_formal.json'
SMOKE_PREFLIGHT = ROOT / 'results/experiments/gcn/weibo/protocol_v1_paper_hidden64/smoke/seed_0/preflight.json'
OUTPUT_ROOT = ROOT / 'results/experiments/gcn/weibo/protocol_v1_paper_hidden64/formal'
REFERENCE_FILE = ROOT / 'audit/mlp_reference/GADBench/models/gnn.py'
REFERENCE_COMMIT = 'f9aa021ce9b6c6580427fb633b596843be76ddc6'


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def choose_f1_threshold(labels, probabilities, thresholds):
    best = (-1.0, thresholds[0])
    for threshold in thresholds:
        f1 = f1_score(labels, (probabilities > threshold).astype(np.int64), average='macro')
        if f1 > best[0]:
            best = (float(f1), float(threshold))
    return best


def write_csv(path, row):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def run(seed):
    config = json.loads(CONFIG_PATH.read_text())
    if seed not in config['training_seeds']:
        raise ValueError(f'seed {seed} is not frozen in the formal configuration')
    output = OUTPUT_ROOT / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'environment').mkdir()
    result = {
        'method': 'GCN', 'protocol_version': config['protocol_version'], 'dataset': 'weibo', 'seed': seed,
        'run_type': 'formal', 'status': 'ERROR', 'edge_access': config['edge_access'],
        'config_sha256': sha_json(config), 'code_sha256': file_sha256(str(Path(__file__))),
    }
    try:
        setup_seed(seed)
        raw_graph = dgl.load_graphs(str(ROOT / 'datasets/weibo'))[0][0]
        feature, label, masks = load_feature_data(ROOT / 'datasets/weibo')
        fingerprints = {
            'dataset_file_sha256': file_sha256(str(ROOT / 'datasets/weibo')),
            'feature_sha256': tensor_sha256(feature), 'label_sha256': tensor_sha256(label),
            **{name + '_sha256': tensor_sha256(mask) for name, mask in masks.items()},
        }
        if any(fingerprints[name + '_sha256'] != digest for name, digest in EXPECTED_WEIBO_MASKS.items()):
            raise RuntimeError('frozen Weibo mask mismatch')
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
        graph.ndata['feature'], graph.ndata['label'] = feature, label
        smoke = json.loads(SMOKE_PREFLIGHT.read_text())
        graph_contract = {
            'input_sha256_equal_to_hidden64_smoke': smoke['input_sha256'] == fingerprints,
            'raw_graph_equal_to_hidden64_smoke': smoke['raw_graph'] == {'nodes': raw_graph.num_nodes(), 'edges': raw_graph.num_edges()},
            'training_graph_equal_to_hidden64_smoke': smoke['training_graph']['nodes'] == graph.num_nodes()
                and smoke['training_graph']['edges'] == graph.num_edges()
                and smoke['training_graph']['preprocess'] == config['graph_preprocess'],
        }
        if not all(graph_contract.values()):
            raise RuntimeError('formal input or graph contract differs from hidden64 smoke: ' + json.dumps(graph_contract))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        graph, label = graph.to(device), label.to(device)
        masks = {name: mask.to(device) for name, mask in masks.items()}
        run_config = dict(config)
        run_config['active_seed'] = seed
        preflight = {
            'reference_repo': 'https://github.com/squareRoot3/GADBench.git',
            'reference_commit': REFERENCE_COMMIT, 'reference_file_sha256': file_sha256(str(REFERENCE_FILE)),
            'runner_code_sha256': result['code_sha256'], 'input_sha256': fingerprints,
            'raw_graph': {'nodes': raw_graph.num_nodes(), 'edges': raw_graph.num_edges()},
            'training_graph': {'nodes': graph.num_nodes(), 'edges': graph.num_edges(), 'preprocess': config['graph_preprocess']},
            'comparison_to_hidden64_smoke': graph_contract, 'edge_access': config['edge_access'], 'config': run_config,
        }
        (output / 'preflight.json').write_text(json.dumps(preflight, indent=2))
        (output / 'config_snapshot.json').write_text(json.dumps(run_config, indent=2))
        (output / 'environment/framework_versions.json').write_text(json.dumps({
            'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'dgl': dgl.__version__, 'gpu': torch.cuda.get_device_name(device),
        }, indent=2))
        model = GCN(feature.shape[1], config['hidden_dim'], 2, config['num_layers'], config['mlp_layers'], config['dropout'], config['activation']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        f1_state = {'value': -1.0, 'state': None, 'epoch': 0}
        auprc_state = {'value': -1.0, 'state': None, 'epoch': 0}
        patience_count, history = 0, []
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for epoch in range(1, config['max_epoch'] + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(graph)
            loss = F.cross_entropy(logits[masks['train_mask']], label[masks['train_mask']])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probability = torch.softmax(model(graph), dim=1)[:, 1].cpu().numpy()
            validation_mask = masks['val_mask'].cpu().numpy()
            validation_label = label[masks['val_mask']].cpu().numpy()
            validation_f1, threshold = choose_f1_threshold(validation_label, probability[validation_mask], config['threshold_candidates'])
            validation_auroc = float(roc_auc_score(validation_label, probability[validation_mask]))
            validation_auprc = float(average_precision_score(validation_label, probability[validation_mask]))
            history.append({'epoch': epoch, 'loss': float(loss.detach().cpu()), 'val_f1_macro': validation_f1,
                            'val_auroc': validation_auroc, 'val_auprc': validation_auprc, 'threshold': threshold})
            if validation_f1 > f1_state['value']:
                f1_state = {'value': validation_f1, 'state': clone_state(model), 'epoch': epoch}
                patience_count = 0
            else:
                patience_count += 1
            if validation_auprc > auprc_state['value']:
                auprc_state = {'value': validation_auprc, 'state': clone_state(model), 'epoch': epoch}
            print(json.dumps({'seed': seed, 'epoch': epoch, 'loss': float(loss.detach().cpu()),
                              'val_f1_macro': validation_f1, 'val_auroc': validation_auroc, 'val_auprc': validation_auprc,
                              'f1_patience_count': patience_count}), flush=True)
            if patience_count > config['patience']:
                break
        f1_checkpoint = output / 'checkpoint_val_f1_earlystop_best.pt'
        auprc_checkpoint = output / 'checkpoint_val_auprc_best.pt'
        torch.save({'model_state_dict': f1_state['state'], 'epoch': f1_state['epoch']}, f1_checkpoint)
        torch.save({'model_state_dict': auprc_state['state'], 'epoch': auprc_state['epoch']}, auprc_checkpoint)
        model.load_state_dict(auprc_state['state'])
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(graph), dim=1)[:, 1].cpu().numpy()
        validation_mask = masks['val_mask'].cpu().numpy()
        validation_label = label[masks['val_mask']].cpu().numpy()
        _, threshold = choose_f1_threshold(validation_label, probability[validation_mask], config['threshold_candidates'])
        test_mask = masks['test_mask'].cpu().numpy()
        test_label = label[masks['test_mask']].cpu().numpy()
        test_probability = probability[test_mask]
        result.update({
            'status': 'OK', 'f1_macro': float(f1_score(test_label, (test_probability > threshold).astype(np.int64), average='macro')),
            'auroc': float(roc_auc_score(test_label, test_probability)), 'threshold': threshold,
            'best_epoch': auprc_state['epoch'], 'validation_auprc': auprc_state['value'],
            'early_stop_best_epoch': f1_state['epoch'], 'epochs_executed': len(history),
            'wall_time_sec': time.perf_counter() - start,
            'peak_gpu_mb': float(torch.cuda.max_memory_allocated(device) / 1024 ** 2),
            'checkpoint_sha256': file_sha256(str(auprc_checkpoint)),
            'f1_checkpoint_sha256': file_sha256(str(f1_checkpoint)), **fingerprints,
            'raw_nodes': raw_graph.num_nodes(), 'raw_edges': raw_graph.num_edges(),
            'training_nodes': graph.num_nodes(), 'training_edges': graph.num_edges(),
        })
        (output / 'validation_history.json').write_text(json.dumps(history, indent=2))
    except Exception:
        result['error'] = traceback.format_exc()
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    write_csv(output / 'runs.csv', result)
    print(json.dumps(result), flush=True)
    if result['status'] != 'OK':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    run(parser.parse_args().seed)
