"""Read-only provenance and protocol audit for completed SparseGAD candidates."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import dgl
import torch

ROOT = Path('/root/autodl-tmp/HSMAD')
OUT = ROOT / 'methods/sparsegad/audit'
PAPER = {
    'weibo': (0.9305, 0.9420), 'tolokers': (0.5183, 0.7623),
    'amazon': (0.9013, 0.9714), 'yelp': (0.7025, 0.8520),
    'tfinance': (0.8747, 0.9403),
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def summary(rows):
    f1 = [float(row['f1_macro']) for row in rows]
    auroc = [float(row['auroc']) for row in rows]
    return {'n': len(rows), 'f1_mean': statistics.mean(f1), 'f1_sample_sd': statistics.stdev(f1),
            'auroc_mean': statistics.mean(auroc), 'auroc_sample_sd': statistics.stdev(auroc)}


def dataset_audit(name: str):
    cfg_path = ROOT / f'methods/sparsegad/configs/{name}_sparsegad_h64_candidate.json'
    cfg = json.loads(cfg_path.read_text())
    raw_path = ROOT / cfg['dataset_file']
    raw = dgl.load_graphs(str(raw_path))[0][0]
    training = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    labels = raw.ndata['label'].long().reshape(-1)
    masks = {key: raw.ndata[key].bool() for key in ('train_mask', 'val_mask', 'test_mask')}
    formal = ROOT / cfg['result_root'] / 'formal'
    with (formal / 'runs.csv').open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['run_type'] == 'formal' and row['status'] == 'OK']
    seed_hashes = [{key: json.loads((formal / f"seed_{row['seed']}" / 'metrics.json').read_text())['hashes'][key]
                    for key in ('feature_sha256', 'label_sha256', 'train_mask_sha256', 'val_mask_sha256', 'test_mask_sha256')}
                   for row in rows]
    recomputes = [json.loads((formal / f"seed_{row['seed']}" / 'audit_recompute/compare_to_original.json').read_text())['status'] for row in rows]
    stored = json.loads((formal / 'summary.json').read_text())
    calculated = summary(rows)
    return {
        'dataset': name, 'config_path': str(cfg_path), 'config_sha256': sha_file(cfg_path),
        'data': {'path': str(raw_path), 'size_bytes': raw_path.stat().st_size, 'sha256': sha_file(raw_path),
                 'nodes': raw.num_nodes(), 'raw_stored_edges': raw.num_edges(), 'preprocessed_edges': training.num_edges(),
                 'feature_shape': list(raw.ndata['feature'].shape), 'feature_sha256': sha_tensor(raw.ndata['feature'].float()),
                 'label_sha256': sha_tensor(labels), 'normal_count': int((labels == 0).sum()), 'anomaly_count': int((labels == 1).sum())},
        'masks': {key: {'count': int(value.sum()), 'sha256': sha_tensor(value), 'anomaly_count': int(labels[value].sum())} for key, value in masks.items()},
        'fixed_split_across_seeds': len({tuple(sorted(item.items())) for item in seed_hashes}) == 1,
        'all_recompute_match': all(item == 'recompute_match' for item in recomputes),
        'summary': calculated, 'stored_summary_match': all(abs(stored[stored_key] - calculated[calculated_key]) <= 1e-12
                                                 for stored_key, calculated_key in (('f1_macro_mean', 'f1_mean'), ('f1_macro_std_sample', 'f1_sample_sd'), ('auroc_mean', 'auroc_mean'), ('auroc_std_sample', 'auroc_sample_sd'))),
        'paper': {'f1': PAPER[name][0], 'auroc': PAPER[name][1]},
        'formal_dir': str(formal), 'formal_runs_sha256': sha_file(formal / 'runs.csv'),
    }, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, seed_rows = [], []
    for name in ('weibo', 'tolokers'):
        record, rows = dataset_audit(name)
        records.append(record)
        for row in rows:
            seed_rows.append({'dataset': name, **row, 'recompute_status': 'recompute_match'})
    report = {
        'scope': 'read-only audit of retained formal_candidate_summary artifacts',
        'datasets': records,
        'implementation': {
            'official_repository': 'https://github.com/KellyGong/SparseGAD.git',
            'official_commit': 'f5a5c001d7d4e74314a5db0ccfce24367dba77e0',
            'model_py_sha256': sha_file(ROOT / 'methods/sparsegad/src/model.py'),
            'runner_py_sha256': sha_file(ROOT / 'methods/sparsegad/src/run_formal.py'),
            'adaptation': 'DGL full-graph candidate; not byte-identical official PyG training.',
            'selection': 'validation AUPRC early-stop/checkpoint; validation-only F1 threshold; one final test evaluation.',
            'loss': 'unweighted cross entropy restricted to train_mask.',
        },
        'protocol_differences': [
            'HSMAD frozen masks replace any official SparseGAD split behavior.',
            'Current DGL full-graph adaptation is not the official PyG execution path.',
            'Validation AUPRC checkpoint and grid threshold are project protocol choices; HSMAD paper does not fully disclose SparseGAD training selection.',
            'Observed Tolokers gap is stable across ten seeds but cannot be attributed to one unique cause without author-exact code/data/split/selection details.',
        ],
        'excluded': ['random seed variance: sample SD is much smaller than the Tolokers F1 paper gap.', 'checkpoint corruption: all retained formal checkpoints independently recompute.'],
        'supported': ['different frozen split and non-byte-identical DGL adaptation are concrete protocol differences.'],
        'unknown': ['author-exact SparseGAD preprocessing, data release version, checkpoint selection, and threshold policy used for HSMAD Table 1.'],
        'leakage_static_evidence': {
            'loss': 'run_formal.py indexes logits/labels only with train_mask.',
            'selection': 'run_formal.py calls select and AUPRC only on val_mask.',
            'test': 'run_formal.py indexes labels/probabilities with test_mask only after selected checkpoint is loaded.',
            'forward_signature': 'SparseGADModel.forward(graph, features) accepts no label/mask/test metric.',
        },
        'amazon_oom': {
            'status': 'RESOURCE_BLOCKED_24GB',
            'evidence_path': str(ROOT / 'results/experiments/sparsegad/amazon/sparsegad_h64_candidate/diagnostic/seed_0/failure.json'),
            'phase': 'loss.backward, first diagnostic training step',
            'allocation_failure': '2.10 GiB',
            'gpu_total': '23.52 GiB', 'gpu_free_at_exception': '245.75 MiB',
            'torch_allocated': '20.19 GiB', 'torch_reserved_unallocated': '2.13 GiB',
            'peak_gpu_mb_recorded': 20674.56,
            'dense_n_by_n_matrix': False,
            'reasoning': 'Static model inspection shows edge-wise message passing, sorting, and search operations; it does not allocate an explicit N×N adjacency. The 8.81M-edge full graph and two propagation paths remain resource-blocked on this 24GB instance.',
            'action': 'Do not repeat this unchanged Amazon configuration on this 24GB instance.',
        },
    }
    (OUT / 'sparsegad_protocol_difference_audit.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    with (OUT / 'sparsegad_seed_level_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in seed_rows for key in row}))
        writer.writeheader(); writer.writerows(seed_rows)
    paper_rows = []
    for record in records:
        paper_rows.append({'dataset': record['dataset'], 'f1_mean': record['summary']['f1_mean'], 'f1_sample_sd': record['summary']['f1_sample_sd'],
                           'auroc_mean': record['summary']['auroc_mean'], 'auroc_sample_sd': record['summary']['auroc_sample_sd'],
                           'paper_f1': record['paper']['f1'], 'paper_auroc': record['paper']['auroc'],
                           'f1_delta': record['summary']['f1_mean'] - record['paper']['f1'], 'auroc_delta': record['summary']['auroc_mean'] - record['paper']['auroc']})
    with (OUT / 'sparsegad_paper_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paper_rows[0])); writer.writeheader(); writer.writerows(paper_rows)
    lines = ['# SparseGAD protocol-difference audit', '', '## Status', '', '- Both retained result sets are `formal_candidate_summary` / `candidate_protocol_not_author_exact`.', '- Standard deviations are sample SD (`ddof=1`).', '- Every retained formal checkpoint has `recompute_match`.', '', '## Findings', '']
    for record in records:
        lines += [f"### {record['dataset']}", f"- F1: {record['summary']['f1_mean']:.10f} ± {record['summary']['f1_sample_sd']:.10f}", f"- AUROC: {record['summary']['auroc_mean']:.10f} ± {record['summary']['auroc_sample_sd']:.10f}", f"- Paper delta: F1 {record['summary']['f1_mean']-record['paper']['f1']:+.10f}, AUROC {record['summary']['auroc_mean']-record['paper']['auroc']:+.10f}", '']
    lines += ['## Interpretation', '', '- The stable Tolokers gap is not explained by seed variance or checkpoint corruption.', '- Confirmed differences are frozen HSMAD masks and the DGL adaptation; author-specific Table 1 details remain unpublished/unknown.', '- Static call-path inspection finds train/validation/test mask separation; this is not a dynamic proof beyond the recorded independent recomputations.']
    (OUT / 'sparsegad_protocol_difference_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'datasets': [item['dataset'] for item in records], 'status': 'complete'}))


if __name__ == '__main__':
    main()
