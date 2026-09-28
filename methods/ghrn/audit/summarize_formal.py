"""Create GHRN formal summaries solely from saved seed artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/root/autodl-tmp/HSMAD')
PAPER = {
    'weibo': (0.9150, 0.9666),
    'tolokers': (0.6597, 0.7898),
    'amazon': (0.9207, 0.9741),
    'tfinance': (0.8618, 0.9393),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(PAPER))
    args = parser.parse_args()
    base = ROOT / 'results/experiments/ghrn' / args.dataset / 'ghrn_official_h64_frozen_mask_candidate/formal'
    rows = []
    for seed in range(10):
        run = base / f'seed_{seed}'
        metrics = json.loads((run / 'metrics.json').read_text())
        audit = json.loads((run / 'audit_recompute' / 'compare_to_original.json').read_text())
        metrics['audit_status'] = audit['status']
        metrics['metrics_sha256'] = digest(run / 'metrics.json')
        metrics['checkpoint_sha256'] = digest(run / 'checkpoint_auprc_best.pt')
        rows.append(metrics)
    ok = [row for row in rows if row['status'] == 'OK' and row['audit_status'] == 'recompute_match']
    f1_paper, auc_paper = PAPER[args.dataset]
    summary = {
        'method': 'GHRN-official-core-h64',
        'dataset': args.dataset,
        'positioning': 'candidate_protocol_not_author_exact',
        'n_formal_ok': len([row for row in rows if row['status'] == 'OK']),
        'n_independently_audited': len(ok),
        'f1_macro_mean': statistics.mean(row['f1_macro'] for row in ok),
        'f1_macro_std_sample': statistics.stdev(row['f1_macro'] for row in ok),
        'auroc_mean': statistics.mean(row['auroc'] for row in ok),
        'auroc_std_sample': statistics.stdev(row['auroc'] for row in ok),
        'paper_f1_macro': f1_paper,
        'paper_auroc': auc_paper,
        'f1_delta_vs_paper': statistics.mean(row['f1_macro'] for row in ok) - f1_paper,
        'auroc_delta_vs_paper': statistics.mean(row['auroc'] for row in ok) - auc_paper,
    }
    (base / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    fields = ['seed', 'status', 'audit_status', 'f1_macro', 'auroc', 'auprc', 'best_epoch', 'threshold', 'actual_epochs', 'wall_time_sec', 'peak_gpu_mb', 'predicted_anomaly_count', 'checkpoint_sha256', 'metrics_sha256']
    with (base / 'runs.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    (base / 'formal_audit.md').write_text('# GHRN formal candidate audit\n\n' + json.dumps(summary, indent=2, sort_keys=True) + '\n')
    manifest = {str(path.relative_to(base)): digest(path) for path in base.rglob('*') if path.is_file()}
    (base / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
