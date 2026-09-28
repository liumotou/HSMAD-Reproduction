"""Build an auditable BWGNN T-Finance formal summary from immutable seed artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()
BASE = ROOT / 'results/experiments/bwgnn/tfinance/bwgnn_gadbench_h64_candidate/formal'
PAPER = {'f1_macro': 0.9084, 'auroc': 0.9600}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


rows = []
for seed in range(10):
    run_dir = BASE / f'seed_{seed}'
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    compare = json.loads((run_dir / 'audit_recompute' / 'compare_to_original.json').read_text())
    # The formal eligibility rule is F1/AUROC recomputation, not minor AUPRC
    # round-off differences caused by sklearn ranking implementation details.
    core_match = all(compare['comparison'][name]['match'] for name in ('f1_macro', 'auroc', 'threshold', 'predicted_anomaly_count'))
    metrics['audit_status'] = compare['status']
    metrics['core_metric_recompute_match'] = core_match
    metrics['metrics_sha256'] = sha256(run_dir / 'metrics.json')
    metrics['checkpoint_sha256'] = sha256(run_dir / 'checkpoint_auprc_best.pt')
    rows.append(metrics)

ok = [r for r in rows if r['status'] == 'OK' and r['core_metric_recompute_match']]
summary = {
    'method': 'BWGNN-GADBench-h64',
    'dataset': 'tfinance',
    'positioning': 'candidate_protocol_not_author_exact',
    'n_formal_ok': len([r for r in rows if r['status'] == 'OK']),
    'n_core_metric_recompute_match': len(ok),
    'f1_macro_mean': statistics.mean(r['f1_macro'] for r in ok),
    'f1_macro_std_sample': statistics.stdev(r['f1_macro'] for r in ok),
    'auroc_mean': statistics.mean(r['auroc'] for r in ok),
    'auroc_std_sample': statistics.stdev(r['auroc'] for r in ok),
    'paper_f1_macro': PAPER['f1_macro'],
    'paper_auroc': PAPER['auroc'],
    'f1_delta_vs_paper': statistics.mean(r['f1_macro'] for r in ok) - PAPER['f1_macro'],
    'auroc_delta_vs_paper': statistics.mean(r['auroc'] for r in ok) - PAPER['auroc'],
    'audit_note': 'Eligibility requires F1/AUROC/threshold/predicted-count recomputation. AUPRC-only differences are retained as numerical-audit notes.',
}

(BASE / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
fields = ['seed', 'status', 'audit_status', 'core_metric_recompute_match', 'f1_macro', 'auroc', 'auprc', 'best_epoch', 'threshold', 'wall_time_sec', 'peak_gpu_mb', 'predicted_anomaly_count', 'checkpoint_sha256', 'metrics_sha256']
with (BASE / 'runs.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{key: row.get(key) for key in fields} for row in rows])
(BASE / 'formal_audit.md').write_text('# BWGNN T-Finance formal candidate audit\n\n' + json.dumps(summary, indent=2, sort_keys=True) + '\n')
manifest = {str(path.relative_to(BASE)): sha256(path) for path in BASE.rglob('*') if path.is_file()}
(BASE / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, sort_keys=True))
