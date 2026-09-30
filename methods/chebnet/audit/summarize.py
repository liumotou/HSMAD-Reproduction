"""Automatic formal-result summarization for ChebNet artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def sample_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    f1 = np.asarray([float(row['f1_macro']) for row in rows])
    auroc = np.asarray([float(row['auroc']) for row in rows])
    return {'n': int(len(rows)), 'f1_macro_mean': float(f1.mean()), 'f1_macro_std_sample': float(f1.std(ddof=1)), 'auroc_mean': float(auroc.mean()), 'auroc_std_sample': float(auroc.std(ddof=1))}


def eligible_audit_path(output: Path) -> Path:
    primary = output / 'audit_recompute/recompute_result.json'
    if json.loads(primary.read_text())['status'] == 'recompute_match':
        return primary
    fallback = output / 'audit_recompute_tolerance_v1/recompute_result.json'
    if fallback.exists() and json.loads(fallback.read_text())['status'] == 'recompute_match':
        return fallback
    return primary


def summarize(formal: Path) -> dict[str, object]:
    rows = []
    for seed in range(10):
        output = formal / f'seed_{seed}'
        metrics = json.loads((output / 'metrics.json').read_text())
        audit = json.loads(eligible_audit_path(output).read_text())
        if metrics['run_type'] != 'formal' or metrics['status'] != 'OK' or audit['status'] != 'recompute_match':
            raise RuntimeError(f'seed {seed} is not eligible for formal summary')
        rows.append(metrics)
    result = sample_summary(rows)
    result.update({'method': 'ChebNet', 'dataset': rows[0]['dataset'], 'protocol_version': rows[0]['protocol_version'], 'candidate_protocol_not_author_exact': True})
    fields = ['seed','f1_macro','auroc','best_epoch','threshold','actual_epochs','peak_gpu_mb','wall_time_sec','predicted_anomaly_count']
    with (formal / 'seed_level.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{field: row[field] for field in fields} for row in rows])
    with (formal / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result)); writer.writeheader(); writer.writerow(result)
    (formal / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--formal', required=True); args = parser.parse_args()
    print(json.dumps(summarize(Path(args.formal)), sort_keys=True))

if __name__ == '__main__':
    main()
