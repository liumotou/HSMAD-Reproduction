"""Automatic formal-result summarization for isolated GIN artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sample_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    f1 = np.asarray([float(row['f1_macro']) for row in rows])
    auroc = np.asarray([float(row['auroc']) for row in rows])
    return {
        'n': int(len(rows)),
        'f1_macro_mean': float(f1.mean()),
        'f1_macro_std_sample': float(f1.std(ddof=1)),
        'auroc_mean': float(auroc.mean()),
        'auroc_std_sample': float(auroc.std(ddof=1)),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_audit_path(output: Path) -> Path:
    primary = output / 'audit_recompute/recompute_result.json'
    primary_result = json.loads(primary.read_text(encoding='utf-8'))
    if primary_result.get('status') == 'recompute_match':
        return primary
    secondary = output / 'audit_recompute_tolerance_v1/recompute_result.json'
    if secondary.exists():
        secondary_result = json.loads(secondary.read_text(encoding='utf-8'))
        if secondary_result.get('status') == 'recompute_match':
            return secondary
    return primary


def summarize(formal: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    recompute_sha256: dict[str, str] = {}
    for seed in range(10):
        output = formal / f'seed_{seed}'
        metrics_path = output / 'metrics.json'
        audit_path = eligible_audit_path(output)
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if (
            metrics['run_type'] != 'formal'
            or metrics['status'] != 'OK'
            or audit['status'] != 'recompute_match'
        ):
            raise RuntimeError(f'seed {seed} is not eligible for formal summary')
        rows.append(metrics)
        recompute_sha256[f'seed_{seed}'] = sha256_file(audit_path)
    result = sample_summary(rows)
    result.update(
        {
            'method': 'GIN',
            'dataset': rows[0]['dataset'],
            'protocol_version': rows[0]['protocol_version'],
            'candidate_protocol_not_author_exact': True,
            'eligibility': 'formal_OK_and_checkpoint_recompute_match_only',
        }
    )
    fields = [
        'seed',
        'f1_macro',
        'auroc',
        'best_epoch',
        'threshold',
        'actual_epochs',
        'peak_gpu_mb',
        'wall_time_sec',
        'predicted_anomaly_count',
    ]
    with (formal / 'seed_level.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])
    with (formal / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    (formal / 'summary.json').write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding='utf-8'
    )
    audit = {
        'status': 'FORMAL_CANDIDATE_SUMMARY',
        'summary': result,
        'recompute_audit_sha256': recompute_sha256,
    }
    (formal / 'formal_audit.json').write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding='utf-8'
    )
    artifact_paths = [
        path for path in formal.rglob('*') if path.is_file() and path.name != 'artifact_manifest.json'
    ]
    manifest = {
        str(path.relative_to(formal)): sha256_file(path) for path in sorted(artifact_paths)
    }
    (formal / 'artifact_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.formal)), sort_keys=True))


if __name__ == '__main__':
    main()
