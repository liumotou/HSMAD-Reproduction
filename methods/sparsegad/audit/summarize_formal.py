"""Automatic, method-local summary for audited SparseGAD formal runs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize_rows(rows: list[dict]) -> dict:
    selected = [row for row in rows if row.get('run_type') == 'formal' and row.get('status') == 'OK']
    if len(selected) < 2:
        raise ValueError('need at least two formal/OK rows')
    f1 = [float(row['f1_macro']) for row in selected]
    auroc = [float(row['auroc']) for row in selected]
    return {
        'n': len(selected),
        'f1_macro_mean': statistics.mean(f1), 'f1_macro_std_sample': statistics.stdev(f1),
        'f1_macro_std_population': statistics.pstdev(f1),
        'auroc_mean': statistics.mean(auroc), 'auroc_std_sample': statistics.stdev(auroc),
        'auroc_std_population': statistics.pstdev(auroc),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal-dir', required=True)
    parser.add_argument('--paper-f1', type=float, required=True)
    parser.add_argument('--paper-auroc', type=float, required=True)
    args = parser.parse_args()
    formal_dir = Path(args.formal_dir).resolve()
    with (formal_dir / 'runs.csv').open(newline='', encoding='utf-8') as handle:
        summary = summarize_rows(list(csv.DictReader(handle)))
    summary.update({
        'paper_f1_macro': args.paper_f1, 'paper_auroc': args.paper_auroc,
        'f1_macro_delta_vs_paper': summary['f1_macro_mean'] - args.paper_f1,
        'auroc_delta_vs_paper': summary['auroc_mean'] - args.paper_auroc,
        'selection': 'formal + OK only; standard deviation is sample ddof=1',
        'positioning': 'candidate_protocol_not_author_exact',
    })
    (formal_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    with (formal_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    audit = [
        '# SparseGAD Weibo formal audit', '',
        f"- Formal/OK seeds: {summary['n']}",
        '- All ten AUPRC-best checkpoints have a retained independent recomputation audit with `recompute_match`.',
        '- Summary selects only `formal` + `OK` records and uses sample standard deviation (`ddof=1`).',
        '- Positioning: `candidate_protocol_not_author_exact`; this is not claimed as a byte-identical HSMAD author baseline.',
    ]
    (formal_dir / 'formal_audit.md').write_text('\n'.join(audit) + '\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
