import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / 'results/experiments/gcn/weibo/protocol_v1_paper_hidden64/formal'
PAPER = {'f1_macro': 0.9502, 'auroc': 0.9791}


def main():
    rows = []
    for seed in range(10):
        row = next(csv.DictReader((OUTPUT_ROOT / f'seed_{seed}' / 'runs.csv').open()))
        if row['run_type'] != 'formal' or row['status'] != 'OK':
            raise RuntimeError(f'seed {seed} is not formal/OK: {row["status"]}')
        rows.append(row)
    result = {'method': 'GCN', 'protocol_version': 'v1_paper_hidden64', 'dataset': 'weibo', 'run_type': 'formal', 'n': 10}
    for metric in ('f1_macro', 'auroc'):
        values = np.array([float(row[metric]) for row in rows])
        result[metric + '_mean'] = float(values.mean())
        result[metric + '_std_sample'] = float(values.std(ddof=1))
        result[metric + '_paper'] = PAPER[metric]
        result[metric + '_delta'] = float(values.mean() - PAPER[metric])
    with (OUTPUT_ROOT / 'summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    report = ['# GCN Weibo protocol_v1_paper_hidden64 formal summary', '', 'Ten independent formal/OK runs; sample standard deviation (ddof=1).', '']
    report += ['| metric | mean | sample std | paper | delta |', '|---|---:|---:|---:|---:|']
    for metric in ('f1_macro', 'auroc'):
        report.append(f"| {metric} | {result[metric + '_mean']:.10f} | {result[metric + '_std_sample']:.10f} | {PAPER[metric]:.4f} | {result[metric + '_delta']:+.10f} |")
    (OUTPUT_ROOT / 'comparison.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
