"""Generate MLP-only formal summary and Weibo paper comparison."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "results/experiments/mlp/weibo/formal"
rows = list(csv.DictReader((FORMAL / "runs.csv").open(encoding="utf-8")))
ok = sorted((row for row in rows if row["run_type"] == "formal" and row["status"] == "OK"), key=lambda row: int(row["seed"]))
if len(ok) != 10 or [int(row["seed"]) for row in ok] != list(range(10)):
    raise SystemExit("summary requires exactly formal/OK seeds 0..9")
f1 = np.array([float(row["f1_macro"]) for row in ok]); auc = np.array([float(row["auroc"]) for row in ok])
summary = {"method":"MLP", "dataset":"weibo", "run_type":"formal", "n":10,
           "f1_macro_mean":float(f1.mean()), "f1_macro_std_sample":float(f1.std(ddof=1)),
           "auroc_mean":float(auc.mean()), "auroc_std_sample":float(auc.std(ddof=1)),
           "paper_f1_macro":0.9223, "paper_auroc":0.9801,
           "delta_f1_macro":float(f1.mean()-0.9223), "delta_auroc":float(auc.mean()-0.9801)}
(FORMAL / "summary.csv").write_text(",".join(summary.keys()) + "\n" + ",".join(str(value) for value in summary.values()) + "\n", encoding="utf-8")
lines=["# MLP Weibo formal comparison", "", "All ten records are `formal / OK`; every run records `edge_access=none`.", "",
       "| Metric | This project (mean ± sample std) | Paper | Difference |", "| --- | --- | ---: | ---: |",
       f"| F1-Macro | {summary['f1_macro_mean']:.10f} ± {summary['f1_macro_std_sample']:.10f} | 0.9223 | {summary['delta_f1_macro']:+.10f} |",
       f"| AUROC | {summary['auroc_mean']:.10f} ± {summary['auroc_std_sample']:.10f} | 0.9801 | {summary['delta_auroc']:+.10f} |", "",
       "| seed | F1-Macro | AUROC | best_epoch | threshold | peak_gpu_mb | wall_time_sec |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for row in ok:
    lines.append(f"| {row['seed']} | {float(row['f1_macro']):.10f} | {float(row['auroc']):.10f} | {row['best_epoch']} | {float(row['threshold']):.2f} | {float(row['peak_gpu_mb']):.4f} | {float(row['wall_time_sec']):.4f} |")
(FORMAL / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
