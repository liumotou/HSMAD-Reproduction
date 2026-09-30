#!/usr/bin/env python3
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()
FORMAL = ROOT / "results/experiments/dsgad/weibo/dsgad_hsmad_candidate/formal"
PAPER_F1 = 0.9406
PAPER_AUROC = 0.9776


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for seed in range(10):
    run_dir = FORMAL / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    compare_path = run_dir / "audit_recompute/compare_to_original.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    row = {
        "seed": seed,
        "run_type": metrics["run_type"],
        "status": metrics["status"],
        "audit_status": compare["status"],
        "f1_macro": metrics["f1_macro"],
        "auroc": metrics["auroc"],
        "best_epoch": metrics["best_epoch"],
        "actual_epochs": metrics["actual_epochs"],
        "threshold": metrics["threshold"],
        "validation_auprc": metrics["validation_auprc"],
        "validation_f1_macro": metrics["validation_f1_macro"],
        "predicted_anomaly_count": metrics["predicted_anomaly_count"],
        "actual_anomaly_count": metrics["actual_anomaly_count"],
        "peak_gpu_mb": metrics["peak_gpu_mb"],
        "wall_time_sec": metrics["wall_time_sec"],
        "metrics_sha256": sha256(metrics_path),
        "checkpoint_sha256": sha256(run_dir / "checkpoint_auprc_best.pt"),
        "audit_sha256": sha256(compare_path),
    }
    rows.append(row)

eligible = [
    row for row in rows
    if row["run_type"] == "formal"
    and row["status"] == "OK"
    and row["audit_status"] == "recompute_match"
]
if len(eligible) != 10:
    raise SystemExit(f"expected 10 audited formal/OK rows, found {len(eligible)}")

f1 = [row["f1_macro"] for row in eligible]
auc = [row["auroc"] for row in eligible]
summary = {
    "method": "DSGAD-official-topology-h64",
    "dataset": "weibo",
    "protocol_version": "dsgad_hsmad_candidate",
    "positioning": "candidate_protocol_not_author_exact",
    "n": 10,
    "audit_policy": "strict checkpoint recomputation; absolute float tolerance 1e-12",
    "audit_pass_count": 10,
    "f1_macro_mean": statistics.mean(f1),
    "f1_macro_std_sample": statistics.stdev(f1),
    "auroc_mean": statistics.mean(auc),
    "auroc_std_sample": statistics.stdev(auc),
    "paper_f1_macro": PAPER_F1,
    "paper_auroc": PAPER_AUROC,
    "f1_macro_delta": statistics.mean(f1) - PAPER_F1,
    "auroc_delta": statistics.mean(auc) - PAPER_AUROC,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "original_runs_csv_sha256": sha256(FORMAL / "runs.csv"),
}

with (FORMAL / "audited_runs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

(FORMAL / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
with (FORMAL / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)

lines = [
    "# DSGAD Weibo formal audit",
    "",
    "- Status: FORMAL_CANDIDATE_SUMMARY",
    "- Positioning: candidate_protocol_not_author_exact",
    "- Seeds: 0..9; formal/OK: 10/10; checkpoint recompute_match: 10/10",
    "- Original runs.csv was preserved byte-for-byte; audited rows are in audited_runs.csv.",
    f"- F1-Macro: {summary['f1_macro_mean']:.10f} ± {summary['f1_macro_std_sample']:.10f} (sample std, ddof=1)",
    f"- AUROC: {summary['auroc_mean']:.10f} ± {summary['auroc_std_sample']:.10f} (sample std, ddof=1)",
    f"- Paper target: F1={PAPER_F1:.4f}, AUROC={PAPER_AUROC:.4f}",
    f"- Delta: F1={summary['f1_macro_delta']:+.10f}, AUROC={summary['auroc_delta']:+.10f}",
    "",
    "## Seed-level results",
    "",
    "| seed | F1-Macro | AUROC | best epoch | threshold | audit |",
    "|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append(
        f"| {row['seed']} | {row['f1_macro']:.10f} | {row['auroc']:.10f} | "
        f"{row['best_epoch']} | {row['threshold']:.2f} | {row['audit_status']} |"
    )
(FORMAL / "formal_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {}
for path in sorted(FORMAL.rglob("*")):
    if path.is_file() and path.name != "artifact_manifest.json":
        manifest[str(path.relative_to(ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
(FORMAL / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
