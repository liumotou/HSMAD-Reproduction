#!/usr/bin/env python3
import argparse, csv, hashlib, json, statistics
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--paper-f1", type=float, required=True)
p.add_argument("--paper-auroc", type=float, required=True)
a = p.parse_args()
root = Path("/root/autodl-tmp/HSMAD")
formal = root / f"results/experiments/dsgad/{a.dataset}/dsgad_hsmad_candidate/formal"
rows = []
for seed in range(10):
    rd = formal / f"seed_{seed}"
    mpath = rd / "metrics.json"
    apath = rd / "audit_recompute/compare_to_original.json"
    m = json.loads(mpath.read_text())
    audit = json.loads(apath.read_text())
    rows.append({
        "seed": seed, "run_type": m["run_type"], "status": m["status"],
        "audit_status": audit["status"], "audit_differences": json.dumps(audit.get("differences", {}), sort_keys=True),
        "f1_macro": m["f1_macro"], "auroc": m["auroc"], "best_epoch": m["best_epoch"],
        "actual_epochs": m["actual_epochs"], "threshold": m["threshold"],
        "validation_auprc": m["validation_auprc"], "validation_f1_macro": m["validation_f1_macro"],
        "predicted_anomaly_count": m["predicted_anomaly_count"], "actual_anomaly_count": m["actual_anomaly_count"],
        "peak_gpu_mb": m["peak_gpu_mb"], "wall_time_sec": m["wall_time_sec"],
        "metrics_sha256": sha256(mpath), "checkpoint_sha256": sha256(rd / "checkpoint_auprc_best.pt"),
        "audit_sha256": sha256(apath),
    })
strict = [r for r in rows if r["status"] == "OK" and r["run_type"] == "formal" and r["audit_status"] == "recompute_match"]
raw = [r for r in rows if r["status"] == "OK" and r["run_type"] == "formal"]
def stats(rs, key):
    vals = [float(r[key]) for r in rs]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else None
sf1m, sf1s = stats(strict, "f1_macro"); saucm, saucs = stats(strict, "auroc")
rf1m, rf1s = stats(raw, "f1_macro"); raucm, raucs = stats(raw, "auroc")
summary = {
    "status": "PARTIAL_AUDIT" if len(strict) < 10 else "FORMAL_CANDIDATE_SUMMARY",
    "method": "DSGAD-official-topology-h64", "dataset": a.dataset,
    "protocol_version": "dsgad_hsmad_candidate", "positioning": "candidate_protocol_not_author_exact",
    "formal_ok_count": len(raw), "strict_audit_count": len(strict),
    "strict_seed_ids": [r["seed"] for r in strict], "mismatch_seed_ids": [r["seed"] for r in rows if r["audit_status"] != "recompute_match"],
    "strict_f1_macro_mean": sf1m, "strict_f1_macro_std_sample": sf1s,
    "strict_auroc_mean": saucm, "strict_auroc_std_sample": saucs,
    "raw_all10_f1_macro_mean": rf1m, "raw_all10_f1_macro_std_sample": rf1s,
    "raw_all10_auroc_mean": raucm, "raw_all10_auroc_std_sample": raucs,
    "paper_f1_macro": a.paper_f1, "paper_auroc": a.paper_auroc,
    "raw_all10_f1_delta": rf1m - a.paper_f1, "raw_all10_auroc_delta": raucm - a.paper_auroc,
    "audit_policy": "frozen strict absolute float tolerance 1e-12; no widening",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "original_runs_csv_sha256": sha256(formal / "runs.csv"),
}
with (formal / "audited_runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(formal / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (formal / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
lines = [f"# DSGAD {a.dataset} formal audit", "", f"- Status: {summary['status']}",
         f"- formal/OK: {len(raw)}/10; strict recompute_match: {len(strict)}/10",
         f"- mismatch seeds: {summary['mismatch_seed_ids']}",
         "- Frozen tolerance remains 1e-12; no current-result tolerance widening.",
         f"- Strict F1: {sf1m:.10f} ± {sf1s:.10f}; strict AUROC: {saucm:.10f} ± {saucs:.10f}",
         f"- Raw all-10 F1: {rf1m:.10f} ± {rf1s:.10f}; raw all-10 AUROC: {raucm:.10f} ± {raucs:.10f}",
         f"- Paper: {a.paper_f1:.4f} / {a.paper_auroc:.4f}; raw deltas: {rf1m-a.paper_f1:+.10f} / {raucm-a.paper_auroc:+.10f}",
         "", "| seed | F1 | AUROC | best epoch | threshold | audit |", "|---:|---:|---:|---:|---:|---|"]
for r in rows:
    lines.append(f"| {r['seed']} | {r['f1_macro']:.10f} | {r['auroc']:.10f} | {r['best_epoch']} | {r['threshold']:.2f} | {r['audit_status']} |")
(formal / "formal_audit.md").write_text("\n".join(lines) + "\n")
manifest = {}
for path in sorted(formal.rglob("*")):
    if path.is_file() and path.name != "artifact_manifest.json":
        manifest[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
(formal / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
