from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from methods.project_paths import project_root

import numpy as np

ROOT = project_root()
BASE = ROOT / "results/experiments/spacegnn_hsmad/weibo/spacegnn_hsmad_candidate/formal"
PAPER = {"f1_macro": 0.9476, "auroc": 0.9887}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


rows = []
for seed in range(10):
    directory = BASE / f"seed_{seed}"
    metrics = json.loads((directory / "metrics.json").read_text())
    audit = json.loads((directory / "audit_recompute/recompute.json").read_text())
    rows.append({"seed": seed, "status": metrics["status"], "audit_status": audit["status"],
                 "f1_macro": metrics["f1_macro"], "auroc": metrics["auroc"],
                 "best_epoch": metrics["best_epoch"], "threshold": metrics["threshold"],
                 "wall_time_sec": metrics["wall_time_sec"], "peak_gpu_mb": metrics["peak_gpu_mb"],
                 "checkpoint_sha256": sha(directory / "checkpoint_best.pt")})

f1 = np.array([row["f1_macro"] for row in rows]); auc = np.array([row["auroc"] for row in rows])
summary = {"method": "SpaceGNN-official-topology-h64", "dataset": "weibo", "n": 10,
           "positioning": "candidate_protocol_not_author_exact", "formal_ok": sum(r["status"] == "OK" for r in rows),
           "strict_recompute_match": sum(r["audit_status"] == "recompute_match" for r in rows),
           "f1_macro_mean": float(f1.mean()), "f1_macro_std_sample": float(f1.std(ddof=1)),
           "auroc_mean": float(auc.mean()), "auroc_std_sample": float(auc.std(ddof=1)),
           "paper_f1_macro": PAPER["f1_macro"], "paper_auroc": PAPER["auroc"],
           "f1_delta": float(f1.mean() - PAPER["f1_macro"]), "auroc_delta": float(auc.mean() - PAPER["auroc"])}
(BASE / "audited_summary.json").write_text(json.dumps({"summary": summary, "seeds": rows}, indent=2, sort_keys=True) + "\n")
with (BASE / "audited_seed_level.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
with (BASE / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary); writer.writeheader(); writer.writerow(summary)
(BASE / "formal_audit.md").write_text(
    "# SpaceGNN Weibo formal audit\n\n"
    f"- formal/OK: {summary['formal_ok']}/10\n- strict checkpoint recompute: {summary['strict_recompute_match']}/10\n"
    f"- F1-Macro: {summary['f1_macro_mean']:.10f} ± {summary['f1_macro_std_sample']:.10f}\n"
    f"- AUROC: {summary['auroc_mean']:.10f} ± {summary['auroc_std_sample']:.10f}\n"
    f"- Paper: {PAPER['f1_macro']:.4f} / {PAPER['auroc']:.4f}\n"
    f"- Delta: {summary['f1_delta']:+.10f} / {summary['auroc_delta']:+.10f}\n"
    "- Status: FORMAL_CANDIDATE_SUMMARY; not author-exact.\n")
manifest = {str(path.relative_to(BASE)): sha(path) for path in BASE.rglob("*") if path.is_file()}
(BASE / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
