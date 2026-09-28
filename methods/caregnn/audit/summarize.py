"""Formal summary for isolated CARE-GNN candidate artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(formal: Path) -> dict[str, object]:
    rows = []
    audit_hashes = {}
    for seed in range(10):
        metrics_path = formal / f"seed_{seed}" / "metrics.json"
        audit_path = formal / f"seed_{seed}" / "audit_recompute" / "recompute_result.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if metrics["run_type"] != "formal" or metrics["status"] != "OK":
            raise RuntimeError(f"seed {seed} is not formal/OK")
        if audit["status"] != "recompute_match":
            raise RuntimeError(f"seed {seed} checkpoint recomputation did not match")
        rows.append(metrics)
        audit_hashes[f"seed_{seed}"] = sha256_file(audit_path)
    f1_values = np.asarray([row["f1_macro"] for row in rows], dtype=float)
    auc_values = np.asarray([row["auroc"] for row in rows], dtype=float)
    summary = {
        "method": "CARE-GNN",
        "dataset": rows[0]["dataset"],
        "protocol_version": rows[0]["protocol_version"],
        "candidate_protocol_not_author_exact": True,
        "eligibility": "formal_OK_and_checkpoint_recompute_match_only",
        "n": len(rows),
        "f1_macro_mean": float(f1_values.mean()),
        "f1_macro_std_sample": float(f1_values.std(ddof=1)),
        "auroc_mean": float(auc_values.mean()),
        "auroc_std_sample": float(auc_values.std(ddof=1)),
    }
    fields = [
        "seed", "f1_macro", "auroc", "best_epoch", "threshold", "actual_epochs",
        "peak_gpu_mb", "wall_time_sec", "predicted_anomaly_count", "relation_threshold",
    ]
    with (formal / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    with (formal / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (formal / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (formal / "formal_audit.json").write_text(
        json.dumps({
            "status": "FORMAL_CANDIDATE_SUMMARY",
            "summary": summary,
            "recompute_audit_sha256": audit_hashes,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [path for path in formal.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"]
    (formal / "artifact_manifest.json").write_text(
        json.dumps({
            str(path.relative_to(formal)): sha256_file(path) for path in sorted(files)
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.formal)), sort_keys=True))


if __name__ == "__main__":
    main()
