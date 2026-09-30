"""Build a non-manual SVM formal summary from seed artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(rows: list[dict], dataset: str) -> dict:
    accepted = [row for row in rows if row.get("status") == "OK" and row.get("audit_status") == "recompute_match"]
    if len(accepted) < 2:
        raise ValueError("need at least two independently audited formal records")
    return {
        "method": "SVM",
        "dataset": dataset,
        "positioning": "candidate_protocol_not_author_exact",
        "n_formal_ok": len(accepted),
        "f1_macro_mean": statistics.mean(row["f1_macro"] for row in accepted),
        "f1_macro_std_sample": statistics.stdev(row["f1_macro"] for row in accepted),
        "auroc_mean": statistics.mean(row["auroc"] for row in accepted),
        "auroc_std_sample": statistics.stdev(row["auroc"] for row in accepted),
        "paper_table1_mapping": "not_available_in_current_HSMAD_Table1_extraction",
        "failed_or_unaudited_seeds": [row.get("seed") for row in rows if row not in accepted],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    rows = []
    for seed in range(10):
        run_dir = args.base / f"seed_{seed}"
        metrics = json.loads((run_dir / "metrics.json").read_text())
        audit = json.loads((run_dir / "audit_recompute" / "compare_to_original.json").read_text())
        metrics["audit_status"] = audit["status"]
        metrics["checkpoint_sha256"] = sha256_file(run_dir / "checkpoint_svc.joblib")
        rows.append(metrics)
    summary = summarize_rows(rows, args.dataset)
    fields = ["seed", "status", "audit_status", "f1_macro", "auroc", "threshold", "predicted_anomaly_count", "checkpoint_sha256", "data_sha256"]
    with (args.base / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    (args.base / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.base / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (args.base / "formal_audit.md").write_text("# SVM Weibo formal candidate audit\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {str(path.relative_to(args.base)): sha256_file(path) for path in args.base.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"}
    (args.base / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
