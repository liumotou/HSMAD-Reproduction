"""Summarize only independently recomputed formal/OK PC-GNN records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True)
    args = parser.parse_args()
    root = Path(args.formal_dir)
    rows = []
    artifacts = {}
    for seed in range(10):
        folder = root / f"seed_{seed}"
        metrics_path = folder / "metrics.json"
        recompute_path = folder / "audit_recompute" / "recompute_result.json"
        if not metrics_path.exists() or not recompute_path.exists():
            raise RuntimeError(f"missing audited result for seed {seed}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
        if metrics["run_type"] != "formal" or metrics["status"] != "OK":
            raise RuntimeError(f"seed {seed} is not formal/OK")
        if recompute["status"] != "recompute_match":
            raise RuntimeError(f"seed {seed} did not independently recompute")
        rows.append({
            "seed": seed,
            "run_type": metrics["run_type"],
            "status": metrics["status"],
            "f1_macro": metrics["f1_macro"],
            "auroc": metrics["auroc"],
            "best_epoch": metrics["best_epoch"],
            "threshold": metrics["threshold"],
            "predicted_anomaly_count": metrics["predicted_anomaly_count"],
            "wall_time_sec": metrics["wall_time_sec"],
            "peak_gpu_mb": metrics["peak_gpu_mb"],
            "config_sha256": metrics["hashes"]["config_sha256"],
            "checkpoint_sha256": recompute["recomputed"]["checkpoint_sha256"],
        })
        artifacts[f"seed_{seed}/metrics.json"] = sha256(metrics_path)
        artifacts[f"seed_{seed}/checkpoint_validation_auroc_best.pt"] = sha256(folder / "checkpoint_validation_auroc_best.pt")
        artifacts[f"seed_{seed}/terminal.log"] = sha256(folder / "terminal.log")
    fields = list(rows[0])
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    f1 = [float(row["f1_macro"]) for row in rows]
    auc = [float(row["auroc"]) for row in rows]
    datasets = {json.loads((root / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))["dataset"] for seed in range(10)}
    protocols = {json.loads((root / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))["protocol_version"] for seed in range(10)}
    if len(datasets) != 1 or len(protocols) != 1:
        raise RuntimeError("mixed datasets or protocols in formal directory")
    dataset = datasets.pop()
    protocol = protocols.pop()
    summary = {
        "method": "PC-GNN",
        "protocol_version": protocol,
        "candidate_protocol_not_author_exact": True,
        "dataset": dataset,
        "n": 10,
        "f1_macro_mean": statistics.mean(f1),
        "f1_macro_std_sample": statistics.stdev(f1),
        "auroc_mean": statistics.mean(auc),
        "auroc_std_sample": statistics.stdev(auc),
        "audit": "strict_recompute_match_10_of_10",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (root / "formal_audit.md").write_text(
        f"# PC-GNN {dataset} formal audit\n\n"
        "All ten rows are independent formal/OK runs and each saved checkpoint was independently recomputed.\n\n"
        f"- F1-Macro: {summary['f1_macro_mean']:.10f} ± {summary['f1_macro_std_sample']:.10f}\n"
        f"- AUROC: {summary['auroc_mean']:.10f} ± {summary['auroc_std_sample']:.10f}\n"
        "- Standard deviation: sample (`ddof=1`).\n"
        "- Status: `candidate_protocol_not_author_exact`.\n",
        encoding="utf-8",
    )
    (root / "artifact_manifest.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
