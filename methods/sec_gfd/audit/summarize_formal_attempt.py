"""Aggregate an isolated SEC-GFD formal attempt from immutable seed artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


PAPER = {
    "weibo": {"f1_macro": 0.9305, "auroc": 0.9420},
    "amazon": {"f1_macro": 0.9235, "auroc": 0.9823},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root: Path = args.root
    rows: list[dict[str, object]] = []
    for seed in range(10):
        artifact = root / f"seed_{seed}"
        metrics_path = artifact / "metrics.json"
        row: dict[str, object] = {"seed": seed, "artifact": str(artifact)}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update({key: metrics.get(key) for key in (
                "method", "protocol_version", "positioning", "dataset", "run_type", "status",
                "f1_macro", "auroc", "best_epoch", "threshold", "actual_epochs",
                "validation_auprc", "validation_f1_macro", "wall_time_sec", "peak_gpu_mb",
                "predicted_anomaly_count", "actual_anomaly_count",
            )})
            row["metrics_sha256"] = sha256(metrics_path)
            checkpoint = artifact / "checkpoint_auprc_best.pt"
            row["checkpoint_sha256"] = sha256(checkpoint) if checkpoint.exists() else None
            recompute = artifact / "audit_recompute/compare_to_original.json"
            row["recompute_status"] = json.loads(recompute.read_text())["status"] if recompute.exists() else "MISSING"
        else:
            row["status"] = "MISSING"
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    prior_runs = root / "runs.csv"
    preserved = root / "runs_runner_last_seed_preserved.csv"
    if prior_runs.exists() and not preserved.exists():
        preserved.write_bytes(prior_runs.read_bytes())
    with prior_runs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = [row for row in rows if row.get("status") == "OK" and row.get("run_type") == "formal"]
    if len(ok) != 10:
        raise RuntimeError(f"formal summary requires 10 formal/OK rows, found {len(ok)}")
    f1 = [float(row["f1_macro"]) for row in ok]
    auc = [float(row["auroc"]) for row in ok]
    paper = PAPER[args.dataset]
    summary = {
        "method": ok[0]["method"],
        "protocol_version": ok[0]["protocol_version"],
        "positioning": "candidate_protocol_not_author_exact",
        "dataset": args.dataset,
        "n": 10,
        "f1_macro_mean": statistics.mean(f1),
        "f1_macro_std_sample": statistics.stdev(f1),
        "auroc_mean": statistics.mean(auc),
        "auroc_std_sample": statistics.stdev(auc),
        "paper_f1_macro": paper["f1_macro"],
        "paper_auroc": paper["auroc"],
        "f1_delta": statistics.mean(f1) - paper["f1_macro"],
        "auroc_delta": statistics.mean(auc) - paper["auroc"],
        "all_checkpoint_recomputations_match": all(row["recompute_status"] == "recompute_match" for row in ok),
    }
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}
    (root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "formal_audit.md").write_text(
        "# SEC-GFD Weibo formal-attempt audit\n\n"
        "- Attempt: `formal_attempt_2`; the earlier `formal/seed_*` launch-guard errors are retained unchanged.\n"
        "- Ten independent `formal/OK` seeds are present.\n"
        f"- Checkpoint recomputation: `{summary['all_checkpoint_recomputations_match']}` for all ten.\n"
        f"- F1-Macro: `{summary['f1_macro_mean']:.10f} ± {summary['f1_macro_std_sample']:.10f}` (sample std, ddof=1).\n"
        f"- AUROC: `{summary['auroc_mean']:.10f} ± {summary['auroc_std_sample']:.10f}` (sample std, ddof=1).\n"
        "- Positioning: `candidate_protocol_not_author_exact`; not an author-exact claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
