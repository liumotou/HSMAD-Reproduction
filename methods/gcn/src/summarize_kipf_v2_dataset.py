import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PROTOCOL = "GADBench_style_AUPRC_best"
EARLY_STOP_PROTOCOL = "validation_F1_macro"
THRESHOLD_PROTOCOL = "validation_F1_macro_grid_0.05_to_0.95"


def summarize_rows(rows: list[dict], dataset: str, protocol_version: str, paper: dict) -> dict:
    result = {
        "method": "GCN",
        "protocol_version": protocol_version,
        "dataset": dataset,
        "run_type": "formal",
        "n": len(rows),
        "checkpoint_protocol": CHECKPOINT_PROTOCOL,
        "early_stop_protocol": EARLY_STOP_PROTOCOL,
        "threshold_protocol": THRESHOLD_PROTOCOL,
    }
    for metric in ("f1_macro", "auroc"):
        values = [float(row[metric]) for row in rows]
        mean = statistics.fmean(values)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std_sample"] = statistics.stdev(values)
        result[f"{metric}_paper"] = paper[metric]
        result[f"{metric}_delta"] = mean - paper[metric]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--paper-f1", required=True, type=float)
    parser.add_argument("--paper-auroc", required=True, type=float)
    args = parser.parse_args()
    output_root = ROOT / "results/experiments/gcn" / args.dataset / args.protocol / "formal"
    rows = list(csv.DictReader((output_root / "runs.csv").open()))
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 formal rows, found {len(rows)}")
    for seed, row in enumerate(rows):
        if int(row["seed"]) != seed or row["run_type"] != "formal" or row["status"] != "OK":
            raise RuntimeError(f"invalid formal row {seed}")
        if row["protocol_version"] != args.protocol:
            raise RuntimeError(f"protocol mismatch at seed {seed}")
        if row["checkpoint_protocol"] != CHECKPOINT_PROTOCOL:
            raise RuntimeError(f"checkpoint protocol mismatch at seed {seed}")
    paper = {"f1_macro": args.paper_f1, "auroc": args.paper_auroc}
    result = summarize_rows(rows, args.dataset, args.protocol, paper)
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    report = [
        f"# GCN {args.dataset} {args.protocol} formal summary",
        "",
        "GCN baseline / GADBench_style_AUPRC_best; sample standard deviation (ddof=1).",
        "",
        "| metric | mean | sample std | paper | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("f1_macro", "auroc"):
        report.append(
            f"| {metric} | {result[metric + '_mean']:.10f} | "
            f"{result[metric + '_std_sample']:.10f} | {result[metric + '_paper']:.4f} | "
            f"{result[metric + '_delta']:+.10f} |"
        )
    (output_root / "comparison.md").write_text("\n".join(report) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
