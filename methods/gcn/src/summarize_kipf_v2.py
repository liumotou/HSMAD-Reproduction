import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PAPER = {"f1_macro": 0.9502, "auroc": 0.9791}


def summarize_rows(rows: list[dict], paper: dict) -> dict:
    result = {"n": len(rows)}
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
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    output_root = ROOT / "results/experiments/gcn/weibo" / args.protocol / "formal"
    rows = []
    for seed in range(10):
        row = next(csv.DictReader((output_root / f"seed_{seed}" / "runs.csv").open()))
        if row["run_type"] != "formal" or row["status"] != "OK":
            raise RuntimeError(f"seed {seed} is not formal/OK")
        if row["protocol_version"] != args.protocol:
            raise RuntimeError(f"seed {seed} protocol mismatch")
        rows.append(row)
    result = {
        "method": "GCN",
        "protocol_version": args.protocol,
        "dataset": "weibo",
        "run_type": "formal",
        **summarize_rows(rows, PAPER),
    }
    with (output_root / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    report = [
        f"# GCN Weibo {args.protocol} formal summary",
        "",
        "Ten independent formal/OK runs; sample standard deviation (ddof=1).",
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
