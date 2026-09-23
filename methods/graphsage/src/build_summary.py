"""Build a formal-only GraphSAGE summary after all ten successful seeds exist."""

import argparse
import csv
import json
from pathlib import Path

from run_smoke import file_sha256, write_json
from summary import summarize_rows


def summary_metadata(formal_dir):
    """Read immutable method/dataset targets from the first formal artifact."""
    snapshot = Path(formal_dir) / "seed_0" / "config_snapshot.json"
    config = json.loads(snapshot.read_text(encoding="utf-8"))
    return {
        "method": config["method"],
        "dataset": config["dataset"],
        "protocol_version": config["protocol_version"],
        "paper_f1_macro": config["paper_target"]["f1_macro"],
        "paper_auroc": config["paper_target"]["auroc"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True)
    args = parser.parse_args()
    formal_dir = Path(args.formal_dir).resolve()
    with (formal_dir / "runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize_rows(rows)
    if summary["n"] != 10:
        raise RuntimeError(f"Refusing to summarize: expected 10 formal/OK rows, found {summary['n']}")
    metadata = summary_metadata(formal_dir)
    summary.update({**metadata,
                    "f1_macro_delta": summary["f1_macro_mean"] - metadata["paper_f1_macro"], "auroc_delta": summary["auroc_mean"] - metadata["paper_auroc"],
                    "standard_deviation": "sample ddof=1", "input_runs_csv_sha256": file_sha256(formal_dir / "runs.csv")})
    write_json(formal_dir / "summary.json", summary)
    fields = list(summary.keys())
    with (formal_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary)
    audit = [f"# {summary['method']} {summary['dataset']} formal summary", "", "Only `formal` + `OK` rows from `runs.csv` are included.", "",
             f"- n: {summary['n']}", f"- F1-Macro: {summary['f1_macro_mean']:.10f} ± {summary['f1_macro_std_sample']:.10f} (sample std, ddof=1)",
             f"- AUROC: {summary['auroc_mean']:.10f} ± {summary['auroc_std_sample']:.10f} (sample std, ddof=1)",
             f"- Paper delta: F1 {summary['f1_macro_delta']:+.10f}; AUROC {summary['auroc_delta']:+.10f}",
             "", "Positioning: HSMAD frozen data protocol + GADBench GraphSAGE training protocol; not author-exact HSMAD or byte-identical GADBench."]
    (formal_dir / "summary_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
