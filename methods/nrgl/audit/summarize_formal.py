"""Build NRGL candidate formal summaries from independently recomputed rows."""

import csv
import json
from pathlib import Path

import numpy as np


def summarize_records(rows):
    eligible = [row for row in rows if row.get("run_type") == "formal" and row.get("status") == "OK"]
    f1 = np.asarray([float(row["f1_macro"]) for row in eligible], dtype=float)
    auroc = np.asarray([float(row["auroc"]) for row in eligible], dtype=float)
    return {
        "n": int(len(eligible)),
        "f1_macro_mean": float(f1.mean()) if len(f1) else None,
        "f1_macro_std_sample": float(f1.std(ddof=1)) if len(f1) > 1 else None,
        "auroc_mean": float(auroc.mean()) if len(auroc) else None,
        "auroc_std_sample": float(auroc.std(ddof=1)) if len(auroc) > 1 else None,
    }


def generate(formal_directory):
    formal = Path(formal_directory)
    with (formal / "runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    recompute = {}
    for row in rows:
        path = formal / f"seed_{row['seed']}" / "audit_recompute_tolerance" / "compare_to_original.json"
        if not path.exists():
            path = formal / f"seed_{row['seed']}" / "audit_recompute" / "compare_to_original.json"
        recompute[row["seed"]] = json.loads(path.read_text(encoding="utf-8"))["status"] if path.exists() else "missing"
    eligible = [row for row in rows if recompute.get(row["seed"]) == "RECOMPUTE_MATCH_WITHIN_TOLERANCE"]
    summary = summarize_records(eligible)
    summary.update({"method": "NRGL-HSMAD-adapted", "protocol_version": "nrgl_hsmad_candidate", "positioning": "candidate_protocol_not_author_exact", "recompute_status_by_seed": recompute})
    with (formal / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader(); writer.writerow(summary)
    (formal / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (formal / "formal_audit.md").write_text(
        f"# NRGL Weibo formal candidate audit\n\nIndependent checkpoint recomputation matches within 1e-6 tolerance: `{sum(value == 'RECOMPUTE_MATCH_WITHIN_TOLERANCE' for value in recompute.values())}/{len(rows)}`.\n\n"
        f"F1-Macro: `{summary['f1_macro_mean']} ± {summary['f1_macro_std_sample']}` (sample standard deviation).\n\n"
        f"AUROC: `{summary['auroc_mean']} ± {summary['auroc_std_sample']}` (sample standard deviation).\n\n"
        "This is `candidate_protocol_not_author_exact`; it adapts NRGL's official core to frozen HSMAD data/masks and validation-only selection.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    import sys
    print(json.dumps(generate(sys.argv[1]), sort_keys=True))
