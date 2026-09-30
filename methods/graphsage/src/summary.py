"""Formal-result aggregation for the isolated GraphSAGE candidate."""

import statistics


def summarize_rows(rows):
    accepted = [row for row in rows if row.get("run_type") == "formal" and row.get("status") == "OK"]
    f1_values = [float(row["f1_macro"]) for row in accepted]
    auroc_values = [float(row["auroc"]) for row in accepted]
    if not accepted:
        raise ValueError("No formal/OK rows to summarize")
    return {
        "n": len(accepted),
        "f1_macro_mean": statistics.mean(f1_values),
        "f1_macro_std_sample": statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0,
        "f1_macro_std_population": statistics.pstdev(f1_values) if len(f1_values) > 1 else 0.0,
        "auroc_mean": statistics.mean(auroc_values),
        "auroc_std_sample": statistics.stdev(auroc_values) if len(auroc_values) > 1 else 0.0,
        "auroc_std_population": statistics.pstdev(auroc_values) if len(auroc_values) > 1 else 0.0,
    }
