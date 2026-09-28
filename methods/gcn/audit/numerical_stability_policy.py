"""Separate GCN GPU numerical-stability audit; never rewrites strict evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


AUROC_ABSOLUTE_TOLERANCE = 1e-6
EXACT_FIELDS = (
    "f1_macro",
    "threshold",
    "predicted_anomaly_count",
    "actual_anomaly_count",
    "validation_threshold",
)


def classify_comparison(comparison: dict[str, dict[str, object]]) -> dict[str, object]:
    exact_discrete = all(comparison[field]["kind"] == "exact" for field in EXACT_FIELDS)
    auroc = comparison["auroc"]
    auroc_difference = abs(float(auroc["original"]) - float(auroc["recomputed"]))
    accepted = exact_discrete and auroc_difference <= AUROC_ABSOLUTE_TOLERANCE
    return {
        "status": "numerical_stability_match" if accepted else "numerical_stability_mismatch",
        "auroc_absolute_difference": auroc_difference,
        "auroc_absolute_tolerance": AUROC_ABSOLUTE_TOLERANCE,
        "exact_discrete_fields_required": list(EXACT_FIELDS),
        "exact_discrete_fields_observed": exact_discrete,
        "strict_audit_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--strict-audit-name", default="audit_recompute")
    parser.add_argument("--output-name", default="audit_numerical_stability_v1")
    parser.add_argument("--evidence-reference", default="")
    args = parser.parse_args()

    strict_path = args.run_dir / args.strict_audit_name / "compare_to_original.json"
    strict = json.loads(strict_path.read_text())
    result = classify_comparison(strict["comparison"])
    result.update({
        "strict_audit_path": str(strict_path),
        "strict_audit_status": strict["status"],
        "evidence_reference": args.evidence_reference,
    })
    output = args.run_dir / args.output_name
    output.mkdir(exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output / "audit.md").write_text(
        "# GCN GPU numerical-stability audit\n\n"
        f"Status: `{result['status']}`.\n\n"
        "The original strict audit is retained unchanged. Discrete metrics must match "
        "exactly; only AUROC drift up to the predeclared 1e-6 bound is eligible.\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
