"""Explicit secondary metric comparison for documented CUDA scatter variance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_metrics_with_numerical_tolerance(original, recomputed):
    tolerances = {"f1_macro": 1e-12, "auroc": 1e-6, "threshold": 1e-12}
    differences = {}
    for field in ("f1_macro", "auroc", "threshold", "predicted_anomaly_count", "best_epoch"):
        left, right = original[field], recomputed[field]
        equal = abs(float(left) - float(right)) <= tolerances[field] if field in tolerances else left == right
        if not equal:
            differences[field] = {"original": left, "recomputed": right}
    return {
        "status": "recompute_match" if not differences else "recompute_mismatch",
        "differences": differences,
        "policy": {
            "auroc_absolute_tolerance": 1e-6,
            "f1_and_threshold_absolute_tolerance": 1e-12,
            "integer_fields_exact": True,
            "strict_audit_preserved": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    strict_path = output / "audit_recompute/recompute_result.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    result = compare_metrics_with_numerical_tolerance(strict["original"], strict["recomputed"])
    result["strict_audit_path"] = str(strict_path)
    result["strict_audit_status"] = strict["status"]
    destination = output / "audit_recompute_tolerance_v1"
    destination.mkdir(exist_ok=True)
    (destination / "recompute_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "recompute_match" else 2)


if __name__ == "__main__":
    main()
