"""Secondary audit for documented CUDA GINConv AUROC reduction-order variance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AUROC_ABS_TOLERANCE = 1e-6


def classify_secondary_audit(strict: dict[str, object]) -> dict[str, object]:
    if strict.get("status") == "recompute_match":
        return {
            "status": "recompute_match",
            "strict_status_preserved": True,
            "auroc_absolute_tolerance": AUROC_ABS_TOLERANCE,
        }
    differences = dict(strict.get("differences", {}))
    accepted = set(differences) == {"auroc"}
    delta = None
    if accepted:
        values = differences["auroc"]
        delta = abs(float(values["original"]) - float(values["recomputed"]))
        accepted = delta <= AUROC_ABS_TOLERANCE
    for field in ("f1_macro", "threshold", "predicted_anomaly_count", "best_epoch"):
        accepted = accepted and strict["original"][field] == strict["recomputed"][field]
    return {
        "status": (
            "recompute_match_with_documented_cuda_auroc_tolerance"
            if accepted else "recompute_mismatch"
        ),
        "strict_status": strict.get("status"),
        "strict_status_preserved": True,
        "auroc_absolute_difference": delta,
        "auroc_absolute_tolerance": AUROC_ABS_TOLERANCE,
        "policy_source": "methods/gin/audit/cuda_scatter_numerical_variance.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    strict_path = output / "audit_recompute" / "recompute_result.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    result = classify_secondary_audit(strict)
    audit = output / "audit_recompute_tolerance"
    audit.mkdir(exist_ok=True)
    (audit / "tolerance_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] != "recompute_mismatch" else 2)


if __name__ == "__main__":
    main()
