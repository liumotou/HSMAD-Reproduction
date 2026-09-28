"""Read-only checkpoint recomputation for one saved SVM experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from runner import frozen_masks_from_graph  # noqa: E402
from dgl.data.utils import load_graphs  # noqa: E402


CORE_FIELDS = ("f1_macro", "auroc", "threshold", "predicted_anomaly_count")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_metrics(original: dict, recomputed: dict, tolerance: float = 1e-12) -> dict:
    comparison = {}
    for field in CORE_FIELDS:
        left, right = original[field], recomputed[field]
        match = abs(left - right) <= tolerance if isinstance(left, float) else left == right
        comparison[field] = {"original": left, "recomputed": right, "match": match}
    return {
        "status": "recompute_match" if all(item["match"] for item in comparison.values()) else "recompute_mismatch",
        "comparison": comparison,
    }


def recompute(root: Path, dataset: str, run_dir: Path) -> tuple[dict, dict]:
    original = json.loads((run_dir / "metrics.json").read_text())
    graph = load_graphs(str(root / "datasets" / dataset))[0][0]
    masks = {name: tensor.cpu().numpy().astype(bool) for name, tensor in frozen_masks_from_graph(graph).items()}
    features = graph.ndata["feature"].cpu().numpy()
    labels = graph.ndata["label"].cpu().numpy().reshape(-1).astype(int)
    checkpoint = run_dir / "checkpoint_svc.joblib"
    model = joblib.load(checkpoint)
    scores = model.predict_proba(features[masks["test"]])[:, 1]
    threshold = float(original["threshold"])
    predicted = (scores >= threshold).astype(int)
    recomputed = {
        "f1_macro": float(f1_score(labels[masks["test"]], predicted, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels[masks["test"]], scores)),
        "threshold": threshold,
        "predicted_anomaly_count": int(predicted.sum()),
        "test_anomaly_count": int(labels[masks["test"]].sum()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return recomputed, compare_metrics(original, recomputed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/HSMAD"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result, comparison = recompute(args.root, args.dataset, args.run_dir)
    audit_dir = args.run_dir / "audit_recompute"
    audit_dir.mkdir(exist_ok=False)
    (audit_dir / "recomputed_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (audit_dir / "compare_to_original.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
