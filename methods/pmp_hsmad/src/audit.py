"""Independent, evaluation-only recomputation of a saved PMP candidate checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from methods.dsgad.src.protocol import setup_seed
from methods.dsgad.src.runner import load_data
from methods.pmp_hsmad.src.protocol import test_metrics, validation_metrics
from methods.pmp_hsmad.src.runner import _forward_batches, build_model, make_loader, prepare_pmp_graph


NUMERIC_TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_metrics(original: dict, recomputed: dict) -> dict:
    differences = {}
    for field in ("f1_macro", "auroc", "threshold"):
        delta = abs(float(original[field]) - float(recomputed[field]))
        differences[field] = {"absolute_difference": delta, "match": delta <= NUMERIC_TOLERANCE}
    field = "predicted_anomaly_count"
    differences[field] = {
        "difference": int(recomputed[field]) - int(original[field]),
        "match": int(recomputed[field]) == int(original[field]),
    }
    return {"match": all(item["match"] for item in differences.values()), "differences": differences}


def recompute_artifact(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config_snapshot.json").read_text())
    original = json.loads((artifact_dir / "metrics.json").read_text())
    checkpoint_path = artifact_dir / "checkpoint_validation_auroc_best.pt"
    setup_seed(int(config["seed"]))
    raw, _ = load_data(config["dataset"])
    graph = prepare_pmp_graph(raw)
    val_ids = torch.nonzero(graph.ndata["val_mask"].bool(), as_tuple=False).reshape(-1)
    test_ids = torch.nonzero(graph.ndata["test_mask"].bool(), as_tuple=False).reshape(-1)
    val_loader = make_loader(graph, val_ids, 4096, False)
    test_loader = make_loader(graph, test_ids, 4096, False)
    device = torch.device("cuda")
    model = build_model(raw.ndata["feature"].shape[1], config["batch_size"], config["dropout"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        val_probability, val_labels = _forward_batches(model, val_loader, list(graph.etypes), device)
        test_probability, test_labels = _forward_batches(model, test_loader, list(graph.etypes), device)
    val = validation_metrics(val_probability, val_labels, torch.ones_like(val_labels, dtype=torch.bool))
    recomputed = test_metrics(
        test_probability,
        test_labels,
        torch.ones_like(test_labels, dtype=torch.bool),
        val["threshold"],
    )
    comparison = compare_metrics(original, recomputed)
    result = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "comparison": comparison,
        "dataset": config["dataset"],
        "recomputed": recomputed,
        "seed": int(config["seed"]),
        "status": "recompute_match" if comparison["match"] else "recompute_mismatch",
        "validation": val,
    }
    output = artifact_dir / "audit_recompute"
    output.mkdir(exist_ok=False)
    (output / "recompute.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(recompute_artifact(args.artifact_dir), sort_keys=True))


if __name__ == "__main__":
    main()
