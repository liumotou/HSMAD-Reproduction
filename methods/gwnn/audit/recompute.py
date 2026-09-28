"""Strict read-only checkpoint recomputation for GWNN candidate artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/HSMAD")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
site.addsitedir("/root/miniconda3/lib/python3.10/site-packages")

import dgl
import torch

from methods.gwnn.src.model import GWNNPaperFormulaCandidate, build_paper_formula_wavelets
from methods.gwnn.src.protocol import select_validation_threshold, test_metrics
from methods.gwnn.src.run import prepare_training_graph, write_json


FIELDS = ("f1_macro", "auroc", "threshold", "best_epoch", "predicted_anomaly_count")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_metrics(original: dict[str, object], recomputed: dict[str, object]) -> dict[str, object]:
    differences = {
        field: {"original": original[field], "recomputed": recomputed[field]}
        for field in FIELDS if original[field] != recomputed[field]
    }
    return {
        "status": "recompute_match" if not differences else "recompute_mismatch",
        "differences": differences,
        "original": {field: original[field] for field in FIELDS},
        "recomputed": recomputed,
    }


def recompute(output: Path) -> dict[str, object]:
    config = json.loads((output / "config_snapshot.json").read_text(encoding="utf-8"))
    original = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    raw = dgl.load_graphs(str(ROOT / str(config["dataset_file"])))[0][0]
    graph = prepare_training_graph(raw)
    source, destination = graph.edges(order="eid")
    device = torch.device("cuda")
    edge_index = torch.stack((source, destination)).to(device)
    wavelet, inverse = build_paper_formula_wavelets(
        edge_index, graph.num_nodes(), float(config["wavelet_scale"]),
        float(config["wavelet_threshold"]), torch.float32,
    )
    features = graph.ndata["feature"].float().to(device)
    labels = graph.ndata["label"].long().to(device)
    val_mask = graph.ndata["val_mask"].bool().to(device)
    test_mask = graph.ndata["test_mask"].bool().to(device)
    model = GWNNPaperFormulaCandidate(features.shape[1], int(config["hidden_dim"]), 2, graph.num_nodes(), float(config["dropout"])).to(device)
    checkpoint_path = output / "checkpoint_validation_loss_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(features, wavelet, inverse), dim=1)[:, 1]
    threshold, _ = select_validation_threshold(labels, probabilities, val_mask)
    metrics = test_metrics(labels, probabilities, test_mask, threshold)
    metrics.update({"threshold": threshold, "best_epoch": int(checkpoint["epoch"]), "checkpoint_sha256": sha256_file(checkpoint_path)})
    result = compare_metrics(original, metrics)
    audit_dir = output / "audit_recompute"
    write_json(audit_dir / "recompute_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    result = recompute(Path(args.output))
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "recompute_match" else 2)


if __name__ == "__main__":
    main()
