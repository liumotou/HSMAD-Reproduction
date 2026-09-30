"""Independent checkpoint-only recomputation for PC-GNN candidate artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import dgl
import torch
from sklearn.metrics import f1_score, roc_auc_score

from methods.pcgnn.src.model import SingleRelationPCGNN
from methods.pcgnn.src.protocol import build_train_positive_nodes
from methods.pcgnn.src.run import (
    build_single_relation_adjacency,
    choose_threshold,
    evaluate_nodes,
    official_row_sum_normalize,
    prepare_training_graph,
    write_json,
)


FIELDS = ("f1_macro", "auroc", "threshold", "best_epoch", "predicted_anomaly_count")


def recompute(output: Path) -> dict[str, object]:
    config = json.loads((output / "config_snapshot.json").read_text(encoding="utf-8"))
    original = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    graph = prepare_training_graph(dgl.load_graphs(str(ROOT / config["dataset_file"]))[0][0])
    source, destination = graph.edges(order="eid")
    adjacency = build_single_relation_adjacency(source.cpu(), destination.cpu(), graph.num_nodes())
    device = torch.device("cuda")
    features = official_row_sum_normalize(graph.ndata["feature"].float()).to(device)
    labels = graph.ndata["label"].long().to(device)
    train_mask = graph.ndata["train_mask"].bool()
    train_positive = build_train_positive_nodes(graph.ndata["label"], train_mask).to(device)
    val_nodes = graph.ndata["val_mask"].bool().nonzero(as_tuple=False).flatten().to(device)
    test_nodes = graph.ndata["test_mask"].bool().nonzero(as_tuple=False).flatten().to(device)
    train_mask_device = train_mask.to(device)
    model = SingleRelationPCGNN(
        features.shape[1], int(config["hidden_dim"]), 2, train_positive,
        float(config["rho"]), float(config["alpha"]),
    ).to(device)
    checkpoint_path = output / "checkpoint_validation_auroc_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_logits = evaluate_nodes(
        model, features, adjacency, labels, train_mask_device, val_nodes, int(config["batch_size"])
    )
    threshold, _ = choose_threshold(labels[val_nodes], torch.sigmoid(val_logits)[:, 1])
    test_logits = evaluate_nodes(
        model, features, adjacency, labels, train_mask_device, test_nodes, int(config["batch_size"])
    )
    probability = torch.sigmoid(test_logits)[:, 1]
    truth = labels[test_nodes]
    prediction = (probability >= threshold).long()
    values = {
        "f1_macro": float(f1_score(truth.cpu().numpy(), prediction.cpu().numpy(), average="macro")),
        "auroc": float(roc_auc_score(truth.cpu().numpy(), probability.cpu().numpy())),
        "threshold": float(threshold),
        "best_epoch": int(checkpoint["epoch"]),
        "predicted_anomaly_count": int(prediction.sum()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    differences = {
        key: {"original": original[key], "recomputed": values[key]}
        for key in FIELDS if original[key] != values[key]
    }
    result = {
        "status": "recompute_match" if not differences else "recompute_mismatch",
        "differences": differences,
        "original": {key: original[key] for key in FIELDS},
        "recomputed": values,
    }
    write_json(output / "audit_recompute" / "recompute_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = recompute(Path(args.output))
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "recompute_match" else 2)


if __name__ == "__main__":
    main()
