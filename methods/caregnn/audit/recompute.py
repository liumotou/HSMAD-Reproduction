"""Independent checkpoint-only recomputation for CARE-GNN candidate artifacts."""
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
from sklearn.metrics import f1_score, roc_auc_score

from methods.caregnn.src.model import SingleRelationCareGNN
from methods.caregnn.src.run import (
    build_single_relation_adjacency,
    choose_threshold,
    evaluate_nodes,
    prepare_training_graph,
    row_l2_normalize,
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
    features = row_l2_normalize(graph.ndata["feature"].float()).to(device)
    labels = graph.ndata["label"].long().to(device)
    val_nodes = graph.ndata["val_mask"].bool().nonzero(as_tuple=False).flatten().to(device)
    test_nodes = graph.ndata["test_mask"].bool().nonzero(as_tuple=False).flatten().to(device)
    model = SingleRelationCareGNN(
        features.shape[1], int(config["hidden_dim"]), 2,
        float(config["lambda_1"]), float(config["rl_step_size"]),
    ).to(device)
    checkpoint_path = output / "checkpoint_validation_auprc_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.thresholds = [float(value) for value in checkpoint["relation_thresholds"]]
    val_logits = evaluate_nodes(model, features, adjacency, val_nodes, int(config["batch_size"]))
    threshold, _ = choose_threshold(labels[val_nodes], torch.softmax(val_logits, 1)[:, 1])
    test_logits = evaluate_nodes(model, features, adjacency, test_nodes, int(config["batch_size"]))
    test_probability = torch.softmax(test_logits, 1)[:, 1]
    test_truth = labels[test_nodes]
    prediction = (test_probability >= threshold).long()
    values = {
        "f1_macro": float(f1_score(test_truth.cpu().numpy(), prediction.cpu().numpy(), average="macro")),
        "auroc": float(roc_auc_score(test_truth.cpu().numpy(), test_probability.cpu().numpy())),
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
