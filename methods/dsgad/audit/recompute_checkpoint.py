"""Checkpoint-only DSGAD evaluator. It never trains or changes checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from methods.dsgad.src.model import DSGADModel
from methods.dsgad.src.protocol import prepare_training_graph, test_values, validation_values
from methods.dsgad.src.runner import load_data


def compare_metrics(original, recomputed):
    differences = {}
    for field in ("f1_macro", "auroc"):
        if not math.isclose(float(original[field]), float(recomputed[field]), rel_tol=0.0, abs_tol=1e-12):
            differences[field] = {"original": original[field], "recomputed": recomputed[field], "abs_diff": abs(float(original[field]) - float(recomputed[field]))}
    for field in ("threshold", "predicted_anomaly_count", "actual_anomaly_count", "confusion_matrix"):
        if original.get(field) != recomputed.get(field): differences[field] = {"original": original.get(field), "recomputed": recomputed.get(field)}
    return {"status": "recompute_match" if not differences else "recompute_mismatch", "differences": differences, "float_abs_tolerance": 1e-12}


def recompute(run_dir):
    run_dir = Path(run_dir)
    original = json.loads((run_dir / "metrics.json").read_text())
    config = json.loads((run_dir / "config_snapshot.json").read_text())
    raw, _ = load_data(config["dataset"])
    graph = prepare_training_graph(raw)
    features = raw.ndata["feature"].float(); labels = raw.ndata["label"].long().reshape(-1)
    masks = {name: raw.ndata[name].bool() for name in ("val_mask", "test_mask")}
    device = torch.device("cuda")
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = DSGADModel(graph.num_nodes(), features.shape[1], config["hidden_dim"], degree=config["degree"], mix_beta=config["mix_beta"]).to(device)
    checkpoint = torch.load(run_dir / "checkpoint_auprc_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.no_grad(): probability = model(graph, features).softmax(1)[:, 1]
    selected_threshold, validation_f1, validation_auprc = validation_values(labels, probability, masks["val_mask"])
    values = test_values(labels, probability, masks["test_mask"], original["threshold"])
    values.update({"threshold": original["threshold"], "validation_selected_threshold": selected_threshold, "validation_f1_macro": validation_f1, "validation_auprc": validation_auprc})
    comparison = compare_metrics(original, values)
    if selected_threshold != original["threshold"]:
        comparison["differences"]["validation_selected_threshold"] = {"original": original["threshold"], "recomputed": selected_threshold}
        comparison["status"] = "recompute_mismatch"
    output = run_dir / "audit_recompute"; output.mkdir(exist_ok=False)
    (output / "recompute_metrics.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    (output / "compare_to_original.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    (output / "audit.md").write_text("# DSGAD checkpoint-only recomputation\n\n" + json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    print(json.dumps(recompute(args.run_dir), sort_keys=True))


if __name__ == "__main__": main()
