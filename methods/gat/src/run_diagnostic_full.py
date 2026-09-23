"""Independent full seed-0 diagnostic for the approved GAT Weibo candidate."""

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F

from model import OriginalGAT8x8
from run_smoke import (
    PAPER_AUROC, PAPER_F1, framework_versions, graph_sha256, load_json,
    probability_metrics, save_json, setup_seed, sha256_file, tensor_sha256,
    threshold_metrics,
)
from selection import update_selection


ROOT = Path(__file__).resolve().parents[3]


def evaluate(model, graph, feature, label, masks):
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph, feature), dim=1)[:, 1]
    output = {}
    for split in ("val_mask", "test_mask"):
        index = masks[split]
        split_label = label[index].detach().cpu().numpy()
        split_probability = probability[index].detach().cpu().numpy()
        threshold, f1 = threshold_metrics(split_label, split_probability)
        values = probability_metrics(split_label, split_probability, threshold)
        values["threshold"] = threshold
        values["f1_macro"] = f1
        output[split] = values
    return output


def save_checkpoint(path, epoch, model, config, validation):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config,
                "validation": validation}, path)


def test_from_checkpoint(path, model, graph, feature, label, masks):
    checkpoint = torch.load(path, map_location=feature.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph, feature), dim=1)[:, 1]
    val_probability = probability[masks["val_mask"]].detach().cpu().numpy()
    test_probability = probability[masks["test_mask"]].detach().cpu().numpy()
    val_label = label[masks["val_mask"]].detach().cpu().numpy()
    test_label = label[masks["test_mask"]].detach().cpu().numpy()
    threshold, validation_f1 = threshold_metrics(val_label, val_probability)
    test = probability_metrics(test_label, test_probability, threshold)
    test["threshold"] = threshold
    test["validation_f1_macro"] = validation_f1
    test["epoch"] = checkpoint["epoch"]
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    output_dir = ROOT / config["result_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic directory: {output_dir}")
    output_dir.mkdir(parents=True)
    start = time.monotonic()
    save_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    save_json(output_dir / "environment" / "framework_versions.json", framework_versions())

    setup_seed(config["seed"])
    raw_path = ROOT / config["dataset_file"]
    raw_graph = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    feature, label = raw_graph.ndata["feature"], raw_graph.ndata["label"].long()
    raw_masks = {name: raw_graph.ndata[name] for name in ("train_mask", "val_mask", "test_mask")}
    masks = {name: value.bool() for name, value in raw_masks.items()}
    hashes = {
        "dataset_file_sha256": sha256_file(raw_path), "feature_sha256": tensor_sha256(feature),
        "label_sha256": tensor_sha256(label),
        **{f"{name}_sha256": tensor_sha256(value) for name, value in raw_masks.items()},
        "raw_graph_edges_sha256": graph_sha256(raw_graph), "training_graph_edges_sha256": graph_sha256(graph),
        "model_py_sha256": sha256_file(ROOT / "methods/gat/src/model.py"),
        "run_diagnostic_full_py_sha256": sha256_file(Path(__file__)),
        "selection_py_sha256": sha256_file(ROOT / "methods/gat/src/selection.py"),
    }
    for name, expected in config["expected_hashes"].items():
        if hashes[name] != expected:
            raise RuntimeError(f"Frozen input mismatch for {name}: {hashes[name]} != {expected}")
    if graph.num_edges() != config["expected_training_graph_edges"]:
        raise RuntimeError(f"Training graph edge mismatch: {graph.num_edges()}")
    save_json(output_dir / "preflight.json", {
        "passed": True, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "mask_counts": {name: int(value.sum()) for name, value in masks.items()}, "hashes": hashes,
        "protocol_version": config["protocol_version"], "run_type": config["run_type"],
        "early_stop_protocol": config["early_stop_protocol"], "checkpoint_protocol": config["checkpoint_protocol"],
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, feature, label = graph.to(device), feature.to(device), label.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = OriginalGAT8x8(feature.shape[1], config["feature_dropout"], config["attention_dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    state = {"best_f1": -1.0, "f1_best_epoch": None, "best_auprc": -1.0,
             "checkpoint_epoch": None, "patience_counter": 0}
    history = []
    f1_checkpoint_path = output_dir / "checkpoint_validation_f1_best.pt"
    checkpoint_path = output_dir / "checkpoint_validation_auprc_best.pt"
    for epoch in range(1, config["max_epoch"] + 1):
        model.train()
        logits = model(graph, feature)
        train_loss = F.cross_entropy(logits[masks["train_mask"]], label[masks["train_mask"]])
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        validation = evaluate(model, graph, feature, label, masks)["val_mask"]
        state, f1_improved, auprc_improved = update_selection(
            state, epoch, validation["f1_macro"], validation["auprc"]
        )
        if f1_improved:
            save_checkpoint(f1_checkpoint_path, epoch, model, config, validation)
        if auprc_improved:
            save_checkpoint(checkpoint_path, epoch, model, config, validation)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
        record = {
            "epoch": epoch, "train_loss": float(train_loss.item()),
            "validation_f1_macro": validation["f1_macro"], "validation_auroc": validation["auroc"],
            "validation_auprc": validation["auprc"], "validation_threshold": validation["threshold"],
            "f1_best_epoch": state["f1_best_epoch"], "checkpoint_selected_epoch": state["checkpoint_epoch"],
            "patience_counter": state["patience_counter"], "learning_rate": optimizer.param_groups[0]["lr"],
            "peak_gpu_mb": peak_mb,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if state["patience_counter"] >= config["patience"]:
            break

    f1_test = test_from_checkpoint(f1_checkpoint_path, model, graph, feature, label, masks)
    checkpoint_test = test_from_checkpoint(checkpoint_path, model, graph, feature, label, masks)
    elapsed = time.monotonic() - start
    metrics = {
        "status": "diagnostic", "run_type": "diagnostic_full", "method": "GAT", "dataset": config["dataset"],
        "seed": config["seed"], "actual_epochs": len(history), "f1_best_epoch": state["f1_best_epoch"],
        "checkpoint_selected_epoch": state["checkpoint_epoch"], "early_stop_protocol": config["early_stop_protocol"],
        "checkpoint_protocol": config["checkpoint_protocol"], "threshold_protocol": config["threshold_protocol"],
        "f1_best_checkpoint_test": f1_test, "checkpoint_selected_test": checkpoint_test,
        "wall_time_sec": elapsed,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "edge_access": config["edge_access"], "hashes": hashes,
        "paper_weibo_gat": {"f1_macro": PAPER_F1, "auroc": PAPER_AUROC},
        "paper_delta_selected_checkpoint": {"f1_macro": checkpoint_test["f1_macro"] - PAPER_F1,
                                            "auroc": checkpoint_test["auroc"] - PAPER_AUROC},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "validation_history.json", history)
    save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["method", "dataset", "seed", "run_type", "status", "actual_epochs", "f1_best_epoch",
                   "checkpoint_selected_epoch", "wall_time_sec", "peak_gpu_mb", "edge_access"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: metrics[column] for column in columns})
    save_json(output_dir / "artifact_sha256s.json", {p.name: sha256_file(p) for p in output_dir.iterdir() if p.is_file()})
    print("DIAGNOSTIC_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
