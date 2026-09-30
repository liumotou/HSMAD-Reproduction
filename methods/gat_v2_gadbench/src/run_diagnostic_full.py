"""Independent GAT-v2 Weibo seed-0 full diagnostic with dual checkpoints."""

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

from methods.gat_v2_gadbench.src.model import GADBenchGATV2, architecture_contract
from methods.gat_v2_gadbench.src.run_smoke import (
    ROOT, best_threshold, file_sha256, framework, graph_sha256, protected_manifest,
    set_seed, split_metrics, tensor_sha256, write_json,
)
from methods.gat_v2_gadbench.src.selection import update_selection


PAPER_F1 = 0.9408
PAPER_AUROC = 0.9614


def protected_manifest_with_smoke():
    manifest = protected_manifest()
    smoke_root = ROOT / "results/experiments/gat_v2_gadbench/weibo/protocol_v2_gadbench_hidden64/smoke"
    for entry in sorted(path for path in smoke_root.rglob("*") if path.is_file()):
        manifest["files"][str(entry.relative_to(ROOT))] = file_sha256(entry)
    import hashlib
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    return manifest


def evaluate_probabilities(model, graph, features):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(graph, features), dim=1)[:, 1]


def checkpoint_test(checkpoint_path, model, graph, features, labels, masks):
    checkpoint = torch.load(checkpoint_path, map_location=features.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    probability = evaluate_probabilities(model, graph, features).cpu().numpy()
    val_index, test_index = masks["val_mask"].cpu().numpy(), masks["test_mask"].cpu().numpy()
    val_label, test_label = labels[masks["val_mask"]].cpu().numpy(), labels[masks["test_mask"]].cpu().numpy()
    threshold, validation_f1 = best_threshold(val_label, probability[val_index])
    test = split_metrics(test_label, probability[test_index], threshold)
    test.update({"epoch": checkpoint["epoch"], "threshold": threshold, "validation_f1_macro": validation_f1})
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["result_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    protected_before = protected_manifest_with_smoke()
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    write_json(output_dir / "environment" / "framework.json", framework())
    write_json(output_dir / "protected_manifest_before.json", protected_before)

    set_seed(config["seed"])
    raw_path = ROOT / config["dataset_file"]
    raw_graph = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    features, labels = raw_graph.ndata["feature"], raw_graph.ndata["label"].long()
    raw_masks = {name: raw_graph.ndata[name] for name in ("train_mask", "val_mask", "test_mask")}
    masks = {name: value.bool() for name, value in raw_masks.items()}
    hashes = {
        "dataset_file_sha256": file_sha256(raw_path), "feature_sha256": tensor_sha256(features),
        "label_sha256": tensor_sha256(labels), **{f"{name}_sha256": tensor_sha256(value) for name, value in raw_masks.items()},
        "raw_graph_edges_sha256": graph_sha256(raw_graph), "training_graph_edges_sha256": graph_sha256(graph),
        "model_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/model.py"),
        "run_diagnostic_full_py_sha256": file_sha256(Path(__file__)),
        "selection_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/selection.py"),
    }
    for name, expected in config["expected_hashes"].items():
        if hashes[name] != expected:
            raise RuntimeError(f"Frozen input mismatch: {name}")
    if graph.num_edges() != config["expected_training_graph_edges"]:
        raise RuntimeError("Unexpected preprocessed graph edge count")
    train_labels = labels[masks["train_mask"]]
    anomaly_count, normal_count = int(train_labels.sum()), int((train_labels == 0).sum())
    class_weight = [1.0, normal_count / anomaly_count]
    if class_weight[1] != config["expected_anomaly_weight"]:
        raise RuntimeError("Class weight differs from frozen smoke configuration")
    write_json(output_dir / "preflight.json", {
        "passed": True, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "feature_shape": list(features.shape), "mask_counts": {name: int(value.sum()) for name, value in masks.items()},
        "hashes": hashes, "class_weight": class_weight, "model_contract": architecture_contract(),
        "early_stop_protocol": config["early_stop_protocol"], "checkpoint_protocol": config["checkpoint_protocol"],
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = GADBenchGATV2(features.shape[1], 64, 4, 0.0, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = torch.tensor(class_weight, device=device, dtype=torch.float32)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    state = {"best_auprc": -1.0, "auprc_best_epoch": None, "best_f1": -1.0,
             "f1_best_epoch": None, "patience_counter": 0}
    history = []
    auprc_checkpoint = output_dir / "checkpoint_validation_auprc_best.pt"
    f1_checkpoint = output_dir / "checkpoint_validation_f1_best_diagnostic.pt"
    for epoch in range(1, config["max_epoch"] + 1):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        probability = evaluate_probabilities(model, graph, features).cpu().numpy()
        val_index = masks["val_mask"].cpu().numpy()
        val_label = labels[masks["val_mask"]].cpu().numpy()
        threshold, val_f1 = best_threshold(val_label, probability[val_index])
        val = split_metrics(val_label, probability[val_index], threshold)
        state, auprc_improved, f1_improved = update_selection(state, epoch, val["auprc"], val_f1)
        checkpoint_payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "validation": val}
        if auprc_improved:
            torch.save(checkpoint_payload, auprc_checkpoint)
        if f1_improved:
            torch.save(checkpoint_payload, f1_checkpoint)
        record = {"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": val_f1,
                  "validation_auroc": val["auroc"], "validation_auprc": val["auprc"], "validation_threshold": threshold,
                  "auprc_best_epoch": state["auprc_best_epoch"], "f1_best_epoch": state["f1_best_epoch"],
                  "patience_counter": state["patience_counter"], "learning_rate": optimizer.param_groups[0]["lr"],
                  "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if state["patience_counter"] > config["patience"]:
            break

    auprc_test = checkpoint_test(auprc_checkpoint, model, graph, features, labels, masks)
    f1_test = checkpoint_test(f1_checkpoint, model, graph, features, labels, masks)
    protected_after = protected_manifest_with_smoke()
    metrics = {
        "method": "GAT-v2-GADBench", "protocol_version": config["protocol_version"], "dataset": "weibo", "seed": 0,
        "run_type": "diagnostic_full", "status": "diagnostic", "actual_epochs": len(history),
        "early_stop_protocol": config["early_stop_protocol"], "checkpoint_protocol": config["checkpoint_protocol"],
        "auprc_best_epoch": state["auprc_best_epoch"], "f1_best_epoch": state["f1_best_epoch"],
        "same_epoch": state["auprc_best_epoch"] == state["f1_best_epoch"],
        "auprc_best_checkpoint_test": auprc_test, "f1_best_checkpoint_test": f1_test,
        "paper": {"f1_macro": PAPER_F1, "auroc": PAPER_AUROC},
        "paper_delta_auprc_best": {"f1_macro": auprc_test["f1_macro"] - PAPER_F1, "auroc": auprc_test["auroc"] - PAPER_AUROC},
        "paper_delta_f1_best": {"f1_macro": f1_test["f1_macro"] - PAPER_F1, "auroc": f1_test["auroc"] - PAPER_AUROC},
        "class_weight": class_weight, "edge_access": config["edge_access"], "hashes": hashes,
        "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
        "protected_manifest_unchanged": protected_before == protected_after,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "validation_history.json", history)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "protected_manifest_after.json", protected_after)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs", "auprc_best_epoch", "f1_best_epoch", "wall_time_sec", "peak_gpu_mb", "edge_access"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: metrics[name] for name in fields})
    write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
    print("DIAGNOSTIC_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
