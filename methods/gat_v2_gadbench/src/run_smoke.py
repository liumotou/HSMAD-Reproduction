"""Five-epoch, seed-zero, isolated Weibo smoke for GAT-v2-GADBench."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from model import GADBenchGATV2, architecture_contract


ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def graph_sha256(graph):
    source, destination = graph.edges(order="eid")
    return hashlib.sha256(source.cpu().contiguous().numpy().tobytes() + destination.cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def best_threshold(labels, probabilities):
    chosen, score = 0.05, -1.0
    for threshold in [round(value / 100, 2) for value in range(5, 100, 5)]:
        candidate = f1_score(labels, probabilities >= threshold, average="macro", zero_division=0)
        if candidate > score:
            chosen, score = threshold, candidate
    return chosen, float(score)


def split_metrics(labels, probabilities, threshold):
    prediction = probabilities >= threshold
    return {
        "f1_macro": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "predicted_anomaly_count": int(prediction.sum()),
        "predicted_anomaly_ratio": float(prediction.mean()),
    }


def protected_manifest():
    protected = [
        ROOT / "main.py", ROOT / "dataset.py", ROOT / "model.py", ROOT / "utils.py",
        ROOT / "results/runs.csv", ROOT / "results/summary.csv",
        ROOT / "methods/mlp", ROOT / "methods/gcn", ROOT / "methods/gat",
        ROOT / "results/experiments/mlp", ROOT / "results/experiments/gcn", ROOT / "results/experiments/gat",
    ]
    values = {}
    for target in protected:
        if target.is_file():
            values[str(target.relative_to(ROOT))] = file_sha256(target)
        elif target.exists():
            for entry in sorted(path for path in target.rglob("*") if path.is_file()):
                values[str(entry.relative_to(ROOT))] = file_sha256(entry)
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return {"manifest_sha256": digest, "files": values}


def framework():
    values = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "dgl": dgl.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        values.update({"gpu": torch.cuda.get_device_name(0), "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2})
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["result_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    protected_before = protected_manifest()
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
        "run_smoke_py_sha256": file_sha256(Path(__file__)),
    }
    for name, expected in config["expected_hashes"].items():
        if hashes[name] != expected:
            raise RuntimeError(f"Frozen input mismatch: {name}")
    if graph.num_edges() != config["expected_training_graph_edges"]:
        raise RuntimeError("Unexpected preprocessed graph edge count")
    train_labels = labels[masks["train_mask"]]
    anomaly_count, normal_count = int(train_labels.sum()), int((train_labels == 0).sum())
    if anomaly_count == 0:
        raise RuntimeError("Cannot derive class weight: no train anomalies")
    class_weight = [1.0, normal_count / anomaly_count]
    write_json(output_dir / "preflight.json", {
        "passed": True, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "feature_shape": list(features.shape), "label_shape": list(labels.shape),
        "mask_counts": {name: int(value.sum()) for name, value in masks.items()}, "hashes": hashes,
        "train_normal_count": normal_count, "train_anomaly_count": anomaly_count, "class_weight": class_weight,
        "edge_access": config["edge_access"], "model_contract": architecture_contract(),
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = GADBenchGATV2(features.shape[1], 64, 4, 0.0, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = torch.tensor(class_weight, device=device, dtype=torch.float32)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    for epoch in range(1, config["max_epoch"] + 1):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(graph, features), dim=1)[:, 1].detach().cpu().numpy()
        val_index = masks["val_mask"].cpu().numpy()
        val_label = labels[masks["val_mask"]].cpu().numpy()
        threshold, validation_f1 = best_threshold(val_label, probability[val_index])
        validation = split_metrics(val_label, probability[val_index], threshold)
        record = {"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": validation_f1,
                  "validation_auroc": validation["auroc"], "validation_auprc": validation["auprc"],
                  "validation_threshold": threshold, "learning_rate": optimizer.param_groups[0]["lr"],
                  "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint = output_dir / "checkpoint_final_epoch_5.pt"
    torch.save({"epoch": config["max_epoch"], "model_state_dict": model.state_dict(), "config": config}, checkpoint)
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph, features), dim=1)[:, 1].detach().cpu().numpy()
    val_index, test_index = masks["val_mask"].cpu().numpy(), masks["test_mask"].cpu().numpy()
    val_label, test_label = labels[masks["val_mask"]].cpu().numpy(), labels[masks["test_mask"]].cpu().numpy()
    threshold, validation_f1 = best_threshold(val_label, probability[val_index])
    test = split_metrics(test_label, probability[test_index], threshold)
    protected_after = protected_manifest()
    metrics = {
        "method": "GAT-v2-GADBench", "protocol_version": config["protocol_version"], "dataset": "weibo", "seed": 0,
        "run_type": "smoke", "status": "smoke", "actual_epochs": 5, "checkpoint_epoch": 5,
        "checkpoint_protocol": "smoke_final_epoch_5_no_early_stop", "threshold": threshold,
        "validation_f1_macro": validation_f1, "f1_macro": test["f1_macro"], "auroc": test["auroc"], "auprc": test["auprc"],
        "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_tensor_shapes": {"input": ["N", int(features.shape[1])], "post_input_linear": ["N", 64],
                                "per_head": ["N", 16, 4], "post_each_block": ["N", 64], "logits": ["N", 2]},
        "edge_access": config["edge_access"], "class_weight": class_weight, "hashes": hashes,
        "protected_manifest_unchanged": protected_before == protected_after,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "validation_history.json", history)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "protected_manifest_after.json", protected_after)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs", "f1_macro", "auroc", "threshold", "checkpoint_epoch", "wall_time_sec", "peak_gpu_mb", "edge_access"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: metrics[name] for name in fields})
    write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
    print("SMOKE_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
