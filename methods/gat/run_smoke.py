"""Run the approved GAT Weibo seed-0 five-epoch smoke experiment."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from model import OriginalGAT8x8


ROOT = Path(__file__).resolve().parents[3]
PAPER_F1 = 0.9408
PAPER_AUROC = 0.9614


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def threshold_metrics(labels, anomaly_probability):
    candidates = [round(i / 100, 2) for i in range(5, 100, 5)]
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        predicted = (anomaly_probability >= threshold).astype(np.int64)
        value = f1_score(labels, predicted, average="macro", zero_division=0)
        if value > best_f1:
            best_f1, best_threshold = value, threshold
    return best_threshold, best_f1


def probability_metrics(labels, probability, threshold):
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "f1_macro": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "predicted_anomaly_count": int(prediction.sum()),
        "predicted_anomaly_ratio": float(prediction.mean()),
    }


def graph_sha256(graph):
    src, dst = graph.edges(order="eid")
    digest = hashlib.sha256()
    digest.update(src.detach().cpu().contiguous().numpy().tobytes())
    digest.update(dst.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def framework_versions():
    output = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dgl": dgl.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        output["gpu_name"] = torch.cuda.get_device_name(0)
        output["gpu_total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    output_dir = ROOT / config["result_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke directory: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    save_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    save_json(output_dir / "environment" / "framework_versions.json", framework_versions())

    setup_seed(config["seed"])
    raw_path = ROOT / config["dataset_file"]
    raw_graph = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    feature = raw_graph.ndata["feature"]
    label = raw_graph.ndata["label"].long()
    raw_masks = {key: raw_graph.ndata[key] for key in ("train_mask", "val_mask", "test_mask")}
    masks = {key: value.bool() for key, value in raw_masks.items()}
    hashes = {
        "dataset_file_sha256": sha256_file(raw_path),
        "feature_sha256": tensor_sha256(feature),
        "label_sha256": tensor_sha256(label),
        **{f"{key}_sha256": tensor_sha256(value) for key, value in raw_masks.items()},
        "raw_graph_edges_sha256": graph_sha256(raw_graph),
        "training_graph_edges_sha256": graph_sha256(graph),
        "model_py_sha256": sha256_file(ROOT / "methods/gat/src/model.py"),
        "run_smoke_py_sha256": sha256_file(Path(__file__)),
        "reference_models_gat_py_sha256": config["source_reference"]["models_gat_py_sha256"],
        "reference_layers_py_sha256": config["source_reference"]["layers_py_sha256"],
    }
    preflight = {
        "passed": True,
        "protocol_version": config["protocol_version"],
        "run_type": config["run_type"],
        "edge_access": config["edge_access"],
        "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "feature_shape": list(feature.shape),
        "label_shape": list(label.shape),
        "mask_counts": {key: int(value.sum()) for key, value in masks.items()},
        "hashes": hashes,
    }
    expected_hashes = config["expected_hashes"]
    for name, expected in expected_hashes.items():
        actual = hashes[name]
        if actual != expected:
            raise RuntimeError(f"Frozen input mismatch for {name}: {actual} != {expected}")
    if graph.num_edges() != config["expected_training_graph_edges"]:
        raise RuntimeError(f"Training graph edge mismatch: {graph.num_edges()}")
    save_json(output_dir / "preflight.json", preflight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph.to(device)
    feature, label = feature.to(device), label.to(device)
    masks = {key: value.to(device) for key, value in masks.items()}
    model = OriginalGAT8x8(feature.shape[1], config["feature_dropout"], config["attention_dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    for epoch in range(1, config["max_epoch"] + 1):
        model.train()
        logits = model(graph, feature)
        loss = F.cross_entropy(logits[masks["train_mask"]], label[masks["train_mask"]])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(graph, feature), dim=1)[:, 1].detach().cpu().numpy()
        val_labels = label[masks["val_mask"]].detach().cpu().numpy()
        val_probabilities = probabilities[masks["val_mask"].detach().cpu().numpy()]
        threshold, val_f1 = threshold_metrics(val_labels, val_probabilities)
        val_metrics = probability_metrics(val_labels, val_probabilities, threshold)
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "validation_threshold": threshold,
                        "validation_f1_macro": val_f1, "validation_auroc": val_metrics["auroc"],
                        "validation_auprc": val_metrics["auprc"]})
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    checkpoint_path = output_dir / "checkpoint_final_epoch_5.pt"
    torch.save({"epoch": config["max_epoch"], "model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(graph, feature), dim=1)[:, 1].detach().cpu().numpy()
    val_index, test_index = masks["val_mask"].detach().cpu().numpy(), masks["test_mask"].detach().cpu().numpy()
    val_labels, test_labels = label[masks["val_mask"]].detach().cpu().numpy(), label[masks["test_mask"]].detach().cpu().numpy()
    threshold, validation_f1 = threshold_metrics(val_labels, probabilities[val_index])
    test_metrics = probability_metrics(test_labels, probabilities[test_index], threshold)
    elapsed = time.monotonic() - started
    peak_gpu_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
    metrics = {
        "status": "smoke", "run_type": "smoke", "method": "GAT", "dataset": config["dataset"], "seed": config["seed"],
        "best_epoch": config["max_epoch"], "checkpoint_protocol": "smoke_final_epoch_5", "threshold": threshold,
        "validation_f1_macro": validation_f1, "f1_macro": test_metrics["f1_macro"], "auroc": test_metrics["auroc"],
        "auprc": test_metrics["auprc"], "test_predicted_anomaly_count": test_metrics["predicted_anomaly_count"],
        "test_predicted_anomaly_ratio": test_metrics["predicted_anomaly_ratio"], "wall_time_sec": elapsed,
        "peak_gpu_mb": peak_gpu_mb, "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "edge_access": config["edge_access"], "hashes": hashes,
        "paper_weibo_gat": {"f1_macro": PAPER_F1, "auroc": PAPER_AUROC},
        "paper_delta": {"f1_macro": test_metrics["f1_macro"] - PAPER_F1, "auroc": test_metrics["auroc"] - PAPER_AUROC},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "validation_history.json", history)
    save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "dataset", "seed", "run_type", "status", "f1_macro", "auroc", "threshold", "best_epoch", "wall_time_sec", "peak_gpu_mb", "edge_access"])
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in writer.fieldnames})
    save_json(output_dir / "artifact_sha256s.json", {path.name: sha256_file(path) for path in output_dir.iterdir() if path.is_file()})
    print("SMOKE_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
