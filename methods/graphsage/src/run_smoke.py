"""Isolated five-epoch GraphSAGE-GADBench-h64 Weibo smoke runner."""

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

from model import GraphSAGEGADBench, architecture_contract
from protocol import class_weight_from_train_labels, numpy_bool_mask, numpy_labels, numpy_probabilities, select_validation_f1_threshold


ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def graph_sha256(graph):
    source, destination = graph.edges(order="eid")
    value = source.cpu().contiguous().numpy().tobytes() + destination.cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        ROOT / "methods/mlp", ROOT / "methods/gcn", ROOT / "methods/gat", ROOT / "methods/gat_v2_gadbench",
        ROOT / "results/experiments/mlp", ROOT / "results/experiments/gcn", ROOT / "results/experiments/gat", ROOT / "results/experiments/gat_v2_gadbench",
    ]
    values = {}
    for target in protected:
        if target.is_file():
            values[str(target.relative_to(ROOT))] = file_sha256(target)
        elif target.exists():
            for entry in sorted(path for path in target.rglob("*") if path.is_file()):
                values[str(entry.relative_to(ROOT))] = file_sha256(entry)
    return {"manifest_sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(), "files": values}


def framework_info():
    value = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "dgl": dgl.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        value.update({"gpu": torch.cuda.get_device_name(0), "gpu_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2, "gpu_free_mb_at_start": (torch.cuda.mem_get_info(0)[0] / 1024**2)})
    return value


def recoverable_output_dir(output_dir):
    """Permit continuation only when a prior attempt produced no result artifact."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return True
    result_artifacts = {
        "metrics.json", "validation_history.json", "runs.csv", "artifact_sha256s.json",
        "checkpoint_final_epoch_5.pt", "protected_manifest_after.json",
    }
    return not any((output_dir / name).exists() for name in result_artifacts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["result_dir"]
    if not recoverable_output_dir(output_dir):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protected_before = protected_manifest()
    write_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    write_json(output_dir / "environment" / "framework.json", framework_info())
    write_json(output_dir / "protected_manifest_before.json", protected_before)

    setup_seed(config["seed"])
    raw_path = ROOT / config["dataset_file"]
    raw_graph = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    features = raw_graph.ndata["feature"]
    labels = raw_graph.ndata["label"].long()
    masks = {name: raw_graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    graph.ndata["feature"] = features
    hashes = {
        "dataset_file_sha256": file_sha256(raw_path), "feature_sha256": tensor_sha256(features),
        "label_sha256": tensor_sha256(labels), "raw_graph_edges_sha256": graph_sha256(raw_graph),
        "training_graph_edges_sha256": graph_sha256(graph), "model_py_sha256": file_sha256(ROOT / "methods/graphsage/src/model.py"),
        "protocol_py_sha256": file_sha256(ROOT / "methods/graphsage/src/protocol.py"), "run_smoke_py_sha256": file_sha256(Path(__file__)),
    }
    hashes.update({f"{name}_sha256": tensor_sha256(value) for name, value in masks.items()})
    for name, expected in config["expected_hashes"].items():
        if hashes[name] != expected:
            raise RuntimeError(f"Frozen input mismatch: {name}: {hashes[name]} != {expected}")
    if graph.num_edges() != config["expected_training_graph_edges"]:
        raise RuntimeError(f"Unexpected training graph edges: {graph.num_edges()}")
    class_weight, normal_count, anomaly_count = class_weight_from_train_labels(labels[masks["train_mask"]])
    model = GraphSAGEGADBench(features.shape[1], config["h_feats"], 2, config["num_layers"], config["aggregation"], config["dropout"], config["activation"])
    model_contract = architecture_contract()
    if len(model.layers) != 2 or model.layers[0]._aggre_type != "pool" or model.layers[0]._out_feats != 64 or model.layers[1]._out_feats != 64:
        raise RuntimeError("Model contract mismatch: expected two pool SAGEConv layers with h_feats=64")
    write_json(output_dir / "preflight.json", {
        "passed": True, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()}, "feature_shape": list(features.shape),
        "label_shape": list(labels.shape), "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "hashes": hashes,
        "train_normal_count": normal_count, "train_anomaly_count": anomaly_count, "class_weight": class_weight,
        "edge_access": config["edge_access"], "model_contract": model_contract,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, labels = graph.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    for epoch in range(1, config["max_epoch"] + 1):
        model.train()
        logits = model(graph)
        loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(graph), dim=1)[:, 1]
        val_mask = numpy_bool_mask(masks["val_mask"])
        val_labels = numpy_labels(labels[masks["val_mask"]])
        values = numpy_probabilities(probabilities)
        threshold, validation_f1 = select_validation_f1_threshold(val_labels, values[val_mask])
        validation = split_metrics(val_labels, values[val_mask], threshold)
        record = {"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": validation_f1,
                  "validation_auroc": validation["auroc"], "validation_auprc": validation["auprc"], "validation_threshold": threshold,
                  "learning_rate": optimizer.param_groups[0]["lr"], "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint = output_dir / "checkpoint_final_epoch_5.pt"
    torch.save({"epoch": config["max_epoch"], "model_state_dict": model.state_dict(), "config": config}, checkpoint)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(graph), dim=1)[:, 1]
    values = numpy_probabilities(probabilities)
    val_mask, test_mask = numpy_bool_mask(masks["val_mask"]), numpy_bool_mask(masks["test_mask"])
    val_labels, test_labels = numpy_labels(labels[masks["val_mask"]]), numpy_labels(labels[masks["test_mask"]])
    threshold, validation_f1 = select_validation_f1_threshold(val_labels, values[val_mask])
    test = split_metrics(test_labels, values[test_mask], threshold)
    protected_after = protected_manifest()
    metrics = {
        "method": config["method"], "protocol_version": config["protocol_version"], "dataset": config["dataset"], "seed": config["seed"],
        "run_type": "smoke", "status": "smoke", "actual_epochs": config["max_epoch"], "checkpoint_epoch": config["max_epoch"],
        "checkpoint_protocol": config["checkpoint_protocol"], "threshold": threshold, "validation_f1_macro": validation_f1,
        "f1_macro": test["f1_macro"], "auroc": test["auroc"], "auprc": test["auprc"], "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()), "edge_access": config["edge_access"],
        "class_weight": class_weight, "train_normal_count": normal_count, "train_anomaly_count": anomaly_count, "hashes": hashes,
        "protected_manifest_unchanged": protected_before == protected_after, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "validation_history.json", history)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "protected_manifest_after.json", protected_after)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs", "f1_macro", "auroc", "threshold", "checkpoint_epoch", "wall_time_sec", "peak_gpu_mb", "train_normal_count", "train_anomaly_count", "class_weight", "edge_access"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: metrics[name] for name in fields})
    write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
    print("SMOKE_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
