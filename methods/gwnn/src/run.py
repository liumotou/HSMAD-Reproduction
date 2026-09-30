"""Isolated GWNN paper-formula candidate runner for HSMAD frozen inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from methods.project_paths import project_root

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from methods.gwnn.src.model import GWNNPaperFormulaCandidate, build_paper_formula_wavelets
from methods.gwnn.src.protocol import masked_cross_entropy, select_validation_threshold, test_metrics


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    run_type: str
    result_dir: Path
    max_epoch: int
    patience: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha256_edges(graph) -> str:
    source, destination = graph.edges(order="eid")
    return hashlib.sha256(source.cpu().numpy().tobytes() + destination.cpu().numpy().tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_sha256"] = sha256_file(path)
    return config


def build_run_spec(config: dict[str, object]) -> RunSpec:
    if not config.get("_config_sha256"):
        raise ValueError("missing config SHA256 provenance")
    if config["run_type"] not in {"smoke", "diagnostic", "formal"}:
        raise ValueError("unsupported run_type")
    return RunSpec(
        str(config["dataset"]), int(config["seed"]), str(config["run_type"]),
        Path(str(config["result_dir"])), int(config["max_epoch"]), int(config["patience"]),
    )


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def setup_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def prepare_training_graph(raw):
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    for name in ("feature", "label", "train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = raw.ndata[name]
    return graph


def append_row(path: Path, row: dict[str, object]) -> None:
    fields = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs", "best_epoch", "f1_macro", "auroc", "threshold", "wall_time_sec", "peak_gpu_mb", "edge_access"]
    exists = path.exists(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerow({field: row[field] for field in fields})


def train(config: dict[str, object]) -> dict[str, object]:
    spec = build_run_spec(config)
    output = ROOT / spec.result_dir
    ensure_new_output(output)
    setup_seed(spec.seed)
    raw_path = ROOT / str(config["dataset_file"])
    raw = dgl.load_graphs(str(raw_path))[0][0]
    graph = prepare_training_graph(raw)
    observed = {"nodes": int(graph.num_nodes()), "training_edges": int(graph.num_edges())}
    for field, expected in dict(config["expected"]).items():
        if observed[field] != expected:
            raise RuntimeError(f"frozen input mismatch for {field}: {observed[field]} != {expected}")
    masks = {name: graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    if any(torch.logical_and(masks[a], masks[b]).any() for a, b in (("train_mask", "val_mask"), ("train_mask", "test_mask"), ("val_mask", "test_mask"))):
        raise RuntimeError("frozen masks overlap")
    source, destination = graph.edges(order="eid")
    edge_index = torch.stack((source, destination))
    hashes = {
        "dataset_file_sha256": sha256_file(raw_path), "feature_sha256": sha256_tensor(graph.ndata["feature"]),
        "label_sha256": sha256_tensor(graph.ndata["label"]), "train_mask_sha256": sha256_tensor(masks["train_mask"]),
        "val_mask_sha256": sha256_tensor(masks["val_mask"]), "test_mask_sha256": sha256_tensor(masks["test_mask"]),
        "raw_graph_edges_sha256": sha256_edges(raw), "training_graph_edges_sha256": sha256_edges(graph),
        "model_py_sha256": sha256_file(ROOT / "methods/gwnn/src/model.py"),
        "protocol_py_sha256": sha256_file(ROOT / "methods/gwnn/src/protocol.py"),
        "runner_py_sha256": sha256_file(Path(__file__)), "config_sha256": str(config["_config_sha256"]),
    }
    output.mkdir(parents=True)
    write_json(output / "config_snapshot.json", config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GWNN smoke requires the audited CUDA environment")
    torch.cuda.reset_peak_memory_stats(device)
    preflight = {
        "passed": True, "candidate_protocol_not_author_exact": True,
        "formula_choice": "paper exp(-sL)/exp(+sL), not literal in-place source mutation",
        "raw_graph": {"nodes": raw.num_nodes(), "edges": raw.num_edges()}, "training_graph": observed,
        "feature_shape": list(graph.ndata["feature"].shape), "mask_counts": {name: int(value.sum()) for name, value in masks.items()},
        "hashes": hashes, "edge_access": "graph_edges_required",
        "dense_basis_estimate_mb_float32_each": graph.num_nodes() ** 2 * 4 / 1024 ** 2,
    }
    write_json(output / "preflight.json", preflight)
    started = time.monotonic()
    edge_index = edge_index.to(device)
    basis_started = time.monotonic()
    wavelet, inverse = build_paper_formula_wavelets(
        edge_index, graph.num_nodes(), float(config["wavelet_scale"]),
        float(config["wavelet_threshold"]), torch.float32,
    )
    basis_time = time.monotonic() - basis_started
    features = graph.ndata["feature"].float().to(device)
    labels = graph.ndata["label"].long().to(device)
    train_mask, val_mask, test_mask = (masks[name].to(device) for name in ("train_mask", "val_mask", "test_mask"))
    model = GWNNPaperFormulaCandidate(features.shape[1], int(config["hidden_dim"]), 2, graph.num_nodes(), float(config["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history: list[dict[str, object]] = []
    best_epoch, best_val_loss = 0, float("inf")
    checkpoint_path = output / "checkpoint_validation_loss_best.pt"
    for epoch in range(1, spec.max_epoch + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(features, wavelet, inverse)
        classification = masked_cross_entropy(logits, labels, train_mask)
        first_layer_l2 = 0.5 * sum(parameter.square().sum() for parameter in model.conv1.parameters())
        loss = classification + float(config["weight_decay_first_layer"]) * first_layer_l2
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(features, wavelet, inverse)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            validation_loss = float(masked_cross_entropy(logits, labels, val_mask).item())
        val_y = labels[val_mask].cpu().numpy(); val_p = probabilities[val_mask].cpu().numpy()
        threshold, val_f1 = select_validation_threshold(labels, probabilities, val_mask)
        record = {"epoch": epoch, "train_loss": float(loss.item()), "validation_loss": validation_loss,
                  "validation_f1_macro": val_f1, "validation_auroc": float(roc_auc_score(val_y, val_p)),
                  "validation_auprc": float(average_precision_score(val_y, val_p)), "validation_threshold": threshold,
                  "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        if validation_loss < best_val_loss:
            best_epoch, best_val_loss = epoch, validation_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
        if epoch - best_epoch >= spec.patience: break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.no_grad(): probabilities = torch.softmax(model(features, wavelet, inverse), dim=1)[:, 1]
    threshold, validation_f1 = select_validation_threshold(labels, probabilities, val_mask)
    result = test_metrics(labels, probabilities, test_mask, threshold)
    metrics = {
        "method": "GWNN", "protocol_version": str(config["protocol_version"]), "candidate_protocol_not_author_exact": True,
        "dataset": spec.dataset, "seed": spec.seed, "run_type": spec.run_type,
        "status": "smoke" if spec.run_type == "smoke" else "OK", "actual_epochs": len(history),
        "best_epoch": int(checkpoint["epoch"]), "f1_macro": result["f1_macro"], "auroc": result["auroc"],
        "auprc": result["auprc"], "threshold": threshold, "validation_f1_macro": validation_f1,
        "predicted_anomaly_count": result["predicted_anomaly_count"], "actual_anomaly_count": result["actual_anomaly_count"],
        "confusion_matrix": result["confusion_matrix"], "basis_build_time_sec": basis_time,
        "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024 ** 2),
        "edge_access": "graph_edges_required", "hashes": hashes, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "validation_history.json", history); write_json(output / "metrics.json", metrics)
    append_row(output.parent / "runs.csv", metrics)
    write_json(output / "artifact_sha256s.json", {item.name: sha256_file(item) for item in output.iterdir() if item.is_file()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    result = train(load_config(Path(args.config)))
    print("RUN_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
