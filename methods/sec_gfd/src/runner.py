"""Isolated SEC-GFD candidate runner with frozen HSMAD masks.

This is not a byte-identical official entrypoint: it disables the official
label-driven edge deletion and per-epoch test selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import dgl
import numpy as np
import torch

from methods.sec_gfd.src.model import SECGFDModel
from methods.sec_gfd.src.protocol import final_test_values, masked_loss, validation_selection

ROOT = Path('/root/autodl-tmp/HSMAD')
DATASETS = {"weibo", "amazon", "tolokers", "tfinance"}


def candidate_config(dataset: str, run_type: str) -> dict[str, object]:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported SEC-GFD candidate dataset: {dataset}")
    if run_type not in {"smoke", "diagnostic", "formal"}:
        raise ValueError(f"unsupported run type: {run_type}")
    return {
        "method": "SEC-GFD-HSMAD-adapted",
        "protocol_version": "sec_gfd_hsmad_h64_candidate",
        "positioning": "candidate_protocol_not_author_exact",
        "dataset": dataset,
        "dataset_file": f"datasets/{dataset}",
        "hidden_dim": 64,
        "order": 2,
        "high_order": 2,
        "beta": 0.2,
        "class_weight": "train_normal_over_train_anomaly",
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "graph_preprocess": "to_bidirected -> remove_self_loop -> add_self_loop",
        "edge_access": "graph_edges_required",
        "checkpoint_protocol": "validation_AUPRC_best",
        "early_stop_protocol": "validation_AUPRC_patience",
        "threshold_protocol": "validation_F1_macro_grid_0.05_to_0.95",
        "test_protocol": "final_only_after_checkpoint_and_validation_threshold",
        "run_type": run_type,
        "max_epoch": 5 if run_type == "smoke" else 100,
        "patience": None if run_type == "smoke" else 50,
    }


def artifact_root(dataset: str, run_type: str, artifact_label: str | None = None) -> Path:
    """Return an isolated artifact root without changing training semantics.

    ``artifact_label`` exists solely to preserve a failed launch attempt while
    letting a clean formal attempt start in a new, non-overwriting directory.
    """
    label = artifact_label or run_type
    return ROOT / "results/experiments/sec_gfd" / dataset / "sec_gfd_hsmad_h64_candidate" / label


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare_graph(raw: dgl.DGLGraph) -> dgl.DGLGraph:
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    graph.ndata["feature"] = raw.ndata["feature"].float()
    return graph


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fingerprints(raw_file: Path, graph: dgl.DGLGraph, features: torch.Tensor, labels: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "dataset_file_sha256": sha256_file(raw_file),
        "feature_sha256": sha256_tensor(features),
        "label_sha256": sha256_tensor(labels),
        **{f"{name}_sha256": sha256_tensor(mask) for name, mask in masks.items()},
        "model_py_sha256": sha256_file(ROOT / "methods/sec_gfd/src/model.py"),
        "protocol_py_sha256": sha256_file(ROOT / "methods/sec_gfd/src/protocol.py"),
        "runner_py_sha256": sha256_file(ROOT / "methods/sec_gfd/src/runner.py"),
        "training_nodes": graph.num_nodes(),
        "training_edges": graph.num_edges(),
    }


def run_one(dataset: str, seed: int, run_type: str, artifact_label: str | None = None) -> dict[str, object]:
    config = candidate_config(dataset, run_type)
    config["seed"] = seed
    output = artifact_root(dataset, run_type, artifact_label) / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing SEC-GFD artifact: {output}")
    setup_seed(seed)
    raw_file = ROOT / str(config["dataset_file"])
    raw = dgl.load_graphs(str(raw_file))[0][0]
    graph = prepare_graph(raw)
    features = raw.ndata["feature"].float()
    labels = raw.ndata["label"].long().reshape(-1)
    masks = {name: raw.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    if any(torch.logical_and(masks[left], masks[right]).any() for left, right in (("train_mask", "val_mask"), ("train_mask", "test_mask"), ("val_mask", "test_mask"))):
        raise RuntimeError("frozen masks overlap")
    train_labels = labels[masks["train_mask"]]
    anomaly_count = int((train_labels == 1).sum())
    normal_count = int((train_labels == 0).sum())
    if not anomaly_count:
        raise RuntimeError("frozen train mask has no anomaly labels")
    class_weight = [1.0, normal_count / anomaly_count]
    fingerprints = _fingerprints(raw_file, graph, features, labels, masks)
    output.mkdir(parents=True)
    _write_json(output / "config_snapshot.json", config)
    _write_json(output / "preflight.json", {"passed": True, "raw_graph": {"nodes": raw.num_nodes(), "edges": raw.num_edges()}, "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()}, "feature_shape": list(features.shape), "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "train_normal_count": normal_count, "train_anomaly_count": anomaly_count, "class_weight": class_weight, "hashes": fingerprints})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: mask.to(device) for name, mask in masks.items()}
    weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
    model = SECGFDModel(features.shape[1], 64, 2, graph, order=2, high_order=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    best_auprc, best_epoch, stalled, best_state = -float("inf"), None, 0, None
    history: list[dict[str, object]] = []
    for epoch in range(1, int(config["max_epoch"]) + 1):
        model.train()
        logits, embedding = model(graph, features)
        loss = masked_loss(logits, labels, masks["train_mask"], weights, embedding, features, beta=0.2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(graph, features)[0], dim=1)[:, 1]
        threshold, validation_f1, validation_auprc = validation_selection(labels, probability, masks["val_mask"])
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": validation_f1, "validation_auprc": validation_auprc, "validation_threshold": threshold, "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2)})
        if validation_auprc > best_auprc:
            best_auprc, best_epoch, stalled = validation_auprc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stalled += 1
        if config["patience"] is not None and stalled >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was created")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph, features)[0], dim=1)[:, 1]
    threshold, validation_f1, validation_auprc = validation_selection(labels, probability, masks["val_mask"])
    metrics = final_test_values(labels, probability, masks["test_mask"], threshold)
    metrics.update({"method": config["method"], "protocol_version": config["protocol_version"], "positioning": config["positioning"], "dataset": dataset, "seed": seed, "run_type": run_type, "status": "smoke" if run_type == "smoke" else "OK", "actual_epochs": len(history), "best_epoch": best_epoch, "validation_auprc": validation_auprc, "validation_f1_macro": validation_f1, "threshold": threshold, "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2), "edge_access": config["edge_access"], "hashes": fingerprints})
    torch.save({"epoch": best_epoch, "model_state_dict": best_state, "config": config}, output / "checkpoint_auprc_best.pt")
    _write_json(output / "validation_history.json", history)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "artifact_sha256s.json", {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--run-type", required=True, choices=("smoke", "diagnostic", "formal"))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--artifact-label", default=None)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for seed in (int(item) for item in args.seeds.split(",")):
        try:
            rows.append(run_one(args.dataset, seed, args.run_type, args.artifact_label))
        except Exception as error:
            rows.append({"dataset": args.dataset, "seed": seed, "run_type": args.run_type, "status": "ERROR", "error": repr(error)})
            if args.run_type == "smoke":
                raise
    base = artifact_root(args.dataset, args.run_type, args.artifact_label)
    base.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (base / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
