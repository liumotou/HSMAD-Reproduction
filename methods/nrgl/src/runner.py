"""Isolated NRGL candidate runner using frozen HSMAD inputs only."""

import csv
import hashlib
import json
import random
import time
from pathlib import Path

import dgl
import numpy as np
import torch
from torch import nn

from .data import ROOT, frozen_dataset_contract, load_frozen_dgl_graph
from .model import NRGLCore
from .selection import compute_test_metrics, select_validation_checkpoint_metrics, should_replace_checkpoint

RUN_FIELDS = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs", "best_epoch", "f1_macro", "auroc", "threshold", "wall_time_sec", "peak_gpu_mb"]

def execution_contract(run_type):
    if run_type in {"smoke", "smoke_edge_weight_fix"}:
        return {"run_type": "smoke", "max_epoch": 5, "patience": 5, "checkpoint_metric": "validation_auprc"}
    if run_type in {"diagnostic", "formal"}:
        return {"run_type": run_type, "max_epoch": 200, "patience": 50, "checkpoint_metric": "validation_auprc"}
    raise ValueError(f"unsupported run type: {run_type}")


def output_directory(dataset, run_type, seed):
    return ROOT / "results" / "experiments" / "nrgl" / dataset / "nrgl_hsmad_candidate" / run_type / f"seed_{seed}"


def masked_cross_entropy(logits, labels, train_mask, class_weight):
    return nn.CrossEntropyLoss(weight=class_weight)(logits[train_mask], labels[train_mask])


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def append_run_record(path, metrics):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: metrics.get(field) for field in RUN_FIELDS})


def prepare_training_graph(raw):
    return dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))


def run(config, seed, run_type):
    contract = {**execution_contract(run_type), **config, "seed": int(seed), "positioning": "candidate_protocol_not_author_exact"}
    dataset = contract["dataset"]
    frozen_dataset_contract(dataset)
    output = output_directory(dataset, run_type, seed)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    setup_seed(seed)
    raw, _ = load_frozen_dgl_graph(dataset)
    graph = prepare_training_graph(raw)
    feature = raw.ndata["feature"].float()
    label = raw.ndata["label"].long().reshape(-1)
    masks = {key: raw.ndata[key].bool() for key in ("train_mask", "val_mask", "test_mask")}
    train_labels = label[masks["train_mask"]]
    normal_count, anomaly_count = int((train_labels == 0).sum()), int((train_labels == 1).sum())
    if anomaly_count == 0:
        raise RuntimeError("frozen train mask has no anomaly label")
    hashes = {
        "dataset_file_sha256": _sha_file(ROOT / frozen_dataset_contract(dataset)["dataset_file"]),
        "feature_sha256": _sha_tensor(feature), "label_sha256": _sha_tensor(label),
        **{f"{key}_sha256": _sha_tensor(value) for key, value in masks.items()},
        "model_py_sha256": _sha_file(ROOT / "methods/nrgl/src/model.py"),
        "runner_py_sha256": _sha_file(Path(__file__)), "selection_py_sha256": _sha_file(ROOT / "methods/nrgl/src/selection.py"),
    }
    output.mkdir(parents=True)
    _dump(output / "config_snapshot.json", contract)
    _dump(output / "preflight.json", {
        "passed": True, "edge_access": "graph_edges_required",
        "raw_graph": {"nodes": raw.num_nodes(), "edges": raw.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "feature_shape": list(feature.shape), "mask_counts": {key: int(value.sum()) for key, value in masks.items()},
        "train_normal_count": normal_count, "train_anomaly_count": anomaly_count,
        "class_weight": "none", "hashes": hashes,
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, feature, label = graph.to(device), feature.to(device), label.to(device)
    masks = {key: value.to(device) for key, value in masks.items()}
    model = NRGLCore(graph.num_nodes(), feature.shape[1], contract["hidden_dim"], contract["order"], contract["alpha"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=contract["learning_rate"], weight_decay=contract["weight_decay"])
    criterion_weight = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, best_auprc, best_epoch, best_state, bad_epochs = time.monotonic(), None, 0, None, 0
    history = []
    for epoch in range(1, contract["max_epoch"] + 1):
        model.train()
        logits = model(graph, feature)
        loss = masked_cross_entropy(logits, label, masks["train_mask"], criterion_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = select_validation_checkpoint_metrics(model(graph, feature), label, masks["val_mask"], contract["threshold_candidates"])
        record = {"epoch": epoch, "train_loss": float(loss.item()), **validation}
        history.append(record)
        if should_replace_checkpoint(validation["validation_auprc"], best_auprc):
            best_auprc, best_epoch, best_state, bad_epochs = validation["validation_auprc"], epoch, {key: value.detach().cpu() for key, value in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if not run_type.startswith("smoke") and bad_epochs >= contract["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(graph, feature)
        validation = select_validation_checkpoint_metrics(final_logits, label, masks["val_mask"], contract["threshold_candidates"])
        test = compute_test_metrics(final_logits, label, masks["test_mask"], validation["threshold"])
    checkpoint = output / "checkpoint_auprc_best.pt"
    torch.save({"epoch": best_epoch, "model_state_dict": best_state, "config": contract}, checkpoint)
    _dump(output / "validation_history.json", history)
    metrics = {"method": "NRGL-HSMAD-adapted", "protocol_version": "nrgl_hsmad_candidate", "dataset": dataset, "seed": seed,
               "run_type": contract["run_type"], "status": "smoke" if run_type.startswith("smoke") else "OK", "actual_epochs": epoch,
               "best_epoch": best_epoch, "validation_auprc": best_auprc, "threshold": validation["threshold"],
               "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
               "edge_access": "graph_edges_required", "hashes": hashes, **test}
    _dump(output / "metrics.json", metrics)
    append_run_record(output.parent / "runs.csv", metrics)
    _dump(output / "artifact_sha256s.json", {path.name: _sha_file(path) for path in output.iterdir() if path.is_file()})
    return metrics
