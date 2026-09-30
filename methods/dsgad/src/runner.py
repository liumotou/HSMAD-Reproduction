"""DSGAD candidate runner: official topology plus frozen HSMAD no-leakage governance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import dgl
import torch

from methods.dsgad.src.model import DSGADModel
from methods.dsgad.src.protocol import class_weight, masked_weighted_loss, prepare_training_graph, setup_seed, test_values, validation_values
from methods.project_paths import project_root

ROOT = project_root()
DATASETS = {
    "weibo": (8405, 400), "tolokers": (11758, 10),
    "amazon": (11944, 25), "tfinance": (39357, 10),
}


def candidate_config(dataset, run_type="formal"):
    if dataset not in DATASETS: raise ValueError(dataset)
    if run_type not in {"smoke", "diagnostic", "formal"}: raise ValueError(run_type)
    return {
        "dataset": dataset, "run_type": run_type,
        "positioning": "candidate_protocol_not_author_exact",
        "protocol_version": "dsgad_hsmad_candidate",
        "official_commit": "bf7e0ac31a4d28c82c796b339d06b4a1a5308158",
        "hidden_dim": 64, "degree": 2, "mix_beta": 2,
        "optimizer": "Adam", "learning_rate": 0.01, "weight_decay": 0.0,
        "class_weight": "train_normal_over_train_anomaly",
        "max_epoch": 5 if run_type == "smoke" else 100,
        "patience": 5 if run_type == "smoke" else 50,
        "checkpoint_metric": "validation_auprc",
        "threshold_protocol": "validation_F1_macro_grid_0.05_to_0.95",
        "test_access": "once_after_selection",
        "graph_preprocess": "to_bidirected->remove_self_loop->add_self_loop",
        "edge_access": "graph_edges_required",
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def result_directory(dataset, run_type, seed):
    return ROOT / "results/experiments/dsgad" / dataset / "dsgad_hsmad_candidate" / run_type / f"seed_{seed}"


def load_data(dataset):
    raw_path = ROOT / "datasets" / dataset
    raw = dgl.load_graphs(str(raw_path))[0][0]
    nodes, feature_dim = DATASETS[dataset]
    if raw.num_nodes() != nodes or raw.ndata["feature"].shape[1] != feature_dim: raise RuntimeError("dataset contract mismatch")
    return raw, raw_path


def run_one(dataset, seed, run_type):
    config = candidate_config(dataset, run_type)
    output = result_directory(dataset, run_type, seed)
    output.mkdir(parents=True, exist_ok=False)
    setup_seed(seed)
    raw, raw_path = load_data(dataset)
    graph = prepare_training_graph(raw)
    features = raw.ndata["feature"].float()
    labels = raw.ndata["label"].long().reshape(-1)
    masks = {name: raw.ndata[name].bool().clone() for name in ("train_mask", "val_mask", "test_mask")}
    meta = {
        "dataset_sha256": sha256_file(raw_path), "feature_sha256": sha256_tensor(features), "label_sha256": sha256_tensor(labels),
        "mask_sha256": {name: sha256_tensor(mask) for name, mask in masks.items()},
        "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "raw_nodes": raw.num_nodes(), "raw_edges": raw.num_edges(), "training_nodes": graph.num_nodes(), "training_edges": graph.num_edges(),
        "model_sha256": sha256_file(Path(__file__).with_name("model.py")), "protocol_sha256": sha256_file(Path(__file__).with_name("protocol.py")), "runner_sha256": sha256_file(__file__),
    }
    dump(output / "config_snapshot.json", {**config, "seed": seed})
    dump(output / "preflight.json", {"passed": True, "meta": meta})
    device = torch.device("cuda")
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: mask.to(device) for name, mask in masks.items()}
    model = DSGADModel(graph.num_nodes(), features.shape[1], config["hidden_dim"], degree=config["degree"], mix_beta=config["mix_beta"]).to(device)
    mixing, other = [], []
    for name, parameter in model.named_parameters(): (mixing if name == "weights" else other).append(parameter)
    optimizer = torch.optim.Adam([{"params": mixing, "lr": config["learning_rate"]}, {"params": other, "lr": config["learning_rate"]}], weight_decay=config["weight_decay"])
    weight = class_weight(labels, masks["train_mask"])
    best, best_epoch, best_state, stale, history = float("-inf"), 0, None, 0, []
    torch.cuda.reset_peak_memory_stats(device); started = time.monotonic()
    for epoch in range(1, config["max_epoch"] + 1):
        model.train(); logits = model(graph, features); loss = masked_weighted_loss(logits, labels, masks["train_mask"])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): probability = model(graph, features).softmax(1)[:, 1]
        threshold, validation_f1, validation_auprc = validation_values(labels, probability, masks["val_mask"])
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "validation_f1_macro": validation_f1, "validation_auprc": validation_auprc, "validation_threshold": threshold})
        if validation_auprc > best:
            best, best_epoch, stale = validation_auprc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else: stale += 1
        if run_type != "smoke" and stale >= config["patience"]: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): probability = model(graph, features).softmax(1)[:, 1]
    threshold, validation_f1, validation_auprc = validation_values(labels, probability, masks["val_mask"])
    metrics = test_values(labels, probability, masks["test_mask"], threshold)
    metrics.update({"method": "DSGAD-official-topology-h64", "dataset": dataset, "seed": seed, "run_type": run_type, "status": "smoke" if run_type == "smoke" else "OK", "positioning": config["positioning"], "protocol_version": config["protocol_version"], "best_epoch": best_epoch, "actual_epochs": epoch, "validation_auprc": validation_auprc, "validation_f1_macro": validation_f1, "threshold": threshold, "class_weight": [float(x) for x in weight], "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2, "hashes": meta, "edge_access": "graph_edges_required"})
    torch.save({"model_state_dict": best_state, "config": config, "best_epoch": best_epoch}, output / "checkpoint_auprc_best.pt")
    dump(output / "validation_history.json", history); dump(output / "metrics.json", metrics)
    dump(output / "artifact_sha256s.json", {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()})
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, choices=tuple(DATASETS)); parser.add_argument("--seeds", default="0"); parser.add_argument("--run-type", default="smoke", choices=("smoke", "diagnostic", "formal")); args = parser.parse_args()
    rows = []
    for raw_seed in args.seeds.split(","):
        seed = int(raw_seed)
        try: rows.append(run_one(args.dataset, seed, args.run_type))
        except Exception as error: rows.append({"dataset": args.dataset, "seed": seed, "run_type": args.run_type, "status": "ERROR", "error": repr(error)})
        base = result_directory(args.dataset, args.run_type, seed).parent; base.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with (base / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__": main()
