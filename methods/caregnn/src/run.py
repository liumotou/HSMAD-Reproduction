"""Auditable single-flattened-relation CARE-GNN candidate runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from methods.project_paths import project_root

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from methods.caregnn.src.model import SingleRelationCareGNN


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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_sha256"] = sha256_file(path)
    return config


def build_run_spec(config: dict[str, object]) -> RunSpec:
    if not config.get("_config_sha256"):
        raise ValueError("missing config provenance")
    if not config.get("candidate_protocol_not_author_exact"):
        raise ValueError("candidate provenance marker required")
    if config.get("relation_policy") != "single_flattened_relation_project_adaptation":
        raise ValueError("exactly one flattened relation is required")
    if config.get("checkpoint_protocol") != "validation_auprc_project_choice":
        raise ValueError("validation-only checkpoint protocol required")
    if config.get("threshold_protocol") != "validation_f1_macro_grid_0.05_to_0.95":
        raise ValueError("validation-only threshold protocol required")
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_graph(raw):
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    for name in ("feature", "label", "train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = raw.ndata[name]
    return graph


def assert_uncovered_prefix(masks: dict[str, torch.Tensor], count: int) -> None:
    for name, mask in masks.items():
        if bool(mask[:count].any()):
            raise RuntimeError(f"{name} includes a node in the frozen uncovered prefix 0:{count}")


def build_single_relation_adjacency(
    source: torch.Tensor, destination: torch.Tensor, num_nodes: int
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in zip(source.tolist(), destination.tolist()):
        adjacency[src].append(dst)
    return adjacency


def balanced_training_nodes(
    labels: torch.Tensor, train_mask: torch.Tensor, seed: int
) -> torch.Tensor:
    train_nodes = train_mask.bool().nonzero(as_tuple=False).flatten()
    positive = train_nodes[labels[train_nodes].long() == 1]
    negative = train_nodes[labels[train_nodes].long() == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise RuntimeError("both classes required in frozen train_mask")
    generator = torch.Generator().manual_seed(seed)
    chosen_negative = negative[
        torch.randperm(negative.numel(), generator=generator)[: positive.numel()]
    ]
    combined = torch.cat((positive, chosen_negative))
    return combined[torch.randperm(combined.numel(), generator=generator)]


def row_l2_normalize(features: torch.Tensor) -> torch.Tensor:
    denominator = features.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return features / denominator


def threshold_grid() -> list[float]:
    return [index / 100 for index in range(5, 100, 5)]


def choose_threshold(labels: torch.Tensor, probabilities: torch.Tensor) -> tuple[float, float]:
    truth = labels.detach().cpu().numpy()
    score = probabilities.detach().cpu().numpy()
    candidates = [(threshold, f1_score(truth, score >= threshold, average="macro"))
                  for threshold in threshold_grid()]
    return max(candidates, key=lambda item: (item[1], -item[0]))


def evaluate_nodes(model, features, adjacency, nodes, batch_size: int) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, nodes.numel(), batch_size):
            logits, _, _ = model(features, adjacency, nodes[start : start + batch_size])
            outputs.append(logits)
    return torch.cat(outputs)


def evaluate_split(model, features, adjacency, labels, nodes, batch_size: int) -> dict[str, object]:
    logits = evaluate_nodes(model, features, adjacency, nodes, batch_size)
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    truth = labels[nodes]
    threshold, f1 = choose_threshold(truth, probabilities)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "labels": truth,
        "threshold": threshold,
        "f1_macro": f1,
        "auroc": float(roc_auc_score(truth.cpu().numpy(), probabilities.cpu().numpy())),
        "auprc": float(average_precision_score(truth.cpu().numpy(), probabilities.cpu().numpy())),
    }


def train(config: dict[str, object]) -> dict[str, object]:
    spec = build_run_spec(config)
    output = ROOT / spec.result_dir
    ensure_new_output(output)
    setup_seed(spec.seed)
    raw_path = ROOT / str(config["dataset_file"])
    raw = dgl.load_graphs(str(raw_path))[0][0]
    graph = prepare_training_graph(raw)
    observed = {"nodes": int(graph.num_nodes()), "training_edges": int(graph.num_edges())}
    if observed != dict(config["expected"]):
        raise RuntimeError(f"frozen input mismatch: {observed} != {config['expected']}")
    masks = {name: graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    if any(torch.logical_and(masks[a], masks[b]).any() for a, b in (
        ("train_mask", "val_mask"), ("train_mask", "test_mask"), ("val_mask", "test_mask")
    )):
        raise RuntimeError("frozen masks overlap")
    uncovered_prefix = int(config.get("uncovered_prefix_nodes", 0))
    if uncovered_prefix:
        assert_uncovered_prefix(masks, uncovered_prefix)
    source, destination = graph.edges(order="eid")
    adjacency = build_single_relation_adjacency(source.cpu(), destination.cpu(), graph.num_nodes())
    if sum(map(len, adjacency)) != graph.num_edges():
        raise RuntimeError("single-relation adjacency edge count mismatch")

    output.mkdir(parents=True)
    write_json(output / "config_snapshot.json", config)
    hashes = {
        "dataset_file_sha256": sha256_file(raw_path),
        "feature_sha256": sha256_tensor(graph.ndata["feature"]),
        "label_sha256": sha256_tensor(graph.ndata["label"]),
        **{f"{name}_sha256": sha256_tensor(value) for name, value in masks.items()},
        "model_py_sha256": sha256_file(ROOT / "methods/caregnn/src/model.py"),
        "protocol_py_sha256": sha256_file(ROOT / "methods/caregnn/src/protocol.py"),
        "runner_py_sha256": sha256_file(Path(__file__)),
        "config_sha256": str(config["_config_sha256"]),
    }
    expected_hashes = dict(config.get("expected_hashes", {}))
    for key, value in expected_hashes.items():
        if hashes.get(key) != value:
            raise RuntimeError(f"frozen hash mismatch for {key}: {hashes.get(key)} != {value}")
    train_labels = graph.ndata["label"][masks["train_mask"]].long()
    preflight = {
        "passed": True,
        "candidate_protocol_not_author_exact": True,
        "source_commit": config["source_commit"],
        "relation_policy": config["relation_policy"],
        "graph": observed,
        "feature_shape": list(graph.ndata["feature"].shape),
        "mask_counts": {key: int(value.sum()) for key, value in masks.items()},
        "uncovered_prefix_nodes": uncovered_prefix,
        "uncovered_prefix_in_any_mask": bool(
            torch.stack([value[:uncovered_prefix].any() for value in masks.values()]).any()
        ) if uncovered_prefix else False,
        "train_class_counts": {"normal": int((train_labels == 0).sum()), "anomaly": int((train_labels == 1).sum())},
        "model": {"layers": 1, "hidden_dim": int(config["hidden_dim"]), "relations": 1,
                  "initial_threshold": 0.5, "lambda_1": float(config["lambda_1"]),
                  "rl_step_size": float(config["rl_step_size"])},
        "feature_preprocess": config["feature_preprocess"],
        "batch_size_source": config["batch_size_source"],
        "edge_access": "single_flattened_graph_relation_required",
        "hashes": hashes,
    }
    write_json(output / "preflight.json", preflight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("audited CUDA environment required")
    torch.cuda.reset_peak_memory_stats(device)
    features = row_l2_normalize(graph.ndata["feature"].float()).to(device)
    labels = graph.ndata["label"].long().to(device)
    val_nodes = masks["val_mask"].nonzero(as_tuple=False).flatten().to(device)
    test_nodes = masks["test_mask"].nonzero(as_tuple=False).flatten().to(device)
    model = SingleRelationCareGNN(
        features.shape[1], int(config["hidden_dim"]), 2,
        float(config["lambda_1"]), float(config["rl_step_size"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history = []
    best_epoch = 0
    best_auprc = -float("inf")
    batch_size = int(config["batch_size"])
    checkpoint_path = output / "checkpoint_validation_auprc_best.pt"
    started = time.monotonic()

    for epoch in range(1, spec.max_epoch + 1):
        model.train()
        epoch_nodes = balanced_training_nodes(graph.ndata["label"], masks["train_mask"], spec.seed + epoch).to(device)
        batch_num = max(1, math.ceil(epoch_nodes.numel() / batch_size))
        losses = []
        for start in range(0, epoch_nodes.numel(), batch_size):
            batch = epoch_nodes[start : start + batch_size]
            if not bool(masks["train_mask"][batch.cpu()].all()):
                raise RuntimeError("loss batch escaped frozen train_mask")
            optimizer.zero_grad(set_to_none=True)
            logits, label_logits, relation_scores = model(features, adjacency, batch)
            batch_labels = labels[batch]
            loss = model.loss(logits, label_logits, batch_labels)
            loss.backward()
            optimizer.step()
            model.update_threshold(relation_scores, batch_labels.detach().cpu(), batch_num)
            losses.append(float(loss.item()))

        validation = evaluate_split(model, features, adjacency, labels, val_nodes, batch_size)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_f1_macro": validation["f1_macro"],
            "validation_auroc": validation["auroc"],
            "validation_auprc": validation["auprc"],
            "validation_threshold": validation["threshold"],
            "relation_threshold": model.thresholds[0],
            "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if validation["auprc"] > best_auprc:
            best_auprc = float(validation["auprc"])
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "relation_thresholds": list(model.thresholds),
                "config": config,
            }, checkpoint_path)
        if epoch - best_epoch >= spec.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.thresholds = [float(value) for value in checkpoint["relation_thresholds"]]
    validation = evaluate_split(model, features, adjacency, labels, val_nodes, batch_size)
    threshold = float(validation["threshold"])
    test_logits = evaluate_nodes(model, features, adjacency, test_nodes, batch_size)
    test_probabilities = torch.softmax(test_logits, dim=1)[:, 1]
    test_truth = labels[test_nodes]
    predicted = (test_probabilities >= threshold).long()
    matrix = confusion_matrix(test_truth.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])
    result = {
        "f1_macro": float(f1_score(test_truth.cpu().numpy(), predicted.cpu().numpy(), average="macro")),
        "auroc": float(roc_auc_score(test_truth.cpu().numpy(), test_probabilities.cpu().numpy())),
        "predicted_anomaly_count": int(predicted.sum()),
        "actual_anomaly_count": int(test_truth.sum()),
        "confusion_matrix": matrix.tolist(),
        "probability_min": float(test_probabilities.min()),
        "probability_max": float(test_probabilities.max()),
        "probability_mean": float(test_probabilities.mean()),
    }
    metrics = {
        "method": "CARE-GNN",
        "protocol_version": config["protocol_version"],
        "candidate_protocol_not_author_exact": True,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "run_type": spec.run_type,
        "status": "smoke" if spec.run_type == "smoke" else "OK",
        "actual_epochs": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "threshold": threshold,
        "validation_f1_macro": float(validation["f1_macro"]),
        "validation_auprc": float(validation["auprc"]),
        "relation_threshold": model.thresholds[0],
        **result,
        "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "edge_access": "single_flattened_graph_relation_required",
        "hashes": hashes,
    }
    write_json(output / "validation_history.json", history)
    write_json(output / "metrics.json", metrics)
    write_json(output / "artifact_sha256s.json", {
        path.name: sha256_file(path) for path in output.iterdir()
        if path.is_file() and path.name != "artifact_sha256s.json"
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("RUN_COMPLETE " + json.dumps(train(load_config(Path(args.config))), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
