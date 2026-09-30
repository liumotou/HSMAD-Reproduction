"""Auditable single-flattened-relation PC-GNN candidate runner."""
from __future__ import annotations

import argparse
import hashlib
import json
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

from methods.pcgnn.src.model import SingleRelationPCGNN
from methods.pcgnn.src.protocol import build_train_positive_nodes, labels_for_forward


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    run_type: str
    result_dir: Path
    max_epoch: int
    validation_interval: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    if config.get("checkpoint_protocol") != "validation_auroc_official_pcgnn":
        raise ValueError("official PC-GNN validation AUROC checkpoint protocol required")
    if config.get("threshold_protocol") != "validation_f1_macro_grid_0.05_to_0.95":
        raise ValueError("validation-only threshold protocol required")
    if config["run_type"] not in {"smoke", "diagnostic", "formal"}:
        raise ValueError("unsupported run_type")
    return RunSpec(
        str(config["dataset"]), int(config["seed"]), str(config["run_type"]),
        Path(str(config["result_dir"])), int(config["max_epoch"]),
        int(config["validation_interval"]),
    )


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def assert_uncovered_prefix_excluded(
    masks: dict[str, torch.Tensor], prefix_nodes: int
) -> None:
    """Require the dataset-specific uncovered prefix to remain outside every split."""
    if prefix_nodes < 0:
        raise ValueError("uncovered prefix must be non-negative")
    for name, mask in masks.items():
        if bool(mask[:prefix_nodes].any()):
            raise RuntimeError(f"uncovered prefix unexpectedly appears in {name}")


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


def build_single_relation_adjacency(
    source: torch.Tensor, destination: torch.Tensor, num_nodes: int
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in zip(source.tolist(), destination.tolist()):
        adjacency[src].append(dst)
    return adjacency


def pick_training_nodes(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    adjacency: list[list[int]],
    seed: int,
) -> torch.Tensor:
    train_nodes = train_mask.bool().nonzero(as_tuple=False).flatten().tolist()
    y_train = labels[train_nodes].long().cpu().numpy()
    positive_count = int(y_train.sum())
    if positive_count == 0 or positive_count == len(train_nodes):
        raise RuntimeError("both classes required in frozen train_mask")
    label_frequency = (positive_count - len(y_train)) * y_train + len(y_train)
    weights = [len(adjacency[node]) / float(frequency)
               for node, frequency in zip(train_nodes, label_frequency)]
    sampled = random.Random(seed).choices(train_nodes, weights=weights, k=2 * positive_count)
    return torch.tensor(sampled, dtype=torch.long)


def official_row_sum_normalize(features: torch.Tensor) -> torch.Tensor:
    denominator = features.sum(dim=1, keepdim=True)
    inverse = torch.zeros_like(denominator)
    nonzero = denominator != 0
    inverse[nonzero] = denominator[nonzero].reciprocal()
    return features * inverse


def threshold_grid() -> list[float]:
    return [index / 100 for index in range(5, 100, 5)]


def choose_threshold(labels: torch.Tensor, probabilities: torch.Tensor) -> tuple[float, float]:
    truth = labels.detach().cpu().numpy()
    score = probabilities.detach().cpu().numpy()
    candidates = [(threshold, f1_score(truth, score >= threshold, average="macro"))
                  for threshold in threshold_grid()]
    return max(candidates, key=lambda item: (item[1], -item[0]))


def evaluate_nodes(model, features, adjacency, labels, train_mask, nodes, batch_size: int):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, nodes.numel(), batch_size):
            batch = nodes[start : start + batch_size]
            placeholder = labels_for_forward(labels, batch, train_mask, train_flag=False)
            logits, _ = model(features, adjacency, batch, placeholder, train_flag=False)
            outputs.append(logits)
    return torch.cat(outputs)


def split_scores(model, features, adjacency, labels, train_mask, nodes, batch_size: int):
    logits = evaluate_nodes(model, features, adjacency, labels, train_mask, nodes, batch_size)
    probability = torch.sigmoid(logits)[:, 1]
    truth = labels[nodes]
    threshold, macro_f1 = choose_threshold(truth, probability)
    return {
        "probability": probability,
        "labels": truth,
        "threshold": threshold,
        "f1_macro": macro_f1,
        "auroc": float(roc_auc_score(truth.cpu().numpy(), probability.cpu().numpy())),
        "auprc": float(average_precision_score(truth.cpu().numpy(), probability.cpu().numpy())),
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
    uncovered_prefix_nodes = int(config.get("uncovered_prefix_nodes", 0))
    assert_uncovered_prefix_excluded(masks, uncovered_prefix_nodes)
    if any(torch.logical_and(masks[a], masks[b]).any() for a, b in (
        ("train_mask", "val_mask"), ("train_mask", "test_mask"), ("val_mask", "test_mask")
    )):
        raise RuntimeError("frozen masks overlap")
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
        "model_py_sha256": sha256_file(ROOT / "methods/pcgnn/src/model.py"),
        "protocol_py_sha256": sha256_file(ROOT / "methods/pcgnn/src/protocol.py"),
        "runner_py_sha256": sha256_file(Path(__file__)),
        "config_sha256": str(config["_config_sha256"]),
    }
    for key, value in dict(config.get("expected_hashes", {})).items():
        if hashes.get(key) != value:
            raise RuntimeError(f"frozen hash mismatch for {key}")
    train_positive = build_train_positive_nodes(graph.ndata["label"], masks["train_mask"])
    write_json(output / "preflight.json", {
        "passed": True,
        "candidate_protocol_not_author_exact": True,
        "source_commit": config["source_commit"],
        "source_hash_note": "audit hashes are CRLF forms; Git checkout LF hashes are separately recorded",
        "relation_policy": config["relation_policy"],
        "graph": observed,
        "feature_shape": list(graph.ndata["feature"].shape),
        "mask_counts": {key: int(value.sum()) for key, value in masks.items()},
        "uncovered_prefix_nodes": uncovered_prefix_nodes,
        "uncovered_prefix_in_any_mask": any(
            bool(mask[:uncovered_prefix_nodes].any()) for mask in masks.values()
        ),
        "train_positive_count": int(train_positive.numel()),
        "model": {"layers": 1, "hidden_dim": int(config["hidden_dim"]), "relations": 1,
                  "top_p": float(config["relation_threshold"]), "rho": float(config["rho"]),
                  "alpha": float(config["alpha"])},
        "feature_preprocess": config["feature_preprocess"],
        "checkpoint_protocol": config["checkpoint_protocol"],
        "edge_access": "single_flattened_graph_relation_required",
        "hashes": hashes,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("audited CUDA environment required")
    torch.cuda.reset_peak_memory_stats(device)
    features = official_row_sum_normalize(graph.ndata["feature"].float()).to(device)
    labels = graph.ndata["label"].long().to(device)
    train_mask_device = masks["train_mask"].to(device)
    val_nodes = masks["val_mask"].nonzero(as_tuple=False).flatten().to(device)
    test_nodes = masks["test_mask"].nonzero(as_tuple=False).flatten().to(device)
    model = SingleRelationPCGNN(
        features.shape[1], int(config["hidden_dim"]), 2, train_positive.to(device),
        float(config["rho"]), float(config["alpha"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["batch_size"])
    history = []
    best_epoch = -1
    best_auroc = -float("inf")
    checkpoint_path = output / "checkpoint_validation_auroc_best.pt"
    started = time.monotonic()

    for epoch in range(spec.max_epoch):
        model.train()
        epoch_nodes = pick_training_nodes(
            graph.ndata["label"], masks["train_mask"], adjacency, spec.seed + epoch
        ).to(device)
        losses = []
        for start in range(0, epoch_nodes.numel(), batch_size):
            batch = epoch_nodes[start : start + batch_size]
            batch_labels = labels_for_forward(labels, batch, train_mask_device, train_flag=True)
            optimizer.zero_grad(set_to_none=True)
            logits, label_logits = model(
                features, adjacency, batch, batch_labels, train_flag=True
            )
            loss = model.loss(logits, label_logits, batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2)}
        if epoch % spec.validation_interval == 0:
            validation = split_scores(
                model, features, adjacency, labels, train_mask_device, val_nodes, batch_size
            )
            record.update({
                "validation_f1_macro": validation["f1_macro"],
                "validation_auroc": validation["auroc"],
                "validation_auprc": validation["auprc"],
                "validation_threshold": validation["threshold"],
            })
            if validation["auroc"] > best_auroc:
                best_auroc = float(validation["auroc"])
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                }, checkpoint_path)
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = split_scores(
        model, features, adjacency, labels, train_mask_device, val_nodes, batch_size
    )
    threshold = float(validation["threshold"])
    test_logits = evaluate_nodes(
        model, features, adjacency, labels, train_mask_device, test_nodes, batch_size
    )
    test_probability = torch.sigmoid(test_logits)[:, 1]
    test_truth = labels[test_nodes]
    prediction = (test_probability >= threshold).long()
    matrix = confusion_matrix(test_truth.cpu().numpy(), prediction.cpu().numpy(), labels=[0, 1])
    metrics = {
        "method": "PC-GNN",
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
        "validation_auroc": float(validation["auroc"]),
        "validation_auprc": float(validation["auprc"]),
        "f1_macro": float(f1_score(test_truth.cpu().numpy(), prediction.cpu().numpy(), average="macro")),
        "auroc": float(roc_auc_score(test_truth.cpu().numpy(), test_probability.cpu().numpy())),
        "predicted_anomaly_count": int(prediction.sum()),
        "actual_anomaly_count": int(test_truth.sum()),
        "confusion_matrix": matrix.tolist(),
        "probability_min": float(test_probability.min()),
        "probability_max": float(test_probability.max()),
        "probability_mean": float(test_probability.mean()),
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
