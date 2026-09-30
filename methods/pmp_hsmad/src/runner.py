from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader as DGLDataLoader
from dgl.dataloading import MultiLayerFullNeighborSampler

from methods.dsgad.src.protocol import prepare_training_graph, setup_seed
from methods.dsgad.src.runner import load_data
from methods.pmp_hsmad.src.model import PMPCandidate
from methods.pmp_hsmad.src.protocol import build_label_unknown, frozen_config, test_metrics, validation_metrics
from methods.project_paths import project_root

ROOT = project_root()


def candidate_config(dataset: str, run_type: str = "formal") -> dict:
    if dataset not in {"weibo", "tolokers", "amazon", "tfinance"}:
        raise ValueError(dataset)
    if run_type not in {"smoke", "diagnostic", "formal"}:
        raise ValueError(run_type)
    config = frozen_config(dataset)
    config.update({
        "protocol_version": "pmp_hsmad_candidate",
        "run_type": run_type,
        "max_epoch": 5 if run_type == "smoke" else 100,
        "patience": 20,
        "batch_size": 128,
        "dropout": 0.6,
        "parameter_source_for_unsupported_dataset": "official_amazon_config",
    })
    return config


def _row_normalize(features: torch.Tensor) -> torch.Tensor:
    denominator = features.sum(dim=1, keepdim=True) + 0.01
    safe = torch.where(torch.isfinite(denominator.reciprocal()), denominator, torch.ones_like(denominator))
    return features / safe


def prepare_pmp_graph(raw: dgl.DGLGraph) -> dgl.DGLGraph:
    graph = prepare_training_graph(raw)
    graph.ndata["feature"] = _row_normalize(raw.ndata["feature"].float())
    graph.ndata["label"] = raw.ndata["label"].reshape(-1).long().clone()
    for name in ("train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = raw.ndata[name].bool().clone()
    graph.ndata["label_unk"] = build_label_unknown(graph.ndata["label"], graph.ndata["train_mask"])
    return graph


def make_loader(graph, node_ids, batch_size: int, shuffle: bool):
    sampler = MultiLayerFullNeighborSampler(num_layers=1)
    return DGLDataLoader(
        graph, node_ids, sampler, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=0,
    )


def build_model(input_dim: int, batch_size: int = 128, dropout: float = 0.6) -> PMPCandidate:
    return PMPCandidate(input_dim=input_dim, num_relations=1, batch_size=batch_size, dropout=dropout)


def result_directory(dataset: str, run_type: str, seed: int) -> Path:
    suffix = os.environ.get("PMP_OUTPUT_SUFFIX", "")
    return ROOT / "results/experiments/pmp_hsmad" / dataset / "pmp_hsmad_candidate" / run_type / f"seed_{seed}{suffix}"


def checkpoint_is_better(current_validation_auroc: float, best_validation_auroc: float) -> bool:
    return float(current_validation_auroc) > float(best_validation_auroc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _forward_batches(model, loader, relations, device):
    probabilities, labels = [], []
    for _, _, blocks in loader:
        blocks = [block.to(device) for block in blocks]
        logits = model(blocks, relations, blocks[0].srcdata["feature"])
        probabilities.append(logits.softmax(1)[:, 1].detach().cpu())
        labels.append(blocks[-1].dstdata["label"].detach().cpu())
    return torch.cat(probabilities), torch.cat(labels)


def run_one(dataset: str, seed: int, run_type: str):
    config = candidate_config(dataset, run_type)
    output = result_directory(dataset, run_type, seed)
    output.mkdir(parents=True, exist_ok=False)
    setup_seed(seed)
    raw, raw_path = load_data(dataset)
    graph = prepare_pmp_graph(raw)
    masks = {name: graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    meta = {
        "dataset_sha256": _sha256_file(raw_path),
        "feature_sha256": _sha256_tensor(raw.ndata["feature"]),
        "label_sha256": _sha256_tensor(raw.ndata["label"].reshape(-1)),
        "mask_sha256": {name: _sha256_tensor(mask) for name, mask in masks.items()},
        "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "label_unk_counts": {str(value): int((graph.ndata["label_unk"] == value).sum()) for value in (0, 1, 2)},
        "raw_nodes": raw.num_nodes(), "raw_edges": raw.num_edges(),
        "training_nodes": graph.num_nodes(), "training_edges": graph.num_edges(),
        "model_sha256": _sha256_file(Path(__file__).with_name("model.py")),
        "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py")),
        "runner_sha256": _sha256_file(Path(__file__)),
    }
    _dump(output / "config_snapshot.json", {**config, "seed": seed})
    _dump(output / "preflight.json", {"passed": True, "meta": meta})
    device = torch.device("cuda")
    relations = list(graph.etypes)
    train_ids = torch.nonzero(masks["train_mask"], as_tuple=False).reshape(-1)
    val_ids = torch.nonzero(masks["val_mask"], as_tuple=False).reshape(-1)
    test_ids = torch.nonzero(masks["test_mask"], as_tuple=False).reshape(-1)
    train_loader = make_loader(graph, train_ids, config["batch_size"], True)
    val_loader = make_loader(graph, val_ids, 4096, False)
    test_loader = make_loader(graph, test_ids, 4096, False)
    model = build_model(raw.ndata["feature"].shape[1], config["batch_size"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best, best_epoch, best_state, stale, history = float("-inf"), 0, None, 0, []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for epoch in range(1, config["max_epoch"] + 1):
        model.train(); epoch_loss = 0.0; seen = 0
        for _, _, blocks in train_loader:
            blocks = [block.to(device) for block in blocks]
            logits = model(blocks, relations, blocks[0].srcdata["feature"])
            labels = blocks[-1].dstdata["label"]
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            count = int(labels.numel()); epoch_loss += float(loss.detach().cpu()) * count; seen += count
        model.eval()
        with torch.no_grad(): val_probability, val_labels = _forward_batches(model, val_loader, relations, device)
        val = validation_metrics(val_probability, val_labels, torch.ones_like(val_labels, dtype=torch.bool))
        history.append({"epoch": epoch, "train_loss": epoch_loss / seen, "validation_auroc": val["auroc"], "validation_auprc": val["auprc"], "validation_f1_macro": val["f1_macro"], "validation_threshold": val["threshold"]})
        if checkpoint_is_better(val["auroc"], best):
            best, best_epoch, stale = val["auroc"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if run_type != "smoke" and stale >= config["patience"]:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        val_probability, val_labels = _forward_batches(model, val_loader, relations, device)
        test_probability, test_labels = _forward_batches(model, test_loader, relations, device)
    val = validation_metrics(val_probability, val_labels, torch.ones_like(val_labels, dtype=torch.bool))
    metrics = test_metrics(test_probability, test_labels, torch.ones_like(test_labels, dtype=torch.bool), val["threshold"])
    metrics.update({"method": "PMP-official-LASAGE-S-h64", "dataset": dataset, "seed": seed,
                    "run_type": run_type, "status": "smoke" if run_type == "smoke" else "OK",
                    "positioning": config["positioning"], "protocol_version": config["protocol_version"],
                    "best_epoch": best_epoch, "actual_epochs": epoch,
                    "validation_auroc": val["auroc"], "validation_auprc": val["auprc"],
                    "validation_f1_macro": val["f1_macro"], "threshold": val["threshold"],
                    "wall_time_sec": time.monotonic() - started,
                    "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                    "hashes": meta, "edge_access": "graph_edges_required", "class_weight": None})
    torch.save({"model_state_dict": best_state, "config": config, "best_epoch": best_epoch,
                "threshold": val["threshold"]}, output / "checkpoint_validation_auroc_best.pt")
    _dump(output / "validation_history.json", history); _dump(output / "metrics.json", metrics)
    _dump(output / "artifact_sha256s.json", {path.name: _sha256_file(path) for path in output.iterdir() if path.is_file()})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("weibo", "tolokers", "amazon", "tfinance"))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--run-type", default="smoke", choices=("smoke", "diagnostic", "formal"))
    args = parser.parse_args(); rows = []
    for raw_seed in args.seeds.split(","):
        seed = int(raw_seed)
        try:
            rows.append(run_one(args.dataset, seed, args.run_type))
        except Exception as error:
            rows.append({"dataset": args.dataset, "seed": seed, "run_type": args.run_type, "status": "ERROR", "error": repr(error)})
        base = result_directory(args.dataset, args.run_type, seed).parent
        fields = sorted({key for row in rows for key in row})
        with (base / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
