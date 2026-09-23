"""Bounded CGADM candidate runner helpers and CLI entry point."""

from __future__ import annotations

import os
import random
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import dgl
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx

from .adapter import load_frozen_data
from .model import build_full_model, build_optimizer, build_scheduler
from .protocol import test_values, validation_values


def setup_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_new_output(path) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")


def build_candidate_data(graph):
    """Build the official Model's data interface without changing frozen masks."""
    network = to_networkx(graph, to_undirected=True)
    centrality = nx.degree_centrality(network)
    node_centrality = torch.tensor(
        [centrality[index] for index in range(graph.num_nodes)], dtype=torch.float32
    )
    return SimpleNamespace(
        graph=graph,
        num_features=graph.x.shape[1],
        val_labels=graph.y[graph.val_mask],
        test_labels=graph.y[graph.test_mask],
        node_centrality=node_centrality,
    )


def checkpoint_payload(model, prior, epoch: int, val_auprc: float, threshold: float, evaluation_seed: int):
    return {
        "model_state_dict": model.state_dict(),
        "prior": prior.detach().cpu().clone(),
        "epoch": int(epoch),
        "validation_auprc": float(val_auprc),
        "threshold": float(threshold),
        "selection_metric": "validation_AUPRC",
        "threshold_source": "validation_F1_macro",
        "evaluation_seed": int(evaluation_seed),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _options(config):
    return SimpleNamespace(
        num_layers=int(config["gnn_layers"]),
        num_steps=int(config["diffusion_steps"]),
        beta_1=float(config["beta_1"]),
        beta_T=float(config["beta_T"]),
        hiddim=int(config["hidden_dim"]),
        gnndp=float(config["dropout"]),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        step_size=int(config["scheduler_step_size"]),
        gamma=float(config["scheduler_gamma"]),
    )


def run_one(root: Path, config_path: Path, seed: int, run_type: str, max_epoch: int, patience: int, output_suffix: str = ""):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    seed_directory = f"seed_{seed}" + (f"_{output_suffix}" if output_suffix else "")
    output = root / "results" / "experiments" / "cgadm_hsmad" / dataset / config["protocol"] / run_type / seed_directory
    ensure_new_output(output)
    output.mkdir(parents=True)
    started = time.time()
    runtime_config = dict(config)
    runtime_config.update({"seed": seed, "run_type": run_type, "max_epoch": max_epoch, "patience": patience})
    (output / "config.json").write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    metrics = {"method": "CGADM", "dataset": dataset, "seed": seed, "run_type": run_type, "status": "ERROR"}
    history = []
    try:
        setup_seed(seed)
        source_graph, graph = load_frozen_data(root / "datasets" / dataset)
        data = build_candidate_data(graph)
        preflight = {
            "nodes": source_graph.num_nodes(),
            "edges": source_graph.num_edges(),
            "feature_shape": list(graph.x.shape),
            "mask_counts": {name: int(getattr(graph, f"{name}_mask").sum()) for name in ("train", "val", "test")},
            "dataset_sha256": _file_sha256(root / "datasets" / dataset),
            "config_sha256": _file_sha256(output / "config.json"),
            "protocol_label": config["status_label"],
            "test_used_during_training": False,
        }
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CGADM candidate requires CUDA for the official XGBoost prior")
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        model = build_full_model(root / "methods" / "cgadm" / "official_snapshot", _options(config), data, device).to(device)
        optimizer = build_optimizer(model, config["learning_rate"], config["weight_decay"])
        scheduler = build_scheduler(optimizer, config["scheduler_step_size"], config["scheduler_gamma"])
        best = None
        stale = 0
        evaluation_seed = 10000 + seed
        checkpoint_path = output / "best_checkpoint.pt"
        for epoch in range(max_epoch):
            model.train()
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.eval()
            setup_seed(evaluation_seed)
            with torch.no_grad():
                scores = model.predict().detach().cpu()
            val = validation_values(graph.y, scores, graph.val_mask)
            row = {"epoch": epoch, "train_loss": float(loss.item()), **{f"validation_{key}": value for key, value in val.items()}}
            history.append(row)
            if best is None or val["auprc"] > best:
                best = val["auprc"]
                stale = 0
                torch.save(checkpoint_payload(model, model.prior, epoch, val["auprc"], val["threshold"], evaluation_seed), checkpoint_path)
            else:
                stale += 1
                if stale >= patience:
                    break
        (output / "validation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.prior = checkpoint["prior"].to(device)
        model.eval()
        setup_seed(checkpoint["evaluation_seed"])
        with torch.no_grad():
            scores = model.predict().detach().cpu()
        final = test_values(graph.y, scores, graph.test_mask, checkpoint["threshold"])
        metrics.update(
            status="smoke" if run_type == "smoke" else "OK",
            best_epoch=int(checkpoint["epoch"]),
            validation_auprc=float(checkpoint["validation_auprc"]),
            wall_time_sec=time.time() - started,
            peak_gpu_mb=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            checkpoint_sha256=_file_sha256(checkpoint_path),
            **final,
        )
    except Exception as exc:
        metrics.update(error_type=type(exc).__name__, error=str(exc), wall_time_sec=time.time() - started)
        (output / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(metrics))
            writer.writeheader(); writer.writerow(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-type", choices=("smoke", "diagnostic", "formal"), required=True)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    result = run_one(args.root, args.config, args.seed, args.run_type, args.max_epoch, args.patience, args.output_suffix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
