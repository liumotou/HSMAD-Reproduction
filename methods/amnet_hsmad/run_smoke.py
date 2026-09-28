from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .frozen_cache import load_frozen_npz
from .integrity import data_integrity_record, file_sha256
from .model import AMNetHSMAD, build_optimizer
from .protocol import (
    final_test_metrics,
    marginal_constraint_loss,
    masked_training_loss,
    select_validation_threshold,
    validation_scores,
)
from .reproducibility import setup_seed

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = [round(value / 100, 2) for value in range(5, 100, 5)]


def validate_smoke_config(config):
    required = {
        "protocol_status": "candidate_protocol_not_author_exact",
        "run_type": "smoke",
        "seed": 0,
        "max_epoch": 5,
        "beta": 1.0,
    }
    mismatches = {key: (required[key], config.get(key)) for key in required if config.get(key) != required[key]}
    if mismatches:
        raise ValueError(f"invalid frozen smoke config: {mismatches}")


def refuse_existing_output(path):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite {path}")


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU mode is required for AMNet smoke training")


def train_epoch(model, data, optimizer, beta):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(data.x, data.edge_index)
    classification = masked_training_loss(logits, data.y, data.train_mask)
    marginal = marginal_constraint_loss(model.last_filter_scores, data.y, data.train_mask)
    loss = classification + float(beta) * marginal
    loss.backward()
    optimizer.step()
    return float(loss.detach()), float(classification.detach()), float(marginal.detach())


@torch.no_grad()
def probabilities(model, data):
    model.eval()
    return torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    validate_smoke_config(config)
    require_cuda()

    output = ROOT / config["result_dir"]
    refuse_existing_output(output)
    cache = ROOT / config["cache_file"]
    if file_sha256(cache) != config["cache_sha256"]:
        raise RuntimeError("frozen bridge cache SHA256 mismatch")
    data, cache_metadata = load_frozen_npz(cache)
    if cache_metadata["preprocess"] != ["to_bidirected", "remove_self_loop", "add_self_loop"]:
        raise RuntimeError("unexpected graph preprocessing provenance")

    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_snapshot.json")
    started = time.monotonic()
    setup_seed(config["seed"])
    device = torch.device("cuda:0")
    data = data.to(device)
    model = AMNetHSMAD(
        int(data.x.shape[1]), config["hidden_channels"], 2,
        config["filter_num_K"], config["bernstein_order_M"], config["dropout"],
    ).to(device)
    optimizer = build_optimizer(model, config["filter_lr"], config["non_filter_weight_decay"])
    torch.cuda.reset_peak_memory_stats(device)

    preflight = {
        "passed": True,
        "protocol_status": config["protocol_status"],
        "edge_access": "graph_edges_required",
        "cache_metadata": cache_metadata,
        "data_integrity": data_integrity_record(data),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_groups": [
            {"lr": group["lr"], "weight_decay": group["weight_decay"], "parameter_tensors": len(group["params"])}
            for group in optimizer.param_groups
        ],
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_mb": torch.cuda.get_device_properties(device).total_memory / 1024**2,
    }
    write_json(output / "preflight.json", preflight)

    history = []
    best = None
    checkpoint = output / "checkpoint_validation_auprc_best.pt"
    for epoch in range(1, config["max_epoch"] + 1):
        loss, classification, marginal = train_epoch(model, data, optimizer, config["beta"])
        score = probabilities(model, data)
        validation = validation_scores(score, data.y, data.val_mask)
        threshold = select_validation_threshold(score, data.y, data.val_mask, THRESHOLDS)
        validation_prediction = score[data.val_mask] >= threshold
        validation_f1 = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
            data.y[data.val_mask].detach().cpu().numpy(),
            validation_prediction.detach().cpu().numpy(), average="macro", zero_division=0,
        )
        record = {
            "epoch": epoch, "train_loss": loss, "classification_loss": classification,
            "marginal_loss": marginal, "validation_f1_macro": float(validation_f1),
            "validation_auroc": validation["auroc"], "validation_auprc": validation["auprc"],
            "validation_threshold": threshold,
            "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or validation["auprc"] >= best["validation_auprc"]:
            best = record
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, checkpoint)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    score = probabilities(model, data)
    threshold = select_validation_threshold(score, data.y, data.val_mask, THRESHOLDS)
    test = final_test_metrics(score, data.y, data.test_mask, threshold)
    metrics = {
        "method": "AMNet-HSMAD-adapted", "protocol_status": config["protocol_status"],
        "dataset": config["dataset"], "seed": 0, "run_type": "smoke", "status": "smoke",
        "actual_epochs": config["max_epoch"], "best_epoch": int(state["epoch"]),
        "checkpoint_metric": "validation_AUPRC", "threshold_protocol": "validation_only_grid_0.05_to_0.95",
        **test, "validation_auprc": best["validation_auprc"],
        "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "checkpoint_sha256": file_sha256(checkpoint),
        "cache_sha256": config["cache_sha256"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "validation_history.json", history)
    write_json(output / "metrics.json", metrics)
    write_json(output / "artifact_sha256.json", {
        path.name: file_sha256(path) for path in output.iterdir() if path.is_file()
    })
    with (output / "runs.csv").open("w", newline="") as stream:
        fields = ["method", "protocol_status", "dataset", "seed", "run_type", "status", "f1_macro", "auroc", "best_epoch", "threshold", "wall_time_sec", "peak_gpu_mb"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in fields})
    print("AMNET_SMOKE_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
