from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from .frozen_cache import load_frozen_npz
from .integrity import data_integrity_record, file_sha256
from .model import AMNetHSMAD, build_optimizer
from .protocol import final_test_metrics, select_validation_threshold, validation_scores
from .reproducibility import setup_seed
from .run_smoke import THRESHOLDS, probabilities, refuse_existing_output, require_cuda, train_epoch, write_json

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AUPRCEarlyStopper:
    patience: int
    best_value: float = float("-inf")
    best_epoch: int = 0
    stale: int = 0

    def update(self, value, epoch):
        improved = float(value) >= self.best_value
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.stale = 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def validate_full_config(config):
    if config.get("protocol_status") != "candidate_protocol_not_author_exact":
        raise ValueError("protocol status mismatch")
    if config.get("max_epoch") != 2000 or config.get("patience") != 200:
        raise ValueError("official Yelp epoch/patience must be 2000/200")
    if config.get("checkpoint_metric") != "validation_AUPRC":
        raise ValueError("checkpoint metric must be validation AUPRC")
    run_type, seed = config.get("run_type"), config.get("seed")
    if run_type == "diagnostic" and seed != 0:
        raise ValueError("diagnostic must use seed 0")
    if run_type == "formal" and seed not in range(10):
        raise ValueError("formal seed must be 0..9")
    if run_type not in ("diagnostic", "formal"):
        raise ValueError("run_type must be diagnostic or formal")


def protected_manifest():
    targets = [
        "main.py", "dataset.py", "model.py", "utils.py", "results/runs.csv", "results/summary.csv",
        "methods/mlp", "methods/gcn", "methods/gat", "methods/gat_v2_gadbench", "methods/graphsage",
        "results/experiments/mlp", "results/experiments/gcn", "results/experiments/gat",
        "results/experiments/gat_v2_gadbench", "results/experiments/graphsage",
    ]
    files = {}
    for relative in targets:
        target = ROOT / relative
        if target.is_file():
            files[relative] = file_sha256(target)
        elif target.exists():
            for path in sorted(item for item in target.rglob("*") if item.is_file()):
                files[str(path.relative_to(ROOT))] = file_sha256(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"manifest_sha256": digest, "file_count": len(files), "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    validate_full_config(config)
    require_cuda()
    output = ROOT / config["result_dir"]
    refuse_existing_output(output)

    protected_before = protected_manifest()
    cache = ROOT / config["cache_file"]
    if file_sha256(cache) != config["cache_sha256"]:
        raise RuntimeError("frozen bridge cache SHA256 mismatch")
    data, cache_metadata = load_frozen_npz(cache)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_snapshot.json")
    write_json(output / "protected_manifest_before.json", protected_before)
    started = time.monotonic()
    setup_seed(config["seed"])
    device = torch.device("cuda:0")
    data = data.to(device)
    model = AMNetHSMAD(data.x.shape[1], config["hidden_channels"], 2, config["filter_num_K"], config["bernstein_order_M"], config["dropout"]).to(device)
    optimizer = build_optimizer(model, config["filter_lr"], config["non_filter_weight_decay"])
    torch.cuda.reset_peak_memory_stats(device)
    code_files = [
        ROOT / "methods/amnet_hsmad/model.py", ROOT / "methods/amnet_hsmad/protocol.py",
        ROOT / "methods/amnet_hsmad/run_smoke.py", ROOT / "methods/amnet_hsmad/run_full.py",
        ROOT / "methods/amnet_hsmad/frozen_cache.py", ROOT / "methods/amnet_hsmad/dataset_specs.py",
    ]
    preflight = {
        "passed": True, "protocol_status": config["protocol_status"], "run_type": config["run_type"],
        "seed": config["seed"], "edge_access": "graph_edges_required", "cache_metadata": cache_metadata,
        "data_integrity": data_integrity_record(data), "config_sha256": file_sha256(config_path),
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files},
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_groups": [{"lr": group["lr"], "weight_decay": group["weight_decay"]} for group in optimizer.param_groups],
        "gpu": torch.cuda.get_device_name(device), "gpu_total_mb": torch.cuda.get_device_properties(device).total_memory / 1024**2,
    }
    write_json(output / "preflight.json", preflight)

    stopper = AUPRCEarlyStopper(config["patience"])
    checkpoint = output / "checkpoint_validation_auprc_best.pt"
    history = []
    stop_reason = "max_epoch"
    for epoch in range(1, config["max_epoch"] + 1):
        loss, classification, marginal = train_epoch(model, data, optimizer, config["beta"])
        score = probabilities(model, data)
        validation = validation_scores(score, data.y, data.val_mask)
        threshold = select_validation_threshold(score, data.y, data.val_mask, THRESHOLDS)
        y_val = data.y[data.val_mask].detach().cpu().numpy()
        pred_val = (score[data.val_mask] >= threshold).detach().cpu().numpy()
        validation_f1 = float(f1_score(y_val, pred_val, average="macro", zero_division=0))
        improved, should_stop = stopper.update(validation["auprc"], epoch)
        record = {
            "epoch": epoch, "train_loss": loss, "classification_loss": classification, "marginal_loss": marginal,
            "validation_f1_macro": validation_f1, "validation_auroc": validation["auroc"],
            "validation_auprc": validation["auprc"], "validation_threshold": threshold,
            "checkpoint_improved": improved, "early_stop_stale": stopper.stale,
            "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if improved:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, checkpoint)
        if should_stop:
            stop_reason = "validation_AUPRC_patience"
            break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    score = probabilities(model, data)
    threshold = select_validation_threshold(score, data.y, data.val_mask, THRESHOLDS)
    test = final_test_metrics(score, data.y, data.test_mask, threshold)
    protected_after = protected_manifest()
    metrics = {
        "method": "AMNet-HSMAD-adapted", "protocol_status": config["protocol_status"], "dataset": config["dataset"],
        "seed": config["seed"], "run_type": config["run_type"],
        "status": "OK" if config["run_type"] == "diagnostic" else "formal/OK",
        "actual_epochs": len(history), "stop_reason": stop_reason, "best_epoch": int(state["epoch"]),
        "checkpoint_metric": "validation_AUPRC", "validation_auprc": stopper.best_value,
        "threshold_protocol": "validation_only_grid_0.05_to_0.95", **test,
        "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "checkpoint_sha256": file_sha256(checkpoint), "cache_sha256": config["cache_sha256"],
        "protected_manifest_unchanged": protected_before == protected_after,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "validation_history.json", history)
    write_json(output / "metrics.json", metrics)
    write_json(output / "protected_manifest_after.json", protected_after)
    with (output / "runs.csv").open("w", newline="") as stream:
        fields = ["method", "protocol_status", "dataset", "seed", "run_type", "status", "f1_macro", "auroc", "best_epoch", "threshold", "actual_epochs", "wall_time_sec", "peak_gpu_mb"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerow({key: metrics[key] for key in fields})
    write_json(output / "artifact_sha256.json", {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()})
    if protected_before != protected_after:
        raise RuntimeError("protected manifest changed during AMNet run")
    print("AMNET_FULL_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
