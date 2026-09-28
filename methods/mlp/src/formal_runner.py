"""Formal, feature-only Weibo MLP runner isolated from HSMAD results."""

import argparse
import contextlib
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import f1_score, roc_auc_score

from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.train import EXPECTED_WEIBO_MASKS, best_validation_threshold, load_feature_data
from methods.mlp.src.utils import file_sha256, setup_seed, tensor_sha256


ROOT = Path(__file__).resolve().parents[3]
FORMAL_ROOT = ROOT / "results" / "experiments" / "mlp" / "weibo" / "formal"
REFERENCE_COMMIT = "f9aa021ce9b6c6580427fb633b596843be76ddc6"
REFERENCE_FILE_SHA256 = "6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d"
SMOKE_BASELINE = ROOT / "results" / "experiments" / "mlp" / "weibo" / "smoke" / "seed_0" / "metrics.json"


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_path(path: Path):
    return file_sha256(str(path))


def code_sha256s():
    paths = [ROOT / "methods/mlp/src/model.py", ROOT / "methods/mlp/src/utils.py", ROOT / "methods/mlp/src/train.py", ROOT / "methods/mlp/src/formal_runner.py"]
    return {str(path.relative_to(ROOT)): sha256_path(path) for path in paths}


def environment(device):
    return {"python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "dgl": dgl.__version__, "cuda_available": torch.cuda.is_available(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None}


def fingerprints():
    dataset_path = ROOT / "datasets/weibo"
    features, labels, masks = load_feature_data(dataset_path)
    result = {"dataset_file_sha256": sha256_path(dataset_path), "feature_sha256": tensor_sha256(features),
              "label_sha256": tensor_sha256(labels), "nodes": int(labels.numel()), "feature_shape": list(features.shape)}
    result.update({name + "_sha256": tensor_sha256(value) for name, value in masks.items()})
    return features, labels, masks, result


def preflight(config, config_sha):
    features, labels, masks, fp = fingerprints()
    smoke = json.loads(SMOKE_BASELINE.read_text(encoding="utf-8"))
    for key in ("dataset_file_sha256", "feature_sha256", "label_sha256", "train_mask_sha256", "val_mask_sha256", "test_mask_sha256"):
        if fp[key] != smoke[key]:
            raise RuntimeError(f"smoke baseline mismatch for {key}: {fp[key]} != {smoke[key]}")
    for key, expected in EXPECTED_WEIBO_MASKS.items():
        if fp[key + "_sha256"] != expected:
            raise RuntimeError(f"frozen mask mismatch for {key}")
    ref_file = ROOT / "audit/mlp_reference/GADBench/models/gnn.py"
    if sha256_path(ref_file) != REFERENCE_FILE_SHA256:
        raise RuntimeError("GADBench reference file SHA256 mismatch")
    current_code = code_sha256s()
    if current_code != config["frozen_code_sha256"]:
        raise RuntimeError(f"frozen MLP code SHA256 mismatch: {current_code} != {config['frozen_code_sha256']}")
    payload = {"status": "PASS", "reference_commit": REFERENCE_COMMIT, "reference_file_sha256": REFERENCE_FILE_SHA256,
               "formal_config_sha256": config_sha, "code_sha256": current_code, "fingerprints": fp,
               "environment": environment(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
               "smoke_metrics_path": str(SMOKE_BASELINE.relative_to(ROOT)), "edge_access": "none"}
    return features, labels, masks, fp, payload


def append_record(record):
    path = FORMAL_ROOT / "runs.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def clone_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=range(10))
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_json(config)
    if args.preflight_only:
        setup_seed(args.seed)
        _, _, _, _, payload = preflight(config, config_sha)
        FORMAL_ROOT.mkdir(parents=True, exist_ok=True)
        (FORMAL_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        return
    seed_dir = FORMAL_ROOT / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    log_path = seed_dir / "terminal.log"
    record = {"method": "MLP", "dataset": "weibo", "seed": args.seed, "run_type": "formal", "status": "ERROR",
              "edge_access": "none", "config_sha256": config_sha, "error": ""}
    try:
        # Seed precedes data, model, and optimizer creation by protocol.
        setup_seed(args.seed)
        features, labels, masks, fp, preflight_record = preflight(config, config_sha)
        (seed_dir / "preflight.json").write_text(json.dumps(preflight_record, indent=2), encoding="utf-8")
        shutil.copy2(config_path, seed_dir / "config_snapshot.json")
        env_dir = seed_dir / "environment"; env_dir.mkdir()
        (env_dir / "framework_versions.json").write_text(json.dumps(preflight_record["environment"], indent=2), encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout
        (env_dir / "packages_freeze.txt").write_text(freeze, encoding="utf-8")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        features, labels = features.to(device), labels.to(device)
        masks = {name: value.to(device) for name, value in masks.items()}
        model = FeatureMLP(features.shape[1], config["hidden_dim"], config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        best, best_state, stalled = None, None, 0
        start = time.perf_counter()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, config["max_epoch"] + 1):
            model.train(); optimizer.zero_grad()
            loss = functional.cross_entropy(model(features)[masks["train_mask"]], labels[masks["train_mask"]])
            loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad(): probabilities = torch.softmax(model(features), dim=1)[:, 1].detach().cpu().numpy()
            val_mask = masks["val_mask"].detach().cpu().numpy()
            val_labels = labels[masks["val_mask"]].detach().cpu().numpy()
            val_f1, threshold = best_validation_threshold(val_labels, probabilities[val_mask])
            if best is None or val_f1 > best["val_f1"]:
                test_mask = masks["test_mask"].detach().cpu().numpy()
                test_labels, test_probs = labels[masks["test_mask"]].detach().cpu().numpy(), probabilities[test_mask]
                best = {"val_f1": val_f1, "threshold": threshold, "best_epoch": epoch,
                        "f1_macro": float(f1_score(test_labels, (test_probs > threshold).astype(np.int64), average="macro")),
                        "auroc": float(roc_auc_score(test_labels, np.nan_to_num(test_probs)))}
                best_state, stalled = clone_state_dict(model), 0
            else: stalled += 1
            if stalled >= config["patience"]: break
        torch.save({"model_state_dict": best_state, "seed": args.seed, "best_epoch": best["best_epoch"], "threshold": best["threshold"], "config_sha256": config_sha}, seed_dir / "best_checkpoint.pt")
        record.update(best); record.update(fp); record.update({"status": "OK", "wall_time_sec": time.perf_counter()-start,
            "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device)/1024**2) if torch.cuda.is_available() else 0.0,
            "parameter_count": parameter_count, "reference_commit": REFERENCE_COMMIT,
            "reference_file_sha256": REFERENCE_FILE_SHA256, "code_sha256": json.dumps(code_sha256s(), sort_keys=True)})
        (seed_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    except Exception:
        record["error"] = traceback.format_exc()
        record["wall_time_sec"] = record.get("wall_time_sec", 0.0)
        (seed_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        log_path.write_text(record["error"], encoding="utf-8")
    append_record(record)
    print(json.dumps(record, sort_keys=True))
    if record["status"] != "OK": raise SystemExit(1)


if __name__ == "__main__":
    main()
