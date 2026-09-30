"""Isolated feature-only MLP runner; it never reads graph edges."""

import argparse
import csv
import json
import os
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
from methods.mlp.src.utils import file_sha256, setup_seed, tensor_sha256


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_WEIBO_MASKS = {
    "train_mask": "017423ff367b7e4580e4a705e86c056394a6a4abf448af1e037340ef104abe86",
    "val_mask": "4c04ebc905a8a7790b506488f75a9ef405681d7cfb27b740349a987047a1244c",
    "test_mask": "e00c8e697ed4828a6ab12d64df73f9d1c639638c482f94bcbef87a5313d985f2",
}
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def best_validation_threshold(labels, probabilities):
    best_f1, best_threshold = -1.0, 0.05
    for threshold in THRESHOLDS:
        score = f1_score(labels, (probabilities > threshold).astype(np.int64), average="macro")
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_f1, best_threshold


def load_feature_data(dataset_path: Path):
    """Load only ndata fields; do not inspect or transform graph structure."""
    loaded, _ = dgl.load_graphs(str(dataset_path))
    ndata = loaded[0].ndata
    required = ("feature", "label", "train_mask", "val_mask", "test_mask")
    missing = [name for name in required if name not in ndata]
    if missing:
        raise ValueError(f"missing persisted fields: {missing}")
    features = ndata["feature"].float().contiguous()
    labels = ndata["label"].long().reshape(-1).contiguous()
    masks = {name: ndata[name].bool().reshape(-1).contiguous() for name in required[2:]}
    return features, labels, masks


def write_csv(path: Path, row: dict) -> None:
    columns = list(row)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_environment(path: Path, device: torch.device) -> None:
    output = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout
    (path / "packages_freeze.txt").write_text(output, encoding="utf-8")
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dgl": dgl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
    }
    (path / "framework_versions.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["weibo"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-type", choices=["smoke", "formal"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.run_type != "smoke" or args.seed != 0:
        raise ValueError("This approved invocation is restricted to Weibo smoke seed=0")

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = ROOT / "results" / "experiments" / "mlp" / args.dataset / args.run_type / f"seed_{args.seed}"
    env_dir = run_dir / "environment"
    run_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, run_dir / "config_snapshot.json")

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = ROOT / "datasets" / args.dataset
    features, labels, masks = load_feature_data(dataset_path)
    fingerprints = {
        "dataset_file_sha256": file_sha256(str(dataset_path)),
        "feature_sha256": tensor_sha256(features),
        "label_sha256": tensor_sha256(labels),
        **{name + "_sha256": tensor_sha256(mask) for name, mask in masks.items()},
        "nodes": int(labels.numel()), "feature_shape": list(features.shape),
    }
    for name, expected in EXPECTED_WEIBO_MASKS.items():
        actual = fingerprints[name + "_sha256"]
        if actual != expected:
            raise RuntimeError(f"frozen {name} SHA256 mismatch: {actual} != {expected}")
    (run_dir / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")
    write_environment(env_dir, device)

    features, labels = features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = FeatureMLP(features.shape[1], config["hidden_dim"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best, stalled = None, 0
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config["epoch"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = functional.cross_entropy(model(features)[masks["train_mask"]], labels[masks["train_mask"]])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(features), dim=1)[:, 1].detach().cpu().numpy()
        val_mask = masks["val_mask"].detach().cpu().numpy()
        val_labels = labels[masks["val_mask"]].detach().cpu().numpy()
        val_f1, threshold = best_validation_threshold(val_labels, probabilities[val_mask])
        if best is None or val_f1 > best["val_f1"]:
            test_mask = masks["test_mask"].detach().cpu().numpy()
            test_labels = labels[masks["test_mask"]].detach().cpu().numpy()
            test_probs = probabilities[test_mask]
            best = {"val_f1": val_f1, "threshold": threshold, "best_epoch": epoch,
                    "f1_macro": float(f1_score(test_labels, (test_probs > threshold).astype(np.int64), average="macro")),
                    "auroc": float(roc_auc_score(test_labels, np.nan_to_num(test_probs)))}
            stalled = 0
        else:
            stalled += 1
        if stalled >= config["patience"]:
            break
    result = {"method": "MLP", "dataset": args.dataset, "seed": args.seed, "run_type": args.run_type,
              "status": "smoke", **best, "wall_time_sec": time.perf_counter() - start,
              "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024 ** 2) if torch.cuda.is_available() else 0.0,
              "parameter_count": parameter_count, **fingerprints, "edge_access": "none"}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(run_dir / "runs.csv", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
