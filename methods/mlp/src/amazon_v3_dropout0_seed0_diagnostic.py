"""Isolated Amazon seed-0 diagnostic using the frozen MLP Weibo-v3 protocol."""
import argparse
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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/mlp/src"))
from methods.mlp.src.model import FeatureMLP
from methods.mlp.src.utils import file_sha256, setup_seed, tensor_sha256

OUT = ROOT / "results/experiments/mlp/amazon/protocol_v3_dropout0_seed0_diagnostic"
MASKS = {
    "train_mask": "fb95bd68eda65b33435b2214bd1ceff41dfa324b5fbed8bcfeb72a1950ca0c4a",
    "val_mask": "2175e7133a0b272f26416cf46b11e08b771d899af50d222724036ed090d9acfb",
    "test_mask": "bc42f2677dbf8357e949a30330b0236fbe6435c91583aa0232a249a20f95e9d1",
}
REF_COMMIT = "f9aa021ce9b6c6580427fb633b596843be76ddc6"
REF_SHA = "6f81e05c4f924e8b8a047e7d052bee7b358b9473dee4b2ddfac3153eba53704d"


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_hashes():
    return {path: file_sha256(str(ROOT / path)) for path in [
        "methods/mlp/src/model.py", "methods/mlp/src/utils.py", "methods/mlp/src/train.py"
    ]}


def environment(device):
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dgl": dgl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
    }


def validation_threshold(labels, probabilities):
    best_f1, best_threshold = -1.0, 0.05
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(labels, (probabilities > threshold).astype(np.int64), average="macro")
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_f1, best_threshold


def write_row(path, row):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("authorization permits only Amazon seed=0")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    run = OUT / "seed_0"
    run.mkdir(parents=True, exist_ok=False)
    env_dir = run / "environment"
    env_dir.mkdir()
    record = {
        "method": "MLP", "dataset": "amazon", "seed": 0, "run_type": "diagnostic",
        "protocol_version": "v3_dropout0_amazon_seed0_diagnostic", "status": "ERROR",
        "edge_access": "none", "config_sha256": config_sha256(config),
    }
    try:
        # This is intentionally before data, model and optimizer creation.
        setup_seed(0)
        graphs, _ = dgl.load_graphs(str(ROOT / "datasets/amazon"))
        ndata = graphs[0].ndata  # only ndata is read; no edge accessor or graph preprocessing follows.
        feature = ndata["feature"].float().contiguous()
        label = ndata["label"].long().reshape(-1).contiguous()
        masks = {name: ndata[name].bool().reshape(-1).contiguous() for name in MASKS}
        input_sha = {
            "dataset_file_sha256": file_sha256(str(ROOT / "datasets/amazon")),
            "feature_sha256": tensor_sha256(feature), "label_sha256": tensor_sha256(label),
            **{name + "_sha256": tensor_sha256(mask) for name, mask in masks.items()},
            "nodes": int(label.numel()), "feature_shape": list(feature.shape),
        }
        for name, expected in MASKS.items():
            if input_sha[name + "_sha256"] != expected:
                raise RuntimeError("frozen HSMAD mask SHA mismatch: " + name)
        prefix = int(config["amazon_excluded_prefix_nodes"])
        covered = masks["train_mask"] | masks["val_mask"] | masks["test_mask"]
        prefix_counts = {name: int(mask[:prefix].sum()) for name, mask in masks.items()}
        if any(prefix_counts.values()) or int(covered[:prefix].sum()) != 0:
            raise RuntimeError("official Amazon excluded-prefix rule violated")
        if feature.shape != (label.numel(), 25) or config["input_dim"] != 25:
            raise RuntimeError("unexpected Amazon feature shape/input_dim")
        v3 = json.loads((ROOT / "methods/mlp/configs/weibo_protocol_v3_dropout0_candidate.json").read_text())
        shared = ["model", "architecture", "hidden_dim", "activation", "dropout", "loss", "class_weight", "sampling", "optimizer", "learning_rate", "weight_decay", "max_epoch", "patience", "selection_metric", "threshold_candidates", "edge_access"]
        difference = {key: {"weibo_v3": v3[key], "amazon_v3": config[key]} for key in shared if v3[key] != config[key]}
        if difference:
            raise RuntimeError("non-data configuration differs from frozen Weibo v3: " + json.dumps(difference, sort_keys=True))
        reference = ROOT / "audit/mlp_reference/GADBench/models/gnn.py"
        if file_sha256(str(reference)) != REF_SHA:
            raise RuntimeError("GADBench reference file SHA mismatch")
        got_commit = subprocess.run(["git", "-C", str(reference.parents[1]), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
        if got_commit != REF_COMMIT:
            raise RuntimeError("GADBench reference commit mismatch")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        preflight = {
            "status": "PASS", "input_sha256": input_sha, "frozen_hsmad_masks_match": True,
            "amazon_excluded_prefix_nodes": prefix, "prefix_mask_counts": prefix_counts,
            "prefix_covered_count": int(covered[:prefix].sum()),
            "only_allowed_config_differences_from_weibo_v3": True,
            "semantic_config_diff": {"dataset": "weibo -> amazon", "input_dim": "400 -> 25"},
            "reference_commit": got_commit, "reference_file_sha256": REF_SHA,
            "code_sha256": code_hashes(), "environment": environment(device), "edge_access": "none",
            "edge_access_proof": "runner only reads graph.ndata; no graph edge accessor, preprocessing, aggregation, or message passing",
        }
        (OUT / "config_diff_preflight.json").write_text(json.dumps(preflight, indent=2))
        (run / "preflight.json").write_text(json.dumps(preflight, indent=2))
        shutil.copy2(config_path, run / "config_snapshot.json")
        (env_dir / "framework_versions.json").write_text(json.dumps(preflight["environment"], indent=2))
        (env_dir / "packages_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout)
        feature, label = feature.to(device), label.to(device)
        masks = {name: mask.to(device) for name, mask in masks.items()}
        model = FeatureMLP(25, 64, 0.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        best, state, stall, history = None, None, 0, []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for epoch in range(1, 1001):
            model.train(); optimizer.zero_grad()
            loss = F.cross_entropy(model(feature)[masks["train_mask"]], label[masks["train_mask"]])
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad():
                probability = torch.softmax(model(feature), dim=1)[:, 1].cpu().numpy()
            validation = masks["val_mask"].cpu().numpy()
            validation_f1, threshold = validation_threshold(label[masks["val_mask"]].cpu().numpy(), probability[validation])
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "val_f1_macro": validation_f1, "threshold": threshold})
            if best is None or validation_f1 > best["val_f1"]:
                test = masks["test_mask"].cpu().numpy(); test_labels = label[masks["test_mask"]].cpu().numpy(); test_probability = probability[test]
                predicted = (test_probability > threshold).astype(np.int64)
                best = {"val_f1": validation_f1, "threshold": threshold, "best_epoch": epoch, "f1_macro": float(f1_score(test_labels, predicted, average="macro")), "auroc": float(roc_auc_score(test_labels, np.nan_to_num(test_probability))), "test_predicted_anomaly_count": int(predicted.sum()), "test_predicted_anomaly_ratio": float(predicted.mean()), "test_confusion_matrix_labels_0_1": confusion_matrix(test_labels, predicted, labels=[0, 1]).tolist()}
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
            if stall >= 100:
                break
        (run / "validation_history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model_state_dict": state, "seed": 0, "best_epoch": best["best_epoch"], "threshold": best["threshold"], "config_sha256": record["config_sha256"]}, run / "best_checkpoint.pt")
        record.update(best)
        record.update(input_sha)
        record.update({"status": "OK", "wall_time_sec": time.perf_counter() - start, "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "class_weight": None, "dropout": 0.0, "reference_commit": got_commit, "reference_file_sha256": REF_SHA, "code_sha256": json.dumps(code_hashes(), sort_keys=True), "auroc_score": "softmax(logits)[:,1]", "paper_f1_delta": best["f1_macro"] - 0.9223, "paper_auroc_delta": best["auroc"] - 0.9801})
    except Exception:
        record["error"] = traceback.format_exc()
    (run / "metrics.json").write_text(json.dumps(record, indent=2))
    write_row(OUT / "runs.csv", record)
    print(json.dumps(record, sort_keys=True))
    if record["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
