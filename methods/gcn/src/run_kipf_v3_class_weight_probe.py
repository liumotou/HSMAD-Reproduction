"""Isolated GCN-v3 probe: GCN-v2 with only frozen train-ratio CE weights."""
import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def train_ratio_class_weight(train_labels):
    """Return [normal=1, anomaly=train_normal/train_anomaly] from train labels."""
    normal_count = int((train_labels == 0).sum())
    anomaly_count = int((train_labels == 1).sum())
    if anomaly_count == 0:
        raise ValueError("frozen train_mask contains no anomaly labels")
    return [1.0, normal_count / anomaly_count]


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def best_threshold(f1_score, labels, probabilities, thresholds):
    best = (-1.0, thresholds[0])
    for threshold in thresholds:
        score = f1_score(labels, (probabilities > threshold).astype("int64"), average="macro")
        if score > best[0]:
            best = (float(score), float(threshold))
    return best


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["run_type"] != "diagnostic" or config["seed"] != 0:
        raise ValueError("v3 class-weight probe is frozen to diagnostic seed=0")
    output = ROOT / "results/experiments/gcn" / config["dataset"] / config["protocol_version"] / "diagnostic" / "seed_0"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "environment").mkdir()

    import dgl
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    sys.path.insert(0, str(ROOT / "methods/mlp/src"))
    sys.path.insert(0, str(ROOT / "methods/gcn/src"))
    from utils import file_sha256, setup_seed, tensor_sha256
    from kipf_two_layer import KipfTwoLayerGCN

    result = {
        "method": "GCN", "protocol_version": config["protocol_version"], "dataset": config["dataset"],
        "seed": 0, "run_type": "diagnostic", "status": "ERROR", "edge_access": config["edge_access"],
        "checkpoint_protocol": config["checkpoint_protocol"], "early_stop_protocol": config["early_stop_protocol"],
        "threshold_protocol": config["threshold_protocol"], "config_sha256": json_sha256(config),
        "code_sha256": file_sha256(str(Path(__file__))),
        "v2_runner_sha256": file_sha256(str(ROOT / "methods/gcn/src/run_kipf_v2.py")),
        "model_sha256": file_sha256(str(ROOT / "methods/gcn/src/kipf_two_layer.py")),
        "class_weight": "", "train_normal_count": "", "train_anomaly_count": "", "f1_macro": "", "auroc": "",
        "threshold": "", "best_epoch": "", "early_stop_best_epoch": "", "validation_auprc": "",
        "epochs_executed": "", "wall_time_sec": "", "peak_gpu_mb": "", "checkpoint_sha256": "",
        "dataset_file_sha256": "", "feature_sha256": "", "label_sha256": "", "train_mask_sha256": "",
        "val_mask_sha256": "", "test_mask_sha256": "", "raw_nodes": "", "raw_edges": "",
        "training_nodes": "", "training_edges": "", "error": "",
    }
    try:
        setup_seed(0)
        raw = dgl.load_graphs(str(ROOT / "datasets" / config["dataset"]))[0][0]
        features = raw.ndata["feature"].float().contiguous()
        labels = raw.ndata["label"].long().reshape(-1).contiguous()
        masks = {key: raw.ndata[key].bool().reshape(-1).contiguous() for key in ("train_mask", "val_mask", "test_mask")}
        if features.shape[1] != config["input_dim"]:
            raise RuntimeError("input dimension mismatch")
        if config["dataset"] == "amazon" and any(int(mask[:3305].sum()) for mask in masks.values()):
            raise RuntimeError("Amazon uncovered prefix is present in a frozen mask")
        train_labels = labels[masks["train_mask"]]
        normal_count, anomaly_count = int((train_labels == 0).sum()), int((train_labels == 1).sum())
        weights = train_ratio_class_weight(train_labels)
        fingerprints = {
            "dataset_file_sha256": file_sha256(str(ROOT / "datasets" / config["dataset"])),
            "feature_sha256": tensor_sha256(features), "label_sha256": tensor_sha256(labels),
            **{f"{key}_sha256": tensor_sha256(mask) for key, mask in masks.items()},
        }
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
        graph.ndata["feature"], graph.ndata["label"] = features, labels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        graph, labels = graph.to(device), labels.to(device)
        masks = {key: mask.to(device) for key, mask in masks.items()}
        preflight = {
            "input_sha256": fingerprints, "raw_graph": {"nodes": raw.num_nodes(), "edges": raw.num_edges()},
            "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges(), "preprocess": config["graph_preprocess"]},
            "mask_counts": {key: int(mask.sum().cpu()) for key, mask in masks.items()},
            "edge_access": config["edge_access"], "config": config,
            "train_normal_count": normal_count, "train_anomaly_count": anomaly_count,
            "class_weight": weights, "class_weight_source": "frozen train_mask labels only",
        }
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
        (output / "config_snapshot.json").write_text(json.dumps(config, indent=2))
        (output / "environment/framework_versions.json").write_text(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "dgl": dgl.__version__, "gpu": torch.cuda.get_device_name(device)}, indent=2))
        model = KipfTwoLayerGCN(config["input_dim"], config["hidden_dim"], config["output_dim"], config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        f1_best, auprc_best, patience_count, history = ({"value": -1.0, "state": None, "epoch": 0}, {"value": -1.0, "state": None, "epoch": 0}, 0, [])
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for epoch in range(1, config["max_epoch"] + 1):
            model.train(); optimizer.zero_grad()
            loss = F.cross_entropy(model(graph)[masks["train_mask"]], labels[masks["train_mask"]], weight=class_weight_tensor)
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad(): probabilities = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
            val_mask, val_labels = masks["val_mask"].cpu().numpy(), labels[masks["val_mask"]].cpu().numpy()
            val_f1, threshold = best_threshold(f1_score, val_labels, probabilities[val_mask], config["threshold_candidates"])
            val_auroc, val_auprc = float(roc_auc_score(val_labels, probabilities[val_mask])), float(average_precision_score(val_labels, probabilities[val_mask]))
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "val_f1_macro": val_f1, "val_auroc": val_auroc, "val_auprc": val_auprc, "threshold": threshold})
            if val_f1 > f1_best["value"]:
                f1_best, patience_count = {"value": val_f1, "state": cpu_state(model), "epoch": epoch}, 0
            else:
                patience_count += 1
            if val_auprc > auprc_best["value"]:
                auprc_best = {"value": val_auprc, "state": cpu_state(model), "epoch": epoch}
            print(json.dumps({"epoch": epoch, "val_f1_macro": val_f1, "val_auroc": val_auroc, "val_auprc": val_auprc, "f1_patience_count": patience_count, "class_weight": weights}), flush=True)
            if patience_count > config["patience"]:
                break
        f1_checkpoint, auprc_checkpoint = output / "checkpoint_val_f1_best.pt", output / "checkpoint_val_auprc_best.pt"
        torch.save({"model_state_dict": f1_best["state"], "epoch": f1_best["epoch"]}, f1_checkpoint)
        torch.save({"model_state_dict": auprc_best["state"], "epoch": auprc_best["epoch"]}, auprc_checkpoint)
        model.load_state_dict(auprc_best["state"]); model.eval()
        with torch.no_grad(): probabilities = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
        val_mask, val_labels = masks["val_mask"].cpu().numpy(), labels[masks["val_mask"]].cpu().numpy()
        _, threshold = best_threshold(f1_score, val_labels, probabilities[val_mask], config["threshold_candidates"])
        test_mask, test_labels = masks["test_mask"].cpu().numpy(), labels[masks["test_mask"]].cpu().numpy()
        test_probabilities = probabilities[test_mask]
        result.update({"status": "diagnostic", "class_weight": weights, "train_normal_count": normal_count, "train_anomaly_count": anomaly_count,
                       "f1_macro": float(f1_score(test_labels, (test_probabilities > threshold).astype(np.int64), average="macro")),
                       "auroc": float(roc_auc_score(test_labels, test_probabilities)), "threshold": threshold,
                       "best_epoch": auprc_best["epoch"], "early_stop_best_epoch": f1_best["epoch"], "validation_auprc": auprc_best["value"],
                       "epochs_executed": len(history), "wall_time_sec": time.perf_counter() - started,
                       "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2), "checkpoint_sha256": file_sha256(str(auprc_checkpoint)),
                       **fingerprints, "raw_nodes": raw.num_nodes(), "raw_edges": raw.num_edges(), "training_nodes": graph.num_nodes(), "training_edges": graph.num_edges()})
        (output / "validation_history.json").write_text(json.dumps(history, indent=2))
    except Exception:
        result["error"] = traceback.format_exc()
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    with (output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result)); writer.writeheader(); writer.writerow(result)
    print(json.dumps(result))
    if result["status"] != "diagnostic":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
