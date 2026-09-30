import argparse
import csv
import hashlib
import json
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def run_contract(config: dict) -> tuple[int, Path]:
    seed = int(config["seed"])
    suffix = Path(
        "results/experiments/gcn"
    ) / config["dataset"] / config["protocol_version"] / config["run_type"] / f"seed_{seed}"
    return seed, suffix


def protocol_metadata(config: dict) -> dict:
    return {
        "checkpoint_protocol": config["checkpoint_protocol"],
        "early_stop_protocol": config["early_stop_protocol"],
        "threshold_protocol": config["threshold_protocol"],
    }


def json_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def best_threshold(f1_score, labels, probabilities, thresholds):
    result = (-1.0, thresholds[0])
    for threshold in thresholds:
        score = f1_score(labels, (probabilities > threshold).astype("int64"), average="macro")
        if score > result[0]:
            result = (float(score), float(threshold))
    return result


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def test_prediction_counts(labels, predictions) -> dict[str, int]:
    """Metadata only: count labels/predictions already restricted to test_mask."""
    return {
        "actual_anomaly_count": int(sum(int(value) for value in labels)),
        "predicted_anomaly_count": int(sum(int(value) for value in predictions)),
    }


def append_formal_record(output: Path, result: dict) -> None:
    if result["run_type"] != "formal":
        return
    parent_runs = output.parent / "runs.csv"
    write_header = not parent_runs.exists()
    with parent_runs.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result), extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    seed, suffix = run_contract(config)
    output = ROOT / suffix
    output.mkdir(parents=True, exist_ok=False)
    (output / "environment").mkdir()

    import dgl
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    from methods.mlp.src.utils import file_sha256, setup_seed, tensor_sha256
    from methods.gcn.src.kipf_two_layer import KipfTwoLayerGCN

    result = {
        "method": "GCN",
        "protocol_version": config["protocol_version"],
        "dataset": config["dataset"],
        "seed": seed,
        "run_type": config["run_type"],
        "status": "ERROR",
        "edge_access": config["edge_access"],
        **protocol_metadata(config),
        "config_sha256": json_sha256(config),
        "code_sha256": file_sha256(str(Path(__file__))),
        "model_sha256": file_sha256(str(ROOT / "methods/gcn/src/kipf_two_layer.py")),
        "f1_macro": "",
        "auroc": "",
        "threshold": "",
        "best_epoch": "",
        "early_stop_best_epoch": "",
        "validation_auprc": "",
        "epochs_executed": "",
        "wall_time_sec": "",
        "peak_gpu_mb": "",
        "checkpoint_sha256": "",
        "dataset_file_sha256": "",
        "feature_sha256": "",
        "label_sha256": "",
        "train_mask_sha256": "",
        "val_mask_sha256": "",
        "test_mask_sha256": "",
        "raw_nodes": "",
        "raw_edges": "",
        "training_nodes": "",
        "training_edges": "",
        "actual_anomaly_count": "",
        "predicted_anomaly_count": "",
        "error": "",
    }
    try:
        setup_seed(seed)
        raw = dgl.load_graphs(str(ROOT / "datasets" / config["dataset"]))[0][0]
        x = raw.ndata["feature"].float().contiguous()
        y = raw.ndata["label"].long().reshape(-1).contiguous()
        masks = {
            key: raw.ndata[key].bool().reshape(-1).contiguous()
            for key in ("train_mask", "val_mask", "test_mask")
        }
        if x.shape[1] != config["input_dim"]:
            raise RuntimeError("input dimension mismatch")
        if config["dataset"] == "amazon" and any(int(mask[:3305].sum()) for mask in masks.values()):
            raise RuntimeError("Amazon prefix covered")
        fingerprints = {
            "dataset_file_sha256": file_sha256(str(ROOT / "datasets" / config["dataset"])),
            "feature_sha256": tensor_sha256(x),
            "label_sha256": tensor_sha256(y),
            **{f"{key}_sha256": tensor_sha256(mask) for key, mask in masks.items()},
        }
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
        graph.ndata["feature"] = x
        graph.ndata["label"] = y
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        graph = graph.to(device)
        y = y.to(device)
        masks = {key: mask.to(device) for key, mask in masks.items()}
        preflight = {
            "input_sha256": fingerprints,
            "raw_graph": {"nodes": raw.num_nodes(), "edges": raw.num_edges()},
            "training_graph": {
                "nodes": graph.num_nodes(),
                "edges": graph.num_edges(),
                "preprocess": config["graph_preprocess"],
            },
            "mask_counts": {key: int(mask.sum().cpu()) for key, mask in masks.items()},
            "edge_access": config["edge_access"],
            "config": config,
        }
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
        (output / "config_snapshot.json").write_text(json.dumps(config, indent=2))
        (output / "environment/framework_versions.json").write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "dgl": dgl.__version__,
                    "gpu": torch.cuda.get_device_name(device),
                },
                indent=2,
            )
        )
        model = KipfTwoLayerGCN(
            config["input_dim"], config["hidden_dim"], config["output_dim"], config["dropout"]
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        f1_best = {"value": -1.0, "state": None, "epoch": 0}
        auprc_best = {"value": -1.0, "state": None, "epoch": 0}
        patience_count = 0
        history = []
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for epoch in range(1, config["max_epoch"] + 1):
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(graph)[masks["train_mask"]], y[masks["train_mask"]])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probabilities = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
            val_mask = masks["val_mask"].cpu().numpy()
            val_y = y[masks["val_mask"]].cpu().numpy()
            val_f1, threshold = best_threshold(f1_score, val_y, probabilities[val_mask], config["threshold_candidates"])
            val_auroc = float(roc_auc_score(val_y, probabilities[val_mask]))
            val_auprc = float(average_precision_score(val_y, probabilities[val_mask]))
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "val_f1_macro": val_f1, "val_auroc": val_auroc, "val_auprc": val_auprc, "threshold": threshold})
            if val_f1 > f1_best["value"]:
                f1_best = {"value": val_f1, "state": cpu_state(model), "epoch": epoch}
                patience_count = 0
            else:
                patience_count += 1
            if val_auprc > auprc_best["value"]:
                auprc_best = {"value": val_auprc, "state": cpu_state(model), "epoch": epoch}
            print(json.dumps({"epoch": epoch, "val_f1_macro": val_f1, "val_auroc": val_auroc, "val_auprc": val_auprc, "f1_patience_count": patience_count}), flush=True)
            if patience_count > config["patience"]:
                break
        f1_checkpoint = output / "checkpoint_val_f1_best.pt"
        auprc_checkpoint = output / "checkpoint_val_auprc_best.pt"
        torch.save({"model_state_dict": f1_best["state"], "epoch": f1_best["epoch"]}, f1_checkpoint)
        torch.save({"model_state_dict": auprc_best["state"], "epoch": auprc_best["epoch"]}, auprc_checkpoint)
        model.load_state_dict(auprc_best["state"])
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
        val_mask = masks["val_mask"].cpu().numpy()
        val_y = y[masks["val_mask"]].cpu().numpy()
        _, threshold = best_threshold(f1_score, val_y, probabilities[val_mask], config["threshold_candidates"])
        test_mask = masks["test_mask"].cpu().numpy()
        test_y = y[masks["test_mask"]].cpu().numpy()
        test_probabilities = probabilities[test_mask]
        test_predictions = (test_probabilities > threshold).astype(np.int64)
        result.update(
            {
                "status": "OK" if config["run_type"] == "formal" else config["run_type"],
                "f1_macro": float(f1_score(test_y, test_predictions, average="macro")),
                "auroc": float(roc_auc_score(test_y, test_probabilities)),
                "threshold": threshold,
                "best_epoch": auprc_best["epoch"],
                "early_stop_best_epoch": f1_best["epoch"],
                "validation_auprc": auprc_best["value"],
                "epochs_executed": len(history),
                "wall_time_sec": time.perf_counter() - started,
                "peak_gpu_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
                "checkpoint_sha256": file_sha256(str(auprc_checkpoint)),
                **fingerprints,
                "raw_nodes": raw.num_nodes(),
                "raw_edges": raw.num_edges(),
                "training_nodes": graph.num_nodes(),
                "training_edges": graph.num_edges(),
                **test_prediction_counts(test_y, test_predictions),
            }
        )
        (output / "validation_history.json").write_text(json.dumps(history, indent=2))
    except Exception:
        result["error"] = traceback.format_exc()
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    with (output / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    append_formal_record(output, result)
    print(json.dumps(result))
    if result["status"] not in {"OK", "smoke", "diagnostic", "candidate_full"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
