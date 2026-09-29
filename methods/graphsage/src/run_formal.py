"""One isolated formal GraphSAGE-GADBench-h64 training run."""

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from methods.graphsage.src.model import GraphSAGEGADBench, architecture_contract
from methods.graphsage.src.protocol import class_weight_from_train_labels, numpy_bool_mask, numpy_labels, numpy_probabilities, select_validation_f1_threshold
from methods.graphsage.src.run_smoke import file_sha256, framework_info, graph_sha256, protected_manifest, setup_seed, tensor_sha256, write_json
from methods.graphsage.src.selection import update_auprc_selection


ROOT = Path(__file__).resolve().parents[3]
RUN_FIELDS = [
    "method", "protocol_version", "dataset", "seed", "run_type", "status", "f1_macro", "auroc", "auprc",
    "threshold", "best_epoch", "actual_epochs", "wall_time_sec", "peak_gpu_mb", "train_normal_count",
    "train_anomaly_count", "class_weight", "config_sha256", "checkpoint_sha256", "split_sha256", "environment_id",
    "edge_access", "error",
]


def strict_determinism_contract():
    """The pre-launch and in-process requirements for reproducible DGL runs."""
    return {
        "required_cublas_workspace_config": ":4096:8",
        "required_pythonhashseed": "0",
        "dgl_seed": True,
        "torch_deterministic_algorithms": True,
        "warn_only": False,
    }


def configure_strict_determinism(seed):
    """Activate the contract before graph loading or model construction."""
    contract = strict_determinism_contract()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract["required_cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Python starts")
    if os.environ.get("PYTHONHASHSEED") != contract["required_pythonhashseed"]:
        raise RuntimeError("PYTHONHASHSEED must be 0 before Python starts")
    setup_seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    return contract


def validated_run_type(config):
    """Keep legacy formal behavior while allowing isolated full diagnostics."""
    run_type = config.get("run_type", "formal")
    if run_type not in {"formal", "diagnostic"}:
        raise ValueError(f"Unsupported run_type: {run_type}")
    return run_type


def split_metrics(labels, probabilities, threshold):
    prediction = probabilities >= threshold
    return {
        "f1_macro": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "predicted_anomaly_count": int(prediction.sum()),
        "predicted_anomaly_ratio": float(prediction.mean()),
    }


def append_run(base_dir, record):
    path = Path(base_dir) / "runs.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in RUN_FIELDS})


def input_hashes(raw_path, raw_graph, graph, labels, masks):
    values = {
        "dataset_file_sha256": file_sha256(raw_path), "feature_sha256": tensor_sha256(raw_graph.ndata["feature"]),
        "label_sha256": tensor_sha256(labels), "raw_graph_edges_sha256": graph_sha256(raw_graph),
        "training_graph_edges_sha256": graph_sha256(graph), "model_py_sha256": file_sha256(ROOT / "methods/graphsage/src/model.py"),
        "protocol_py_sha256": file_sha256(ROOT / "methods/graphsage/src/protocol.py"), "selection_py_sha256": file_sha256(ROOT / "methods/graphsage/src/selection.py"),
        "run_formal_py_sha256": file_sha256(Path(__file__)),
    }
    values.update({f"{name}_sha256": tensor_sha256(value) for name, value in masks.items()})
    return values


def execute(config, config_path, seed):
    run_type = validated_run_type(config)
    base_dir = ROOT / config["result_base_dir"]
    output_dir = base_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protected_before = protected_manifest()
    config_snapshot = dict(config, seed=seed, run_type=run_type)
    config_sha256 = file_sha256(config_path)
    write_json(output_dir / "config_snapshot.json", config_snapshot)
    shutil.copy2(config_path, output_dir / "config_source.json")
    environment = framework_info()
    environment_id = file_sha256(Path(__file__))[:16] + "-" + str(environment.get("torch"))
    write_json(output_dir / "environment" / "framework.json", environment)
    write_json(output_dir / "protected_manifest_before.json", protected_before)
    record = {"method": config["method"], "protocol_version": config["protocol_version"], "dataset": config["dataset"], "seed": seed,
              "run_type": run_type, "config_sha256": config_sha256, "environment_id": environment_id, "edge_access": config["edge_access"], "error": ""}
    try:
        strict_contract = None
        if config.get("strict_determinism", False):
            strict_contract = configure_strict_determinism(seed)
        else:
            setup_seed(seed)
        raw_path = ROOT / config["dataset_file"]
        raw_graph = dgl.load_graphs(str(raw_path))[0][0]
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
        graph.ndata["feature"] = raw_graph.ndata["feature"]
        labels = raw_graph.ndata["label"].long()
        masks = {name: raw_graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
        hashes = input_hashes(raw_path, raw_graph, graph, labels, masks)
        for name, expected in config["expected_hashes"].items():
            if hashes[name] != expected:
                raise RuntimeError(f"Frozen input mismatch: {name}")
        if graph.num_edges() != config["expected_training_graph_edges"]:
            raise RuntimeError(f"Unexpected training graph edge count: {graph.num_edges()}")
        class_weight, normal_count, anomaly_count = class_weight_from_train_labels(labels[masks["train_mask"]])
        model = GraphSAGEGADBench(raw_graph.ndata["feature"].shape[1], config["h_feats"], 2, config["num_layers"], config["aggregation"], config["dropout"], config["activation"])
        if len(model.layers) != 2 or model.layers[0]._aggre_type != "pool" or model.layers[0]._out_feats != 64 or model.layers[1]._out_feats != 64:
            raise RuntimeError("Model contract mismatch")
        split_sha256 = ":".join(hashes[f"{name}_sha256"] for name in ("train_mask", "val_mask", "test_mask"))
        write_json(output_dir / "preflight.json", {"passed": True, "seed": seed, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
                   "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()}, "feature_shape": list(raw_graph.ndata["feature"].shape),
                   "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "hashes": hashes, "class_weight": class_weight,
                   "train_normal_count": normal_count, "train_anomaly_count": anomaly_count, "edge_access": config["edge_access"],
                   "model_contract": architecture_contract(), "strict_determinism": strict_contract})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph, labels = graph.to(device), labels.to(device)
        masks = {name: value.to(device) for name, value in masks.items()}
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        selection = {"best_auprc": -1.0, "best_epoch": None, "patience_counter": 0}
        history = []
        checkpoint = output_dir / "checkpoint_validation_auprc_best.pt"
        for epoch in range(1, config["max_epoch"] + 1):
            model.train()
            logits = model(graph)
            loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probabilities = torch.softmax(model(graph), dim=1)[:, 1]
            values = numpy_probabilities(probabilities)
            val_mask = numpy_bool_mask(masks["val_mask"])
            val_labels = numpy_labels(labels[masks["val_mask"]])
            threshold, validation_f1 = select_validation_f1_threshold(val_labels, values[val_mask])
            validation = split_metrics(val_labels, values[val_mask], threshold)
            selection, improved, should_stop = update_auprc_selection(selection, epoch, validation["auprc"], config["patience"])
            if improved:
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config_snapshot, "validation_auprc": validation["auprc"]}, checkpoint)
            row = {"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": validation_f1, "validation_auroc": validation["auroc"],
                   "validation_auprc": validation["auprc"], "validation_threshold": threshold, "auprc_best_epoch_so_far": selection["best_epoch"],
                   "patience_counter": selection["patience_counter"], "learning_rate": optimizer.param_groups[0]["lr"],
                   "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if should_stop:
                break
        if selection["best_epoch"] is None:
            raise RuntimeError("No validation AUPRC checkpoint was saved")
        write_json(output_dir / "validation_history.json", history)
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(graph), dim=1)[:, 1]
        values = numpy_probabilities(probabilities)
        val_mask, test_mask = numpy_bool_mask(masks["val_mask"]), numpy_bool_mask(masks["test_mask"])
        val_labels, test_labels = numpy_labels(labels[masks["val_mask"]]), numpy_labels(labels[masks["test_mask"]])
        threshold, validation_f1 = select_validation_f1_threshold(val_labels, values[val_mask])
        test = split_metrics(test_labels, values[test_mask], threshold)
        checkpoint_sha256 = file_sha256(checkpoint)
        protected_after = protected_manifest()
        metrics = {**record, "status": "OK", "f1_macro": test["f1_macro"], "auroc": test["auroc"], "auprc": test["auprc"], "threshold": threshold,
                   "best_epoch": selection["best_epoch"], "actual_epochs": len(history), "wall_time_sec": time.monotonic() - started,
                   "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
                   "train_normal_count": normal_count, "train_anomaly_count": anomaly_count, "class_weight": class_weight, "checkpoint_sha256": checkpoint_sha256,
                   "split_sha256": split_sha256, "hashes": hashes, "validation_f1_macro": validation_f1, "validation_auprc_best": selection["best_auprc"],
                   "protected_manifest_unchanged": protected_before == protected_after, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "test_metrics": test}
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "protected_manifest_after.json", protected_after)
        append_run(base_dir, metrics)
        write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
        print("FORMAL_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)
    except Exception as error:
        protected_after = protected_manifest()
        status = "OOM" if "out of memory" in str(error).lower() else "ERROR"
        metrics = {**record, "status": status, "f1_macro": None, "auroc": None, "auprc": None, "threshold": None, "best_epoch": None,
                   "actual_epochs": 0, "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": None, "train_normal_count": None,
                   "train_anomaly_count": None, "class_weight": None, "checkpoint_sha256": None, "split_sha256": None, "error": repr(error),
                   "protected_manifest_unchanged": protected_before == protected_after, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "protected_manifest_after.json", protected_after)
        append_run(base_dir, metrics)
        write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    execute(json.loads(config_path.read_text(encoding="utf-8")), config_path, args.seed)


if __name__ == "__main__":
    main()
