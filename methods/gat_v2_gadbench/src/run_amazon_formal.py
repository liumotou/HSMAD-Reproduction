"""Independent Amazon formal runs for the frozen GAT-v2-GADBench protocol."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from contracts import amazon_contract, architecture_contract
from model import GADBenchGATV2
from run_diagnostic_full import checkpoint_test
from run_smoke import ROOT, best_threshold, file_sha256, framework, graph_sha256, protected_manifest, set_seed, split_metrics, tensor_sha256, write_json
from selection import update_selection


def protected_manifest_without_amazon_formal():
    manifest = protected_manifest()
    for relative in (
        "results/experiments/gat_v2_gadbench/weibo",
        "results/experiments/gat_v2_gadbench/amazon/protocol_v2_gadbench_hidden64/diagnostic_full",
    ):
        target = ROOT / relative
        if target.exists():
            for entry in sorted(path for path in target.rglob("*") if path.is_file()):
                manifest["files"][str(entry.relative_to(ROOT))] = file_sha256(entry)
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    return manifest


def evaluate_probability(model, graph, features):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(graph, features), dim=1)[:, 1]


def append_run_and_maybe_summarize(formal_dir, metrics, config):
    fields = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs", "stop_reason", "auprc_best_epoch", "wall_time_sec", "peak_gpu_mb", "f1_macro", "auroc", "threshold", "edge_access", "error"]
    path = formal_dir / "runs.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in fields})
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "OK"]
    if len(rows) != 10 or {int(row["seed"]) for row in rows} != set(range(10)):
        return
    f1 = np.asarray([float(row["f1_macro"]) for row in rows])
    auroc = np.asarray([float(row["auroc"]) for row in rows])
    summary = {
        "method": "GAT-v2-GADBench", "dataset": "amazon", "protocol_version": config["protocol_version"], "n": 10,
        "f1_macro_mean": float(f1.mean()), "f1_macro_std_sample": float(f1.std(ddof=1)),
        "auroc_mean": float(auroc.mean()), "auroc_std_sample": float(auroc.std(ddof=1)),
        "paper_f1_macro": config["paper_f1_macro"], "paper_auroc": config["paper_auroc"],
        "paper_delta_f1_macro": float(f1.mean() - config["paper_f1_macro"]),
        "paper_delta_auroc": float(auroc.mean() - config["paper_auroc"]),
    }
    write_json(formal_dir / "summary.json", summary)
    with (formal_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary.keys())
        writer.writeheader(); writer.writerow(summary)


def assert_frozen_protocol(config):
    diagnostic = json.loads((ROOT / "methods/gat_v2_gadbench/configs/amazon_protocol_v2_gadbench_hidden64_diagnostic_full.json").read_text())
    keys = ("dataset_file", "max_epoch", "patience", "h_feats", "num_heads", "per_head_dim", "drop_rate", "expected_anomaly_weight", "optimizer", "learning_rate", "weight_decay", "edge_access", "graph_preprocess", "early_stop_protocol", "checkpoint_protocol", "threshold_protocol", "expected_training_graph_edges", "expected_hashes")
    if any(config[key] != diagnostic[key] for key in keys):
        raise RuntimeError("Formal configuration differs from the completed Amazon diagnostic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.seed not in range(10):
        raise ValueError("Formal seeds are frozen to 0 through 9")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    assert_frozen_protocol(config)
    config["seed"] = args.seed
    config["result_dir"] = config["result_dir_template"].format(seed=args.seed)
    output_dir, formal_dir = ROOT / config["result_dir"], ROOT / config["result_dir"].rsplit("/seed_", 1)[0]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    protected_before, started = protected_manifest_without_amazon_formal(), time.monotonic()
    output_dir.mkdir(parents=True)
    write_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    write_json(output_dir / "environment" / "framework.json", framework())
    write_json(output_dir / "protected_manifest_before.json", protected_before)
    try:
        set_seed(args.seed)
        raw_path = ROOT / config["dataset_file"]
        raw_graph = dgl.load_graphs(str(raw_path))[0][0]
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
        features, labels = raw_graph.ndata["feature"], raw_graph.ndata["label"].long()
        raw_masks = {name: raw_graph.ndata[name] for name in ("train_mask", "val_mask", "test_mask")}
        masks, contract = {name: value.bool() for name, value in raw_masks.items()}, amazon_contract()
        hashes = {"dataset_file_sha256": file_sha256(raw_path), "feature_sha256": tensor_sha256(features), "label_sha256": tensor_sha256(labels), **{f"{name}_sha256": tensor_sha256(value) for name, value in raw_masks.items()}, "raw_graph_edges_sha256": graph_sha256(raw_graph), "training_graph_edges_sha256": graph_sha256(graph), "model_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/model.py"), "run_amazon_formal_py_sha256": file_sha256(Path(__file__)), "selection_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/selection.py")}
        for name, expected in config["expected_hashes"].items():
            if hashes[name] != expected:
                raise RuntimeError(f"Frozen input mismatch: {name}")
        for split, count in (("train_mask", contract["train_mask_count"]), ("val_mask", contract["val_mask_count"]), ("test_mask", contract["test_mask_count"])):
            if int(masks[split].sum()) != count or int(masks[split][:contract["uncovered_prefix_nodes"]].sum()) != 0:
                raise RuntimeError(f"Amazon {split} violates frozen prefix exclusion")
        if features.shape[1] != 25 or graph.num_edges() != contract["training_graph_edges"]:
            raise RuntimeError("Amazon graph contract mismatch")
        train_labels = labels[masks["train_mask"]]
        anomaly_count, normal_count = int(train_labels.sum()), int((train_labels == 0).sum())
        class_weight = [1.0, normal_count / anomaly_count]
        if class_weight[1] != config["expected_anomaly_weight"]:
            raise RuntimeError("Amazon class weight mismatch")
        write_json(output_dir / "preflight.json", {"passed": True, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()}, "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()}, "feature_shape": list(features.shape), "mask_counts": {name: int(value.sum()) for name, value in masks.items()}, "prefix_mask_counts": {name: int(value[:3305].sum()) for name, value in masks.items()}, "hashes": hashes, "class_weight": class_weight, "model_contract": architecture_contract(), "edge_access": config["edge_access"]})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph, features, labels = graph.to(device), features.to(device), labels.to(device)
        masks = {name: value.to(device) for name, value in masks.items()}
        model = GADBenchGATV2(features.shape[1], 64, 4, 0.0, 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        weights = torch.tensor(class_weight, device=device, dtype=torch.float32)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        state, history = {"best_auprc": -1.0, "auprc_best_epoch": None, "best_f1": -1.0, "f1_best_epoch": None, "patience_counter": 0}, []
        checkpoint = output_dir / "checkpoint_validation_auprc_best.pt"
        for epoch in range(1, config["max_epoch"] + 1):
            model.train(); logits = model(graph, features)
            loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            probability = evaluate_probability(model, graph, features).cpu().numpy()
            val_index, val_label = masks["val_mask"].cpu().numpy(), labels[masks["val_mask"]].cpu().numpy()
            threshold, val_f1 = best_threshold(val_label, probability[val_index]); val = split_metrics(val_label, probability[val_index], threshold)
            state, improved, _ = update_selection(state, epoch, val["auprc"], val_f1)
            if improved: torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "validation": val}, checkpoint)
            record = {"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": val_f1, "validation_auroc": val["auroc"], "validation_auprc": val["auprc"], "validation_threshold": threshold, "auprc_best_epoch": state["auprc_best_epoch"], "patience_counter": state["patience_counter"], "learning_rate": optimizer.param_groups[0]["lr"], "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0}
            history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
            if state["patience_counter"] > config["patience"]: break
        test = checkpoint_test(checkpoint, model, graph, features, labels, masks)
        metrics = {"method": "GAT-v2-GADBench", "protocol_version": config["protocol_version"], "dataset": "amazon", "seed": args.seed, "run_type": "formal", "status": "OK", "actual_epochs": len(history), "stop_reason": "max_epoch_reached" if len(history) == config["max_epoch"] else "validation_AUPRC_patience", "auprc_best_epoch": state["auprc_best_epoch"], "f1_macro": test["f1_macro"], "auroc": test["auroc"], "auprc": test["auprc"], "threshold": test["threshold"], "validation_f1_macro": test["validation_f1_macro"], "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0, "class_weight": class_weight, "edge_access": config["edge_access"], "hashes": hashes, "error": ""}
        write_json(output_dir / "validation_history.json", history)
    except Exception as exc:
        metrics = {"method": "GAT-v2-GADBench", "protocol_version": config["protocol_version"], "dataset": "amazon", "seed": args.seed, "run_type": "formal", "status": "ERROR", "actual_epochs": 0, "stop_reason": "ERROR", "auprc_best_epoch": "", "wall_time_sec": time.monotonic() - started, "peak_gpu_mb": 0.0, "f1_macro": "", "auroc": "", "threshold": "", "edge_access": config["edge_access"], "error": repr(exc)}
        raise
    finally:
        protected_after = protected_manifest_without_amazon_formal()
        metrics["protected_manifest_unchanged"] = protected_before == protected_after
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "protected_manifest_after.json", protected_after)
        write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
        append_run_and_maybe_summarize(formal_dir, metrics, config)
    print("FORMAL_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
