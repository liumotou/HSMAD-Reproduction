"""Independent formal GAT-v2 runs; one explicitly seeded process per call."""

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from contracts import formal_execution_contract
from model import GADBenchGATV2, architecture_contract
from run_diagnostic_full import checkpoint_test
from run_smoke import (
    ROOT, best_threshold, file_sha256, framework, graph_sha256, protected_manifest,
    set_seed, split_metrics, tensor_sha256, write_json,
)
from selection import update_selection


PAPER_F1, PAPER_AUROC = 0.9408, 0.9614


def protected_manifest_without_formal():
    manifest = protected_manifest()
    for relative in (
        "results/experiments/gat_v2_gadbench/weibo/protocol_v2_gadbench_hidden64/smoke",
        "results/experiments/gat_v2_gadbench/weibo/protocol_v2_gadbench_hidden64/diagnostic_full",
    ):
        target = ROOT / relative
        if target.exists():
            for entry in sorted(path for path in target.rglob("*") if path.is_file()):
                manifest["files"][str(entry.relative_to(ROOT))] = file_sha256(entry)
    import hashlib
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    return manifest


def evaluate_probability(model, graph, features):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(graph, features), dim=1)[:, 1]


def append_run_and_maybe_summarize(formal_dir, metrics):
    runs_path = formal_dir / "runs.csv"
    fields = ["method", "protocol_version", "dataset", "seed", "run_type", "status", "actual_epochs",
              "auprc_best_epoch", "wall_time_sec", "peak_gpu_mb", "f1_macro", "auroc", "threshold", "edge_access"]
    exists = runs_path.exists()
    with runs_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: metrics[name] for name in fields})
    with runs_path.open("r", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "OK"]
    if len(rows) == 10 and {int(row["seed"]) for row in rows} == set(range(10)):
        f1 = np.asarray([float(row["f1_macro"]) for row in rows])
        auroc = np.asarray([float(row["auroc"]) for row in rows])
        summary = {
            "method": "GAT-v2-GADBench", "dataset": "weibo", "protocol_version": "protocol_v2_gadbench_hidden64",
            "n": 10, "f1_macro_mean": float(f1.mean()), "f1_macro_std_sample": float(f1.std(ddof=1)),
            "auroc_mean": float(auroc.mean()), "auroc_std_sample": float(auroc.std(ddof=1)),
            "paper_f1_macro": PAPER_F1, "paper_auroc": PAPER_AUROC,
            "paper_delta_f1_macro": float(f1.mean() - PAPER_F1), "paper_delta_auroc": float(auroc.mean() - PAPER_AUROC),
        }
        write_json(formal_dir / "summary.json", summary)
        with (formal_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.seed not in range(10):
        raise ValueError("Formal seeds are frozen to 0 through 9")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["result_dir"] = config["result_dir_template"].format(seed=args.seed)
    output_dir = ROOT / config["result_dir"]
    formal_dir = output_dir.parent
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if {key: config[key] for key in formal_execution_contract()} != formal_execution_contract():
        raise RuntimeError("Formal configuration differs from the frozen diagnostic hyperparameters")
    protected_before = protected_manifest_without_formal()
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    write_json(output_dir / "environment" / "framework.json", framework())
    write_json(output_dir / "protected_manifest_before.json", protected_before)

    set_seed(args.seed)
    raw_path = ROOT / config["dataset_file"]
    raw_graph = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    features, labels = raw_graph.ndata["feature"], raw_graph.ndata["label"].long()
    raw_masks = {name: raw_graph.ndata[name] for name in ("train_mask", "val_mask", "test_mask")}
    masks = {name: value.bool() for name, value in raw_masks.items()}
    hashes = {
        "dataset_file_sha256": file_sha256(raw_path), "feature_sha256": tensor_sha256(features),
        "label_sha256": tensor_sha256(labels), **{f"{name}_sha256": tensor_sha256(value) for name, value in raw_masks.items()},
        "raw_graph_edges_sha256": graph_sha256(raw_graph), "training_graph_edges_sha256": graph_sha256(graph),
        "model_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/model.py"),
        "run_formal_py_sha256": file_sha256(Path(__file__)), "selection_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/selection.py"),
    }
    for name, expected in config["expected_hashes"].items():
        if hashes[name] != expected:
            raise RuntimeError(f"Frozen input mismatch: {name}")
    if graph.num_edges() != config["expected_training_graph_edges"]:
        raise RuntimeError("Unexpected preprocessed graph edge count")
    train_labels = labels[masks["train_mask"]]
    anomaly_count, normal_count = int(train_labels.sum()), int((train_labels == 0).sum())
    class_weight = [1.0, normal_count / anomaly_count]
    if class_weight[1] != config["expected_anomaly_weight"]:
        raise RuntimeError("Class weight differs from frozen diagnostic configuration")
    write_json(output_dir / "preflight.json", {
        "passed": True, "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "mask_counts": {name: int(value.sum()) for name, value in masks.items()}, "hashes": hashes,
        "class_weight": class_weight, "model_contract": architecture_contract(), "edge_access": config["edge_access"],
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, features, labels = graph.to(device), features.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = GADBenchGATV2(features.shape[1], 64, 4, 0.0, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = torch.tensor(class_weight, device=device, dtype=torch.float32)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    state = {"best_auprc": -1.0, "auprc_best_epoch": None, "best_f1": -1.0, "f1_best_epoch": None, "patience_counter": 0}
    history = []
    checkpoint = output_dir / "checkpoint_validation_auprc_best.pt"
    for epoch in range(1, config["max_epoch"] + 1):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        probability = evaluate_probability(model, graph, features).cpu().numpy()
        val_index, val_label = masks["val_mask"].cpu().numpy(), labels[masks["val_mask"]].cpu().numpy()
        threshold, val_f1 = best_threshold(val_label, probability[val_index])
        val = split_metrics(val_label, probability[val_index], threshold)
        state, auprc_improved, _ = update_selection(state, epoch, val["auprc"], val_f1)
        if auprc_improved:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "validation": val}, checkpoint)
        record = {"epoch": epoch, "train_loss": float(loss.item()), "validation_f1_macro": val_f1,
                  "validation_auroc": val["auroc"], "validation_auprc": val["auprc"], "validation_threshold": threshold,
                  "auprc_best_epoch": state["auprc_best_epoch"], "patience_counter": state["patience_counter"],
                  "learning_rate": optimizer.param_groups[0]["lr"], "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if state["patience_counter"] > config["patience"]:
            break
    test = checkpoint_test(checkpoint, model, graph, features, labels, masks)
    protected_after = protected_manifest_without_formal()
    metrics = {
        "method": "GAT-v2-GADBench", "protocol_version": config["protocol_version"], "dataset": "weibo", "seed": args.seed,
        "run_type": "formal", "status": "OK", "actual_epochs": len(history), "auprc_best_epoch": state["auprc_best_epoch"],
        "f1_macro": test["f1_macro"], "auroc": test["auroc"], "auprc": test["auprc"], "threshold": test["threshold"],
        "validation_f1_macro": test["validation_f1_macro"], "wall_time_sec": time.monotonic() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0,
        "class_weight": class_weight, "edge_access": config["edge_access"], "hashes": hashes,
        "protected_manifest_unchanged": protected_before == protected_after, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "validation_history.json", history)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "protected_manifest_after.json", protected_after)
    write_json(output_dir / "artifact_sha256s.json", {entry.name: file_sha256(entry) for entry in output_dir.iterdir() if entry.is_file()})
    append_run_and_maybe_summarize(formal_dir, metrics)
    print("FORMAL_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
