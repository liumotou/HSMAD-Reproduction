"""One-epoch, no-adaptation full-graph T-Social GAT-v2 memory smoke."""
import argparse, csv, hashlib, json, shutil, time, traceback
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F

from model_tsocial_smoke import TSocialGADBenchGATV2
from run_smoke import ROOT, best_threshold, file_sha256, framework, graph_sha256, protected_manifest, set_seed, split_metrics, tensor_sha256, write_json


def protected_tsocial_history_roots():
    """All existing GAT-v2 result roots that the T-Social smoke must not alter."""
    return (
        "results/experiments/gat_v2_gadbench/weibo",
        "results/experiments/gat_v2_gadbench/amazon",
        "results/experiments/gat_v2_gadbench/yelp",
        "results/experiments/gat_v2_gadbench/tolokers",
        "results/experiments/gat_v2_gadbench/tfinance",
    )


def protected_manifest_with_gat_v2_history():
    manifest = protected_manifest()
    for relative in protected_tsocial_history_roots():
        target = ROOT / relative
        if target.exists():
            for entry in sorted(path for path in target.rglob("*") if path.is_file()):
                manifest["files"][str(entry.relative_to(ROOT))] = file_sha256(entry)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    return manifest


def gpu_snapshot():
    if not torch.cuda.is_available():
        return {"cuda_available": False, "gpu_total_mb": None, "gpu_free_mb": None}
    free, total = torch.cuda.mem_get_info()
    return {"cuda_available": True, "gpu_total_mb": total / 1024**2, "gpu_free_mb": free / 1024**2}


def is_cuda_oom(error):
    """DGL reports CUDA OOM as DGLError, rather than torch's OOM subtype."""
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["result_dir"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    protected_before = protected_manifest_with_gat_v2_history()
    started = time.monotonic()
    output_dir.mkdir(parents=True)
    write_json(output_dir / "config_snapshot.json", config)
    shutil.copy2(config_path, output_dir / "config_source.json")
    write_json(output_dir / "environment" / "framework.json", framework())
    write_json(output_dir / "protected_manifest_before.json", protected_before)
    history, hashes, preflight, stage = [], {}, {"gpu_at_start": gpu_snapshot(), "passed": False}, "load"
    try:
        set_seed(config["seed"])
        raw = dgl.load_graphs(str(ROOT / config["dataset_file"]))[0][0]
        stage = "graph_preprocess"
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
        features, labels = raw.ndata["feature"], raw.ndata["label"].long()
        raw_masks = {name: raw.ndata[name] for name in ("train_mask", "val_mask", "test_mask")}
        masks = {name: value.bool() for name, value in raw_masks.items()}
        hashes = {
            "dataset_file_sha256": file_sha256(ROOT / config["dataset_file"]),
            "feature_sha256": tensor_sha256(features), "label_sha256": tensor_sha256(labels),
            **{f"{name}_sha256": tensor_sha256(value) for name, value in raw_masks.items()},
            "raw_graph_edges_sha256": graph_sha256(raw), "training_graph_edges_sha256": graph_sha256(graph),
            "model_tsocial_smoke_py_sha256": file_sha256(ROOT / "methods/gat_v2_gadbench/src/model_tsocial_smoke.py"),
            "run_tsocial_smoke_py_sha256": file_sha256(Path(__file__)),
        }
        train_labels = labels[masks["train_mask"]]
        normal_count, anomaly_count = int((train_labels == 0).sum()), int(train_labels.sum())
        if anomaly_count == 0:
            raise RuntimeError("Frozen T-Social train_mask has no anomaly labels")
        class_weight = [1.0, normal_count / anomaly_count]
        model = TSocialGADBenchGATV2(features.shape[1], config["hidden_dim_total"], config["num_heads"], config["drop_rate"], 2)
        preflight.update({
            "passed": True, "raw_graph": {"nodes": raw.num_nodes(), "edges": raw.num_edges()},
            "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
            "feature_shape": list(features.shape), "mask_counts": {name: int(value.sum()) for name, value in masks.items()},
            "class_weight": class_weight, "hashes": hashes, "edge_access": config["edge_access"],
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_contract": {"hidden_dim_total": 10, "num_heads": 2, "head_dim": 5, "num_gat_blocks": 2, "output_dim": 2},
        })
        write_json(output_dir / "preflight.json", preflight)
        stage = "device_transfer"
        device = torch.device("cuda")
        graph, features, labels = graph.to(device), features.to(device), labels.to(device)
        masks = {name: value.to(device) for name, value in masks.items()}
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        stage = "epoch_1"
        epoch_started = time.monotonic()
        model.train()
        loss = F.cross_entropy(model(graph, features)[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(graph, features), dim=1)[:, 1].cpu().numpy()
        val_index = masks["val_mask"].cpu().numpy()
        val_labels = labels[masks["val_mask"]].cpu().numpy()
        threshold, validation_f1 = best_threshold(val_labels, probability[val_index])
        validation = split_metrics(val_labels, probability[val_index], threshold)
        history.append({"epoch": 1, "train_loss": float(loss.item()), "validation_f1_macro": validation_f1,
                        "validation_auroc": validation["auroc"], "validation_auprc": validation["auprc"],
                        "validation_threshold": threshold, "epoch_wall_time_sec": time.monotonic() - epoch_started,
                        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        test_index = masks["test_mask"].cpu().numpy()
        test = split_metrics(labels[masks["test_mask"]].cpu().numpy(), probability[test_index], threshold)
        checkpoint = output_dir / "checkpoint_epoch_1.pt"
        torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "config": config}, checkpoint)
        metrics = {"status": "smoke", "run_type": "smoke", "dataset": "tsocial", "seed": 0,
                   "actual_epochs": 1, "stage": "completed", "validation": validation, "final_test": test,
                   "threshold": threshold, "wall_time_sec": time.monotonic() - started,
                   "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                   "edge_access": config["edge_access"], "hashes": hashes, "model_parameter_count": sum(p.numel() for p in model.parameters())}
    except torch.cuda.OutOfMemoryError:
        preflight.update({"passed": False, "failure_stage": stage, "gpu_at_failure": gpu_snapshot(), "hashes": hashes})
        write_json(output_dir / "preflight.json", preflight)
        metrics = {"status": "OOM", "run_type": "smoke", "dataset": "tsocial", "seed": 0, "stage": stage,
                   "error": traceback.format_exc(), "gpu": gpu_snapshot(),
                   "allocated_gpu_mb": torch.cuda.memory_allocated() / 1024**2, "reserved_gpu_mb": torch.cuda.memory_reserved() / 1024**2,
                   "wall_time_sec": time.monotonic() - started, "edge_access": config["edge_access"], "hashes": hashes}
    except Exception as error:
        if is_cuda_oom(error):
            preflight.update({"passed": False, "failure_stage": stage, "gpu_at_failure": gpu_snapshot(), "hashes": hashes})
            write_json(output_dir / "preflight.json", preflight)
            metrics = {"status": "OOM", "run_type": "smoke", "dataset": "tsocial", "seed": 0, "stage": stage,
                       "error": traceback.format_exc(), "gpu": gpu_snapshot(),
                       "requested_allocation_mb": None, "requested_allocation_note": "DGL error did not expose allocation size",
                       "allocated_gpu_mb": torch.cuda.memory_allocated() / 1024**2, "reserved_gpu_mb": torch.cuda.memory_reserved() / 1024**2,
                       "wall_time_sec": time.monotonic() - started, "edge_access": config["edge_access"], "hashes": hashes}
        else:
            preflight.update({"passed": False, "failure_stage": stage, "hashes": hashes})
            write_json(output_dir / "preflight.json", preflight)
            metrics = {"status": "ERROR", "run_type": "smoke", "dataset": "tsocial", "seed": 0, "stage": stage,
                       "error": traceback.format_exc(), "wall_time_sec": time.monotonic() - started,
                       "edge_access": config["edge_access"], "hashes": hashes}
    finally:
        write_json(output_dir / "validation_history.json", history)
        protected_after = protected_manifest_with_gat_v2_history()
        metrics["protected_manifest_unchanged"] = protected_before == protected_after
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "protected_manifest_after.json", protected_after)
        write_json(output_dir / "artifact_sha256s.json", {path.name: file_sha256(path) for path in output_dir.iterdir() if path.is_file()})
        with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["dataset", "seed", "run_type", "status", "stage", "actual_epochs", "wall_time_sec", "peak_gpu_mb", "edge_access", "error"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerow({field: metrics.get(field, "") for field in fields})
    print("TSOCIAL_SMOKE_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
