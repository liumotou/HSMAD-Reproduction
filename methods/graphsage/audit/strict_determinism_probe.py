"""Strict-determinism, one-step Amazon GraphSAGE audit probe.

It is deliberately separate from the formal runner and never writes a
checkpoint or evaluates validation/test data.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import traceback
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "methods" / "graphsage" / "src"
sys.path.insert(0, str(SRC))

from determinism_probe import gradient_record, rng_state_hashes, state_dict_sha256  # noqa: E402
from model import GraphSAGEGADBench  # noqa: E402
from protocol import class_weight_from_train_labels  # noqa: E402
from run_smoke import file_sha256, graph_sha256, setup_seed, tensor_sha256, write_json  # noqa: E402


def strict_probe_contract():
    return {
        "optimizer_steps": 1,
        "save_checkpoint": False,
        "compute_validation_or_test_metrics": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "required_cublas_workspace_config": ":4096:8",
        "required_pythonhashseed": "0",
    }


def deterministic_settings():
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def strict_setup():
    contract = strict_probe_contract()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract["required_cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Python starts")
    if os.environ.get("PYTHONHASHSEED") != contract["required_pythonhashseed"]:
        raise RuntimeError("PYTHONHASHSEED must be 0 before Python starts")
    # Existing project setup first; these following calls make the strict
    # probe explicit without changing the formal runner.
    setup_seed(0)
    dgl.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    return deterministic_settings()


def run_probe(config_path, output_dir):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty strict probe output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["dataset"] != "amazon" or config["h_feats"] != 64 or config["max_epoch"] != 200:
            raise RuntimeError("Strict probe requires frozen Amazon h64 formal configuration")
        settings = strict_setup()
        rng_hashes = rng_state_hashes()
        raw_path = ROOT / config["dataset_file"]
        raw_graph = dgl.load_graphs(str(raw_path))[0][0]
        graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
        graph.ndata["feature"] = raw_graph.ndata["feature"]
        features, labels = raw_graph.ndata["feature"], raw_graph.ndata["label"].long()
        masks = {name: raw_graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
        hashes = {
            "dataset_file_sha256": file_sha256(raw_path),
            "feature_sha256": tensor_sha256(features),
            "label_sha256": tensor_sha256(labels),
            "raw_graph_edges_sha256": graph_sha256(raw_graph),
            "training_graph_edges_sha256": graph_sha256(graph),
        }
        hashes.update({f"{name}_sha256": tensor_sha256(value) for name, value in masks.items()})
        for name, expected in config["expected_hashes"].items():
            if hashes[name] != expected:
                raise RuntimeError(f"Frozen input mismatch: {name}")
        if graph.num_edges() != config["expected_training_graph_edges"]:
            raise RuntimeError("Unexpected training graph edge count")
        class_weight, normal_count, anomaly_count = class_weight_from_train_labels(labels[masks["train_mask"]])
        model = GraphSAGEGADBench(features.shape[1], config["h_feats"], 2, config["num_layers"], config["aggregation"], config["dropout"], config["activation"])
        initial_hash = state_dict_sha256(model.state_dict())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph, labels = graph.to(device), labels.to(device)
        masks = {name: value.to(device) for name, value in masks.items()}
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
        model.train()
        logits = model(graph)
        loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = gradient_record(model)
        optimizer.step()
        result = {
            "contract": strict_probe_contract(),
            "strict_settings": settings,
            "rng_state_after_strict_setup": rng_hashes,
            "config_sha256": file_sha256(config_path),
            "input_hashes": hashes,
            "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
            "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
            "initial_model_state_dict_sha256": initial_hash,
            "first_forward_logits_sha256": tensor_sha256(logits),
            "first_loss": float(loss.item()),
            "gradients": gradients,
            "after_first_adam_step_state_dict_sha256": state_dict_sha256(model.state_dict()),
            "checkpoint_saved": False,
            "validation_or_test_metrics_computed": False,
        }
        write_json(output_dir / "strict_probe.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        write_json(output_dir / "error.json", {
            "contract": strict_probe_contract(),
            "strict_settings_at_error": deterministic_settings(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "checkpoint_saved": False,
            "validation_or_test_metrics_computed": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_probe(args.config, args.output_dir)


if __name__ == "__main__":
    main()
