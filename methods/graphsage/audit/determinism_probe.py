"""One-step, independent-process determinism probe for Amazon GraphSAGE.

This script deliberately does not import or invoke the formal runner's execute
function.  It mirrors that runner's seed, loading, preprocessing, model,
loss, and Adam setup through one optimizer step, then stops without a
checkpoint, validation, or test evaluation.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "methods" / "graphsage" / "src"
sys.path.insert(0, str(SRC))

from model import GraphSAGEGADBench  # noqa: E402
from protocol import class_weight_from_train_labels  # noqa: E402
from run_smoke import file_sha256, graph_sha256, setup_seed, tensor_sha256, write_json  # noqa: E402


def probe_contract():
    return {
        "run_type": "determinism_probe",
        "seed": 0,
        "optimizer_steps": 1,
        "save_checkpoint": False,
        "compute_test_metrics": False,
    }


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state_dict[name]).encode("ascii"))
    return digest.hexdigest()


def rng_state_hashes():
    value = {
        "python_random_state_sha256": bytes_sha256(repr(random.getstate()).encode("utf-8")),
        "numpy_random_state_sha256": bytes_sha256(pickle.dumps(np.random.get_state(), protocol=4)),
        "torch_cpu_rng_state_sha256": tensor_sha256(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        value["torch_cuda_rng_state_sha256"] = [tensor_sha256(state) for state in torch.cuda.get_rng_state_all()]
    else:
        value["torch_cuda_rng_state_sha256"] = []
    return value


def determinism_settings():
    return {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def gradient_record(model):
    digest = hashlib.sha256()
    values = {}
    total_squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            values[name] = None
            continue
        item = tensor_sha256(parameter.grad)
        values[name] = item
        digest.update(name.encode("utf-8"))
        digest.update(item.encode("ascii"))
        total_squared_norm += float(parameter.grad.detach().float().pow(2).sum().item())
    return {
        "per_parameter_gradient_sha256": values,
        "all_gradients_sha256": digest.hexdigest(),
        "gradient_l2_norm": total_squared_norm ** 0.5,
    }


def run_probe(config_path, output_dir):
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty probe output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != "amazon" or config["h_feats"] != 64 or config["max_epoch"] != 200:
        raise RuntimeError("Probe requires the frozen Amazon h64 formal configuration")

    before_seed_settings = determinism_settings()
    # This is the existing formal-runner seed function; the probe adds no
    # deterministic flags or environment settings of its own.
    setup_seed(0)
    after_seed_rng = rng_state_hashes()
    after_seed_settings = determinism_settings()

    raw_path = ROOT / config["dataset_file"]
    raw_graph = dgl.load_graphs(str(raw_path))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    graph.ndata["feature"] = raw_graph.ndata["feature"]
    features = raw_graph.ndata["feature"]
    labels = raw_graph.ndata["label"].long()
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
    initial_state_sha256 = state_dict_sha256(model.state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, labels = graph.to(device), labels.to(device)
    masks = {name: value.to(device) for name, value in masks.items()}
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
    model.train()
    logits = model(graph)
    forward_logits_sha256 = tensor_sha256(logits)
    loss = F.cross_entropy(logits[masks["train_mask"]], labels[masks["train_mask"]], weight=weights)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = gradient_record(model)
    optimizer.step()
    after_step_state_sha256 = state_dict_sha256(model.state_dict())

    result = {
        "contract": probe_contract(),
        "config_sha256": file_sha256(config_path),
        "code_sha256": {
            "model_py": file_sha256(SRC / "model.py"),
            "protocol_py": file_sha256(SRC / "protocol.py"),
            "run_smoke_py": file_sha256(SRC / "run_smoke.py"),
            "probe_py": file_sha256(Path(__file__)),
        },
        "determinism_settings_before_seed": before_seed_settings,
        "determinism_settings_after_existing_setup_seed": after_seed_settings,
        "rng_state_after_existing_setup_seed": after_seed_rng,
        "input_hashes": hashes,
        "raw_graph": {"nodes": raw_graph.num_nodes(), "edges": raw_graph.num_edges()},
        "training_graph": {"nodes": graph.num_nodes(), "edges": graph.num_edges()},
        "mask_counts": {name: int(mask.sum().item()) for name, mask in masks.items()},
        "class_weight": class_weight,
        "train_normal_count": normal_count,
        "train_anomaly_count": anomaly_count,
        "initial_model_state_dict_sha256": initial_state_sha256,
        "first_forward_logits_sha256": forward_logits_sha256,
        "first_loss": float(loss.item()),
        "gradients": gradients,
        "after_first_adam_step_state_dict_sha256": after_step_state_sha256,
        "checkpoint_saved": False,
        "test_metrics_computed": False,
    }
    write_json(output_dir / "probe.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_probe(args.config, args.output_dir)


if __name__ == "__main__":
    main()
