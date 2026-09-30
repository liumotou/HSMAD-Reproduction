"""Auditable single-relation GraphConsis candidate runner for frozen HSMAD inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from methods.project_paths import project_root

ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from methods.graphconsis.src.model import GraphConsisSingleRelationCandidate
from methods.graphconsis.src.protocol import select_validation_threshold, test_metrics
from methods.graphconsis.src.scalable_adjacency import build_padded_adjacency_stable_csr


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int
    run_type: str
    result_dir: Path
    max_epoch: int
    patience: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_sha256"] = sha256_file(path)
    return config


def build_run_spec(config: dict[str, object]) -> RunSpec:
    if not config.get("_config_sha256"):
        raise ValueError("missing config provenance")
    if config["run_type"] not in {"smoke", "diagnostic", "formal"}:
        raise ValueError("unsupported run_type")
    return RunSpec(str(config["dataset"]), int(config["seed"]), str(config["run_type"]),
                   Path(str(config["result_dir"])), int(config["max_epoch"]), int(config["patience"]))


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def setup_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_graph(raw):
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    for name in ("feature", "label", "train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = raw.ndata[name]
    return graph


def build_padded_adjacency(source: torch.Tensor, destination: torch.Tensor, num_nodes: int,
                           max_degree: int, seed: int) -> torch.Tensor:
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in zip(source.tolist(), destination.tolist()):
        neighbours[src].append(dst)
    rng = np.random.default_rng(seed)
    table = np.empty((num_nodes, max_degree), dtype=np.int64)
    for node, values in enumerate(neighbours):
        if not values:
            values = [node]
        array = np.asarray(values, dtype=np.int64)
        table[node] = rng.choice(array, size=max_degree, replace=len(array) < max_degree)
    return torch.from_numpy(table)


def select_adjacency_builder(config: dict[str, object]):
    name = str(config.get("adjacency_builder", "legacy_python_lists"))
    if name == "legacy_python_lists":
        return name, build_padded_adjacency
    if name == "stable_csr":
        return name, build_padded_adjacency_stable_csr
    raise ValueError(f"unsupported adjacency_builder: {name}")


def evaluate_nodes(model, features, adjacency, nodes, batch_size: int, seed: int):
    model.eval(); outputs = []
    generator = torch.Generator(device=features.device).manual_seed(seed)
    with torch.no_grad():
        for start in range(0, nodes.numel(), batch_size):
            outputs.append(model(features, adjacency, nodes[start:start + batch_size], generator))
    return torch.cat(outputs)


def train(config: dict[str, object]) -> dict[str, object]:
    spec = build_run_spec(config); output = ROOT / spec.result_dir; ensure_new_output(output)
    setup_seed(spec.seed)
    raw_path = ROOT / str(config["dataset_file"]); raw = dgl.load_graphs(str(raw_path))[0][0]
    graph = prepare_training_graph(raw)
    observed = {"nodes": int(graph.num_nodes()), "training_edges": int(graph.num_edges())}
    if observed != dict(config["expected"]):
        raise RuntimeError(f"frozen input mismatch: {observed} != {config['expected']}")
    masks = {name: graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    if any(torch.logical_and(masks[a], masks[b]).any() for a, b in (("train_mask", "val_mask"), ("train_mask", "test_mask"), ("val_mask", "test_mask"))):
        raise RuntimeError("frozen masks overlap")
    source, destination = graph.edges(order="eid")
    adjacency_builder_name, adjacency_builder = select_adjacency_builder(config)
    adjacency = adjacency_builder(source.cpu(), destination.cpu(), graph.num_nodes(), int(config["max_degree"]), spec.seed)
    output.mkdir(parents=True); write_json(output / "config_snapshot.json", config)
    hashes = {
        "dataset_file_sha256": sha256_file(raw_path), "feature_sha256": sha256_tensor(graph.ndata["feature"]),
        "label_sha256": sha256_tensor(graph.ndata["label"]),
        **{f"{name}_sha256": sha256_tensor(value) for name, value in masks.items()},
        "model_py_sha256": sha256_file(ROOT / "methods/graphconsis/src/model.py"),
        "protocol_py_sha256": sha256_file(ROOT / "methods/graphconsis/src/protocol.py"),
        "runner_py_sha256": sha256_file(Path(__file__)), "config_sha256": str(config["_config_sha256"]),
    }
    write_json(output / "preflight.json", {
        "passed": True, "candidate_protocol_not_author_exact": True,
        "source_commit": config["source_commit"], "relation_policy": config["relation_policy"],
        "graph": observed, "feature_shape": list(graph.ndata["feature"].shape),
        "mask_counts": {k: int(v.sum()) for k, v in masks.items()}, "hashes": hashes,
        "model": {"aggregator": "mean_concat", "layers": 2, "hidden_flag": config["hidden_dim"],
                  "fanouts": config["fanouts"], "max_degree": config["max_degree"]},
        "edge_access": "sampled_graph_edges_required",
        "adjacency_builder": adjacency_builder_name,
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("audited CUDA environment required")
    torch.cuda.reset_peak_memory_stats(device)
    features = graph.ndata["feature"].float().to(device); labels = graph.ndata["label"].long().to(device)
    adjacency = adjacency.to(device)
    train_nodes = masks["train_mask"].nonzero(as_tuple=False).flatten().to(device)
    val_nodes = masks["val_mask"].nonzero(as_tuple=False).flatten().to(device)
    test_nodes = masks["test_mask"].nonzero(as_tuple=False).flatten().to(device)
    model = GraphConsisSingleRelationCandidate(features.shape[1], int(config["hidden_dim"]), 2, float(config["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    batch_size = int(config["batch_size"]); history=[]; best_epoch=0; best_auprc=-float("inf")
    checkpoint_path = output / "checkpoint_validation_auprc_best.pt"; started=time.monotonic()
    train_generator = torch.Generator(device=device).manual_seed(spec.seed)
    order_generator = torch.Generator(device=device).manual_seed(spec.seed)
    for epoch in range(1, spec.max_epoch + 1):
        model.train(); permutation = train_nodes[torch.randperm(train_nodes.numel(), generator=order_generator, device=device)]
        losses=[]
        for start in range(0, permutation.numel(), batch_size):
            batch = permutation[start:start + batch_size]; optimizer.zero_grad(set_to_none=True)
            logits = model(features, adjacency, batch, train_generator)
            loss = torch.nn.functional.cross_entropy(logits, labels[batch]); loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None: parameter.grad.clamp_(-5.0, 5.0)
            optimizer.step(); losses.append(float(loss.item()))
        val_logits = evaluate_nodes(model, features, adjacency, val_nodes, batch_size, spec.seed + 100000)
        val_prob = torch.softmax(val_logits, 1)[:, 1]; val_y = labels[val_nodes]
        global_prob = torch.zeros(graph.num_nodes(), device=device); global_prob[val_nodes] = val_prob
        threshold, val_f1 = select_validation_threshold(labels, global_prob, masks["val_mask"].to(device))
        auprc=float(average_precision_score(val_y.cpu().numpy(), val_prob.cpu().numpy()))
        record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_f1_macro":val_f1,
                "validation_auroc":float(roc_auc_score(val_y.cpu().numpy(),val_prob.cpu().numpy())),
                "validation_auprc":auprc,"validation_threshold":threshold,
                "peak_gpu_mb":float(torch.cuda.max_memory_allocated(device)/1024**2)}
        history.append(record); print(json.dumps(record,sort_keys=True),flush=True)
        if auprc > best_auprc:
            best_auprc=auprc; best_epoch=epoch
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"config":config},checkpoint_path)
        if epoch-best_epoch >= spec.patience: break
    checkpoint=torch.load(checkpoint_path,map_location=device); model.load_state_dict(checkpoint["model_state_dict"])
    val_logits=evaluate_nodes(model,features,adjacency,val_nodes,batch_size,spec.seed+100000)
    val_prob=torch.softmax(val_logits,1)[:,1]; global_prob=torch.zeros(graph.num_nodes(),device=device); global_prob[val_nodes]=val_prob
    threshold,validation_f1=select_validation_threshold(labels,global_prob,masks["val_mask"].to(device))
    test_logits=evaluate_nodes(model,features,adjacency,test_nodes,batch_size,spec.seed+200000)
    test_prob=torch.softmax(test_logits,1)[:,1]; global_test=torch.zeros(graph.num_nodes(),device=device); global_test[test_nodes]=test_prob
    result=test_metrics(labels,global_test,masks["test_mask"].to(device),threshold)
    metrics={"method":"GraphConsis","protocol_version":config["protocol_version"],"candidate_protocol_not_author_exact":True,
             "dataset":spec.dataset,"seed":spec.seed,"run_type":spec.run_type,"status":"smoke" if spec.run_type=="smoke" else "OK",
             "actual_epochs":len(history),"best_epoch":int(checkpoint["epoch"]),"threshold":threshold,"validation_f1_macro":validation_f1,
             **result,"wall_time_sec":time.monotonic()-started,"peak_gpu_mb":float(torch.cuda.max_memory_allocated(device)/1024**2),
             "edge_access":"sampled_graph_edges_required","hashes":hashes}
    write_json(output/"validation_history.json",history); write_json(output/"metrics.json",metrics)
    write_json(output/"artifact_sha256s.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    return metrics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    print("RUN_COMPLETE "+json.dumps(train(load_config(Path(args.config))),sort_keys=True),flush=True)


if __name__ == "__main__": main()
