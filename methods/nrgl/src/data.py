"""Frozen HSMAD DGL-data access for the NRGL candidate."""

from pathlib import Path

import dgl


ROOT = Path("/root/autodl-tmp/HSMAD")
DATASETS = {
    "weibo": {"dataset_file": "datasets/weibo", "nodes": 8405, "feature_dim": 400},
    "amazon": {"dataset_file": "datasets/amazon", "nodes": 11944, "feature_dim": 25},
    "tolokers": {"dataset_file": "datasets/tolokers", "nodes": 11758, "feature_dim": 10},
    "tfinance": {"dataset_file": "datasets/tfinance", "nodes": 39357, "feature_dim": 10},
}


def frozen_dataset_contract(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    return {**DATASETS[dataset], "reuse_frozen_masks": True, "regenerate_masks": False, "inject_label_noise": False}


def load_frozen_dgl_graph(dataset):
    contract = frozen_dataset_contract(dataset)
    graph = dgl.load_graphs(str(ROOT / contract["dataset_file"]))[0][0]
    if graph.num_nodes() != contract["nodes"] or graph.ndata["feature"].shape[1] != contract["feature_dim"]:
        raise RuntimeError("frozen NRGL input contract mismatch")
    return graph, contract
