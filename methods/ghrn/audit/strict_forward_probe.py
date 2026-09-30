"""Strict, checkpoint-only GHRN forward-repeat probe; never trains or saves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import dgl
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from methods.ghrn.src.model import GHRNModel
from methods.ghrn.src.protocol import setup_seed
from methods.ghrn.src.runner import edge_reduction_probability, load_frozen_hsmad_data, official_style_random_walk_update, prepare_training_graph


def strict_probe_contract():
    return {
        "required_cublas_workspace_config": ":4096:8", "required_pythonhashseed": "0",
        "torch_deterministic_algorithms": True, "warn_only": False,
        "backward": False, "optimizer_step": False, "checkpoint_save": False,
    }


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_from_run_dir(run_dir, config):
    if "seed" in config:
        return int(config["seed"])
    name = Path(run_dir).name
    if name.startswith("seed_"):
        return int(name.split("_", 1)[1])
    raise RuntimeError("Cannot determine seed from legacy run artifact")


def strict_setup(seed):
    contract = strict_probe_contract()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract["required_cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Python starts")
    if os.environ.get("PYTHONHASHSEED") != contract["required_pythonhashseed"]:
        raise RuntimeError("PYTHONHASHSEED must be 0 before Python starts")
    setup_seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def observation(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config_snapshot.json").read_text())
    strict_setup(seed_from_run_dir(run_dir, config))
    raw, features, _, _, _ = load_frozen_hsmad_data(config["dataset"])
    device = torch.device("cuda")
    graph = prepare_training_graph(raw).to(device)
    features = features.to(device)
    stage1 = GHRNModel(features.shape[1], int(config["hidden_dim"]), int(config["order"])).to(device)
    stage2 = GHRNModel(features.shape[1], int(config["hidden_dim"]), int(config["order"])).to(device)
    stage1.load_state_dict(torch.load(run_dir / "checkpoint_bootstrap_auprc_best.pt", map_location=device)["model_state_dict"])
    stage2.load_state_dict(torch.load(run_dir / "checkpoint_auprc_best.pt", map_location=device)["model_state_dict"])
    stage1.eval(); stage2.eval()
    with torch.no_grad():
        bootstrap = edge_reduction_probability(stage1(graph, features))
        reduced = official_style_random_walk_update(graph, bootstrap, float(config["delete_ratio"]))
        logits = stage2(reduced, features)
        probability = edge_reduction_probability(logits)[:, 1]
    torch.cuda.synchronize(device)
    src, dst = reduced.edges(order="eid")
    return {"bootstrap_sha256": tensor_hash(bootstrap), "reduced_edges_sha256": tensor_hash(torch.stack((src, dst))),
            "final_logits_sha256": tensor_hash(logits), "final_probability_sha256": tensor_hash(probability),
            "reduced_edges": int(reduced.num_edges())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, second = observation(args.run_dir), observation(args.run_dir)
    result = {"contract": strict_probe_contract(), "first": first, "second": second,
              "all_hashes_match": first == second}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
