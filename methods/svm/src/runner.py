"""SVM runner for frozen HSMAD graphs (feature-only, no graph access in model)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from dgl.data.utils import load_graphs

from protocol import FeatureSVMProtocol


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_mask(root: Path, dataset: str, split: str) -> torch.Tensor:
    candidates = [
        root / "results" / "splits" / f"{dataset}_{split}_mask.pt",
        root / "results" / "splits" / dataset / f"{split}_mask.pt",
        root / "results" / "splits" / f"{dataset}_{split}.pt",
    ]
    for p in candidates:
        if p.exists():
            return torch.load(p, map_location="cpu")
    raise FileNotFoundError(f"frozen {split} mask not found for {dataset}")


def frozen_masks_from_graph(graph) -> dict[str, torch.Tensor]:
    """Read the already-persisted HSMAD split; never generate a split."""
    required = ("train_mask", "val_mask", "test_mask")
    missing = [name for name in required if name not in graph.ndata]
    if missing:
        raise KeyError(f"persisted frozen mask fields missing: {missing}")
    return {name.removesuffix("_mask"): torch.as_tensor(graph.ndata[name], dtype=torch.bool) for name in required}


def sha256_tensor(value) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return sha256_bytes(header + tensor.numpy().tobytes())


def build_preflight(graph) -> dict:
    """Describe frozen inputs without reading graph edges in the SVM model."""
    masks = frozen_masks_from_graph(graph)
    return {
        "passed": True,
        "nodes": int(graph.num_nodes()),
        "stored_edges": int(graph.num_edges()),
        "edge_access": "none",
        "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "hashes": {
            "feature_sha256": sha256_tensor(graph.ndata["feature"]),
            "label_sha256": sha256_tensor(graph.ndata["label"]),
            **{f"{name}_mask_sha256": sha256_tensor(mask) for name, mask in masks.items()},
        },
    }


def write_run_artifacts(output_dir: Path, config: dict, preflight: dict, metrics: dict) -> None:
    """Create a non-overwriting result directory for one isolated run."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    for filename, payload in (
        ("config_snapshot.json", config),
        ("preflight.json", preflight),
        ("metrics.json", metrics),
    ):
        (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fit_with_protocol(features: np.ndarray, labels: np.ndarray, masks: dict[str, np.ndarray], seed: int) -> tuple[dict, FeatureSVMProtocol]:
    protocol = FeatureSVMProtocol(random_state=seed)
    return protocol.fit_evaluate(features, labels, masks), protocol


def run(root: Path, dataset: str, seed: int, return_protocol: bool = False):
    graph_path = root / "datasets" / dataset
    graphs, _ = load_graphs(str(graph_path))
    graph = graphs[0]
    x = graph.ndata["feature"].detach().cpu().numpy()
    y = graph.ndata["label"].detach().cpu().numpy().reshape(-1).astype(int)
    masks = {s: value.detach().cpu().numpy().astype(bool) for s, value in frozen_masks_from_graph(graph).items()}
    result, protocol = fit_with_protocol(x, y, masks, seed)
    result.update({"dataset": dataset, "seed": seed, "data_sha256": sha256_bytes(Path(graph_path).read_bytes())})
    return (result, protocol) if return_protocol else result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-type", choices=("smoke", "diagnostic", "formal"), default="diagnostic")
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    executed = run(args.root, args.dataset, args.seed, return_protocol=args.output_dir is not None)
    result, protocol = executed if args.output_dir is not None else (executed, None)
    result.update({"run_type": args.run_type, "status": "smoke" if args.run_type == "smoke" else "OK"})
    if args.output_dir is not None:
        graph = load_graphs(str(args.root / "datasets" / args.dataset))[0][0]
        config = {
            "dataset": args.dataset,
            "seed": args.seed,
            "run_type": args.run_type,
            "protocol_status": "candidate_protocol_not_author_exact",
            "edge_access": "none",
            "training_unit": "single_SVC_fit_no_epochs",
        }
        preflight = build_preflight(graph)
        preflight["dataset_file_sha256"] = sha256_bytes((args.root / "datasets" / args.dataset).read_bytes())
        write_run_artifacts(args.output_dir, config, preflight, result)
        import joblib
        joblib.dump(protocol.model, args.output_dir / "checkpoint_svc.joblib")
    print(json.dumps(result, indent=2))
