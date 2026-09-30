"""CLI for one-time exact CurvGAD curvature preprocessing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .dataset import load_preprocessed_frozen_graph
from .precompute import compute_or_load_exact_curvature, graph_topology_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--method", choices=["exact_orc"], default="exact_orc")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    graph = load_preprocessed_frozen_graph(args.dataset_path)
    preflight = {
        "alpha": args.alpha,
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path.resolve()),
        "graph_preprocess": ["dgl.to_bidirected", "dgl.remove_self_loop", "dgl.add_self_loop"],
        "graph_sha256": graph_topology_sha256(graph),
        "label_or_mask_access_for_curvature": "none",
        "method": args.method,
        "num_edges": int(graph.num_edges()),
        "num_nodes": int(graph.num_nodes()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps({"preflight": preflight}, sort_keys=True), flush=True)
    if args.preflight:
        args.preflight.parent.mkdir(parents=True, exist_ok=True)
        args.preflight.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact = compute_or_load_exact_curvature(
        graph,
        args.cache_dir,
        alpha=args.alpha,
        method=args.method,
    )
    result = {
        "cache_hit": artifact.cache_hit,
        "elapsed_sec": time.perf_counter() - started,
        "matrix_path": str(artifact.matrix_path.resolve()),
        "matrix_sha256": artifact.matrix_sha256,
        "metadata_path": str(artifact.metadata_path.resolve()),
        "status": "OK",
    }
    print(json.dumps({"result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
