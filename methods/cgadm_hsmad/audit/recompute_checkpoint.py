"""Read-only independent recomputation of a saved CGADM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from methods.cgadm_hsmad.src.adapter import load_frozen_data
from methods.cgadm_hsmad.src.model import build_full_model
from methods.cgadm_hsmad.src.protocol import test_values
from methods.cgadm_hsmad.src.runner import _options, build_candidate_data, setup_seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    config_path = args.run_dir / "config.json"
    metrics_path = args.run_dir / "metrics.json"
    checkpoint_path = args.run_dir / "best_checkpoint.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original = json.loads(metrics_path.read_text(encoding="utf-8"))

    setup_seed(int(config["seed"]))
    _, graph = load_frozen_data(args.root / "datasets" / config["dataset"])
    data = build_candidate_data(graph)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_full_model(
        args.root / "methods" / "cgadm" / "official_snapshot",
        _options(config), data, device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.prior = checkpoint["prior"].to(device)
    model.eval()
    setup_seed(int(checkpoint["evaluation_seed"]))
    with torch.no_grad():
        scores = model.predict().detach().cpu()
    recomputed = test_values(graph.y, scores, graph.test_mask, float(checkpoint["threshold"]))
    recomputed.update(
        best_epoch=int(checkpoint["epoch"]),
        checkpoint_sha256=sha256(checkpoint_path),
        evaluation_seed=int(checkpoint["evaluation_seed"]),
    )

    keys = ["f1_macro", "auroc", "threshold", "predicted_anomaly_count", "actual_anomaly_count", "best_epoch", "checkpoint_sha256"]
    comparisons = {}
    match = True
    for key in keys:
        expected, actual = original[key], recomputed[key]
        if isinstance(expected, float):
            equal = abs(float(expected) - float(actual)) <= 1e-12
        else:
            equal = expected == actual
        comparisons[key] = {"original": expected, "recomputed": actual, "match": equal}
        match = match and equal

    result = {
        "status": "recompute_match" if match else "recompute_mismatch",
        "training_performed": False,
        "threshold_reselected": False,
        "test_mask_only": True,
        "inputs_sha256": {
            "config": sha256(config_path),
            "metrics": sha256(metrics_path),
            "checkpoint": sha256(checkpoint_path),
        },
        "recomputed": recomputed,
        "comparisons": comparisons,
    }
    (args.output_dir / "recompute_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "audit.md").write_text(
        "# CGADM checkpoint recomputation\n\n"
        f"- Status: `{result['status']}`\n"
        "- No backward, optimizer step, checkpoint selection, or threshold search was performed.\n"
        "- Metrics were computed only on the frozen test mask.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if not match:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
