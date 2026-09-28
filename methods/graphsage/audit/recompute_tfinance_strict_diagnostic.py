"""Read-only checkpoint recomputation for strict GraphSAGE T-Finance seed 0."""

import hashlib
import json
import os
import sys
from pathlib import Path

import dgl
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods" / "graphsage" / "src"))

from model import GraphSAGEGADBench  # noqa: E402
from run_formal import configure_strict_determinism  # noqa: E402


def recompute_contract():
    return {
        "checkpoint_read_only": True,
        "backward": False,
        "optimizer_step": False,
        "threshold_search": False,
        "test_mask_only": True,
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    seed_dir = ROOT / "results/experiments/graphsage/tfinance/graphsage_gadbench_h64_candidate_strict_determinism/diagnostic/seed_0"
    output_dir = seed_dir / "audit_recompute"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((seed_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint_path = seed_dir / "checkpoint_validation_auprc_best.pt"
    if not config.get("strict_determinism"):
        raise RuntimeError("Audit requires the strict-determinism diagnostic configuration")
    settings = configure_strict_determinism(int(config["seed"]))
    raw_graph = dgl.load_graphs(str(ROOT / config["dataset_file"]))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    graph.ndata["feature"] = raw_graph.ndata["feature"]
    labels = raw_graph.ndata["label"].long()
    test_mask = raw_graph.ndata["test_mask"].bool()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEGADBench(raw_graph.ndata["feature"].shape[1], config["h_feats"], 2, config["num_layers"], config["aggregation"], config["dropout"], config["activation"]).to(device)
    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(graph.to(device)), dim=1)[:, 1].cpu()
    test_labels = labels[test_mask].numpy()
    test_probabilities = probabilities[test_mask].numpy()
    threshold = float(metrics["threshold"])
    prediction = test_probabilities >= threshold
    recomputed = {
        "contract": recompute_contract(),
        "strict_settings": settings,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(saved["epoch"]),
        "threshold_loaded_from_metrics": threshold,
        "test_mask_count": int(test_mask.sum()),
        "test_label_anomaly_count": int(test_labels.sum()),
        "predicted_anomaly_count": int(prediction.sum()),
        "f1_macro": float(f1_score(test_labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(test_labels, test_probabilities)),
        "auprc": float(average_precision_score(test_labels, test_probabilities)),
        "probability_min": float(test_probabilities.min()),
        "probability_max": float(test_probabilities.max()),
        "probability_mean": float(test_probabilities.mean()),
    }
    original = {
        "f1_macro": metrics["f1_macro"],
        "auroc": metrics["auroc"],
        "threshold": metrics["threshold"],
        "predicted_anomaly_count": metrics["test_metrics"]["predicted_anomaly_count"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
    }
    observed = {
        "f1_macro": recomputed["f1_macro"],
        "auroc": recomputed["auroc"],
        "threshold": threshold,
        "predicted_anomaly_count": recomputed["predicted_anomaly_count"],
        "checkpoint_sha256": recomputed["checkpoint_sha256"],
    }
    comparison = {key: {"original": original[key], "recomputed": observed[key], "matches": original[key] == observed[key]} for key in original}
    comparison["status"] = "recompute_match" if all(item["matches"] for item in comparison.values()) else "recompute_mismatch"
    (output_dir / "recompute_metrics.json").write_text(json.dumps(recomputed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "compare_to_original.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "input_sha256.json").write_text(json.dumps({"checkpoint": sha256(checkpoint_path), "metrics": sha256(seed_dir / "metrics.json"), "config": sha256(seed_dir / "config_snapshot.json"), "script": sha256(__file__)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
