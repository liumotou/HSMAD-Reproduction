"""Read-only strict-determinism GraphSAGE checkpoint recomputation utility."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import dgl
import torch
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods" / "graphsage" / "src"))

from model import GraphSAGEGADBench  # noqa: E402
from run_formal import configure_strict_determinism  # noqa: E402


def recompute_contract():
    return {"checkpoint_read_only": True, "backward": False, "optimizer_step": False, "threshold_search": False, "test_mask_only": True}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(seed_dir):
    seed_dir = Path(seed_dir).resolve()
    output_dir = seed_dir / "audit_recompute"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((seed_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    if not config.get("strict_determinism"):
        raise RuntimeError("This utility requires a strict-determinism formal configuration")
    settings = configure_strict_determinism(int(config["seed"]))
    checkpoint_path = seed_dir / "checkpoint_validation_auprc_best.pt"
    raw_graph = dgl.load_graphs(str(ROOT / config["dataset_file"]))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    graph.ndata["feature"] = raw_graph.ndata["feature"]
    labels, test_mask = raw_graph.ndata["label"].long(), raw_graph.ndata["test_mask"].bool()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEGADBench(raw_graph.ndata["feature"].shape[1], config["h_feats"], 2, config["num_layers"], config["aggregation"], config["dropout"], config["activation"]).to(device)
    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(graph.to(device)), dim=1)[:, 1].cpu()
    truth, score = labels[test_mask].numpy(), probabilities[test_mask].numpy()
    threshold = float(metrics["threshold"])
    prediction = score >= threshold
    observed = {"f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)), "auroc": float(roc_auc_score(truth, score)),
                "threshold": threshold, "predicted_anomaly_count": int(prediction.sum()), "checkpoint_sha256": sha256(checkpoint_path)}
    original = {"f1_macro": metrics["f1_macro"], "auroc": metrics["auroc"], "threshold": metrics["threshold"],
                "predicted_anomaly_count": metrics["test_metrics"]["predicted_anomaly_count"], "checkpoint_sha256": metrics["checkpoint_sha256"]}
    comparison = {name: {"original": original[name], "recomputed": observed[name], "matches": original[name] == observed[name]} for name in original}
    comparison.update({"status": "recompute_match" if all(item["matches"] for item in comparison.values()) else "recompute_mismatch",
                       "seed": config["seed"], "checkpoint_epoch": int(saved["epoch"]), "strict_settings": settings, "contract": recompute_contract(),
                       "test_label_anomaly_count": int(truth.sum()), "test_mask_count": int(test_mask.sum())})
    (output_dir / "compare_to_original.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "input_sha256.json").write_text(json.dumps({"checkpoint": observed["checkpoint_sha256"], "metrics": sha256(seed_dir / "metrics.json"), "config": sha256(seed_dir / "config_snapshot.json"), "script": sha256(__file__)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True)
    recompute(parser.parse_args().seed_dir)


if __name__ == "__main__":
    main()
