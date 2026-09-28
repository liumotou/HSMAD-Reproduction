"""Read-only re-evaluation of the saved GraphSAGE Weibo formal seed=0 checkpoint."""

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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    seed_dir = ROOT / "results/experiments/graphsage/weibo/graphsage_gadbench_h64_candidate/formal/seed_0"
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint_path = seed_dir / "checkpoint_validation_auprc_best.pt"
    raw_graph = dgl.load_graphs(str(ROOT / "datasets/weibo"))[0][0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw_graph)))
    graph.ndata["feature"] = raw_graph.ndata["feature"]
    labels = raw_graph.ndata["label"].long()
    test_mask = raw_graph.ndata["test_mask"].bool()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEGADBench(raw_graph.ndata["feature"].shape[1], 64, 2, 2, "pool", 0.0, "ReLU").to(device)
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
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(saved["epoch"]),
        "threshold_loaded_from_metrics": threshold,
        "test_mask_count": int(test_mask.sum()),
        "test_label_anomaly_count": int(test_labels.sum()),
        "predicted_anomaly_count": int(prediction.sum()),
        "f1_macro": float(f1_score(test_labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(test_labels, test_probabilities)),
        "probability_min": float(test_probabilities.min()),
        "probability_max": float(test_probabilities.max()),
        "probability_mean": float(test_probabilities.mean()),
        "matches_saved_f1_macro": float(f1_score(test_labels, prediction, average="macro", zero_division=0)) == float(metrics["f1_macro"]),
        "matches_saved_auroc": float(roc_auc_score(test_labels, test_probabilities)) == float(metrics["auroc"]),
        "matches_saved_predicted_anomaly_count": int(prediction.sum()) == int(metrics["test_metrics"]["predicted_anomaly_count"]),
    }
    output = ROOT / "methods/graphsage/audit/weibo_seed0_checkpoint_recompute.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recomputed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(recomputed, sort_keys=True))


if __name__ == "__main__":
    main()
