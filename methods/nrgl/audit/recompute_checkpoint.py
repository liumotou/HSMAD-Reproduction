"""Read-only checkpoint recomputation for NRGL candidate artifacts."""

import hashlib
import json
from pathlib import Path

import torch

try:
    from methods.nrgl.src.data import load_frozen_dgl_graph
    from methods.nrgl.src.model import NRGLCore
    from methods.nrgl.src.runner import ROOT, prepare_training_graph
    from methods.nrgl.src.selection import compute_test_metrics, select_validation_checkpoint_metrics
except ModuleNotFoundError:  # Permit the contract suite to run from methods/nrgl.
    from src.data import load_frozen_dgl_graph
    from src.model import NRGLCore
    from src.runner import ROOT, prepare_training_graph
    from src.selection import compute_test_metrics, select_validation_checkpoint_metrics


METRIC_TOLERANCE = 1e-6


def compare_original_and_recomputed(original, recomputed, original_checkpoint_sha256, recomputed_checkpoint_sha256):
    metric_keys = ("f1_macro", "auroc", "validation_auprc")
    exact_keys = ("threshold", "predicted_anomaly_count", "test_anomaly_count")
    differences = {}
    for key in metric_keys:
        if original.get(key) is None or recomputed.get(key) is None:
            differences[key] = {"original": original.get(key), "recomputed": recomputed.get(key), "reason": "missing"}
        else:
            delta = abs(float(original[key]) - float(recomputed[key]))
            if delta > METRIC_TOLERANCE:
                differences[key] = {"original": original[key], "recomputed": recomputed[key], "abs_diff": delta}
    differences.update({key: {"original": original.get(key), "recomputed": recomputed.get(key)} for key in exact_keys if original.get(key) != recomputed.get(key)})
    if original_checkpoint_sha256 != recomputed_checkpoint_sha256:
        differences["checkpoint_sha256"] = {"original": original_checkpoint_sha256, "recomputed": recomputed_checkpoint_sha256}
    return {"status": "RECOMPUTE_MATCH_WITHIN_TOLERANCE" if not differences else "recompute_mismatch", "tolerance": METRIC_TOLERANCE, "differences": differences}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(artifact_directory):
    artifact = Path(artifact_directory)
    original = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "config_snapshot.json").read_text(encoding="utf-8"))
    checkpoint_path = artifact / "checkpoint_auprc_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    raw, _ = load_frozen_dgl_graph(config["dataset"])
    graph = prepare_training_graph(raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph.to(device)
    feature = raw.ndata["feature"].float().to(device)
    label = raw.ndata["label"].long().reshape(-1).to(device)
    test_mask = raw.ndata["test_mask"].bool().to(device)
    model = NRGLCore(graph.num_nodes(), feature.shape[1], config["hidden_dim"], config["order"], config["alpha"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    val_mask = raw.ndata["val_mask"].bool().to(device)
    with torch.no_grad():
        logits = model(graph, feature)
        validation = select_validation_checkpoint_metrics(logits, label, val_mask, config["threshold_candidates"])
        recomputed = compute_test_metrics(logits, label, test_mask, original["threshold"])
    recomputed["validation_auprc"] = validation["validation_auprc"]
    recomputed["threshold"] = original["threshold"]
    checkpoint_hash = sha256_file(checkpoint_path)
    comparison = compare_original_and_recomputed(original, recomputed, checkpoint_hash, checkpoint_hash)
    audit = artifact / "audit_recompute_tolerance"
    audit.mkdir(exist_ok=False)
    (audit / "recompute_metrics.json").write_text(json.dumps(recomputed, indent=2, sort_keys=True), encoding="utf-8")
    (audit / "compare_to_original.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    (audit / "audit_recompute.md").write_text(
        f"# NRGL checkpoint recomputation\n\nStatus: `{comparison['status']}`.\n\nCheckpoint SHA256: `{checkpoint_hash}`.\n",
        encoding="utf-8",
    )
    return comparison


if __name__ == "__main__":
    import sys
    print(json.dumps(recompute(sys.argv[1]), sort_keys=True))
