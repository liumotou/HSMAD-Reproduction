"""Checkpoint-only recomputation for the isolated Kipf GCN-v2 runner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare_scalar(original: object, recomputed: object) -> dict[str, object]:
    if original == recomputed:
        return {"kind": "exact", "original": original, "recomputed": recomputed}
    if isinstance(original, float) and isinstance(recomputed, float) and math.isclose(
        original, recomputed, rel_tol=0.0, abs_tol=1e-12
    ):
        return {"kind": "roundoff", "original": original, "recomputed": recomputed}
    return {"kind": "mismatch", "original": original, "recomputed": recomputed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--audit-name", default="audit_recompute")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    root = Path("/root/autodl-tmp/HSMAD")

    import dgl
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    import sys

    sys.path.insert(0, str(root / "methods/gcn/src"))
    sys.path.insert(0, str(root / "methods/mlp/src"))
    from kipf_two_layer import KipfTwoLayerGCN
    from run_kipf_v2 import best_threshold

    config = json.loads((run_dir / "config_snapshot.json").read_text())
    original = json.loads((run_dir / "metrics.json").read_text())
    raw = dgl.load_graphs(str(root / "datasets" / config["dataset"]))[0][0]
    x = raw.ndata["feature"].float().contiguous()
    y = raw.ndata["label"].long().reshape(-1).contiguous()
    masks = {key: raw.ndata[key].bool().reshape(-1).contiguous() for key in ("train_mask", "val_mask", "test_mask")}
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw)))
    graph.ndata["feature"] = x
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph, y = graph.to(device), y.to(device)
    masks = {key: value.to(device) for key, value in masks.items()}
    model = KipfTwoLayerGCN(config["input_dim"], config["hidden_dim"], config["output_dim"], config["dropout"]).to(device)
    checkpoint = torch.load(run_dir / "checkpoint_val_auprc_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph), 1)[:, 1].cpu().numpy()
    val_mask = masks["val_mask"].cpu().numpy()
    val_y = y[masks["val_mask"]].cpu().numpy()
    _, validation_threshold = best_threshold(f1_score, val_y, probability[val_mask], config["threshold_candidates"])
    test_mask = masks["test_mask"].cpu().numpy()
    test_y = y[masks["test_mask"]].cpu().numpy()
    test_probability = probability[test_mask]
    prediction = (test_probability > float(original["threshold"])).astype(np.int64)
    recomputed = {
        "f1_macro": float(f1_score(test_y, prediction, average="macro")),
        "auroc": float(roc_auc_score(test_y, test_probability)),
        "auprc": float(average_precision_score(test_y, test_probability)),
        "threshold": float(original["threshold"]),
        "validation_threshold": validation_threshold,
        "predicted_anomaly_count": int(prediction.sum()),
        "actual_anomaly_count": int(test_y.sum()),
    }
    comparison = {key: compare_scalar(original.get(key), recomputed.get(key)) for key in (
        "f1_macro", "auroc", "threshold", "predicted_anomaly_count", "actual_anomaly_count"
    )}
    comparison["validation_threshold"] = compare_scalar(original.get("threshold"), validation_threshold)
    kinds = [item["kind"] for item in comparison.values()]
    status = "recompute_match" if all(kind == "exact" for kind in kinds) else (
        "recompute_roundoff_only" if all(kind in {"exact", "roundoff"} for kind in kinds) else "recompute_mismatch"
    )
    output = run_dir / args.audit_name
    output.mkdir(exist_ok=False)
    (output / "recompute_metrics.json").write_text(json.dumps(recomputed, indent=2, sort_keys=True))
    (output / "compare_to_original.json").write_text(json.dumps({"status": status, "comparison": comparison}, indent=2, sort_keys=True))
    (output / "audit.md").write_text(f"# GCN v2 checkpoint recomputation\n\nStatus: `{status}`.\n")
    print(json.dumps({"status": status, "comparison": comparison}, sort_keys=True))


if __name__ == "__main__":
    main()
