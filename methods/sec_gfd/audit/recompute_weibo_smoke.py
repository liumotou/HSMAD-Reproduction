"""Read-only checkpoint recomputation for the isolated SEC-GFD Weibo smoke."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import dgl
import torch

from methods.sec_gfd.src.model import SECGFDModel
from methods.sec_gfd.src.protocol import final_test_values
from methods.sec_gfd.src.runner import ROOT, prepare_graph, sha256_tensor


ARTIFACT = ROOT / "results/experiments/sec_gfd/weibo/sec_gfd_hsmad_h64_candidate/smoke/seed_0"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads((ARTIFACT / "config_snapshot.json").read_text())
    original = json.loads((ARTIFACT / "metrics.json").read_text())
    checkpoint_path = ARTIFACT / "checkpoint_auprc_best.pt"
    raw = dgl.load_graphs(str(ROOT / config["dataset_file"]))[0][0]
    graph = prepare_graph(raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = raw.ndata["feature"].float().to(device)
    labels = raw.ndata["label"].long().reshape(-1).to(device)
    test_mask = raw.ndata["test_mask"].bool().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SECGFDModel(features.shape[1], config["hidden_dim"], 2, graph, config["order"], config["high_order"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(graph.to(device), features)[0], dim=1)[:, 1]
    recomputed = final_test_values(labels, probability, test_mask, original["threshold"])
    comparable = ("f1_macro", "auroc", "predicted_anomaly_count", "actual_anomaly_count", "confusion_matrix")
    differences = {field: {"original": original[field], "recomputed": recomputed[field]} for field in comparable if original[field] != recomputed[field]}
    audit_dir = ARTIFACT / "audit_recompute"
    audit_dir.mkdir(exist_ok=False)
    result = {
        "status": "recompute_match" if not differences else "recompute_mismatch",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "threshold": original["threshold"],
        "test_mask_sha256": sha256_tensor(test_mask),
        "original": {field: original[field] for field in comparable},
        "recomputed": recomputed,
        "differences": differences,
    }
    (audit_dir / "recompute_metrics.json").write_text(json.dumps(recomputed, indent=2, sort_keys=True))
    (audit_dir / "compare_to_original.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (audit_dir / "audit_recompute.md").write_text(
        "# SEC-GFD Weibo smoke checkpoint recomputation\n\n"
        f"Status: `{result['status']}`.\n\n"
        f"Checkpoint SHA256: `{result['checkpoint_sha256']}`.\n\n"
        f"Threshold: `{result['threshold']}` selected earlier on the validation mask and reused unchanged here.\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
