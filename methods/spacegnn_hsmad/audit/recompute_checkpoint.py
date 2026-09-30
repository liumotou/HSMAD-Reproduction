from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import torch

from methods.dsgad.src.runner import load_data
from methods.spacegnn_hsmad.src.protocol import test_values, validation_values
from methods.spacegnn_hsmad.src.runner import build_model, prepare_spacegnn_graph, _probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    directory = Path(args.artifact_dir)
    original = json.loads((directory / "metrics.json").read_text())
    config = json.loads((directory / "config_snapshot.json").read_text())
    checkpoint_path = directory / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw, _ = load_data(config["dataset"])
    graph = prepare_spacegnn_graph(raw, config["layer_num"])
    masks = {name: graph.ndata[name].bool() for name in ("train_mask", "val_mask", "test_mask")}
    val_ids = torch.nonzero(masks["val_mask"], as_tuple=False).reshape(-1)
    test_ids = torch.nonzero(masks["test_mask"], as_tuple=False).reshape(-1)
    model = build_model(raw.ndata["feature"].shape[1], config["hidden_dim"], config["layer_num"],
                        config["dropout"], torch.zeros(config["layer_num"]), torch.zeros(config["layer_num"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda"); model = model.to(device)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    vp, vl = _probability(model, graph, sampler, val_ids, device, config["alpha"], config["beta"])
    threshold, validation_f1, validation_auroc = validation_values(vl, vp, torch.ones_like(vl, dtype=torch.bool))
    tp, tl = _probability(model, graph, sampler, test_ids, device, config["alpha"], config["beta"])
    recomputed = test_values(tl, tp, torch.ones_like(tl, dtype=torch.bool), checkpoint["threshold"])
    recomputed.update({"threshold": checkpoint["threshold"], "validation_threshold": threshold,
                       "validation_f1_macro": validation_f1, "validation_auroc": validation_auroc,
                       "best_epoch": checkpoint["best_epoch"]})
    diffs = {key: abs(float(recomputed[key]) - float(original[key])) for key in
             ("f1_macro", "auroc", "threshold", "validation_f1_macro", "validation_auroc", "best_epoch")}
    discrete = {key: recomputed[key] == original[key] for key in
                ("predicted_anomaly_count", "actual_anomaly_count", "confusion_matrix")}
    status = "recompute_match" if max(diffs.values()) <= args.tolerance and all(discrete.values()) else "recompute_mismatch"
    report = {"status": status, "tolerance": args.tolerance, "differences": diffs,
              "discrete_matches": discrete, "recomputed": recomputed, "original": original}
    output = directory / "audit_recompute"
    output.mkdir(exist_ok=False)
    (output / "recompute.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if status == "recompute_match" else 2)


if __name__ == "__main__":
    main()
