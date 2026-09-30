import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from methods.cgadm_hsmad.src.adapter import load_frozen_data


DATASET = ROOT / "datasets" / "weibo"
OUTPUT = ROOT / "methods" / "cgadm_hsmad" / "audit" / "weibo_preflight.json"


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


graph, data = load_frozen_data(DATASET)
record = {
    "dataset_path": str(DATASET),
    "dataset_sha256": file_sha(DATASET),
    "nodes": graph.num_nodes(),
    "edges_after_hsmad_preprocess": graph.num_edges(),
    "feature_shape": list(data.x.shape),
    "label_anomaly_count": int(data.y.sum()),
    "mask_counts": {
        "train": int(data.train_mask.sum()),
        "val": int(data.val_mask.sum()),
        "test": int(data.test_mask.sum()),
    },
    "sha256": {
        "feature": tensor_sha(data.x),
        "label": tensor_sha(data.y),
        "train_mask": tensor_sha(data.train_mask),
        "val_mask": tensor_sha(data.val_mask),
        "test_mask": tensor_sha(data.test_mask),
        "edge_index": tensor_sha(data.edge_index),
    },
    "mask_disjoint": bool(
        not torch.any(data.train_mask & data.val_mask)
        and not torch.any(data.train_mask & data.test_mask)
        and not torch.any(data.val_mask & data.test_mask)
    ),
}
assert record["nodes"] == 8405
assert record["edges_after_hsmad_preprocess"] == 762947
assert record["mask_counts"] == {"train": 3362, "val": 1664, "test": 3379}
assert record["mask_disjoint"]
OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
