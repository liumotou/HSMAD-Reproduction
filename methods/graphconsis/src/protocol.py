"""Split-isolated evaluation protocol for the GraphConsis HSMAD candidate."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


THRESHOLDS = np.arange(0.05, 1.0, 0.05)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits[mask], labels[mask])


def select_validation_threshold(labels: torch.Tensor, probabilities: torch.Tensor, mask: torch.Tensor):
    y = labels[mask].detach().cpu().numpy()
    p = probabilities[mask].detach().cpu().numpy()
    scores = [(float(f1_score(y, p >= threshold, average="macro")), float(threshold)) for threshold in THRESHOLDS]
    score, threshold = max(scores, key=lambda item: (item[0], -item[1]))
    return threshold, score


def test_metrics(labels: torch.Tensor, probabilities: torch.Tensor, mask: torch.Tensor, threshold: float):
    y = labels[mask].detach().cpu().numpy()
    p = probabilities[mask].detach().cpu().numpy()
    prediction = (p >= threshold).astype(np.int64)
    return {
        "f1_macro": float(f1_score(y, prediction, average="macro")),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "predicted_anomaly_count": int(prediction.sum()),
        "actual_anomaly_count": int(y.sum()),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
