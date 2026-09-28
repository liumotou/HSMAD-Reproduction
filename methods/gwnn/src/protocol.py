"""Mask-isolated evaluation helpers for the GWNN HSMAD candidate."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask])


def select_validation_threshold(
    labels: torch.Tensor, probabilities: torch.Tensor, val_mask: torch.Tensor
) -> tuple[float, float]:
    y_true = labels[val_mask].detach().cpu().numpy()
    scores = probabilities[val_mask].detach().cpu().numpy()
    candidates = [(threshold, f1_score(y_true, scores >= threshold, average="macro")) for threshold in THRESHOLDS]
    return max(candidates, key=lambda item: item[1])


def test_metrics(
    labels: torch.Tensor, probabilities: torch.Tensor, test_mask: torch.Tensor, threshold: float
) -> dict[str, object]:
    y_true = labels[test_mask].detach().cpu().numpy()
    scores = probabilities[test_mask].detach().cpu().numpy()
    prediction = scores >= threshold
    return {
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "predicted_anomaly_count": int(prediction.sum()),
        "actual_anomaly_count": int(y_true.sum()),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }
