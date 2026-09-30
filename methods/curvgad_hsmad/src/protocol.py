"""Frozen-mask selection and final-evaluation helpers for CurvGAD-HSMAD."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 100, 5))


def masked_train_loss(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[train_mask], labels[train_mask])


def validation_values(labels, probabilities, val_mask, thresholds=THRESHOLDS) -> dict:
    y = labels[val_mask].detach().cpu().numpy()
    p = probabilities[val_mask].detach().cpu().numpy()
    candidates = []
    for threshold in thresholds:
        prediction = (p >= threshold).astype(np.int64)
        candidates.append((f1_score(y, prediction, average="macro"), float(threshold)))
    f1_macro, threshold = max(candidates, key=lambda item: (item[0], -item[1]))
    return {
        "count": int(len(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "f1_macro": float(f1_macro),
        "threshold": float(threshold),
    }


def test_values(labels, probabilities, test_mask, *, threshold: float) -> dict:
    y = labels[test_mask].detach().cpu().numpy()
    p = probabilities[test_mask].detach().cpu().numpy()
    prediction = (p >= threshold).astype(np.int64)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "count": int(len(y)),
        "f1_macro": float(f1_score(y, prediction, average="macro")),
        "auroc": float(roc_auc_score(y, p)),
        "threshold": float(threshold),
        "predicted_anomaly_count": int(prediction.sum()),
        "actual_anomaly_count": int(y.sum()),
        "confusion_matrix": matrix.tolist(),
    }
