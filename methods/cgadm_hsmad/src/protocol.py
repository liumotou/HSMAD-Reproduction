"""Mask-isolated selection and final evaluation for the CGADM candidate."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def masked_train_inputs(features: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return features[train_mask], labels[train_mask]


def _numpy_masked(labels: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor):
    y = labels[mask].detach().cpu().numpy()
    p = scores[mask].detach().cpu().numpy()
    return y, p


def validation_values(labels: torch.Tensor, scores: torch.Tensor, val_mask: torch.Tensor) -> Dict[str, float]:
    y, p = _numpy_masked(labels, scores, val_mask)
    thresholds = np.arange(0.05, 1.0, 0.05)
    f1s = [f1_score(y, p >= threshold, average="macro") for threshold in thresholds]
    best_index = int(np.argmax(f1s))
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "f1_macro": float(f1s[best_index]),
        "threshold": float(thresholds[best_index]),
    }


def test_values(labels: torch.Tensor, scores: torch.Tensor, test_mask: torch.Tensor, threshold: float) -> Dict[str, float]:
    y, p = _numpy_masked(labels, scores, test_mask)
    pred = p >= threshold
    return {
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "auroc": float(roc_auc_score(y, p)),
        "threshold": float(threshold),
        "predicted_anomaly_count": int(pred.sum()),
        "actual_anomaly_count": int(y.sum()),
    }

