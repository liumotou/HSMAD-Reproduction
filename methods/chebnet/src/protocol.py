"""Frozen-mask-only optimization and evaluation helpers for ChebNet."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor, class_weight: torch.Tensor | None) -> torch.Tensor:
    return functional.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)


def select_validation_threshold(labels: torch.Tensor, anomaly_probability: torch.Tensor, val_mask: torch.Tensor) -> tuple[float, float]:
    truth = labels[val_mask].detach().cpu().numpy()
    probabilities = anomaly_probability[val_mask].detach().cpu().numpy()
    candidates = [(threshold, f1_score(truth, probabilities > threshold, average='macro', zero_division=0)) for threshold in THRESHOLDS]
    return max(candidates, key=lambda item: item[1])


def test_metrics(labels: torch.Tensor, anomaly_probability: torch.Tensor, test_mask: torch.Tensor, threshold: float) -> dict[str, object]:
    truth = labels[test_mask].detach().cpu().numpy()
    probabilities = anomaly_probability[test_mask].detach().cpu().numpy()
    predicted = (probabilities > threshold).astype(np.int64)
    return {
        'count': int(len(truth)),
        'f1_macro': float(f1_score(truth, predicted, average='macro', zero_division=0)),
        'auroc': float(roc_auc_score(truth, probabilities)),
        'auprc': float(average_precision_score(truth, probabilities)),
        'predicted_anomaly_count': int(predicted.sum()),
        'actual_anomaly_count': int(truth.sum()),
        'confusion_matrix': confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }
