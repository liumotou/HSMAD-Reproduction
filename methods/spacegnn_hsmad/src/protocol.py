from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def _masked(labels, probabilities, mask):
    mask = mask.reshape(-1).bool()
    return (labels.reshape(-1)[mask].detach().cpu().numpy(),
            probabilities.reshape(-1)[mask].detach().cpu().numpy())


def validation_values(labels, probabilities, val_mask):
    y, p = _masked(labels, probabilities, val_mask)
    best_threshold, best_f1 = 0.05, -1.0
    for threshold in np.arange(0.05, 1.0, 0.05):
        value = f1_score(y, (p > threshold).astype(np.int64), average="macro")
        if value > best_f1:
            best_threshold, best_f1 = float(round(threshold, 2)), float(value)
    return best_threshold, best_f1, float(roc_auc_score(y, p))


def test_values(labels, probabilities, test_mask, threshold):
    y, p = _masked(labels, probabilities, test_mask)
    pred = (p > float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"f1_macro": float(f1_score(y, pred, average="macro")),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "predicted_anomaly_count": int(pred.sum()), "actual_anomaly_count": int(y.sum()),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
