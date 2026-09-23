"""Validation-only checkpoint/threshold selection for the NRGL candidate."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _anomaly_probability(logits):
    return torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()


def _masked_arrays(logits, labels, mask):
    probabilities = _anomaly_probability(logits)
    mask_array = mask.detach().cpu().numpy().astype(bool)
    label_array = labels.detach().cpu().numpy()
    return label_array[mask_array], probabilities[mask_array]


def _safe_auroc(labels, probabilities):
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None


def _safe_auprc(labels, probabilities):
    return float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None


def select_validation_checkpoint_metrics(logits, labels, val_mask, threshold_candidates):
    """Return only validation-derived AUPRC/F1/threshold evidence."""
    val_labels, val_probabilities = _masked_arrays(logits, labels, val_mask)
    candidates = [float(value) for value in threshold_candidates]
    best_threshold = max(
        candidates,
        key=lambda value: (f1_score(val_labels, val_probabilities >= value, average="macro", zero_division=0), -value),
    )
    return {
        "validation_auroc": _safe_auroc(val_labels, val_probabilities),
        "validation_auprc": _safe_auprc(val_labels, val_probabilities),
        "validation_f1_macro": float(f1_score(val_labels, val_probabilities >= best_threshold, average="macro", zero_division=0)),
        "threshold": best_threshold,
    }


def should_replace_checkpoint(candidate_auprc, best_auprc):
    return candidate_auprc is not None and (best_auprc is None or candidate_auprc > best_auprc)


def compute_test_metrics(logits, labels, test_mask, threshold):
    """Compute final metrics on test only after checkpoint/threshold are fixed."""
    test_labels, test_probabilities = _masked_arrays(logits, labels, test_mask)
    predictions = (test_probabilities >= float(threshold)).astype(np.int64)
    return {
        "f1_macro": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        "auroc": _safe_auroc(test_labels, test_probabilities),
        "predicted_anomaly_count": int(predictions.sum()),
        "test_anomaly_count": int(test_labels.sum()),
    }
