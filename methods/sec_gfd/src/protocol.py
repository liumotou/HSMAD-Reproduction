"""Frozen-mask-only loss, validation selection and final test metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

THRESHOLDS = tuple(round(step * 0.05, 2) for step in range(1, 20))


def contrastive_loss(embedding: torch.Tensor, features: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Official cosine-ratio auxiliary loss, restricted to frozen training nodes."""
    local_similarity = functional.cosine_similarity(features[train_mask], embedding[train_mask])
    local_labels = labels[train_mask]
    normal = local_similarity[local_labels == 0].mean()
    anomaly = local_similarity[local_labels == 1].mean()
    if torch.isnan(normal) or torch.isnan(anomaly) or anomaly.abs() < torch.finfo(anomaly.dtype).eps:
        raise RuntimeError("SEC-GFD contrastive loss is undefined for this frozen training split")
    return -torch.log(normal / anomaly)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor, class_weight: torch.Tensor, embedding: torch.Tensor | None = None, features: torch.Tensor | None = None, beta: float = 0.2) -> torch.Tensor:
    """Weighted CE on train nodes; optional official auxiliary term also uses train nodes only."""
    loss = functional.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
    if embedding is not None or features is not None:
        if embedding is None or features is None:
            raise ValueError("embedding and features must be provided together")
        loss = loss + beta * contrastive_loss(embedding, features, labels, train_mask)
    return loss


def validation_selection(labels: torch.Tensor, anomaly_probability: torch.Tensor, val_mask: torch.Tensor) -> tuple[float, float, float]:
    """Choose threshold and observe AUPRC using validation nodes only."""
    truth = labels[val_mask].detach().cpu().numpy()
    probability = anomaly_probability[val_mask].detach().cpu().numpy()
    options = [(threshold, f1_score(truth, probability > threshold, average="macro", zero_division=0)) for threshold in THRESHOLDS]
    threshold, f1_macro = max(options, key=lambda item: item[1])
    return threshold, float(f1_macro), float(average_precision_score(truth, probability))


def final_test_values(labels: torch.Tensor, anomaly_probability: torch.Tensor, test_mask: torch.Tensor, threshold: float) -> dict[str, object]:
    """One final, test-mask-only evaluation after checkpoint/threshold selection."""
    truth = labels[test_mask].detach().cpu().numpy()
    probability = anomaly_probability[test_mask].detach().cpu().numpy()
    predicted = (probability > threshold).astype(np.int64)
    return {
        "count": int(len(truth)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(truth, probability)),
        "auprc": float(average_precision_score(truth, probability)),
        "predicted_anomaly_count": int(predicted.sum()),
        "actual_anomaly_count": int(truth.sum()),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }
