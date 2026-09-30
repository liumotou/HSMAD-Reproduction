from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def build_label_unknown(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    labels = labels.reshape(-1).long()
    train_mask = train_mask.reshape(-1).bool()
    if labels.numel() != train_mask.numel():
        raise ValueError("labels and train_mask must have the same number of nodes")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("PMP candidate requires binary labels encoded as 0/1")
    result = torch.full_like(labels, 2)
    result[train_mask] = labels[train_mask]
    return result


def masked_train_loss(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    mask = train_mask.reshape(-1).bool()
    if logits.shape[0] != labels.numel() or labels.numel() != mask.numel():
        raise ValueError("logits, labels and train_mask must align")
    return F.cross_entropy(logits[mask], labels.reshape(-1).long()[mask])


def validation_metrics(probabilities: torch.Tensor, labels: torch.Tensor, val_mask: torch.Tensor) -> dict:
    mask = val_mask.reshape(-1).bool()
    p = probabilities.reshape(-1)[mask].detach().cpu().numpy()
    y = labels.reshape(-1)[mask].detach().cpu().numpy()
    best_threshold, best_f1 = 0.05, -1.0
    for threshold in np.arange(0.05, 1.0, 0.05):
        score = f1_score(y, (p >= threshold).astype(np.int64), average="macro")
        if score > best_f1:
            best_threshold, best_f1 = float(round(threshold, 2)), float(score)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "f1_macro": best_f1,
        "threshold": best_threshold,
        "count": int(mask.sum()),
    }


def test_metrics(probabilities: torch.Tensor, labels: torch.Tensor, test_mask: torch.Tensor,
                 threshold: float) -> dict:
    mask = test_mask.reshape(-1).bool()
    p = probabilities.reshape(-1)[mask].detach().cpu().numpy()
    y = labels.reshape(-1)[mask].detach().cpu().numpy()
    pred = (p >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "auroc": float(roc_auc_score(y, p)),
        "threshold": float(threshold),
        "count": int(mask.sum()),
        "predicted_anomaly_count": int(pred.sum()),
        "actual_anomaly_count": int(y.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def frozen_config(dataset: str) -> dict:
    return {
        "dataset": dataset,
        "official_repository": "https://github.com/Xtra-Computing/PMP",
        "official_commit": "3f7629f6c180891a0bc1bba3c66d94d288a1ddae",
        "positioning": "candidate_protocol_not_author_exact",
        "input_adapter": "trainable_linear_input_dim_to_hidden_dim",
        "input_adapter_output_dim": 64,
        "input_adapter_rationale": "official_LASAGE_S_requires_input_width_not_greater_than_hidden_width",
        "hidden_dim": 64,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "n_layer": 1,
        "aggregator": "mean",
        "num_trans": 1,
        "full_neighbors": True,
        "weighted_loss": False,
        "checkpoint_metric": "validation_auroc",
        "threshold_protocol": "validation_F1_macro_grid_0.05_to_0.95",
        "graph_preprocess": ["to_bidirected", "remove_self_loop", "add_self_loop"],
        "feature_transform": "official_row_normalization",
    }
