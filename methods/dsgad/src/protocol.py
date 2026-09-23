"""Frozen-mask, no-test-leakage protocol helpers for DSGAD."""
from __future__ import annotations

import random

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

THRESHOLDS = [round(value * 0.05, 2) for value in range(1, 20)]


def setup_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed); dgl.seed(seed)


def prepare_training_graph(raw):
    return dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(raw, copy_ndata=True)))


def class_weight(labels, train_mask):
    selected = labels[train_mask]
    normal = int((selected == 0).sum())
    anomaly = int((selected == 1).sum())
    return torch.tensor([1.0, normal / anomaly], dtype=torch.float32, device=labels.device)


def masked_weighted_loss(logits, labels, train_mask):
    return F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight(labels, train_mask))


def validation_values(labels, probability, val_mask):
    truth = labels[val_mask].detach().cpu().numpy()
    score = probability[val_mask].detach().cpu().numpy()
    best_threshold, best_f1 = THRESHOLDS[0], -1.0
    for threshold in THRESHOLDS:
        value = f1_score(truth, score >= threshold, average="macro")
        if value > best_f1:
            best_threshold, best_f1 = threshold, float(value)
    return best_threshold, best_f1, float(average_precision_score(truth, score))


def test_values(labels, probability, test_mask, threshold):
    truth = labels[test_mask].detach().cpu().numpy()
    score = probability[test_mask].detach().cpu().numpy()
    predicted = (score >= threshold).astype("int64")
    return {
        "f1_macro": float(f1_score(truth, predicted, average="macro")),
        "auroc": float(roc_auc_score(truth, score)),
        "predicted_anomaly_count": int(predicted.sum()),
        "actual_anomaly_count": int(truth.sum()),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "count": int(len(truth)),
    }
