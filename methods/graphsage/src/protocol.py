"""Shared, explicit GraphSAGE-GADBench-h64 evaluation primitives."""

import numpy as np
from sklearn.metrics import f1_score


def class_weight_from_train_labels(labels):
    """Return GADBench's [normal, anomaly] CE weights from training labels only."""
    normal_count = int((labels == 0).sum().item())
    anomaly_count = int((labels == 1).sum().item())
    if anomaly_count == 0:
        raise ValueError("Frozen training mask contains no anomaly labels")
    return [1.0, normal_count / anomaly_count], normal_count, anomaly_count


def threshold_grid():
    return [round(value / 100, 2) for value in range(5, 100, 5)]


def select_validation_f1_threshold(labels, probabilities):
    chosen_threshold, chosen_f1 = 0.05, -1.0
    for threshold in threshold_grid():
        candidate = f1_score(labels, probabilities >= threshold, average="macro", zero_division=0)
        if candidate > chosen_f1:
            chosen_threshold, chosen_f1 = threshold, float(candidate)
    return chosen_threshold, chosen_f1


def numpy_bool_mask(mask):
    return mask.detach().cpu().numpy().astype(bool, copy=False)


def numpy_labels(labels):
    return labels.detach().cpu().numpy()


def numpy_probabilities(probabilities):
    return probabilities.detach().cpu().numpy()
