from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC


THRESHOLDS = np.arange(0.05, 1.0, 0.05)


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Choose F1-Macro threshold using validation data only."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    values = [(float(t), f1_score(labels, (scores >= t).astype(int), average="macro", zero_division=0))
              for t in THRESHOLDS]
    return max(values, key=lambda item: (item[1], -item[0]))[0]


class FeatureSVMProtocol:
    """Feature-only SVM; graph, edges and graph masks are never passed to SVC."""

    def __init__(self, random_state: int = 0, C: float = 1.0, kernel: str = "rbf"):
        self.random_state = int(random_state)
        self.C = float(C)
        self.kernel = kernel
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def fit_evaluate(self, features: np.ndarray, labels: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
        features = np.asarray(features)
        labels = np.asarray(labels, dtype=int)
        train = np.asarray(masks["train"], dtype=bool)
        val = np.asarray(masks["val"], dtype=bool)
        test = np.asarray(masks["test"], dtype=bool)
        if np.any(train & val) or np.any(train & test) or np.any(val & test):
            raise ValueError("train/val/test masks overlap")
        self.model.fit(features[train], labels[train])
        val_scores = self.model.predict_proba(features[val])[:, 1]
        threshold = choose_threshold(val_scores, labels[val])
        test_scores = self.model.predict_proba(features[test])[:, 1]
        test_pred = (test_scores >= threshold).astype(int)
        return {
            "method": "SVM",
            "edge_access": "none",
            "kernel": self.kernel,
            "C": self.C,
            "class_weight": "balanced",
            "train_count": int(train.sum()),
            "val_count": int(val.sum()),
            "test_count": int(test.sum()),
            "threshold": float(threshold),
            "f1_macro": float(f1_score(labels[test], test_pred, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(labels[test], test_scores)),
            "predicted_anomaly_count": int(test_pred.sum()),
            "test_anomaly_count": int(labels[test].sum()),
        }
