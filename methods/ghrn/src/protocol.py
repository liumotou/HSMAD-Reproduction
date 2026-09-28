"""Frozen-mask-only selection/evaluation helpers for the GHRN candidate."""
from __future__ import annotations

import random
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from methods.project_paths import project_root

THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(1, 20))
ROOT = project_root()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


class FrozenMaskProtocol:
    @staticmethod
    def mask_roles() -> dict[str, tuple[str, ...]]:
        return {
            'loss': ('train_mask',),
            'class_weight': ('train_mask',),
            'early_stop': ('val_mask',),
            'checkpoint': ('val_mask',),
            'threshold': ('val_mask',),
            'test_metrics': ('test_mask',),
        }


def class_weight(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train = labels[train_mask]
    normal = int((train == 0).sum())
    anomaly = int((train == 1).sum())
    if normal <= 0 or anomaly <= 0:
        raise ValueError('frozen train mask must contain both classes')
    return torch.tensor([1.0, normal / anomaly], dtype=torch.float32, device=labels.device)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(logits[train_mask], labels[train_mask], weight=weights)


def validation_values(labels: torch.Tensor, score: torch.Tensor, val_mask: torch.Tensor) -> tuple[float, float, float]:
    truth = labels[val_mask].detach().cpu().numpy()
    values = score[val_mask].detach().cpu().numpy()
    threshold, f1 = max(
        ((threshold, f1_score(truth, values > threshold, average='macro', zero_division=0)) for threshold in THRESHOLDS),
        key=lambda pair: pair[1],
    )
    return float(threshold), float(f1), float(average_precision_score(truth, values))


def final_test_values(labels: torch.Tensor, score: torch.Tensor, test_mask: torch.Tensor, threshold: float) -> dict[str, object]:
    truth = labels[test_mask].detach().cpu().numpy()
    values = score[test_mask].detach().cpu().numpy()
    predicted = (values > threshold).astype(np.int64)
    return {
        'count': int(len(truth)),
        'f1_macro': float(f1_score(truth, predicted, average='macro', zero_division=0)),
        'auroc': float(roc_auc_score(truth, values)),
        'auprc': float(average_precision_score(truth, values)),
        'predicted_anomaly_count': int(predicted.sum()),
        'actual_anomaly_count': int(truth.sum()),
        'confusion_matrix': confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }


def result_directory(dataset: str, run_type: str, seed: int, retry_id: str | None = None) -> Path:
    base = ROOT / 'results' / 'experiments' / 'ghrn' / dataset / 'ghrn_official_h64_frozen_mask_candidate' / run_type
    return base / retry_id / f'seed_{seed}' if retry_id else base / f'seed_{seed}'


def ensure_new_result_directory(path: str | Path) -> Path:
    value = Path(path)
    if value.exists():
        raise FileExistsError(value)
    value.mkdir(parents=True)
    return value
