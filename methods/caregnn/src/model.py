"""Single-flattened-relation CARE-GNN candidate for frozen HSMAD graphs."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SingleRelationCareGNN(nn.Module):
    """One CARE layer with official label-aware top-p filtering semantics."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        lambda_1: float,
        step_size: float,
    ) -> None:
        super().__init__()
        self.num_relations = 1
        self.thresholds = [0.5]
        self.lambda_1 = float(lambda_1)
        self.step_size = float(step_size)
        self.relation_score_log: list[float] = []
        self.label_classifier = nn.Linear(feature_dim, num_classes)
        self.transform = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: list[list[int]],
        nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
        nodes = nodes.long()
        center_features = features[nodes]
        center_scores = self.label_classifier(center_features)
        aggregated = []
        relation_scores: list[list[float]] = []
        threshold = self.thresholds[0]

        for row, node in enumerate(nodes.detach().cpu().tolist()):
            neighbours = adjacency[node] or [node]
            neighbour_ids = torch.as_tensor(
                neighbours, dtype=torch.long, device=features.device
            )
            neighbour_features = features[neighbour_ids]
            neighbour_scores = self.label_classifier(neighbour_features)
            distances = torch.abs(center_scores[row, 0] - neighbour_scores[:, 0])
            keep = max(1, math.ceil(len(neighbours) * threshold))
            selected = torch.argsort(distances)[:keep]
            aggregated.append(F.relu(neighbour_features[selected].mean(dim=0)))
            relation_scores.append(distances[selected].detach().cpu().tolist())

        neighbour_features = torch.stack(aggregated)
        combined = F.relu(
            self.transform(center_features)
            + threshold * self.transform(neighbour_features)
        )
        logits = self.classifier(combined)
        return logits, center_scores, relation_scores

    def loss(
        self,
        logits: torch.Tensor,
        label_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(logits, labels.long()) + self.lambda_1 * F.cross_entropy(
            label_logits, labels.long()
        )

    def update_threshold(
        self,
        relation_scores: list[list[float]],
        batch_labels: torch.Tensor,
        batch_num: int,
    ) -> None:
        positive_rows = (batch_labels.long() == 1).nonzero(as_tuple=False).flatten()
        if positive_rows.numel() == 0:
            return
        selected = [relation_scores[int(index)] for index in positive_rows]
        flat = [value for row in selected for value in row]
        if not flat:
            return
        if (
            len(self.relation_score_log) % batch_num == 0
            and len(self.relation_score_log) >= 2 * batch_num
        ):
            previous = sum(self.relation_score_log[-2 * batch_num : -batch_num]) / batch_num
            current = sum(self.relation_score_log[-batch_num:]) / batch_num
            reward = 1 if previous - current >= 0 else -1
            updated = self.thresholds[0] + self.step_size * reward
            self.thresholds[0] = min(0.999, max(0.001, updated))
        self.relation_score_log.append(sum(flat) / len(flat))
