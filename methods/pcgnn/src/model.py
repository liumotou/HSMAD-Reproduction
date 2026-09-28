"""Single-flattened-relation PC-GNN candidate for frozen HSMAD graphs."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SingleRelationPCGNN(nn.Module):
    """One Pick-Choose-Aggregate layer with train-only minority oversampling."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        train_positive_nodes: torch.Tensor,
        rho: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.num_relations = 1
        self.relation_threshold = 0.5
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.register_buffer("train_positive_nodes", train_positive_nodes.long().clone())
        self.label_classifier = nn.Linear(feature_dim, num_classes)
        self.intra_weight = nn.Parameter(torch.empty(2 * feature_dim, hidden_dim))
        self.inter_weight = nn.Parameter(torch.empty(feature_dim + hidden_dim, hidden_dim))
        self.classifier_weight = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.intra_weight)
        nn.init.xavier_uniform_(self.inter_weight)
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: list[list[int]],
        nodes: torch.Tensor,
        batch_labels: torch.Tensor | None = None,
        train_flag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = nodes.long()
        center_features = features[nodes]
        center_scores = self.label_classifier(center_features)
        positive_features = features[self.train_positive_nodes]
        positive_scores = self.label_classifier(positive_features)
        relation_embeddings = []

        if train_flag and batch_labels is None:
            raise ValueError("training forward requires train-only batch labels")
        if train_flag and batch_labels.numel() != nodes.numel():
            raise ValueError("batch label count mismatch")

        for row, node in enumerate(nodes.detach().cpu().tolist()):
            neighbours = adjacency[node] or [node]
            neighbour_ids = torch.as_tensor(neighbours, dtype=torch.long, device=features.device)
            neighbour_features = features[neighbour_ids]
            neighbour_scores = self.label_classifier(neighbour_features)
            distances = torch.abs(center_scores[row, 0] - neighbour_scores[:, 0])
            keep = max(1, math.ceil(len(neighbours) * self.relation_threshold))
            selected_positions = torch.argsort(distances)[:keep].detach().cpu().tolist()
            selected_nodes = [neighbours[position] for position in selected_positions]

            if train_flag and int(batch_labels[row]) == 1:
                oversample_count = int(keep * self.rho)
                if oversample_count:
                    positive_distances = torch.abs(center_scores[row, 0] - positive_scores[:, 0])
                    positive_positions = torch.argsort(positive_distances)[:oversample_count]
                    selected_nodes.extend(
                        self.train_positive_nodes[positive_positions].detach().cpu().tolist()
                    )
            selected_nodes = list(dict.fromkeys(selected_nodes))
            selected = torch.as_tensor(selected_nodes, dtype=torch.long, device=features.device)
            neighbour_mean = features[selected].mean(dim=0)
            intra = F.relu(torch.cat((center_features[row], neighbour_mean)) @ self.intra_weight)
            relation_embeddings.append(intra)

        relation_embedding = torch.stack(relation_embeddings)
        combined = F.relu(torch.cat((center_features, relation_embedding), dim=1) @ self.inter_weight)
        logits = combined @ self.classifier_weight.t()
        return logits, center_scores

    def loss(
        self,
        logits: torch.Tensor,
        label_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.long()
        return F.cross_entropy(logits, labels) + self.alpha * F.cross_entropy(
            label_logits, labels
        )
