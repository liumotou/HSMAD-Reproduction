"""Leakage guards for the single-relation PC-GNN HSMAD adapter."""
from __future__ import annotations

import torch


def build_train_positive_nodes(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    if labels.shape[0] != train_mask.shape[0]:
        raise ValueError("labels and train_mask length mismatch")
    return torch.logical_and(train_mask.bool(), labels.long() == 1).nonzero(
        as_tuple=False
    ).flatten()


def labels_for_forward(
    labels: torch.Tensor,
    nodes: torch.Tensor,
    train_mask: torch.Tensor,
    train_flag: bool,
) -> torch.Tensor:
    nodes = nodes.long()
    if train_flag:
        if not bool(train_mask[nodes].all()):
            raise ValueError("training forward includes node outside frozen train_mask")
        return labels[nodes].long()
    return torch.zeros(nodes.numel(), dtype=torch.long, device=nodes.device)


def validate_single_relation(relations: list[object]) -> None:
    if len(relations) != 1:
        raise ValueError("HSMAD adapter requires exactly one flattened relation")
