"""PyG ChebConv candidate, explicitly not author-exact HSMAD code."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import ChebConv


class ChebNetCandidate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, order: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim, K=order, normalization='sym')
        self.conv2 = ChebConv(hidden_dim, output_dim, K=order, normalization='sym')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        return self.conv2(x, edge_index)
