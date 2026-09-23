"""SEC-GFD topology adapted from the fixed official source snapshot.

Reference: Sunxkissed/SEC-GFD commit 97faa51145ed1fbcbdc67cc5d399490da9a9797a,
model/SECGFD.py.  This module deliberately accepts only graph and features;
labels and masks remain in the training protocol.
"""
from __future__ import annotations

import scipy.special
import sympy
import torch
from torch import nn
import dgl
import dgl.function as fn
from dgl.nn import GraphConv


def calculate_theta(order: int) -> list[list[float]]:
    """Official Bernstein-band polynomial coefficients."""
    variable = sympy.symbols("x")
    coefficients: list[list[float]] = []
    for branch in range(order + 1):
        polynomial = sympy.poly(
            (variable / 2) ** branch
            * (1 - variable / 2) ** (order - branch)
            / scipy.special.beta(branch + 1, order + 1 - branch)
        )
        ordered = polynomial.all_coeffs()
        coefficients.append([float(ordered[order - index]) for index in range(order + 1)])
    return coefficients


class BandConv(nn.Module):
    """Parameter-free Bernstein-band filter from official ``SECGFD.py``."""
    def __init__(self, theta: list[float]) -> None:
        super().__init__()
        self.theta = theta

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            inv_degree = graph.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(-1).to(features.device)
            output = self.theta[0] * features
            current = features
            for coefficient in self.theta[1:]:
                graph.ndata["sec_gfd_h"] = current * inv_degree
                graph.update_all(fn.copy_u("sec_gfd_h", "m"), fn.sum("m", "sec_gfd_h"))
                current = current - graph.ndata["sec_gfd_h"] * inv_degree
                output = output + coefficient * current
            return output


class HighConv(nn.Module):
    """High-pass Laplacian branches from official ``SECGFD.py``."""
    def __init__(self, num_layer: int) -> None:
        super().__init__()
        self.num_layer = num_layer

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            inv_degree = graph.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(-1).to(features.device)
            current = features
            for _ in range(self.num_layer):
                graph.ndata["sec_gfd_h"] = current * inv_degree
                graph.update_all(fn.copy_u("sec_gfd_h", "m"), fn.sum("m", "sec_gfd_h"))
                current = current - graph.ndata["sec_gfd_h"] * inv_degree
            return current


class AuxiliaryGCN(nn.Module):
    """The official two-layer GCN embedding head."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.first = GraphConv(input_dim, hidden_dim)
        self.second = GraphConv(hidden_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        graph_without_loops = dgl.remove_self_loop(graph)
        return self.second(graph_without_loops, self.activation(self.first(graph_without_loops, features)))


class SECGFDModel(nn.Module):
    """Official SEC-GFD core topology with a label-free forward signature."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, graph: dgl.DGLGraph, order: int = 2, high_order: int = 2) -> None:
        super().__init__()
        del graph  # graph is supplied at forward time; no mutable label/mask state is retained.
        self.order = order
        self.high_order = high_order
        self.band_filters = nn.ModuleList(BandConv(theta) for theta in calculate_theta(order))
        self.high_filters = nn.ModuleList(HighConv(layer + 1) for layer in range(high_order))
        branch_count = len(self.band_filters) + len(self.high_filters)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim * branch_count, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)
        self.embedding = AuxiliaryGCN(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.linear1(features))
        hidden = self.activation(self.linear2(hidden))
        branches = [branch(graph, hidden) for branch in self.band_filters]
        branches.extend(branch(graph, hidden) for branch in self.high_filters)
        logits = self.linear4(self.activation(self.linear3(torch.cat(branches, dim=-1))))
        return logits, self.embedding(graph, features)
