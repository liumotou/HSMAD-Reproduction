"""Device-safe wrapper of the official GHRN/BWGNN spectral topology.

Reference: blacksingular/GHRN commit d47e047, BWGNN.py:14-142.  The only
mechanical adaptation is allocating the official temporary concatenation
tensor on the feature device; the original CPU allocation cannot concatenate
CUDA tensors.
"""
from __future__ import annotations

import scipy.special
import sympy
import torch
import dgl.function as fn
from torch import nn


def calculate_theta(order: int) -> list[list[float]]:
    variable = sympy.symbols("x")
    output: list[list[float]] = []
    for branch in range(order + 1):
        polynomial = sympy.poly(
            (variable / 2) ** branch
            * (1 - variable / 2) ** (order - branch)
            / scipy.special.beta(branch + 1, order + 1 - branch)
        )
        values = polynomial.all_coeffs()
        output.append([float(values[order - index]) for index in range(order + 1)])
    return output


class PolyConv(nn.Module):
    """Official normalized-Laplacian Bernstein branch, without labels/masks."""
    def __init__(self, theta: list[float]):
        super().__init__()
        self.theta = theta

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            degree = graph.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(-1).to(features.device)
            current = features
            output = self.theta[0] * current
            for coefficient in self.theta[1:]:
                graph.ndata["h"] = current * degree
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                current = current - graph.ndata.pop("h") * degree
                output = output + coefficient * current
            return output


class GHRNModel(nn.Module):
    """Official BWGNN topology used by GHRN, with a graph-only forward API."""
    def __init__(self, in_feats: int, hidden_dim: int = 64, order: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_feats, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.convs = nn.ModuleList(PolyConv(theta) for theta in calculate_theta(order))
        self.linear3 = nn.Linear(hidden_dim * len(self.convs), hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 2)
        self.activation = nn.ReLU()

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.linear(features))
        hidden = self.activation(self.linear2(hidden))
        branches = [branch(graph, hidden) for branch in self.convs]
        hidden = torch.cat(branches, dim=-1)
        hidden = self.activation(self.linear3(hidden))
        return self.linear4(hidden)
