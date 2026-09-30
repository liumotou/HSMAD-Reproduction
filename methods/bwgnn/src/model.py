"""GADBench-style BWGNN, adapted only as an isolated project module.

Reference: squareroot3/GADBench commit f9aa021ce9b6c6580427fb633b596843be76ddc6,
models/gnn.py::PolyConv and ::BWGNN.  This module does not read labels or masks.
"""
from __future__ import annotations

import sympy
import scipy.special
import dgl.function as fn
import torch
from torch import nn


def calculate_theta(order: int) -> list[list[float]]:
    """Return the Beta-wavelet polynomial coefficients used by GADBench BWGNN."""
    variable = sympy.symbols('x')
    thetas: list[list[float]] = []
    for branch in range(order + 1):
        polynomial = sympy.poly(
            (variable / 2) ** branch
            * (1 - variable / 2) ** (order - branch)
            / scipy.special.beta(branch + 1, order + 1 - branch)
        )
        coefficients = polynomial.all_coeffs()
        thetas.append([float(coefficients[order - index]) for index in range(order + 1)])
    return thetas


class PolyConv(nn.Module):
    """Parameter-free normalized-Laplacian polynomial filter."""
    def __init__(self, theta: list[float]):
        super().__init__()
        self.theta = theta

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        def unnormalized_laplacian(value: torch.Tensor, inverse_degree: torch.Tensor) -> torch.Tensor:
            graph.ndata['h'] = value * inverse_degree
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return value - graph.ndata.pop('h') * inverse_degree

        with graph.local_scope():
            inverse_degree = graph.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(-1).to(features.device)
            output = self.theta[0] * features
            current = features
            for coefficient in self.theta[1:]:
                current = unnormalized_laplacian(current, inverse_degree)
                output = output + coefficient * current
            return output


class ClassifierMLP(nn.Module):
    """GADBench MLP classifier semantics with no feature dropout when rate is zero."""
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int, dropout_rate: float, activation: str):
        super().__init__()
        if num_layers < 1:
            raise ValueError('mlp_layers must be at least one')
        layers = [nn.Linear(in_feats, num_classes)] if num_layers == 1 else [nn.Linear(in_feats, h_feats)]
        if num_layers > 1:
            layers.extend(nn.Linear(h_feats, h_feats) for _ in range(1, num_layers - 1))
            layers.append(nn.Linear(h_feats, num_classes))
        self.layers = nn.ModuleList(layers)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            if index:
                value = self.dropout(value)
            value = layer(value)
            if index != len(self.layers) - 1:
                value = self.activation(value)
        return value


class GADBenchBWGNN(nn.Module):
    """BWGNN with GADBench topology and a graph-only forward signature."""
    def __init__(
        self,
        in_feats: int,
        h_feats: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        mlp_layers: int = 2,
        dropout_rate: float = 0.0,
        activation: str = 'ReLU',
    ):
        super().__init__()
        self.thetas = calculate_theta(num_layers)
        self.convs = nn.ModuleList(PolyConv(theta) for theta in self.thetas)
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = ClassifierMLP(h_feats * len(self.convs), h_feats, num_classes, mlp_layers, dropout_rate, activation)

    def forward(self, graph):
        features = graph.ndata['feature']
        hidden = self.activation(self.linear(features))
        hidden = self.activation(self.linear2(hidden))
        filtered = [convolution(graph, hidden) for convolution in self.convs]
        return self.classifier(self.dropout(torch.cat(filtered, dim=-1)))
