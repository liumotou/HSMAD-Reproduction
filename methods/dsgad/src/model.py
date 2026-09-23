"""Traceable DSGAD topology from official commit bf7e0ac31a4d28c82c796b339d06b4a1a5308158."""
from __future__ import annotations

import dgl
import dgl.function as dgl_fn
import scipy.special
import sympy
import torch
from torch import nn


def beta_polynomial_coefficients(degree: int) -> list[list[float]]:
    coefficients = []
    x = sympy.symbols("x")
    for index in range(degree + 1):
        polynomial = sympy.poly(
            (x / 2) ** index * (1 - x / 2) ** (degree - index)
            / scipy.special.beta(index + 1, degree + 1 - index)
        )
        raw = polynomial.all_coeffs()
        coefficients.append([float(raw[degree - power]) for power in range(degree + 1)])
    return coefficients


def polynomial_convolution(theta, graph, features):
    def unnormalized_laplacian(value, degree_inverse_sqrt, local_graph):
        local_graph.ndata["h"] = value * degree_inverse_sqrt
        local_graph.update_all(dgl_fn.copy_u("h", "m"), dgl_fn.sum("m", "h"))
        return value - local_graph.ndata.pop("h") * degree_inverse_sqrt

    with graph.local_scope():
        degree_inverse_sqrt = graph.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(-1).to(features.device)
        output = theta[0] * features
        current = features
        for power in range(1, len(theta)):
            current = unnormalized_laplacian(current, degree_inverse_sqrt, graph)
            output = output + theta[power] * current
    return output


class DSGADModel(nn.Module):
    def __init__(self, in_nodes, in_feats, hidden_dim=64, num_classes=2, degree=2, mix_beta=2):
        super().__init__()
        self.thetas = beta_polynomial_coefficients(degree)
        self.num_filters = len(self.thetas)
        self.total_channels = self.num_filters + mix_beta
        self.input = nn.Sequential(
            nn.Linear(in_feats, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.channel_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(self.total_channels)
        ])
        self.channel_fusion = nn.Sequential(
            nn.Conv1d(self.total_channels, self.total_channels, 3, 1, padding="same"),
            nn.BatchNorm1d(self.total_channels), nn.ReLU(),
            nn.Conv1d(self.total_channels, self.total_channels, 3, 1, padding="same"),
            nn.BatchNorm1d(self.total_channels), nn.ReLU(), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.total_channels * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.weights = nn.Parameter(torch.randn(mix_beta, self.num_filters, in_nodes, hidden_dim))

    def forward(self, graph, features):
        hidden = self.input(features)
        if self.weights.shape[0] > 0:
            expanded = hidden.unsqueeze(0).unsqueeze(0).repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
            mixture_weights = (self.weights * expanded).sum(dim=(2, 3)).softmax(1)
        channels = [polynomial_convolution(theta, graph, hidden) for theta in self.thetas]
        if self.weights.shape[0] > 0:
            channels += [sum(channels[index] * row[index] for index in range(self.num_filters)) for row in mixture_weights]
        channels = [self.channel_mlps[index](channel).unsqueeze(1) for index, channel in enumerate(channels)]
        return self.classifier(self.channel_fusion(torch.cat(channels, dim=1)))
