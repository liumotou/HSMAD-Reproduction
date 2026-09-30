"""DGL adaptation of the official SparseGAD GPR-attention topology.

The source model's only PyG utility is edge-index reordering for symmetric
attention.  This version performs that exact endpoint ordering with torch so
the candidate remains runnable in the project's existing DGL environment.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import dgl.function as fn


class _WeightedConv(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, graph, features: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata['h'] = self.linear(features)
            graph.edata['w'] = edge_weight
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            return graph.ndata['h']


class _Extractor(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, width), nn.Dropout(dropout), nn.ReLU(), nn.Linear(width, width))
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, features: torch.Tensor, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        return self.cosine(self.network(features[source]), self.network(features[destination]))


class SparseGADModel(nn.Module):
    """Official GPR_ATT-style classifier without labels/masks in forward."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 dropout: float, dropout_adj: float) -> None:
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList(_WeightedConv(hidden_dim) for _ in range(num_layers))
        alpha = 0.1
        values = alpha * (1 - alpha) ** torch.arange(num_layers + 1, dtype=torch.float32)
        values[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(values)
        self.extractor = _Extractor(hidden_dim, dropout=0.2)
        self.dropout = float(dropout)
        self.dropout_adj = float(dropout_adj)

    @staticmethod
    def _normalized_weights(graph) -> torch.Tensor:
        source, destination = graph.edges(order='eid')
        out_degree = graph.out_degrees().to(dtype=torch.float32).clamp_min(1)
        return out_degree[source].rsqrt() * out_degree[destination].rsqrt()

    @staticmethod
    def _symmetric_attention(graph, attention: torch.Tensor) -> torch.Tensor:
        source, destination = graph.edges(order='eid')
        node_count = graph.num_nodes()
        edge_key = source * node_count + destination
        reverse_key = destination * node_count + source
        ordered_key, order = torch.sort(edge_key)
        locations = torch.searchsorted(ordered_key, reverse_key)
        reverse_attention = attention[order[locations]]
        return (attention + reverse_attention) / 2

    def _propagate(self, graph, hidden: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        result = hidden * self.temp[0]
        for index, layer in enumerate(self.layers):
            hidden = F.relu(layer(graph, hidden, edge_weight))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            result = result + hidden * self.temp[index + 1]
        return result

    def forward(self, graph, features):
        hidden = self.input_linear(features)
        base_weight = self._normalized_weights(graph)
        embeddings = self._propagate(graph, hidden, F.dropout(base_weight, p=self.dropout_adj, training=self.training))
        source, destination = graph.edges(order='eid')
        attention = self._symmetric_attention(graph, self.extractor(embeddings, source, destination))
        final = self._propagate(graph, hidden, F.dropout(base_weight * attention, p=self.dropout_adj, training=self.training))
        return self.output_linear(final)
