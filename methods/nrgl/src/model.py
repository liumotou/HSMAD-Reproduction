"""Traceable NRGL official-core topology, without source split/noise/test logic."""

import dgl
import dgl.function as fn
import torch
from dgl.nn import EdgeWeightNorm
from torch import nn


class SGCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, order):
        super().__init__()
        self.order = order
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, graph, feature):
        values = [feature]
        current = torch.relu(self.linear(feature))
        with graph.local_scope():
            graph.ndata["h"] = current
            for _ in range(self.order):
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
                current = graph.ndata["h"]
                values.append(current)
        return torch.cat(values, dim=1)


class NRGLCore(nn.Module):
    """Two official-style SGC branches plus an edge discriminator and logits."""
    def __init__(self, num_nodes, input_dim, hidden_dim, order, alpha):
        super().__init__()
        self.num_nodes, self.alpha = num_nodes, alpha
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.edge_mlp = nn.Linear(2 * hidden_dim, 1)
        self.low_encoder = SGCEncoder(input_dim, hidden_dim, order)
        self.high_encoder = SGCEncoder(input_dim, hidden_dim, order)
        embedding_dim = input_dim + order * hidden_dim
        self.classifier = nn.Linear(2 * embedding_dim, 2)

    def _weighted_graphs(self, graph, feature):
        src, dst = graph.edges(order="eid")
        node = torch.relu(self.node_projection(feature))
        score = (self.edge_mlp(torch.cat((node[src], node[dst]), dim=1)).flatten() + self.edge_mlp(torch.cat((node[dst], node[src]), dim=1)).flatten()) / 2
        low = torch.sigmoid(score)
        high = 1 - low
        normalizer = EdgeWeightNorm(norm="both")
        low_graph, high_graph = graph.local_var(), graph.local_var()
        self_loop = src == dst
        low_input = torch.where(self_loop, torch.ones_like(low), low + 1e-10)
        high_input = torch.where(self_loop, torch.ones_like(high), high + 1e-10)
        low_weight = normalizer(low_graph, low_input)
        high_weight = normalizer(high_graph, high_input)
        low_graph.edata["w"] = torch.where(self_loop, torch.ones_like(low_weight), low_weight)
        high_graph.edata["w"] = torch.where(self_loop, torch.ones_like(high_weight), -self.alpha * high_weight)
        return low_graph, high_graph

    def forward(self, graph, feature):
        low_graph, high_graph = self._weighted_graphs(graph, feature)
        return self.classifier(torch.cat((self.low_encoder(low_graph, feature), self.high_encoder(high_graph, feature)), dim=1))
