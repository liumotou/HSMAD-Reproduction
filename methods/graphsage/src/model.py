"""GADBench-equivalent full-graph GraphSAGE model for the isolated candidate."""

import torch.nn as nn
from dgl import nn as dglnn


class GraphSAGEGADBench(nn.Module):
    """Two-or-more-layer SAGEConv model matching GADBench's GraphSAGE class."""

    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        agg="pool",
        dropout_rate=0,
        activation="ReLU",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        act = getattr(nn, activation)()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=act))
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(h_feats, h_feats, agg, activation=act))
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata["feature"]
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        return self.output_linear(h)


def architecture_contract():
    return {
        "reference": "GADBench models/gnn.py::GraphSAGE",
        "aggregation": "pool",
        "sageconv_layers": 2,
        "activation": "ReLU",
        "hidden_dim": 64,
        "output": "Linear(64, 2 logits)",
        "edge_access": "graph_edges_required",
    }
