import torch.nn as nn
from dgl.nn import GraphConv


class KipfTwoLayerGCN(nn.Module):
    """Kipf-Welling topology: GCN(input, hidden, ReLU) then GCN(hidden, classes, logits)."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList([
            GraphConv(input_dim, hidden_dim, norm='both', weight=True, bias=False, activation=nn.ReLU(), allow_zero_in_degree=True),
            GraphConv(hidden_dim, output_dim, norm='both', weight=True, bias=False, activation=None, allow_zero_in_degree=True),
        ])

    def forward(self, graph):
        hidden = self.layers[0](graph, graph.ndata['feature'])
        hidden = self.dropout(hidden)
        return self.layers[1](graph, hidden)
