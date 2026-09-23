"""Isolated 10-wide GAT-v2 model used only by the T-Social smoke probe."""
from torch import nn


class _FeedForward(nn.Module):
    def __init__(self, dim, drop_rate):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(dim, dim)
        self.dropout_2 = nn.Dropout(drop_rate)

    def forward(self, features):
        return self.dropout_2(self.linear_2(self.activation(self.dropout_1(self.linear_1(features)))))


class _GATModule(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super().__init__()
        if dim % num_heads:
            raise ValueError("total hidden dimension must divide evenly into heads")
        self.dim, self.num_heads, self.per_head_dim = dim, num_heads, dim // num_heads
        self.input_linear = nn.Linear(dim, dim)
        self.attn_linear_u = nn.Linear(dim, num_heads)
        self.attn_linear_v = nn.Linear(dim, num_heads, bias=False)
        self.attn_activation = nn.LeakyReLU(negative_slope=0.2)
        self.feed_forward = _FeedForward(dim, drop_rate)

    def forward(self, graph, features):
        from dgl import ops
        from dgl.nn.functional import edge_softmax
        features = self.input_linear(features)
        attention = edge_softmax(graph, self.attn_activation(ops.u_add_v(
            graph, self.attn_linear_u(features), self.attn_linear_v(features))))
        values = features.reshape(-1, self.per_head_dim, self.num_heads)
        return self.feed_forward(ops.u_mul_e_sum(graph, values, attention).reshape(-1, self.dim))


class _ResidualBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super().__init__()
        self.module = _GATModule(dim, num_heads, drop_rate)

    def forward(self, graph, features):
        return features + self.module(graph, features)


class TSocialGADBenchGATV2(nn.Module):
    """Linear -> GELU -> 2 x residual(GAT+FFN) -> two-class logits, width 10."""
    def __init__(self, input_dim, hidden_total_dim=10, num_heads=2, drop_rate=0.0, output_dim=2):
        super().__init__()
        if (hidden_total_dim, num_heads, output_dim) != (10, 2, 2):
            raise ValueError("T-Social smoke is frozen to total hidden=10, two heads, two logits")
        self.hidden_total_dim, self.num_heads = hidden_total_dim, num_heads
        self.per_head_dim = hidden_total_dim // num_heads
        self.input_linear = nn.Linear(input_dim, hidden_total_dim)
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([_ResidualBlock(hidden_total_dim, num_heads, drop_rate) for _ in range(2)])
        self.output_linear = nn.Linear(hidden_total_dim, output_dim)

    def forward(self, graph, features):
        features = self.activation(self.input_linear(features))
        for block in self.blocks:
            features = block(graph, features)
        return self.output_linear(features)
