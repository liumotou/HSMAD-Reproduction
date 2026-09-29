"""Isolated GADBench-style GAT-v2 model used only by this project version."""

from torch import nn

from methods.gat_v2_gadbench.src.contracts import architecture_contract


class FeedForwardModule(nn.Module):
    def __init__(self, dim, drop_rate):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(dim, dim)
        self.dropout_2 = nn.Dropout(drop_rate)

    def forward(self, features):
        features = self.linear_1(features)
        features = self.dropout_1(features)
        features = self.activation(features)
        features = self.linear_2(features)
        return self.dropout_2(features)


class GATModule(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("total hidden dimension must divide evenly into heads")
        self.dim = dim
        self.num_heads = num_heads
        self.per_head_dim = dim // num_heads
        self.input_linear = nn.Linear(dim, dim)
        self.attn_linear_u = nn.Linear(dim, num_heads)
        self.attn_linear_v = nn.Linear(dim, num_heads, bias=False)
        self.attn_activation = nn.LeakyReLU(negative_slope=0.2)
        self.feed_forward = FeedForwardModule(dim, drop_rate)

    def forward(self, graph, features):
        from dgl import ops
        from dgl.nn.functional import edge_softmax

        features = self.input_linear(features)
        score_u = self.attn_linear_u(features)
        score_v = self.attn_linear_v(features)
        scores = self.attn_activation(ops.u_add_v(graph, score_u, score_v))
        attention = edge_softmax(graph, scores)
        values = features.reshape(-1, self.per_head_dim, self.num_heads)
        values = ops.u_mul_e_sum(graph, values, attention).reshape(-1, self.dim)
        return self.feed_forward(values)


class ResidualGATBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super().__init__()
        self.module = GATModule(dim, num_heads, drop_rate)

    def forward(self, graph, features):
        return features + self.module(graph, features)


class GADBenchGATV2(nn.Module):
    """`Linear → GELU → 2 × residual(GAT+FFN) → Linear logits`."""

    def __init__(self, input_dim, hidden_total_dim=64, num_heads=4, drop_rate=0.0, output_dim=2):
        super().__init__()
        expected = architecture_contract()
        if (hidden_total_dim, num_heads, output_dim) != (
            expected["hidden_total_dim"], expected["num_heads"], expected["output_dim"]
        ):
            raise ValueError("This frozen smoke supports only the approved 64-wide four-head contract")
        self.input_linear = nn.Linear(input_dim, hidden_total_dim)
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([
            ResidualGATBlock(hidden_total_dim, num_heads, drop_rate),
            ResidualGATBlock(hidden_total_dim, num_heads, drop_rate),
        ])
        self.output_linear = nn.Linear(hidden_total_dim, output_dim)

    def forward(self, graph, features):
        features = self.activation(self.input_linear(features))
        for block in self.blocks:
            features = block(graph, features)
        return self.output_linear(features)
