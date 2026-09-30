"""DGL implementation of the selected two-layer original-GAT topology."""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

from contracts import selected_candidate_contract


class OriginalGAT8x8(nn.Module):
    """8 x 8 hidden heads (concat=64), then one two-logit output head."""

    def __init__(self, input_dim, feature_dropout=0.6, attention_dropout=0.6):
        super().__init__()
        contract = selected_candidate_contract()
        self.hidden = GATConv(
            input_dim,
            contract["hidden_features_per_head"],
            contract["hidden_heads"],
            feat_drop=feature_dropout,
            attn_drop=attention_dropout,
            negative_slope=contract["attention_negative_slope"],
            residual=False,
            activation=F.elu,
            allow_zero_in_degree=True,
            bias=True,
        )
        self.output = GATConv(
            contract["hidden_concat_dim"],
            contract["output_dim"],
            contract["output_heads"],
            feat_drop=feature_dropout,
            attn_drop=attention_dropout,
            negative_slope=contract["attention_negative_slope"],
            residual=False,
            activation=None,
            allow_zero_in_degree=True,
            bias=True,
        )

    def forward(self, graph, features):
        hidden = self.hidden(graph, features).flatten(1)
        return self.output(graph, hidden).squeeze(1)
