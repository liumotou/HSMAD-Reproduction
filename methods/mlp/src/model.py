"""Feature-only MLP baseline.  This module has no graph dependency."""

import torch.nn as nn


class FeatureMLP(nn.Module):
    """Two-layer node classifier: feature -> hidden -> two logits."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features):
        return self.network(features)
