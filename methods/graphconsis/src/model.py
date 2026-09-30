"""Single-relation HSMAD adaptation of the verified DGFraud GraphConsis core."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MeanConcatAggregator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, final: bool = False) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.neigh_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.final = final

    def forward(self, self_x: torch.Tensor, neigh_x: torch.Tensor) -> torch.Tensor:
        output = torch.cat(
            (self.self_linear(self_x), self.neigh_linear(neigh_x.mean(dim=1))), dim=1
        )
        return output if self.final else F.relu(output)


class GraphConsisSingleRelationCandidate(nn.Module):
    """Official GraphConsis sampling/aggregation semantics on one flattened relation.

    ``hidden_dim`` follows the official ``dim_1``/``dim_2`` flag semantics: each
    self and neighbour branch emits ``hidden_dim`` values and concatenation doubles it.
    """

    fanouts = (25, 10)

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.layer1 = MeanConcatAggregator(input_dim, hidden_dim, final=False)
        self.layer2 = MeanConcatAggregator(hidden_dim * 2, hidden_dim, final=True)
        relation_dim = hidden_dim * 2
        self.relation_augmented_dim = relation_dim * 2
        self.relation_vector = nn.Parameter(torch.empty(relation_dim))
        self.attention_vector = nn.Parameter(torch.empty(self.relation_augmented_dim, 1))
        self.classifier = nn.Linear(self.relation_augmented_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.relation_vector.unsqueeze(0))
        nn.init.xavier_uniform_(self.attention_vector)

    @staticmethod
    def _distance_sample(
        ids: torch.Tensor,
        count: int,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        candidates = adjacency[ids]
        distance = (features[candidates] - features[ids].unsqueeze(1)).square().sum(-1).sqrt()
        probability = torch.exp(-distance)
        probability = probability / probability.sum(1, keepdim=True).clamp_min(1e-12)
        probability = torch.where(probability > 0.001, probability, torch.zeros_like(probability))
        empty = probability.sum(1) == 0
        if empty.any():
            probability[empty] = 1.0 / probability.shape[1]
        positions = torch.multinomial(probability, count, replacement=True, generator=generator)
        return candidates.gather(1, positions)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        seeds: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Official recursive order: 10 one-hop neighbours, then 25 neighbours
        # for each one-hop node (layer-info flags are samples_1=25, samples_2=10).
        one_hop = self._distance_sample(seeds, 10, features, adjacency, generator)
        flat_one = one_hop.reshape(-1)
        two_hop = self._distance_sample(flat_one, 25, features, adjacency, generator)
        seed_h1 = self.layer1(features[seeds], features[one_hop])
        one_h1 = self.layer1(features[flat_one], features[two_hop])
        seed_h2 = self.layer2(seed_h1, one_h1.reshape(seeds.numel(), 10, -1))
        relation = self.relation_vector.unsqueeze(0).expand(seeds.numel(), -1)
        augmented = torch.cat((seed_h2, relation), dim=1)
        attention = augmented @ self.attention_vector
        representation = F.normalize(augmented * attention, p=2, dim=1)
        representation = F.dropout(representation, p=self.dropout, training=self.training)
        return self.classifier(representation)
