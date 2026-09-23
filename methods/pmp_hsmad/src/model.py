import sys
import types

import torch.nn as nn

try:
    from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm  # noqa: F401
except ModuleNotFoundError:
    class _UnavailableOfficialNorm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("PMP frozen candidate sets gn=False; GraphNorm must not be instantiated")

    tg = types.ModuleType("torch_geometric")
    tg_nn = types.ModuleType("torch_geometric.nn")
    tg_norm = types.ModuleType("torch_geometric.nn.norm")
    tg_norm.GraphNorm = _UnavailableOfficialNorm
    tg_norm.GraphSizeNorm = _UnavailableOfficialNorm
    tg.nn = tg_nn
    tg_nn.norm = tg_norm
    sys.modules.setdefault("torch_geometric", tg)
    sys.modules.setdefault("torch_geometric.nn", tg_nn)
    sys.modules.setdefault("torch_geometric.nn.norm", tg_norm)

from methods.pmp.official_snapshot.model.LASAGE_S import LASAGE_S


class PMPCandidate(LASAGE_S):
    """HSMAD-width input adapter followed by the unmodified official LA-SAGE-S core."""

    def __init__(self, input_dim: int, num_relations: int = 1, batch_size: int = 128, dropout: float = 0.6):
        super().__init__(
            in_size=64,
            hid_size=64,
            out_size=64,
            num_layers=1,
            dropout=dropout,
            proj=True,
            num_relations=num_relations,
            batch_size=batch_size,
            num_trans=1,
            out_proj_size=2,
            agg="mean",
            relation_agg="cat",
        )
        self.input_adapter = nn.Linear(input_dim, 64)

    def forward(self, blocks, relations, features):
        return super().forward(blocks, relations, self.input_adapter(features))
