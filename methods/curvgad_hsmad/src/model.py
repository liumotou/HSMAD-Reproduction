"""Thin, traceable access to the fixed official CurvGAD model implementation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "methods" / "curvgad" / "official_snapshot"
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

from models.curvgad_gnn import CurvGAD  # noqa: E402


def balanced_mixed_manifold_config(hidden_dim: int) -> str:
    """Preserve H/S/E topology while making their concatenated width exactly hidden_dim."""
    if hidden_dim < 3:
        raise ValueError("mixed H/S/E CurvGAD requires hidden_dim >= 3")
    base, remainder = divmod(int(hidden_dim), 3)
    dims = [base + int(index < remainder) for index in range(3)]
    return f"H{dims[0]}S{dims[1]}E{dims[2]}"


def build_official_model(
    *,
    in_feats: int,
    hidden_dim: int,
    num_classes: int = 2,
    k: int = 3,
    dropout: float = 0.0,
) -> CurvGAD:
    return CurvGAD(
        in_feats=in_feats,
        num_classes=num_classes,
        K=k,
        num_filters=3,
        manifolds_config_str=balanced_mixed_manifold_config(hidden_dim),
        dropout_rate=dropout,
        activation="ReLU",
    )
