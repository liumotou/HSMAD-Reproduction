"""Traceable thin constructors around the fixed official CGADM model snapshot."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def _load_official_diffuse(path: str):
    source = Path(path).resolve() / "models" / "diffuse.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("cgadm_fixed_official_diffuse", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fixed CGADM source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_denoiser(
    official_snapshot,
    in_channels: int,
    hidden_channels: int = 64,
    num_steps: int = 500,
    dropout: float = 0.0,
):
    """Construct the unmodified official Denoiser with HSMAD hidden=64."""
    module = _load_official_diffuse(str(Path(official_snapshot)))
    return module.Denoiser(
        T=num_steps,
        topo_dim=1,
        gin_h=None,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1,
        dropout=dropout,
        ln=True,
        tailact=True,
    )


def build_optimizer(model, lr: float = 0.01, weight_decay: float = 0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_full_model(official_snapshot, options, data, device):
    """Instantiate the unmodified official CGADM Model class."""
    module = _load_official_diffuse(str(Path(official_snapshot)))
    return module.Model(options, data, device)


def build_scheduler(optimizer, step_size: int = 150, gamma: float = 0.5):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
