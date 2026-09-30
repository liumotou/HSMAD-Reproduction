"""Memory-scalable adjacency-table construction with legacy-exact semantics."""
from __future__ import annotations

import numpy as np
import torch


def build_padded_adjacency_stable_csr(
    source: torch.Tensor,
    destination: torch.Tensor,
    num_nodes: int,
    max_degree: int,
    seed: int,
) -> torch.Tensor:
    """Match the legacy per-node RNG choices without Python edge objects.

    A stable sort by source groups edges while preserving their original EID
    order within every node, which is exactly the ordering produced by the
    legacy append loop.
    """
    source_np = source.detach().cpu().numpy()
    destination_np = destination.detach().cpu().numpy()
    order = np.argsort(source_np, kind="stable")
    grouped_destination = destination_np[order]
    counts = np.bincount(source_np, minlength=num_nodes)
    offsets = np.empty(num_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    rng = np.random.default_rng(seed)
    table = np.empty((num_nodes, max_degree), dtype=np.int64)
    for node in range(num_nodes):
        values = grouped_destination[offsets[node] : offsets[node + 1]]
        if values.size == 0:
            values = np.asarray([node], dtype=np.int64)
        table[node] = rng.choice(
            values, size=max_degree, replace=values.size < max_degree
        )
    return torch.from_numpy(table)
