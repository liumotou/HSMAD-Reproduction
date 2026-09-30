"""Read-only DGL frozen-graph to PyG conversion."""
from __future__ import annotations

import torch
from torch_geometric.data import Data

REQUIRED_NODE_FIELDS = ('feature', 'label', 'train_mask', 'val_mask', 'test_mask')


def dgl_to_pyg_frozen(graph) -> Data:
    """Convert the already-preprocessed graph without changing any edge semantics."""
    missing = [name for name in REQUIRED_NODE_FIELDS if name not in graph.ndata]
    if missing:
        raise KeyError(f'missing frozen graph fields: {missing}')
    source, destination = graph.edges(order='eid')
    return Data(
        x=graph.ndata['feature'].float(),
        edge_index=torch.stack((source.long(), destination.long()), dim=0),
        y=graph.ndata['label'].long().reshape(-1),
        train_mask=graph.ndata['train_mask'].bool(),
        val_mask=graph.ndata['val_mask'].bool(),
        test_mask=graph.ndata['test_mask'].bool(),
        num_nodes=graph.num_nodes(),
    )
