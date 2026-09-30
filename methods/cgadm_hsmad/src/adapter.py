"""Lossless conversion of an already-preprocessed frozen DGL graph to PyG."""

from __future__ import annotations

import torch
import dgl
from torch_geometric.data import Data


def _ndata(graph, singular: str) -> torch.Tensor:
    """Return a required frozen node tensor without deriving or regenerating it."""
    plural = singular + "s"
    if singular in graph.ndata:
        return graph.ndata[singular]
    if plural in graph.ndata:
        return graph.ndata[plural]
    raise KeyError(f"missing frozen node field: {singular}")


def dgl_to_frozen_pyg(graph) -> Data:
    """Convert representation only; do not alter direction, loops, order, or masks."""
    src, dst = graph.edges(order="eid")
    data = Data(
        x=_ndata(graph, "feature"),
        y=_ndata(graph, "label"),
        edge_index=torch.stack((src, dst), dim=0).to(torch.int64),
        num_nodes=graph.num_nodes(),
    )
    data.train_mask = _ndata(graph, "train_mask").bool()
    data.val_mask = _ndata(graph, "val_mask").bool()
    data.test_mask = _ndata(graph, "test_mask").bool()
    # Official CGADM names. These are aliases to the same frozen tensors.
    data.feature = data.x
    data.label = data.y
    data.train_masks = data.train_mask
    data.val_masks = data.val_mask
    data.test_masks = data.test_mask
    return data


def preprocess_frozen_graph(graph):
    """Apply the frozen HSMAD graph path, preserving every node tensor."""
    processed = dgl.to_bidirected(graph, copy_ndata=True)
    processed = dgl.remove_self_loop(processed)
    processed = dgl.add_self_loop(processed)
    return processed


def load_frozen_data(dataset_path):
    """Load one persisted DGL graph, preprocess once, and convert losslessly."""
    graphs, _ = dgl.load_graphs(str(dataset_path))
    if len(graphs) != 1:
        raise ValueError(f"expected one graph, got {len(graphs)}")
    graph = preprocess_frozen_graph(graphs[0])
    return graph, dgl_to_frozen_pyg(graph)
