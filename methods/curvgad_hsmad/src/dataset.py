"""Frozen HSMAD graph loading for CurvGAD."""

from pathlib import Path

import dgl


def load_preprocessed_frozen_graph(dataset_path: Path):
    graphs, _ = dgl.load_graphs(str(dataset_path))
    if len(graphs) != 1:
        raise ValueError(f"expected one graph in {dataset_path}, found {len(graphs)}")
    graph = dgl.to_bidirected(graphs[0], copy_ndata=True)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph
