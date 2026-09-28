import json
import site

site.addsitedir("/root/miniconda3/lib/python3.10/site-packages")

import dgl
import torch


graph = dgl.load_graphs("datasets/tfinance")[0][0]
graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.to_bidirected(graph)))
source, destination = graph.edges(order="eid")
print(json.dumps({
    "nodes": int(graph.num_nodes()),
    "edges": int(graph.num_edges()),
    "source_nondecreasing": bool(torch.all(source[1:] >= source[:-1])),
    "edge_tensor_gib": (
        source.numel() * source.element_size()
        + destination.numel() * destination.element_size()
    ) / 1024**3,
}, sort_keys=True))
