"""Numerical check: DGL GraphConv(norm='both') vs Kipf-Welling normalized adjacency."""
import hashlib
import json
from pathlib import Path

import dgl
import torch
from dgl.nn import GraphConv


OUT = Path(__file__).with_name('gcn_formula_equivalence_result.json')


def sha256_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    torch.manual_seed(17)
    # Undirected base graph with exactly one self-loop per node.
    undirected = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    edges = undirected + [(i, i) for i in range(4)]
    src = torch.tensor([edge[0] for edge in edges])
    dst = torch.tensor([edge[1] for edge in edges])
    graph = dgl.graph((src, dst), num_nodes=4)
    feature = torch.randn(4, 3, dtype=torch.float64)
    weight = torch.randn(3, 2, dtype=torch.float64)
    convolution = GraphConv(3, 2, norm='both', weight=True, bias=False, activation=None, allow_zero_in_degree=True).double()
    with torch.no_grad():
        convolution.weight.copy_(weight)
    convolution.eval()
    dgl_output = convolution(graph, feature)
    adjacency = torch.zeros((4, 4), dtype=torch.float64)
    adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=1)
    normalization = torch.diag(torch.pow(degree, -0.5))
    kipf_welling_output = normalization @ adjacency @ normalization @ feature @ weight
    error = (dgl_output - kipf_welling_output).abs()
    result = {
        'test': 'DGL GraphConv(norm=both, bias=False, activation=None) equals D^-1/2(A+I)D^-1/2 XW on a symmetric graph',
        'dgl_version': dgl.__version__, 'torch_version': torch.__version__,
        'nodes': 4, 'directed_edges_including_self_loops': int(graph.num_edges()),
        'self_loop_count': 4, 'norm': 'both', 'bias': False, 'activation': None,
        'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean()),
        'allclose_atol_1e-12_rtol_1e-12': bool(torch.allclose(dgl_output, kipf_welling_output, atol=1e-12, rtol=1e-12)),
        'feature_sha256': sha256_tensor(feature), 'weight_sha256': sha256_tensor(weight),
        'dgl_output_sha256': sha256_tensor(dgl_output), 'formula_output_sha256': sha256_tensor(kipf_welling_output),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['allclose_atol_1e-12_rtol_1e-12']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
