"""Auditable paper-formula GWNN candidate; not a byte-identical TF port."""
from __future__ import annotations

import torch
from torch import nn


def build_paper_formula_wavelets(
    edge_index: torch.Tensor,
    num_nodes: int,
    scale: float,
    threshold: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exp(-sL) and exp(+sL) from the same unmodified eigensystem.

    This deliberately follows the paper's mathematical inverse rather than the
    official repository's in-place eigenvalue mutation.  Thresholding is by
    absolute magnitude so signs are preserved; threshold=0 is exact.
    """
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=edge_index.device)
    adjacency[edge_index[0], edge_index[1]] = 1
    degree = adjacency.sum(dim=1)
    inv_sqrt = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    normalized_adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    laplacian = torch.eye(num_nodes, dtype=dtype, device=edge_index.device) - normalized_adjacency
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    wavelet = (eigenvectors * torch.exp(-float(scale) * eigenvalues)) @ eigenvectors.T
    inverse = (eigenvectors * torch.exp(float(scale) * eigenvalues)) @ eigenvectors.T
    if threshold > 0:
        wavelet = wavelet.masked_fill(wavelet.abs() < threshold, 0)
        inverse = inverse.masked_fill(inverse.abs() < threshold, 0)
    return wavelet, inverse


class WaveletConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.kernel = nn.Parameter(torch.ones(num_nodes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, wavelet: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        transformed = inverse @ (x @ self.weight)
        return wavelet @ (self.kernel[:, None] * transformed)


class GWNNPaperFormulaCandidate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_nodes: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = WaveletConvolution(input_dim, hidden_dim, num_nodes)
        self.conv2 = WaveletConvolution(hidden_dim, output_dim, num_nodes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, wavelet: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(self.dropout(x), wavelet, inverse))
        return self.conv2(self.dropout(hidden), wavelet, inverse)
