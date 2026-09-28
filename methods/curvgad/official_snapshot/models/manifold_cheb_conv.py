# models/manifold_cheb_conv.py

import torch
import torch.nn as nn
import geoopt
from geoopt import ManifoldParameter

class ManifoldChebConv(nn.Module):
    def __init__(self, in_feats, out_feats, K, manifold):
        super(ManifoldChebConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.K = K  # Chebyshev polynomial order
        self.manifold = manifold
        # Learnable weights for each Chebyshev polynomial order
        self.weight = ManifoldParameter(
            torch.Tensor(K, in_feats, out_feats), manifold=self.manifold
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.in_feats ** 0.5)
        for k in range(self.K):
            self.weight.data[k].uniform_(-stdv, stdv)

    def forward(self, x, laplacian):
        """
        Forward pass for ManifoldChebConv.

        Parameters:
        - x: node features in manifold space (ManifoldTensor)
        - laplacian: precomputed normalized Laplacian matrix, passed as dense or sparse matrix
        """

        # T_0(x) for Chebyshev approximation
        Tx_0 = x
        # Initialize output on the manifold origin
        out = self.manifold.origin(x.size(0), self.out_feats, dtype=x.dtype, device=x.device)

        # Apply first weight [weight[0]]
        out = self.manifold.mobius_matvec(self.weight[0], Tx_0)

        if self.K == 1:
            return out

        # Compute T_1(x) = L @ x using Möbius multiplication with Laplacian
        Tx_1 = self.manifold.mobius_matvec(laplacian, Tx_0)

        # Apply second weight [weight[1]]
        out = self.manifold.mobius_add(out, self.manifold.mobius_matvec(self.weight[1], Tx_1))

        # Higher-order Chebyshev terms
        for k in range(2, self.K):
            Tx_2 = self.manifold.mobius_sub(
                self.manifold.mobius_scalar_mul(torch.tensor(2.0).to(x.device), self.manifold.mobius_matvec(laplacian, Tx_1)),
                Tx_0
            )
            # Apply weight[k]
            out = self.manifold.mobius_add(out, self.manifold.mobius_matvec(self.weight[k], Tx_2))

            # Update T_0 and T_1 for the next iteration
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out