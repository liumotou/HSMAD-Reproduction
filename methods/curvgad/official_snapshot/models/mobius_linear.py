# models/mobius_linear.py

import torch
import geoopt
from torch.nn import Parameter

class MobiusLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True, nonlin=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold  # Should be StereographicProductManifold
        self.nonlin = nonlin
        self.weight = geoopt.ManifoldParameter(
            torch.empty(out_features, in_features), manifold=self.manifold
        )
        if bias:
            self.bias = geoopt.ManifoldParameter(
                torch.empty(out_features), manifold=self.manifold
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases appropriately
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.zero_()

    def forward(self, input):
        # Apply Mobius linear transformation
        output = self.manifold.mobius_matvec(self.weight, input)
        if self.bias is not None:
            output = self.manifold.mobius_add(output, self.bias)
        if self.nonlin is not None:
            output = self.manifold.logmap0(output)
            output = self.nonlin(output)
            output = self.manifold.expmap0(output)
        return output