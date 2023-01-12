import torch
import torch.nn as nn

import cdopt 
from cdopt.nn.modules import Linear_cdopt
from cdopt.manifold_torch import sphere_torch

class Biaffine(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.linear = Linear_cdopt(
            self.dim, 1, bias=False, 
            manifold_class=sphere_torch, penalty_param = 0.02
        )

        #self.linear = nn.Linear(self.dim, 1, bias=False).double()
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x_h, x_w):
        return torch.mul(self.linear(x_h), self.linear(x_w)).squeeze()