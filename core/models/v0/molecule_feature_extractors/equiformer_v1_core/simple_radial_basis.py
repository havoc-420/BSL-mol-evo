"""
Simple radial basis function implementation to replace ocpmodels dependency
"""

import math
import torch
from torch_geometric.nn.models.schnet import GaussianSmearing


class SimpleRadialBasis(torch.nn.Module):
    """
    Simplified radial basis function without dependency on ocpmodels.
    Implements Gaussian smearing with a polynomial envelope.
    
    Parameters
    ----------
    num_radial: int
        Number of radial basis functions
    cutoff: float
        Cutoff distance
    """
    
    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.inv_cutoff = 1.0 / cutoff
        self.num_radial = num_radial
        
        # Use GaussianSmearing from PyG
        self.rbf = GaussianSmearing(
            start=0.0, stop=cutoff, num_gaussians=num_radial
        )
        
        # Polynomial envelope coefficients (5th order)
        self.p = 5
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def envelope(self, d_scaled):
        """
        Polynomial envelope function that ensures a smooth cutoff.
        """
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))

    def forward(self, d):
        """
        Compute radial basis functions with envelope.
        
        Parameters
        ----------
        d : torch.Tensor
            Distance tensor of shape (N,)
            
        Returns
        -------
        torch.Tensor
            Radial basis functions with envelope of shape (N, num_radial)
        """
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        rbf = self.rbf(d)
        return env[:, None] * rbf  # (N, 1) * (N, num_radial) -> (N, num_radial)