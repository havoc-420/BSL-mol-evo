# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_cluster import radius_graph

class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    r"""Applies the ShiftedSoftplus function :math:`\text{ShiftedSoftplus}(x) = \frac{1}{\beta} *
    \log(1 + \exp(\beta * x))-\log(2)` element-wise.

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    """Swish activation function as defined in https://arxiv.org/pdf/1710.05941 :

    .. math::

        \text{Swish}(x) = x \cdot \sigma(\beta x)

    Args:
        beta (float, optional): Scaling factor for Swish activation. Defaults to 1.

    """

    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class OptimizedDistance(nn.Module):
    """Compute the neighbor list for a given cutoff.
    
    This is a simplified version of the OptimizedDistance from torchmdnet,
    using radius_graph for neighbor computation.
    """

    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_pairs=-32,
        return_vecs=False,
        loop=False,
        strategy="brute",
        include_transpose=True,
        resize_to_fit=True,
        check_errors=True,
        box=None,
        long_edge_index=True,
    ):
        super(OptimizedDistance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_pairs = max_num_pairs
        self.return_vecs = return_vecs
        self.loop = loop
        self.include_transpose = include_transpose
        self.resize_to_fit = resize_to_fit
        self.check_errors = check_errors
        self.long_edge_index = long_edge_index

    def forward(self, pos, batch=None, box=None):
        """Compute distances between particles.
        
        Args:
            pos: Particle positions with shape (N, 3)
            batch: Batch indices with shape (N)
            box: Simulation box (not supported in this simplified version)
            
        Returns:
            edge_index: Pairs of particles with shape (2, M)
            edge_weight: Distances between particles with shape (M)
            edge_vec: Vectors between particles with shape (M, 3) or None
        """
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
            
        # Use radius_graph to compute neighbors
        edge_index = radius_graph(
            pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=abs(self.max_num_pairs) if self.max_num_pairs < 0 else self.max_num_pairs,
        )

        # Compute distance vectors
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        
        # Compute distances
        edge_weight = torch.norm(edge_vec, dim=-1)
        
        # Apply lower cutoff
        if self.cutoff_lower > 0.0:
            mask = edge_weight >= self.cutoff_lower
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_vec = edge_vec[mask]
            
        # Return vectors if requested
        if not self.return_vecs:
            edge_vec = None
            
        return edge_index, edge_weight, edge_vec


rbf_class_mapping = {"gauss": None, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
    "mish": nn.Mish,
}

dtype_mapping = {16: torch.float16, 32: torch.float, 64: torch.float64}