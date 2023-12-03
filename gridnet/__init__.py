import math
import warnings
from typing import Tuple

import torch
import torch.nn as nn

from gridnet.backend_torch import gridnet_step_pytorch

try:
    from .backend_cuda import GridnetCudaOp
except ImportError:
    GridnetCudaOp = None


class Gridnet(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        inner_iterations: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.inner_iterations = inner_iterations
        self.block_size = block_size
        self.weight = torch.randn(
            3**3, *shape, device=device, dtype=dtype
        ) / math.sqrt(27)
        self.bias = torch.zeros(*shape, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gridnet_step(
            weight=self.weight,
            bias=self.bias,
            init_activations=x,
            inner_iterations=self.inner_iterations,
            block_size=self.block_size,
            eps=self.eps,
        )


def gridnet_step(
    weight: torch.Tensor,
    bias: torch.Tensor,
    init_activations: torch.Tensor,
    inner_iterations: int,
    block_size: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Apply a forward pass of the model on an activations Tensor
    to produce a new activations Tensor.

    :param weight: a [3^3 x M x N x K] weight tensor. A given entry in the grid
                   is updated by taking a dot product of weights[:, i, j, k]
                   with the (blockwise normalized) neighboring 3x3 grid.
    :param bias: an [M x N x K] bias matrix, which is combined with weights
                 during each recurrent iteration.
    :param init_activations: the input [B x M x N x K] activation grid.
    :param inner_iterations: the number of recurrent iterations to run for
                             each [block_size x block_size x block_size]
                             sub-grid of the full grid before syncing values
                             back amongst all the blocks.
                             All iterations of a given block use the initial
                             values of the surrounding shell of weights around
                             the block, and only update the local values within
                             the block. In other words, each block does not
                             communicate regardless of number of iterations.
    :param block_size: the size of independent blocks which are updated
                       recurrently for multiple iterations. Normalization is
                       also applied only within each block.
    :param eps: a small value to avoid division by zero.
    """
    if weight.device.type == "cuda":
        if GridnetCudaOp is None:
            warnings.warn("gridnet CUDA implementation is not available")
        elif block_size not in (4, 8):
            warnings.warn(
                f"gridnet block_size {block_size} not supported with CUDA kernel. "
                "Using fallback implementation."
            )
        else:
            return GridnetCudaOp.apply(
                weight, bias, init_activations, inner_iterations, block_size, eps
            )

    return gridnet_step_pytorch(
        weight, bias, init_activations, inner_iterations, block_size, eps
    )
