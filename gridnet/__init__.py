import math
import warnings
from typing import Tuple

import torch
import torch.nn as nn

from gridnet.backend_torch import (
    ActivationFn,
    gated_gridnet_step_pytorch,
    gridnet_step_pytorch,
)

try:
    from .backend_cuda import GatedGridnetCudaOp, GridnetCudaOp
except ImportError:
    GridnetCudaOp = None
    GatedGridnetCudaOp = None


class Gridnet(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        inner_iterations: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-5,
        init_scale: float = 1.0,
        residual_scale: float = 1.0,
        normalize: bool = True,
        activation: ActivationFn = "silu",
    ):
        super().__init__()
        self.shape = shape
        self.inner_iterations = inner_iterations
        self.block_size = block_size
        self.eps = eps
        self.normalize = normalize
        self.activation = activation
        self.weight = nn.Parameter(
            torch.randn(3**3, *shape, device=device, dtype=dtype)
            * (init_scale / math.sqrt(27))
        )
        self.bias = nn.Parameter(torch.zeros(*shape, device=device, dtype=dtype))
        self.residual_scale = nn.Parameter(
            torch.ones(*shape, device=device, dtype=dtype) * residual_scale
        )
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gridnet_step(
            weight=self.weight,
            bias=self.bias,
            residual_scale=self.residual_scale,
            init_activations=x,
            inner_iterations=self.inner_iterations,
            block_size=self.block_size,
            eps=self.eps,
            normalize=self.normalize,
            activation=self.activation,
        )


class GatedGridnet(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        inner_iterations: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        init_scale: float = 1.0,
        activation: ActivationFn = "tanh",
    ):
        super().__init__()
        self.shape = shape
        self.inner_iterations = inner_iterations
        self.block_size = block_size
        self.activation = activation
        self.weight = nn.Parameter(
            torch.randn(3**3, 2, *shape, device=device, dtype=dtype)
            * (init_scale / math.sqrt(27))
        )
        self.bias = nn.Parameter(torch.zeros(2, *shape, device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gated_gridnet_step(
            weight=self.weight,
            bias=self.bias,
            init_activations=x,
            inner_iterations=self.inner_iterations,
            block_size=self.block_size,
            activation=self.activation,
        )


class Readout(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        out_channels: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        output_layers: int = 1,
    ):
        super().__init__()
        size = grid_shape[0] * grid_shape[1] * output_layers
        self.output_layers = output_layers
        self.norm = nn.LayerNorm(size, device=device, dtype=dtype)
        self.proj = nn.Linear(size, out_channels, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x[:, :, :, -self.output_layers :].flatten(1)
        h = self.norm(h)
        h = self.proj(h)
        return h


def gridnet_step(
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual_scale: torch.Tensor,
    init_activations: torch.Tensor,
    inner_iterations: int,
    block_size: int,
    eps: float = 1e-5,
    normalize: bool = True,
    activation: ActivationFn = "silu",
) -> torch.Tensor:
    """
    Apply a forward pass of the model on an activations Tensor
    to produce a new activations Tensor.

    :param weight: a [3^3 x M x N x K] weight tensor. A given entry in the grid
                   is updated by taking a dot product of weights[:, i, j, k]
                   with the (blockwise normalized) neighboring 3x3 grid.
    :param bias: an [M x N x K] bias matrix, which is combined with weights
                 during each recurrent iteration.
    :param residual_scale: an [M x N x K] scale parameter to multiply by
                           post-activation results.
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
    :param normalize: if True (default), normalize activations in each block.
    :param activation: the activation function to apply.
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
                weight,
                bias,
                residual_scale,
                init_activations,
                inner_iterations,
                block_size,
                eps,
                normalize,
                activation,
            )

    return gridnet_step_pytorch(
        weight,
        bias,
        residual_scale,
        init_activations,
        inner_iterations,
        block_size,
        eps,
        normalize,
        activation,
    )


def gated_gridnet_step(
    weight: torch.Tensor,
    bias: torch.Tensor,
    init_activations: torch.Tensor,
    inner_iterations: int,
    block_size: int,
    activation: ActivationFn = "tanh",
) -> torch.Tensor:
    """
    Apply a forward pass of the gated version of the model on an
    activations Tensor to produce a new activations Tensor.

    Each update computes a new value, which is modulated with the given
    activation function, and a gate, which is computed with a sigmoid.

    :param weight: a [3^3 x 2 x M x N x K] weight tensor.
    :param bias: an [2 x M x N x K] bias matrix, which is combined with
                 weights during each recurrent iteration.
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
    :param activation: the activation function to apply.
    """
    if weight.device.type == "cuda":
        if GatedGridnetCudaOp is None:
            warnings.warn("gridnet CUDA implementation is not available")
        elif block_size not in (4, 8):
            warnings.warn(
                f"gridnet block_size {block_size} not supported with CUDA kernel. "
                "Using fallback implementation."
            )
        else:
            return GatedGridnetCudaOp.apply(
                weight,
                bias,
                init_activations,
                inner_iterations,
                block_size,
                activation,
            )

    return gated_gridnet_step_pytorch(
        weight,
        bias,
        init_activations,
        inner_iterations,
        block_size,
        activation,
    )
