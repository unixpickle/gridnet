from typing import Callable

import torch
import torch.nn.functional as F


def outer_step_pytorch(
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

    :param weight: an [M x N x K x 3^3] weight tensor. A given entry in the
                   grid is updated by taking a dot product of weights[i, j, k]
                   with all the (normalized) neighboring values to the current
                   one.
    :param bias: an [M x N x K] bias matrix, which is combined with weights
                 during each recurrent iteration.
    :param init_activations: the input [B x M x N  x K] activation
                             grid.
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
    assert weight.shape == (
        block_size,
        block_size,
        block_size,
        3**3,
    ), f"{weight.shape=} invalid for {block_size=}"
    assert bias.shape == weight.shape[:-1], f"{bias.shape=} invalid for {block_size=}"
    batch_size, m, n, k = init_activations.shape
    assert (
        m % block_size == 0 and n % block_size == 0 and k % block_size == 0
    ), f"{block_size=} incompatible with {init_activations.shape=}"

    block_fn = inner_step_fn(block_size, weight, bias, eps)
    output_blocks = []
    padded_init_acts = F.pad(init_activations, (1,) * 6)
    for a in range(1, m + 1, block_size):
        for b in range(1, n + 1, block_size):
            for c in range(1, k + 1, block_size):
                block_in = padded_init_acts[
                    :,
                    a - 1 : a + 1 + block_size,
                    b - 1 : b + 1 + block_size,
                    c - 1 : c + 1 + block_size,
                ]
                block_out = block_in
                for _ in range(inner_iterations):
                    block_out = block_fn(block_out)
                output_blocks.append(block_out[:, 1:-1, 1:-1, 1:-1])
    return (
        torch.stack(output_blocks)
        .reshape(
            m // block_size,
            n // block_size,
            k // block_size,
            batch_size,
            block_size,
            block_size,
            block_size,
        )
        .permute(3, 0, 4, 1, 5, 2, 6)
        .reshape(batch_size, m, n, k)
    )


def inner_step_fn(
    block_size: int, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Extract neighborhoods from a (block_size + 2) ^ 3 tensor.
    def index_in_block(i: int, j: int, k: int) -> int:
        return k + (block_size + 2) * (j + (block_size + 2) * i)

    indices = []
    for i in range(block_size):
        for j in range(block_size):
            for k in range(block_size):
                cell_indices = []
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            cell_indices.append(index_in_block(i + a, j + b, k + c))
                indices.extend(cell_indices)
    input_index_tensor = torch.tensor(indices, device=weight.device, dtype=torch.long)

    def inner_fn(activations: torch.Tensor) -> torch.Tensor:
        batch_size = activations.shape[0]
        flattened = activations.flatten(1)  # [batch_size x block_size^3]
        mean = flattened.mean(1, keepdim=True)
        std = flattened.std(1, keepdim=True)
        inputs = (flattened - mean) / (std + eps)
        patches = inputs.gather(
            1, input_index_tensor[None].repeat(batch_size, 1)
        ).reshape(
            batch_size, block_size**3, 3**3
        )  # [batch_size x block_size^3 x 3^3]
        results = (patches * weight.reshape(-1, 3**3)).sum(-1) + bias.flatten()
        results = F.silu(results)
        results = results.reshape(-1, block_size, block_size, block_size)
        results = F.pad(results, (1, 1, 1, 1, 1, 1))
        return results + activations

    return inner_fn
