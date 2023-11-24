from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from gridnet.backend_torch import outer_step_pytorch


@pytest.mark.parametrize(
    "shape,block_size",
    (
        ((16, 16, 16), 8),
        ((32, 64, 128), 8),
    ),
)
def test_outer_step_zero_iters(shape: Tuple[int, int, int], block_size: int):
    inputs = torch.randn(
        2,
        *shape,
    )
    weights = torch.randn(block_size, block_size, block_size, 3**3)
    biases = torch.randn(block_size, block_size, block_size)
    outputs = outer_step_pytorch(weights, biases, inputs, 0, block_size=block_size)
    assert torch.allclose(outputs, inputs)


@pytest.mark.parametrize(
    "shape,block_size",
    (
        ((16, 16, 16), 8),
        ((32, 64, 128), 8),
    ),
)
def test_outer_step_zero_weights(shape: Tuple[int, int, int], block_size: int):
    inputs = torch.randn(
        2,
        *shape,
    )
    weights = torch.zeros(block_size, block_size, block_size, 3**3)
    biases = torch.zeros(block_size, block_size, block_size)
    outputs = outer_step_pytorch(weights, biases, inputs, 10, block_size=block_size)
    padded = F.pad(inputs, (1,) * 6)
    for i in range(1, shape[0] - 2, block_size):
        for j in range(1, shape[1] - 2, block_size):
            for k in range(1, shape[2] - 2, block_size):
                for _ in range(10):
                    padded[
                        :, i : i + block_size, j : j + block_size, k : k + block_size
                    ] += F.silu(biases)
    assert torch.allclose(outputs, padded[:, 1:-1, 1:-1, 1:-1])
