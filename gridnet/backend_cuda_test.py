from typing import Tuple

import pytest
import torch

from gridnet.backend_torch import outer_step_pytorch


@pytest.mark.parametrize(
    "shape,block_size",
    (
        ((16, 16, 16), 8),
        ((32, 64, 128), 8),
        ((32, 64, 128), 4),
    ),
)
def test_forward_equivalence(shape: Tuple[int, int, int], block_size: int):
    # Import must come after `import torch` to avoid linking issues
    import gridnet_cuda

    eps = 1e-5
    inputs = torch.randn(2, *shape).cuda()
    weights = torch.randn(3**3, *shape).cuda()
    biases = torch.randn(*shape).cuda()
    expected = outer_step_pytorch(weights, biases, inputs, 2, block_size, eps)
    actual = torch.zeros_like(expected)
    gridnet_cuda.forward(weights, biases, inputs, actual, 2, block_size, eps)
    assert (actual - expected).abs().max().item() < 1e-4
