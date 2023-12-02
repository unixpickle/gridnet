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
    from gridnet.backend_cuda import GridnetCudaOp

    eps = 1e-5
    inputs = torch.randn(2, *shape).cuda()
    weights = torch.randn(3**3, *shape).cuda()
    biases = torch.randn(*shape).cuda()
    expected = outer_step_pytorch(weights, biases, inputs, 2, block_size, eps)
    actual = GridnetCudaOp.apply(weights, biases, inputs, 2, block_size, eps)
    assert (actual - expected).abs().max().item() < 1e-4


@pytest.mark.parametrize(
    "shape,inner_iters,block_size",
    (
        ((16, 16, 16), 1, 8),
        ((16, 16, 16), 2, 8),
        ((32, 64, 128), 2, 8),
        ((32, 64, 128), 2, 4),
    ),
)
def test_grad_equivalence(
    shape: Tuple[int, int, int], inner_iters: int, block_size: int
):
    # Import must come after `import torch` to avoid linking issues
    from gridnet.backend_cuda import GridnetCudaOp

    eps = 1e-5
    inputs = torch.randn(2, *shape).cuda().requires_grad_(True)
    weights = torch.randn(3**3, *shape).cuda().requires_grad_(True)
    biases = torch.randn(*shape).cuda().requires_grad_(True)
    expected = outer_step_pytorch(weights, biases, inputs, inner_iters, block_size, eps)
    out_grad = torch.randn_like(expected).cuda()
    expected_grads = torch.autograd.grad(expected, (biases, weights, inputs), out_grad)
    actual = GridnetCudaOp.apply(weights, biases, inputs, inner_iters, block_size, eps)
    actual_grads = torch.autograd.grad(actual, (biases, weights, inputs), out_grad)
    for name, x, a in zip(
        ["biases", "weights", "inputs"], expected_grads, actual_grads
    ):
        err = (x - a).abs().max().item()
        assert err < 1e-4, f"MAE in {name}: {err} ({x=} {a=})"
