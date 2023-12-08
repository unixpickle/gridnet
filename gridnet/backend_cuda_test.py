from typing import Tuple

import pytest
import torch

from gridnet.backend_torch import gridnet_step_pytorch


@pytest.mark.parametrize(
    "shape,block_size,normalize",
    (
        ((16, 16, 16), 8, True),
        ((32, 64, 128), 8, True),
        ((32, 64, 128), 4, True),
        ((32, 16, 8), 4, False),
        ((32, 16, 8), 8, False),
    ),
)
def test_forward_equivalence(
    shape: Tuple[int, int, int], block_size: int, normalize: bool
):
    # Import must come after `import torch` to avoid linking issues
    from gridnet.backend_cuda import GridnetCudaOp

    eps = 1e-5
    inputs = torch.randn(2, *shape).cuda()
    weights = torch.randn(3**3, *shape).cuda()
    biases = torch.randn(*shape).cuda()
    scales = torch.randn(*shape).cuda()
    expected = gridnet_step_pytorch(
        weights, biases, scales, inputs, 2, block_size, eps, normalize
    )
    actual = GridnetCudaOp.apply(
        weights, biases, scales, inputs, 2, block_size, eps, normalize
    )
    assert (actual - expected).abs().max().item() < 2e-4


@pytest.mark.parametrize(
    "shape,inner_iters,block_size,normalize",
    (
        ((16, 16, 16), 1, 8, True),
        ((16, 16, 16), 2, 8, True),
        ((32, 64, 128), 2, 8, True),
        ((32, 64, 128), 2, 4, True),
        ((32, 16, 8), 2, 4, False),
        ((32, 16, 8), 2, 8, False),
    ),
)
def test_grad_equivalence(
    shape: Tuple[int, int, int], inner_iters: int, block_size: int, normalize: bool
):
    # Import must come after `import torch` to avoid linking issues
    from gridnet.backend_cuda import GridnetCudaOp

    eps = 1e-5
    inputs = torch.randn(2, *shape).cuda().requires_grad_(True)
    weights = torch.randn(3**3, *shape).cuda().requires_grad_(True)
    biases = torch.randn(*shape).cuda().requires_grad_(True)
    scales = torch.randn(*shape).cuda().requires_grad_(True)
    expected = gridnet_step_pytorch(
        weights, biases, scales, inputs, inner_iters, block_size, eps, normalize
    )
    out_grad = torch.randn_like(expected).cuda()
    expected_grads = torch.autograd.grad(
        expected, (scales, biases, weights, inputs), out_grad
    )
    actual = GridnetCudaOp.apply(
        weights, biases, scales, inputs, inner_iters, block_size, eps, normalize
    )
    actual_grads = torch.autograd.grad(
        actual, (scales, biases, weights, inputs), out_grad
    )
    for name, x, a in zip(
        ["scales", "biases", "weights", "inputs"], expected_grads, actual_grads
    ):
        err = (x - a).abs().max().item()
        assert err < 3e-4, f"MAE in {name}: {err} ({x=} {a=})"


def test_forward_benchmark(benchmark):
    from gridnet.backend_cuda import GridnetCudaOp

    shape = (32, 32, 32)
    inputs = torch.randn(2, *shape).cuda()
    weights = torch.randn(3**3, *shape).cuda()
    biases = torch.randn(*shape).cuda()
    scales = torch.randn(*shape).cuda()
    torch.cuda.synchronize()

    def fn():
        GridnetCudaOp.apply(weights, biases, scales, inputs, 10, 8, 1e-5, True)
        torch.cuda.synchronize()

    benchmark(fn)


def test_backward_benchmark(benchmark):
    from gridnet.backend_cuda import GridnetCudaOp

    shape = (32, 32, 32)
    inputs = torch.randn(2, *shape).cuda().requires_grad_(True)
    weights = torch.randn(3**3, *shape).cuda().requires_grad_(True)
    biases = torch.randn(*shape).cuda().requires_grad_(True)
    scales = torch.randn(*shape).cuda().requires_grad_(True)
    out_grad = torch.randn_like(inputs)

    def fn():
        out = GridnetCudaOp.apply(weights, biases, scales, inputs, 10, 8, 1e-5, True)
        _grads = torch.autograd.grad(out, (inputs, weights, biases, scales), out_grad)
        torch.cuda.synchronize()

    benchmark(fn)
