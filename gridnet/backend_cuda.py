import gridnet_cuda
import torch


class GridnetCudaOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        bias: torch.Tensor,
        init_activations: torch.Tensor,
        inner_iters: int,
        block_size: int,
        eps: float,
    ):
        batch_size, m, n, k = init_activations.shape
        assert (
            m % block_size == 0 and n % block_size == 0 and k % block_size == 0
        ), f"{block_size=} incompatible with {init_activations.shape=}"
        assert weight.shape == (
            3**3,
            *init_activations.shape[1:],
        ), f"{weight.shape=} invalid for {init_activations.shape=}"
        assert (
            bias.shape == weight.shape[1:]
        ), f"{bias.shape=} invalid for {weight.shape=}"
        assert (
            weight.dtype == bias.dtype
        ), f"mismatching dtypes: {weight.dtype=} {bias.dtype=}"
        assert (
            weight.dtype == init_activations.dtype
        ), f"mismatching dtypes: {weight.dtype=} {init_activations.dtype=}"
        assert (
            weight.device == bias.device
        ), f"mismatching devices: {weight.device=} {bias.device=}"
        assert (
            weight.device == init_activations.device
        ), f"mismatching devices: {weight.devicee=} {init_activations.device=}"
        if not batch_size:
            return init_activations.clone()
        ctx.save_for_backward(weight, bias, init_activations)
        ctx.inner_iters = inner_iters
        ctx.block_size = block_size
        ctx.eps = eps
        outputs = torch.empty_like(init_activations)

        gridnet_cuda.forward(
            weight, bias, init_activations, outputs, inner_iters, block_size, eps
        )
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        (weight, bias, init_activations) = ctx.saved_tensors
        weight_grad = torch.zeros_like(weight)
        bias_grad = torch.zeros_like(bias)
        init_activations_grad = torch.zeros_like(init_activations)
        gridnet_cuda.backward(
            weight,
            bias,
            init_activations,
            grad_output,
            weight_grad,
            bias_grad,
            init_activations_grad,
            ctx.inner_iters,
            ctx.block_size,
            ctx.eps,
        )
        return weight_grad, bias_grad, init_activations_grad, None, None, None
