// Thanks to the tutorial example at
// https://github.com/pytorch/extension-cpp/blob/1031028f3b048fdea84372f3b81498db53d64d98/cuda/lltm_cuda.cpp

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void gridnet_cuda_forward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint blockSize,
    float eps);

void gridnet_forward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint blockSize,
    float eps)
{
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(initActivations);
    CHECK_INPUT(outActivations);
    gridnet_cuda_forward(
        weight,
        bias,
        initActivations,
        outActivations,
        innerIterations,
        blockSize,
        eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &gridnet_cuda_forward, "Gridnet forward (CUDA)");
}
