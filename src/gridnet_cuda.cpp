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
    torch::Tensor scale,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint blockSize,
    float eps,
    bool normalize);

void gridnet_cuda_backward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor initActivations,
    torch::Tensor outGrads,
    torch::Tensor weightGradOut,
    torch::Tensor biasGradOut,
    torch::Tensor scaleGradOut,
    torch::Tensor activationsGradOut,
    uint innerIterations,
    uint blockSize,
    float eps,
    bool normalize);

void gridnet_forward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint blockSize,
    float eps,
    bool normalize)
{
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(initActivations);
    CHECK_INPUT(outActivations);
    gridnet_cuda_forward(
        weight,
        bias,
        scale,
        initActivations,
        outActivations,
        innerIterations,
        blockSize,
        eps,
        normalize);
}

void gridnet_backward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor initActivations,
    torch::Tensor outGrads,
    torch::Tensor weightGradOut,
    torch::Tensor biasGradOut,
    torch::Tensor scaleGradOut,
    torch::Tensor activationsGradOut,
    uint innerIterations,
    uint blockSize,
    float eps,
    bool normalize)
{
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(initActivations);
    CHECK_INPUT(outGrads);
    CHECK_INPUT(weightGradOut);
    CHECK_INPUT(biasGradOut);
    CHECK_INPUT(activationsGradOut);
    gridnet_cuda_backward(
        weight,
        bias,
        scale,
        initActivations,
        outGrads,
        weightGradOut,
        biasGradOut,
        scaleGradOut,
        activationsGradOut,
        innerIterations,
        blockSize,
        eps,
        normalize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &gridnet_forward, "Gridnet forward (CUDA)");
    m.def("backward", &gridnet_backward, "Gridnet backward (CUDA)");
}
