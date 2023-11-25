// Thanks to the tutorial example at
// https://github.com/pytorch/extension-cpp/blob/1031028f3b048fdea84372f3b81498db53d64d98/cuda/lltm_cuda.cpp

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void gridnet_cuda_forward_kernel(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint m,
    uint n,
    uint k,
    float eps);

void gridnet_cuda_forward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    float eps)
{
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(initActivations);
    CHECK_INPUT(outActivations);
    // TODO: check shapes.
    const int blockSize = weight.size(1);
    const int batchSize = initActivations.size(0);
    const int m = initActivations.size(1);
    const int n = initActivations.size(2);
    const int k = initActivations.size(3);

    const int threads = blockSize * blockSize * blockSize;
    const dim3 blocks(batchSize, (m / blockSize) * (n / blockSize) * (k / blockSize));
    AT_DISPATCH_FLOATING_TYPES(
        weight.type(),
        "gridnet_cuda_forward",
        ([&] {
            gridnet_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                weight.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                initActivations.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                outActivations.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                innerIterations,
                m,
                n,
                k,
                eps);
        }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &gridnet_cuda_forward, "Gridnet forward (CUDA)");
}
