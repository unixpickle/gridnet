#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_mean(
    scalar_t *inputBuf,
    scalar_t *aggBuf,
    bool square,
    uint numInputs,
    uint threadOffset,
    uint numThreads)
{
    // Don't overwrite aggBuf before other calls are done.
    __syncthreads();

    scalar_t localSum = 0.0;
    for (uint i = 0; i < 1024; i += numThreads) {
        uint offset = i + threadOffset;
        scalar_t inVal = 0.0;
        if (offset < numInputs) {
            inVal = inputBuf[offset];
        }
        if (square) {
            inVal *= inVal;
        }
        localSum += inVal;
        if (offset < 1024) {
            aggBuf[offset] = 0.0;
        }
    }
    __syncthreads();
    aggBuf[threadOffset] = localSum;

    for (uint j = 1; j < numInputs; j *= 2) {
        __syncthreads();
        uint otherIndex = threadOffset ^ j;
        scalar_t otherValue = aggBuf[otherIndex];
        localSum = localSum + otherValue;
        __syncthreads();
        if (otherIndex >= numThreads) {
            // No other thread is going to set this value.
            aggBuf[otherIndex] = localSum;
        }
        aggBuf[threadOffset] = localSum;
    }
    __syncthreads();
    // Introduce determinism by broadcasting from cell 0.
    return aggBuf[0] / (scalar_t)numInputs;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
silu(scalar_t x)
{
    scalar_t sig;
    if (x > 0) {
        float ex = expf(-(float)x);
        sig = (scalar_t)(1.0 - (ex / (1.0 + ex)));
    } else {
        float ex = expf((float)x);
        sig = (scalar_t)(ex / (1.0 + ex));
    }
    return sig * x;
}

template <typename scalar_t>
__global__ void gridnet_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4> weight,
    const torch::PackedTensorAccessor32<scalar_t, 3> bias,
    const torch::PackedTensorAccessor32<scalar_t, 4> initActivations,
    torch::PackedTensorAccessor32<scalar_t, 4> outActivations,
    uint innerIterations,
    scalar_t eps,
    uint m,
    uint n,
    uint k)
{
    // A tensor of size (blockSize + 2) ^ 3
    // Used for storing the current padded input activations.
    __shared__ scalar_t activations[(8 + 2) * (8 + 2) * (8 + 2)];

    // Temporary buffer used for hypercube sum reductions.
    __shared__ scalar_t aggBuf[1024];

    uint blockSize = blockDim.x;   // side-length of each block
    uint batchIdx = blockIdx.x;    // the index within the batch of this block
    uint actBlockIdx = blockIdx.y; // the index of this block within the activation grid
    uint blockX = actBlockIdx % (k / blockSize);
    uint blockY = (actBlockIdx / (k / blockSize)) % (n / blockSize);
    uint blockZ = (actBlockIdx / (k / blockSize)) / (n / blockSize);
    uint numThreads = blockDim.x * blockDim.y * blockDim.z;
    uint threadOffset = threadIdx.x + threadIdx.y * blockSize + threadIdx.z * blockSize * blockSize;

    // The weights should be loaded directly into registers.
    scalar_t localWeights[27];
    for (uint i = 0; i < 27; i++) {
        localWeights[i] = weight[i][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];
    }
    scalar_t localBias = bias[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];

    // Copy activations or zero-initialize them when out-of-bounds.
    uint activationsSize = (blockSize + 2) * (blockSize + 2) * (blockSize + 2);
    for (uint i = 0; i < activationsSize; i += numThreads) {
        uint offset = threadOffset + i;
        uint globalX = (offset % (blockSize + 2)) + blockX * blockSize;
        uint globalY = (offset / (blockSize + 2)) % (blockSize + 2) + blockY * blockSize;
        uint globalZ = (offset / (blockSize + 2)) / (blockSize + 2) + blockZ * blockSize;
        scalar_t loadedValue = 0.0;
        if (globalX > 0 && globalY > 0 && globalZ > 0 &&
            globalX <= k && globalY <= n && globalZ <= m) {
            loadedValue = initActivations[batchIdx][globalZ - 1][globalY - 1][globalX - 1];
        }
        if (offset < activationsSize) {
            activations[offset] = loadedValue;
        }
    }

    // Perform all iterations in-place.
    uint paddedOffset = threadIdx.x + 1 + (blockSize + 2) * (threadIdx.y + 1 + (blockSize + 2) * (threadIdx.z + 1));
    for (uint i = 0; i < innerIterations; i++) {
        // Wait for activations to be available.
        __syncthreads();

        // Communicate per-block normalization.
        scalar_t localAct = activations[paddedOffset];
        scalar_t mean = reduce_mean(
            activations, aggBuf, false, activationsSize, threadOffset, numThreads);
        scalar_t sqMean = reduce_mean(
            activations, aggBuf, true, activationsSize, threadOffset, numThreads);
        scalar_t std = (scalar_t)sqrtf(fmaxf((float)(sqMean - mean * mean), 0.0));

        // Compute dot product.
        scalar_t dot = localBias;
        for (uint a = 0; a < 3; a++) {
            for (uint b = 0; b < 3; b++) {
                for (uint c = 0; c < 3; c++) {
                    uint cellOffset = threadIdx.x + c + (blockSize + 2) * (threadIdx.y + b + (blockSize + 2) * (threadIdx.z + a));
                    scalar_t act = (activations[cellOffset] - mean) / (std + eps);
                    scalar_t weight = localWeights[a * 9 + b * 3 + c];
                    dot += act * weight;
                }
            }
        }

        // Activation and residual connection.
        scalar_t output = silu(dot) + localAct;

        // Don't overwrite activations while dot products are
        // still being computed.
        __syncthreads();
        activations[paddedOffset] = output;
    }
    __syncthreads();

    // Write the final activations.
    outActivations[batchIdx][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x] = activations[paddedOffset];
}

void gridnet_cuda_forward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint blockSize,
    float eps)
{
    const int batchSize = initActivations.size(0);
    const uint m = initActivations.size(1);
    const uint n = initActivations.size(2);
    const uint k = initActivations.size(3);

    const dim3 threads(blockSize, blockSize, blockSize);
    const dim3 blocks(batchSize, (m / blockSize) * (n / blockSize) * (k / blockSize));
    AT_DISPATCH_FLOATING_TYPES(
        weight.scalar_type(),
        "gridnet_cuda_forward",
        ([&] { gridnet_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                   weight.packed_accessor32<scalar_t, 4>(),
                   bias.packed_accessor32<scalar_t, 3>(),
                   initActivations.packed_accessor32<scalar_t, 4>(),
                   outActivations.packed_accessor32<scalar_t, 4>(),
                   innerIterations,
                   eps,
                   m,
                   n,
                   k); }));
}
