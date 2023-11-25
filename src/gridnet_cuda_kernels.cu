#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

template <typename scalar_t>
__global__ void gridnet_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> initActivations,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> outActivations,
    uint innerIterations,
    uint m,
    uint n,
    uint k,
    scalar_t eps)
{
    // Contains the following:
    //  - activations: a tensor of size (blockSize + 2) ^ 3
    //  - aggBuf: a tensor of size blockSize ^ 3
    extern __shared__ scalar_t activations[(8 + 2) * (8 + 2) * (8 + 2)];
    extern __shared__ scalar_t aggBuf[8 * 8 * 8];

    uint batchSize = gridDim.x;    // outer dimension of activations
    uint blockSize = blockDim.x;   // side-length of each block
    uint batchIdx = blockIdx.x;    // the index within the batch of this block
    uint actBlockIdx = blockIdx.y; // the index of this block within the activation grid
    uint blockX = actBlockIdx % k;
    uint blockY = (actBlockIdx / k) % n;
    uint blockZ = (actBlockIdx / k) / n;
    uint numThreads = threadIdx.x * threadIdx.y * threadIdx.z;
    uint threadOffset = threadIdx.x + threadIdx.y * blockSize + threadIdx.z * blockSize * blockSize;

    uint activationsSize = (blockSize + 2) * (blockSize + 2) * (blockSize + 2);
    scalar_t *activations = shared;
    scalar_t *aggBuf = &shared[activationsSize];

    // The weights should be loaded directly into registers.
    scalar_t localWeights[27];
    for (uint i = 0; i < 27; i++) {
        localWeights[i] = weight[i][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];
    }
    scalar_t localBias = bias[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];

    // Copy activations or zero-initialize them when out-of-bounds.
    for (uint i = 0; i < activationsSize; i += numThreads) {
        uint offset = threadOffset + i;
        uint globalX = (offset % (blockSize + 2)) + blockX;
        uint globalY = (offset / (blockSize + 2)) % (blockSize + 2) + blockY;
        uint globalZ = (offset / (blockSize + 2)) / (blockSize + 2) + blockZ;
        if (globalX > 0 && globalY > 0 && globalZ > 0 &&
            blockX <= k && blockY <= n && blockZ <= m) {
            activations[offset] = initActivations[batchIdx][globalZ - 1][globalY - 1][globalX - 1];
        } else {
            activations[offset] = 0.0;
        }
    }

    // All iterations are performed in-place.
    uint paddedOffset = threadIdx.x + 1 + (blockSize + 2) * (threadIdx.y + 1 + (blockSize + 2) * (threadIdx.z + 1));
    for (uint i = 0; i < innerIterations; i++) {
        __syncthreads(); // wait for activations to be available.

        // Communicate per-block normalization.
        scalar_t localAct = activations[paddedOffset];
        scalar_t mean = reduce_mean(localAct, aggBuf, threadOffset, numThreads);
        scalar_t sqMean = reduce_mean(localAct * localAct, aggBuf, threadOffset, numThreads);
        scalar_t std = (scalar_t)sqrtf(maxf((float)(sqMean - mean * mean), 0.0));

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

        // Write the output back immediately, since all other threads are done
        // reading activations (due to mean reduces).
        activations[paddedOffset] = output;
    }
    __syncthreads();

    // Write the final activations.
    outActivations[batchIdx][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x] = activations[paddedOffset];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_mean(
    scalar_t myValue,
    scalar_t *aggBuf,
    uint threadOffset,
    uint numThreads)
{
    __syncthreads(); // Don't overwrite aggBuf before other calls are done.
    for (uint i = numThreads; i < 1024; i += numThreads) {
        uint offset = i + threadOffset;
        if (offset < 1024) {
            aggBuf[offset] = 0;
        }
    }
    aggBuf[threadOffset] = myValue;
    for (uint j = 0; j < 1024; j *= 2) {
        __syncthreads();
        uint otherIndex = threadOffset ^ j;
        scalar_t otherValue = aggBuf[otherIndex];
        myValue = myValue + otherValue;
        __syncthreads();
        if (otherIndex >= numThreads) {
            // No other thread is going to set this value.
            aggBuf[otherIndex] = myValue;
        }
        aggBuf[threadOffset] = myValue;
    }
    __syncthreads();
    // Introduce determinism by broadcasting from cell 0.
    return aggBuf[0] / (scalar_t)numThreads;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
silu(scalar_t x)
{
    float ex = expf((float)x);
    scalar_t sig = (scalar_t)(ex / (1.0 + ex));
    return sig * x;
}