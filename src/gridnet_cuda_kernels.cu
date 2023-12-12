#include "activations.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_sum(
    scalar_t input,
    scalar_t *aggBuf,
    uint threadOffset,
    uint numThreads)
{
    // Don't overwrite aggBuf before other calls are done.
    __syncthreads();

    // Store one value per thread, and pad with zeros.
    aggBuf[threadOffset] = input;
    for (uint i = numThreads; i < 1024; i += numThreads) {
        uint offset = i + threadOffset;
        if (offset < 1024) {
            aggBuf[offset] = 0.0;
        }
    }

    scalar_t localSum = input;
    for (uint j = 1; j < numThreads; j *= 2) {
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
    return aggBuf[0];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t reduce_mean(
    scalar_t *inputBuf,
    scalar_t *aggBuf,
    bool square,
    uint numInputs,
    uint threadOffset,
    uint numThreads)
{
    scalar_t localSum = 0.0;
    for (uint i = 0; i < numInputs; i += numThreads) {
        uint offset = i + threadOffset;
        scalar_t inVal = 0.0;
        if (offset < numInputs) {
            inVal = inputBuf[offset];
        }
        if (square) {
            inVal *= inVal;
        }
        localSum += inVal;
    }
    return reduce_sum(localSum, aggBuf, threadOffset, numThreads) / (scalar_t)numInputs;
}

template <typename scalar_t, typename Act, bool normalize>
__global__ void __launch_bounds__(512, 1) gridnet_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4> weight,
    const torch::PackedTensorAccessor32<scalar_t, 3> bias,
    const torch::PackedTensorAccessor32<scalar_t, 3> scale,
    const torch::PackedTensorAccessor32<scalar_t, 4> initActivations,
    torch::PackedTensorAccessor32<scalar_t, 4> outActivations,
    uint innerIterations,
    scalar_t eps)
{
    // A tensor of size (blockSize + 2) ^ 3
    // Used for storing the current padded input activations.
    __shared__ scalar_t activations[(8 + 2) * (8 + 2) * (8 + 2)];

    // Temporary buffer used for hypercube sum reductions.
    __shared__ scalar_t aggBuf[1024];

    static_assert(Activation<Act, scalar_t>::implemented);

    uint m = bias.size(0);
    uint n = bias.size(1);
    uint k = bias.size(2);
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
    scalar_t localScale = scale[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];

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
        scalar_t mean;
        scalar_t std;
        if (normalize) {
            mean = reduce_mean(
                activations, aggBuf, false, activationsSize, threadOffset, numThreads);
            scalar_t sqMean = reduce_mean(
                activations, aggBuf, true, activationsSize, threadOffset, numThreads);
            std = (scalar_t)sqrtf(fmaxf((float)(sqMean - mean * mean), 0.0));
        }

        // Compute dot product.
        scalar_t dot = localBias;
        for (uint a = 0; a < 3; a++) {
            for (uint b = 0; b < 3; b++) {
                for (uint c = 0; c < 3; c++) {
                    uint cellOffset = threadIdx.x + c + (blockSize + 2) * (threadIdx.y + b + (blockSize + 2) * (threadIdx.z + a));
                    scalar_t act = activations[cellOffset];
                    if (normalize) {
                        act = (act - mean) / (std + eps);
                    }
                    scalar_t weight = localWeights[a * 9 + b * 3 + c];
                    dot += act * weight;
                }
            }
        }

        // Activation and residual connection.
        scalar_t output = Activation<Act, scalar_t>::forward(dot) * localScale + localAct;

        // Don't overwrite activations while dot products are
        // still being computed.
        __syncthreads();
        activations[paddedOffset] = output;
    }
    __syncthreads();

    // Write the final activations.
    outActivations[batchIdx][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x] = activations[paddedOffset];
}

template <typename scalar_t, typename Act, uint iterationsBufferSize, bool normalize>
__global__ void __launch_bounds__(512, 1) gridnet_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4> weight,
    const torch::PackedTensorAccessor32<scalar_t, 3> bias,
    const torch::PackedTensorAccessor32<scalar_t, 3> scale,
    const torch::PackedTensorAccessor32<scalar_t, 4> initActivations,
    const torch::PackedTensorAccessor32<scalar_t, 4> outGrads,
    torch::PackedTensorAccessor32<scalar_t, 4> weightGradOut,
    torch::PackedTensorAccessor32<scalar_t, 3> biasGradOut,
    torch::PackedTensorAccessor32<scalar_t, 3> scaleGradOut,
    torch::PackedTensorAccessor32<scalar_t, 4> activationsGradOut,
    uint innerIterations,
    scalar_t eps)
{
    // A tensor of size (blockSize + 2) ^ 3
    // Used for storing the current padded input activations.
    __shared__ scalar_t activations[(8 + 2) * (8 + 2) * (8 + 2)];

    // A tensor of activation gradients which are accumulated
    // across each backwards step.
    // The border around the activations is also maintained,
    // since our output depends on these values despite never
    // modifying them.
    __shared__ scalar_t activationsGradAcc[(8 + 2) * (8 + 2) * (8 + 2)];

    // Temporary buffer used for hypercube sum reductions.
    __shared__ scalar_t aggBuf[1024];

    static_assert(Activation<Act, scalar_t>::implemented);

    uint blockSize = blockDim.x;   // side-length of each block
    uint batchIdx = blockIdx.x;    // the index within the batch of this block
    uint actBlockIdx = blockIdx.y; // the index of this block within the activation grid
    uint m = bias.size(0);
    uint n = bias.size(1);
    uint k = bias.size(2);
    uint blockX = actBlockIdx % (k / blockSize);
    uint blockY = (actBlockIdx / (k / blockSize)) % (n / blockSize);
    uint blockZ = (actBlockIdx / (k / blockSize)) / (n / blockSize);
    uint numThreads = blockDim.x * blockDim.y * blockDim.z;
    uint threadOffset = threadIdx.x + threadIdx.y * blockSize + threadIdx.z * blockSize * blockSize;

    // The weights should be loaded directly into registers.
    scalar_t localWeights[27];
    scalar_t localWeightsGradAcc[27];
    for (uint i = 0; i < 27; i++) {
        localWeights[i] = weight[i][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];
        localWeightsGradAcc[i] = 0;
    }
    scalar_t localBias = bias[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];
    scalar_t localBiasGradAcc = 0.0;
    scalar_t localScale = scale[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];
    scalar_t localScaleGradAcc = 0.0;

    // Copy initial activations or zero-initialize them when out-of-bounds.
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
            activationsGradAcc[offset] = 0;
        }
    }

    // Perform all forward iterations, but cache the intermediate
    // activations in a buffer.
    scalar_t activationsBuffer[iterationsBufferSize];
    uint paddedOffset = threadIdx.x + 1 + (blockSize + 2) * (threadIdx.y + 1 + (blockSize + 2) * (threadIdx.z + 1));
    for (uint i = 0; i < innerIterations; i++) {
        // Wait for activations to be available.
        __syncthreads();

        // Communicate per-block normalization.
        scalar_t localAct = activations[paddedOffset];
        activationsBuffer[i] = localAct;
        scalar_t mean;
        scalar_t std;
        if (normalize) {
            mean = reduce_mean(
                activations, aggBuf, false, activationsSize, threadOffset, numThreads);
            scalar_t sqMean = reduce_mean(
                activations, aggBuf, true, activationsSize, threadOffset, numThreads);
            std = (scalar_t)sqrtf(fmaxf((float)(sqMean - mean * mean), 0.0));
        }

        // Compute dot product.
        scalar_t dot = localBias;
        for (uint a = 0; a < 3; a++) {
            for (uint b = 0; b < 3; b++) {
                for (uint c = 0; c < 3; c++) {
                    uint cellOffset = threadIdx.x + c + (blockSize + 2) * (threadIdx.y + b + (blockSize + 2) * (threadIdx.z + a));
                    scalar_t act = activations[cellOffset];
                    if (normalize) {
                        act = (act - mean) / (std + eps);
                    }
                    scalar_t weight = localWeights[a * 9 + b * 3 + c];
                    dot += act * weight;
                }
            }
        }

        // Activation and residual connection.
        scalar_t output = Activation<Act, scalar_t>::forward(dot) * localScale + localAct;

        // Don't overwrite activations while dot products are
        // still being computed.
        __syncthreads();
        activations[paddedOffset] = output;
    }
    __syncthreads();

    // This will be updated per loop iteration to be the gradient
    // of the loss with respect to this thread's output.
    scalar_t outGrad = outGrads[batchIdx][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x];

    // Gradient of residual connection.
    activationsGradAcc[paddedOffset] = outGrad;

    for (int i = innerIterations - 1; i >= 0; i--) {
        // Restore to an intermediate step of the forward pass.
        activations[paddedOffset] = activationsBuffer[i];
        __syncthreads();

        // Communicate per-block normalization.
        scalar_t mean;
        scalar_t std;
        if (normalize) {
            mean = reduce_mean(
                activations, aggBuf, false, activationsSize, threadOffset, numThreads);
            scalar_t sqMean = reduce_mean(
                activations, aggBuf, true, activationsSize, threadOffset, numThreads);
            std = (scalar_t)sqrtf(fmaxf((float)(sqMean - mean * mean), 0.0));
        }

        // Recompute the input to the activation.
        scalar_t dot = localBias;
        for (uint a = 0; a < 3; a++) {
            for (uint b = 0; b < 3; b++) {
                for (uint c = 0; c < 3; c++) {
                    uint cellOffset = threadIdx.x + c + (blockSize + 2) * (threadIdx.y + b + (blockSize + 2) * (threadIdx.z + a));
                    scalar_t act = activations[cellOffset];
                    if (normalize) {
                        act = (act - mean) / (std + eps);
                    }
                    scalar_t weight = localWeights[a * 9 + b * 3 + c];
                    dot += act * weight;
                }
            }
        }

        // Outer grad for all other outputs.
        localScaleGradAcc += Activation<Act, scalar_t>::forward(dot) * outGrad;
        scalar_t innerGrad = Activation<Act, scalar_t>::backward(dot) * outGrad * localScale;

        // Gradients will flow to the mean and std
        // and then we will propagate those gradients
        // separately to the inputs across all threads.
        scalar_t meanGrad = 0.0;
        scalar_t stdGrad = 0.0; // the grad w.r.t. 1/(std+eps)

        // Accumulate bias/weight grads, as well as mean/std grads.
        localBiasGradAcc += innerGrad;
        for (uint a = 0; a < 3; a++) {
            for (uint b = 0; b < 3; b++) {
                for (uint c = 0; c < 3; c++) {
                    uint cellOffset = threadIdx.x + c + (blockSize + 2) * (threadIdx.y + b + (blockSize + 2) * (threadIdx.z + a));
                    scalar_t rawAct = activations[cellOffset];
                    scalar_t act = rawAct;
                    if (normalize) {
                        act = (rawAct - mean) / (std + eps);
                    }
                    localWeightsGradAcc[a * 9 + b * 3 + c] += act * innerGrad;
                    scalar_t weight = localWeights[a * 9 + b * 3 + c];
                    if (normalize) {
                        atomicAdd(&activationsGradAcc[cellOffset], (weight / (std + eps)) * innerGrad);
                    } else {
                        atomicAdd(&activationsGradAcc[cellOffset], weight * innerGrad);
                    }
                    if (normalize) {
                        meanGrad -= (weight / (std + eps)) * innerGrad;
                        stdGrad += (rawAct - mean) * weight * innerGrad;
                    }
                }
            }
        }

        if (normalize) {
            // Accumulate gradients of input activations w.r.t. mean/std
            meanGrad = reduce_sum(meanGrad, aggBuf, threadOffset, numThreads);
            stdGrad = reduce_sum(stdGrad, aggBuf, threadOffset, numThreads);

            //   d/dx of 1/(std+eps)
            // = -1/(x+eps)^2 * d/dx std
            stdGrad *= -1 / ((std + eps) * (std + eps));

            for (uint i = 0; i < activationsSize; i += numThreads) {
                uint offset = i + threadOffset;
                if (offset < activationsSize) {
                    scalar_t activation = activations[offset];
                    scalar_t n = (scalar_t)activationsSize;
                    activationsGradAcc[offset] += meanGrad / n;
                    // The fmaxf() isn't actually correct, but helps us prevent
                    // ill-behaved gradients.
                    activationsGradAcc[offset] += stdGrad * (activation / n - mean / n) / fmaxf(std, eps);
                }
            }
        }

        __syncthreads();
        outGrad = activationsGradAcc[paddedOffset];
    }

    // Accumulate the final activations grad. This is roughly
    // the inverse of loading the activations.
    for (uint i = 0; i < activationsSize; i += numThreads) {
        uint offset = threadOffset + i;
        uint globalX = (offset % (blockSize + 2)) + blockX * blockSize;
        uint globalY = (offset / (blockSize + 2)) % (blockSize + 2) + blockY * blockSize;
        uint globalZ = (offset / (blockSize + 2)) / (blockSize + 2) + blockZ * blockSize;
        if (globalX > 0 && globalY > 0 && globalZ > 0 &&
            globalX <= k && globalY <= n && globalZ <= m &&
            offset < activationsSize) {
            scalar_t acc = activationsGradAcc[offset];
            atomicAdd(&activationsGradOut[batchIdx][globalZ - 1][globalY - 1][globalX - 1], acc);
        }
    }

    // Weight updates are also atomic because they are summed
    // across the batch.
    for (uint i = 0; i < 27; i++) {
        atomicAdd(
            &weightGradOut[i][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x],
            localWeightsGradAcc[i]);
    }
    atomicAdd(
        &biasGradOut[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x],
        localBiasGradAcc);
    atomicAdd(
        &scaleGradOut[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x],
        localScaleGradAcc);
}

void gridnet_cuda_forward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor initActivations,
    torch::Tensor outActivations,
    uint innerIterations,
    uint blockSize,
    float eps,
    bool normalize,
    std::string &activation)
{
    const int batchSize = initActivations.size(0);
    const uint m = initActivations.size(1);
    const uint n = initActivations.size(2);
    const uint k = initActivations.size(3);

    const dim3 threads(blockSize, blockSize, blockSize);
    const dim3 blocks(batchSize, (m / blockSize) * (n / blockSize) * (k / blockSize));

#define FORWARD_WITH_ACT(act)                                                                                                                                                                                                                                                                                                                    \
    if (normalize) {                                                                                                                                                                                                                                                                                                                             \
        AT_DISPATCH_FLOATING_TYPES(                                                                                                                                                                                                                                                                                                              \
            weight.scalar_type(),                                                                                                                                                                                                                                                                                                                \
            "gridnet_cuda_forward",                                                                                                                                                                                                                                                                                                              \
            ([&] { gridnet_cuda_forward_kernel<scalar_t, act, true><<<blocks, threads>>>(                                                                                                                                                                                                                                                        \
                       weight.packed_accessor32<scalar_t, 4>(),                                                                                                                                                                                                                                                                                  \
                       bias.packed_accessor32<scalar_t, 3>(),                                                                                                                                                                                                                                                                                    \
                       scale.packed_accessor32<scalar_t, 3>(),                                                                                                                                                                                                                                                                                   \
                       initActivations.packed_accessor32<scalar_t, 4>(),                                                                                                                                                                                                                                                                         \
                       outActivations.packed_accessor32<scalar_t, 4>(),                                                                                                                                                                                                                                                                          \
                       innerIterations,                                                                                                                                                                                                                                                                                                          \
                       eps); }));  \
    } else {                                                                                                                                                                                                                                                                                                                                     \
        AT_DISPATCH_FLOATING_TYPES(                                                                                                                                                                                                                                                                                                              \
            weight.scalar_type(),                                                                                                                                                                                                                                                                                                                \
            "gridnet_cuda_forward",                                                                                                                                                                                                                                                                                                              \
            ([&] { gridnet_cuda_forward_kernel<scalar_t, act, false><<<blocks, threads>>>(                                                                                                                                                                                                                                                       \
                       weight.packed_accessor32<scalar_t, 4>(),                                                                                                                                                                                                                                                                                  \
                       bias.packed_accessor32<scalar_t, 3>(),                                                                                                                                                                                                                                                                                    \
                       scale.packed_accessor32<scalar_t, 3>(),                                                                                                                                                                                                                                                                                   \
                       initActivations.packed_accessor32<scalar_t, 4>(),                                                                                                                                                                                                                                                                         \
                       outActivations.packed_accessor32<scalar_t, 4>(),                                                                                                                                                                                                                                                                          \
                       innerIterations,                                                                                                                                                                                                                                                                                                          \
                       eps); })); \
    }

    if (activation == "relu") {
        FORWARD_WITH_ACT(ReLU);
    } else if (activation == "leaky_relu") {
        FORWARD_WITH_ACT(LeakyReLU);
    } else if (activation == "silu") {
        FORWARD_WITH_ACT(SiLU);
    }
}

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
    bool normalize,
    std::string &activation)
{
    const int batchSize = initActivations.size(0);
    const uint m = initActivations.size(1);
    const uint n = initActivations.size(2);
    const uint k = initActivations.size(3);

    const dim3 threads(blockSize, blockSize, blockSize);
    const dim3 blocks(batchSize, (m / blockSize) * (n / blockSize) * (k / blockSize));

    // Macro for instantiating many different template variations.
#define BACKWARD(inner, norm, act) AT_DISPATCH_FLOATING_TYPES(                            \
    weight.scalar_type(),                                                                 \
    "gridnet_cuda_backward",                                                              \
    ([&] { gridnet_cuda_backward_kernel<scalar_t, act, inner, norm><<<blocks, threads>>>( \
               weight.packed_accessor32<scalar_t, 4>(),                                   \
               bias.packed_accessor32<scalar_t, 3>(),                                     \
               scale.packed_accessor32<scalar_t, 3>(),                                    \
               initActivations.packed_accessor32<scalar_t, 4>(),                          \
               outGrads.packed_accessor32<scalar_t, 4>(),                                 \
               weightGradOut.packed_accessor32<scalar_t, 4>(),                            \
               biasGradOut.packed_accessor32<scalar_t, 3>(),                              \
               scaleGradOut.packed_accessor32<scalar_t, 3>(),                             \
               activationsGradOut.packed_accessor32<scalar_t, 4>(),                       \
               innerIterations,                                                           \
               eps); }));

#define BACKWARD_FOR_ACT(act)                                                                  \
    if (normalize) {                                                                           \
        if (innerIterations <= 8) {                                                            \
            BACKWARD(8, true, act);                                                            \
        } else if (innerIterations <= 16) {                                                    \
            BACKWARD(16, true, act);                                                           \
        } else if (innerIterations <= 32) {                                                    \
            BACKWARD(32, true, act);                                                           \
        } else {                                                                               \
            throw std::runtime_error("cannot backprop through more than 32 inner iterations"); \
        }                                                                                      \
    } else {                                                                                   \
        if (innerIterations <= 8) {                                                            \
            BACKWARD(8, false, act);                                                           \
        } else if (innerIterations <= 16) {                                                    \
            BACKWARD(16, false, act);                                                          \
        } else if (innerIterations <= 32) {                                                    \
            BACKWARD(32, false, act);                                                          \
        } else {                                                                               \
            throw std::runtime_error("cannot backprop through more than 32 inner iterations"); \
        }                                                                                      \
    }

    if (activation == "relu") {
        BACKWARD_FOR_ACT(ReLU);
    } else if (activation == "leaky_relu") {
        BACKWARD_FOR_ACT(LeakyReLU);
    } else if (activation == "silu") {
        BACKWARD_FOR_ACT(SiLU);
    }
}
