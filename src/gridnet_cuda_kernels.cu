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
__device__ __forceinline__ scalar_t
silu_grad(scalar_t x)
{
    scalar_t sig;
    if (x > 0) {
        float ex = expf(-(float)x);
        sig = (scalar_t)(1.0 - (ex / (1.0 + ex)));
    } else {
        float ex = expf((float)x);
        sig = (scalar_t)(ex / (1.0 + ex));
    }
    //   d/dx x*sigmoid(x)
    // = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    // = sigmoid(x) * (1 + x*(1-sigmoid(x)))
    return sig * (1 + x * (1 - sig));
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

template <typename scalar_t, uint iterationsBufferSize>
__global__ void gridnet_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4> weight,
    const torch::PackedTensorAccessor32<scalar_t, 3> bias,
    const torch::PackedTensorAccessor32<scalar_t, 4> initActivations,
    const torch::PackedTensorAccessor32<scalar_t, 4> outGrads,
    torch::PackedTensorAccessor32<scalar_t, 4> weightGradOut,
    torch::PackedTensorAccessor32<scalar_t, 3> biasGradOut,
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
        }
    }

    // Perform all forward iterations, but cache the intermediate
    // activations in a buffer.
    scalar_t activationsBuffer[iterationsBufferSize];
    uint paddedOffset = threadIdx.x + 1 + (blockSize + 2) * (threadIdx.y + 1 + (blockSize + 2) * (threadIdx.z + 1));
    for (uint i = 0; i < innerIterations - 1; i++) {
        // Wait for activations to be available.
        __syncthreads();

        // Communicate per-block normalization.
        scalar_t localAct = activations[paddedOffset];
        activationsBuffer[i] = localAct;
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
        scalar_t localAct = activations[paddedOffset];
        scalar_t mean = reduce_mean(
            activations, aggBuf, false, activationsSize, threadOffset, numThreads);
        scalar_t sqMean = reduce_mean(
            activations, aggBuf, true, activationsSize, threadOffset, numThreads);
        scalar_t std = (scalar_t)sqrtf(fmaxf((float)(sqMean - mean * mean), 0.0));

        // Recompute the input to the activation.
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

        // Outer grad for all other outputs.
        scalar_t innerGrad = silu_grad(dot) * outGrad;

        // Gradients will flow to the mean and std
        // and then we will propagate those gradients
        // separately to the inputs across all threads.
        scalar_t meanGrad = 0.0;
        scalar_t stdGrad = 0.0;

        // Accumulate bias/weight grads, as well as mean/std grads.
        localBiasGradAcc += innerGrad;
        for (uint a = 0; a < 3; a++) {
            for (uint b = 0; b < 3; b++) {
                for (uint c = 0; c < 3; c++) {
                    uint cellOffset = threadIdx.x + c + (blockSize + 2) * (threadIdx.y + b + (blockSize + 2) * (threadIdx.z + a));
                    scalar_t rawAct = activations[cellOffset];
                    scalar_t act = (rawAct - mean) / (std + eps);
                    localWeightsGradAcc[a * 9 + b * 3 + c] += act * innerGrad;
                    scalar_t weight = localWeights[a * 9 + b * 3 + c];
                    atomicAdd(&activationsGradAcc[cellOffset], (weight / (std + eps)) * innerGrad);
                    meanGrad -= (weight / (std + eps)) * innerGrad;

                    //   d/dx of 1/(std+eps)
                    // = -1/(x+eps)^2 * d/dx std
                    stdGrad += (-1 / ((std + eps) * (std + eps))) * (rawAct - mean) * weight * innerGrad;
                }
            }
        }

        // Accumulate gradients of input activations w.r.t. mean/std
        meanGrad = reduce_sum(meanGrad, aggBuf, threadOffset, numThreads);
        stdGrad = reduce_sum(stdGrad, aggBuf, threadOffset, numThreads);
        for (uint i = 0; i < activationsSize; i += numThreads) {
            uint offset = i + threadOffset;
            if (offset < activationsSize) {
                scalar_t activation = activations[offset];
                scalar_t n = (scalar_t)activationsSize;
                activationsGradAcc[offset] += meanGrad / n;
                // The fmaxf() isn't actually correct, but helps us prevent
                // ill-behaved gradients.
                activationsGradAcc[offset] += stdGrad * (activation / n - 1.0 / (2.0 * n)) / fmaxf(std, eps);
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

    // Outputs for weights needn't be atomic; each thread updates
    // its own parameters.
    for (uint i = 0; i < 27; i++) {
        weightGradOut[i][blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x] += localWeightsGradAcc[i];
    }
    biasGradOut[blockZ * blockSize + threadIdx.z][blockY * blockSize + threadIdx.y][blockX * blockSize + threadIdx.x] += localBiasGradAcc;
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

void gridnet_cuda_backward(
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor initActivations,
    torch::Tensor outGrads,
    torch::Tensor weightGradOut,
    torch::Tensor biasGradOut,
    torch::Tensor activationsGradOut,
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
    if (innerIterations <= 8) {
        AT_DISPATCH_FLOATING_TYPES(
            weight.scalar_type(),
            "gridnet_cuda_backward",
            ([&] { gridnet_cuda_backward_kernel<scalar_t, 8><<<blocks, threads>>>(
                       weight.packed_accessor32<scalar_t, 4>(),
                       bias.packed_accessor32<scalar_t, 3>(),
                       initActivations.packed_accessor32<scalar_t, 4>(),
                       outGrads.packed_accessor32<scalar_t, 4>(),
                       weightGradOut.packed_accessor32<scalar_t, 4>(),
                       biasGradOut.packed_accessor32<scalar_t, 3>(),
                       activationsGradOut.packed_accessor32<scalar_t, 4>(),
                       innerIterations,
                       eps); }));
    } else if (innerIterations <= 16) {
        AT_DISPATCH_FLOATING_TYPES(
            weight.scalar_type(),
            "gridnet_cuda_backward",
            ([&] { gridnet_cuda_backward_kernel<scalar_t, 16><<<blocks, threads>>>(
                       weight.packed_accessor32<scalar_t, 4>(),
                       bias.packed_accessor32<scalar_t, 3>(),
                       initActivations.packed_accessor32<scalar_t, 4>(),
                       outGrads.packed_accessor32<scalar_t, 4>(),
                       weightGradOut.packed_accessor32<scalar_t, 4>(),
                       biasGradOut.packed_accessor32<scalar_t, 3>(),
                       activationsGradOut.packed_accessor32<scalar_t, 4>(),
                       innerIterations,
                       eps); }));
    } else if (innerIterations <= 32) {
        AT_DISPATCH_FLOATING_TYPES(
            weight.scalar_type(),
            "gridnet_cuda_backward",
            ([&] { gridnet_cuda_backward_kernel<scalar_t, 32><<<blocks, threads>>>(
                       weight.packed_accessor32<scalar_t, 4>(),
                       bias.packed_accessor32<scalar_t, 3>(),
                       initActivations.packed_accessor32<scalar_t, 4>(),
                       outGrads.packed_accessor32<scalar_t, 4>(),
                       weightGradOut.packed_accessor32<scalar_t, 4>(),
                       biasGradOut.packed_accessor32<scalar_t, 3>(),
                       activationsGradOut.packed_accessor32<scalar_t, 4>(),
                       innerIterations,
                       eps); }));
    } else {
        throw std::runtime_error("cannot backprop through more than 32 inner iterations");
    }
}
