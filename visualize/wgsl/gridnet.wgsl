@group(0) @binding(0) var<storage, read> iterations: u32;
@group(0) @binding(1) var<storage, read> gridSize: u32;

// [grid x grid x grid]
@group(0) @binding(2) var<storage, read> activationsIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> activationsOut: array<f32>;

// [27 x grid x grid x grid]
@group(0) @binding(4) var<storage, read> weight: array<f32>;

// [grid x grid x grid]
@group(0) @binding(5) var<storage, read> bias: array<f32>;

// [grid x grid x grid]
@group(0) @binding(6) var<storage, read> scale: array<f32>;

// Stores the cube of intermediate activations. (8 + 2) ^ 3
var<workgroup> sharedActivations: array<f32, 1000>;

// We have 8x8x8 = 512 activations, but we can only use a block
// size of 256, so we work on two cells at once.
@compute @workgroup_size(256)
fn gridnet8x8x8(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    let blockSize = 8u;

    let blockX = ctaId.x % (gridSize / blockSize);
    let blockY = (ctaId.x / (gridSize / blockSize)) % (gridSize / blockSize);
    let blockZ = (ctaId.x / (gridSize / blockSize)) / (gridSize / blockSize);
    let activationsSize = (blockSize + 2) * (blockSize + 2) * (blockSize + 2);
    for (var i = 0u; i < activationsSize; i += 256u) {
        let offset = tid + i;
        let globalX = (offset % (blockSize + 2)) + blockX * blockSize;
        let globalY = (offset / (blockSize + 2)) % (blockSize + 2) + blockY * blockSize;
        let globalZ = (offset / (blockSize + 2)) / (blockSize + 2) + blockZ * blockSize;
        var loadedValue: f32 = 0.0;
        if (globalX > 0 && globalY > 0 && globalZ > 0 &&
            globalX <= gridSize && globalY <= gridSize && globalZ <= gridSize) {
            loadedValue = activationsIn[globalX - 1 + gridSize * (
                (globalY - 1) + gridSize * (globalZ - 1)
            )];
        }
        if (offset < activationsSize) {
            sharedActivations[offset] = loadedValue;
        }
    }

    let threadX = (tid % (blockSize / 2)) * 2;
    let threadY = (tid / (blockSize / 2)) % blockSize;
    let threadZ = (tid / (blockSize / 2)) / blockSize;

    // This is a local of thread-local data, which may result
    // in a register spill on some GPUs. We may want to quantize
    // this in the future (e.g. to bf16) to save memory.
    var localWeights: array<f32, 54> = array<f32, 54>(); // 27 * 2
    var localBiases: array<f32, 2> = array<f32, 2>();
    var localScales: array<f32, 2> = array<f32, 2>();

    for (var i = 0u; i < 27; i++) {
        localWeights[i] = weight[blockX*blockSize + threadX + gridSize * (
            (i*gridSize + blockZ*blockSize + threadZ)*gridSize + blockY*blockSize + threadY
        )];
        localWeights[i+27] = weight[blockX*blockSize + threadX + 1 + gridSize * (
            (i*gridSize + blockZ*blockSize + threadZ)*gridSize + blockY*blockSize + threadY
        )];
    }
    for (var i = 0u; i < 2; i++) {
        let idx = blockX*blockSize + threadX + i + gridSize * (
            blockY*blockSize + threadY + gridSize * (blockZ*blockSize + threadZ)
        );
        localBiases[i] = bias[idx];
        localScales[i] = scale[idx];
    }

    let firstOffset = threadX + 1 + (blockSize + 2) * (
        threadY + 1 + (blockSize + 2) * (threadZ + 1)
    );
    for (var i = 0u; i < iterations; i++) {
        // Wait for activations to be available.
        workgroupBarrier();

        var output: array<f32, 2> = array<f32, 2>();
        for (var j = 0u; j < 2; j++) {
            let localAct = sharedActivations[firstOffset + j];
            var dot = localBiases[j];
            for (var a = 0u; a < 3; a++) {
                for (var b = 0u; b < 3; b++) {
                    for (var c = 0u; c < 3; c++) {
                        let cellOffset = threadX + j + c + (blockSize + 2) * (
                            threadY + b + (blockSize + 2) * (threadZ + a)
                        );
                        let act = sharedActivations[cellOffset];
                        let weight = localWeights[j * 27 + a * 9 + b * 3 + c];
                        dot += act * weight;
                    }
                }
            }
            output[j] = leakyReLU(dot) * localScales[j] + localAct;
        }

        // Don't overwrite activations while dot products are
        // still being computed.
        workgroupBarrier();
        for (var j = 0u; j < 2; j++) {
            sharedActivations[firstOffset + j] = output[j];
        }
    }
    workgroupBarrier();

    // Write the final activations.
    for (var j = 0u; j < 2; j++) {
        activationsOut[
            blockX * blockSize + threadX + j + gridSize * (
                (blockY * blockSize + threadY) + gridSize * (blockZ * blockSize + threadZ)
            )
        ] = sharedActivations[firstOffset + j];
    }
}

fn leakyReLU(x: f32) -> f32 {
    if (x < 0) {
        return 0.01 * x;
    } else {
        return x;
    }
}