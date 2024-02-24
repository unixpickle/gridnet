@group(0) @binding(0) var<storage, read> iterations: u32;
@group(0) @binding(1) var<storage, read> gridSize: u32;

// [grid x grid x grid]
@group(0) @binding(2) var<storage, read_write> activations: array<f32>;

// [27 x grid x grid x grid]
@group(0) @binding(3) var<storage, read> weight: array<f32>;

// [grid x grid x grid]
@group(0) @binding(4) var<storage, read> bias: array<f32>;

// [grid x grid x grid]
@group(0) @binding(5) var<storage, read> scale: array<f32>;

// Stores the cube of intermediate activations.
var<workgroup> sharedActivations: array<f32, (8 + 2) * (8 + 2) * (8 + 2)>;

// We have 8x8x8 = 512 activations, but we can only use a block
// size of 256, so we work on two cells at once.
@compute @workgroup_size(256)
fn gridnet8x8x8(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    let blockX = ctaId.x % (gridSize / 8);
    let blockY = (ctaId.x / (gridSize / 8)) % (gridSize / 8);
    let blockZ = ctaId.x / ((gridSize / 8) * (gridSize / 8));
    let activationsSize = (8 + 2) * (8 + 2) * (8 + 2);
    for (var i = u32(0); i < activationsSize; i += 256) {
        let offset = tid + i;
        let globalX = (offset % (8 + 2)) + blockX * blockSize;
        let globalY = (offset / (8 + 2)) % (8 + 2) + blockY * blockSize;
        let globalZ = (offset / (8 + 2)) / (8 + 2) + blockZ * blockSize;
        var loadedValue: f32 = 0.0;
        if (globalX > 0 && globalY > 0 && globalZ > 0 &&
            globalX <= gridSize && globalY <= gridSize && globalZ <= gridSize) {
            loadedValue = activations[((globalZ-1)*gridSize + (globalY-1))*gridSize + globalX - 1];
        }
        if (offset < activationsSize) {
            sharedActivations[offset] = loadedValue;
        }
    }

    let threadX = (tid % 4) * 2;
    let threadY = (tid / 4) % 8;
    let threadZ = (tid / 4) / 8;

    // This is a local of thread-local data, which may result
    // in a register spill on some GPUs. We may want to quantize
    // this in the future (e.g. to bf16) to save memory.
    var localWeights: array<f32; 27*2>;
    var localBiases: array<f32; 2>;
    var localScales: array<f32; 2>;

    for (var i = u32(0); i < 27; i++) {
        localWeights[i] = weight[((i*gridSize + blockZ*8 + threadZ)*gridSize + blockY*8 + threadY)*gridSize + blockX*8 + threadX];
        localWeights[i+27] = weight[((i*gridSize + blockZ*8 + threadZ)*gridSize + blockY*8 + threadY)*gridSize + blockX*8 + threadX + 1];
    }
    localBiases[0] = bias[((blockZ*8 + threadZ)*gridSize + blockY*8 + threadY)*gridSize + blockX*8 + threadX];
    localBiases[1] = bias[((blockZ*8 + threadZ)*gridSize + blockY*8 + threadY)*gridSize + blockX*8 + threadX + 1];
    localScales[0] = scale[((blockZ*8 + threadZ)*gridSize + blockY*8 + threadY)*gridSize + blockX*8 + threadX];
    localScales[1] = scale[((blockZ*8 + threadZ)*gridSize + blockY*8 + threadY)*gridSize + blockX*8 + threadX + 1];

    let paddedOffset: array<u32; 2> = [
        threadX + 1 + (8 + 2) * (threadY + 1 + (8 + 2) * (threadZ + 1)),
        1 + threadX + 1 + (8 + 2) * (threadY + 1 + (8 + 2) * (threadZ + 1)),
    ];
    for (var i: u32 = 0; i < iterations; i++) {
        // Wait for activations to be available.
        workgroupBarrier();

        var output: array<f32; 2>;
        for (var j: u32 = 0; j < 2; j++) {
            let localAct = activations[paddedOffset[j]];
            var dot = localBiases[j];
            for (var a = u32(0); a < 3; a++) {
                for (var b = u32(0); b < 3; b++) {
                    for (var c = u32(0); c < 3; c++) {
                        let cellOffset = threadX + j + c + (8 + 2) * (threadY + b + (8 + 2) * (threadZ + a));
                        let act = activations[cellOffset];
                        let weight = localWeights[j * 27 + a * 9 + b * 3 + c];
                        dot += act * weight;
                    }
                }
            }
            output[j] = leakyReLu(dot) * localScales[j] + localAct;
        }

        // Don't overwrite activations while dot products are
        // still being computed.
        workgroupBarrier();
        for (var j: u32 = 0; j < 2; j++) {
            activations[paddedOffset[j]] = output[j];
        }
    }
    workgroupBarrier();

    // Write the final activations.
    for (var j = u32(0); j < 2; j++) {
        activations[((blockZ*8 + threadZ)*gridSize + (blockY*8 + threadY))*gridSize + blockX*8 + threadX + j] = activations[paddedOffset[j]];
    }
}