override gridSize: u32;
override outChannels: u32;

// [gridSize x gridSize x gridSize]
@group(0) @binding(0) var<storage, read> inputs: array<f32>;

// [gridSize x outChannels]
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;

// Slice a grid like grid[:, :, -outChannels:].
@compute @workgroup_size(256)
fn sliceOutput(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    let outputIndex = ctaId.x * 256 + tid;
    let outerIndex = outputIndex / outChannels;
    let innerIndex = outputIndex % outChannels;
    let inIndex = outerIndex * gridSize + (gridSize - outChannels) + innerIndex;
    if (outputIndex < outChannels * gridSize * gridSize) {
        outputs[outputIndex] = inputs[inIndex];
    }
}
