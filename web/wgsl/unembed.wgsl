@group(0) @binding(0) var<storage, read> inSize: u32;
@group(0) @binding(1) var<storage, read> outSize: u32;

// [inSize]
@group(0) @binding(2) var<storage, read> inputs: array<f32>;

// [inSize x outSize]
@group(0) @binding(3) var<storage, read> weight: array<f32>;

// [outSize]
@group(0) @binding(4) var<storage, read> bias: array<f32>;

// [outSize]
@group(0) @binding(5) var<storage, read_write> outputs: array<f32>;

var<workgroup> localSums: array<f32, 256>;

// Apply a matrix-vector product.
@compute @workgroup_size(256)
fn unembed(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    // Each group of 32 threads works on one output.
    let outputIndex = ctaId.x * 8 + (tid / 32);
    var localSum: f32 = 0.0;
    if (tid % 32 == 0 && outputIndex < outSize) {
        localSum = bias[outputIndex];
    }
    for (var i = 0u; i < inSize; i += 32u) {
        let localIndex = i + (tid % 32);
        var w: f32 = 0.0;
        var x: f32 = 0.0;
        if (localIndex < inSize) {
            x = inputs[localIndex];
        }
        if (outputIndex < outSize && localIndex < inSize) {
            w = weight[outputIndex*inSize + localIndex];
        }
        localSum += x * w;
    }

    // Reduce across each group of 32 threads.
    localSums[tid] = localSum;
    workgroupBarrier();
    for (var i = 1u; i < 32; i *= 2u) {
        let otherValue = localSums[tid ^ i];
        workgroupBarrier();
        localSum += otherValue;
        localSums[tid] = localSum;
        workgroupBarrier();
    }

    if (tid % 32 == 0) {
        outputs[outputIndex] = localSums[tid];
    }
}
