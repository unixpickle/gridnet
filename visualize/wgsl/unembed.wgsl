override inSize: u32;
override outSize: u32;

@group(0) @binding(0) var<storage, read> inputs: array<f32>;

// [inSize x outSize]
@group(0) @binding(1) var<storage, read> weight: array<f32>;

// [outSize]
@group(0) @binding(2) var<storage, read> bias: array<f32>;

// [outSize]
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;

var<workgroup> inputActivations: array<f32, inSize>;
var<workgroup> localSums: array<f32, 256>;

// Apply a matrix-vector product.
@compute @workgroup_size(256)
fn unembed(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    // Cooperatively load all of the inputs in every block.
    for (var i = 0u; i < inSize; i += 256) {
        let localIdx = i + tid.x;
        if (localIdx < inSize) {
            inputActivations[localIdx] = inputs[localIdx];
        }
    }
    workgroupBarrier();

    // Each group of 32 threads works on one output.
    let outputIndex = ctaId.x * 8 + (tid.x / 32);
    var localSum: f32 = 0.0;
    if (tid.x % 32 == 0 && outputIndex < outSize) {
        localSum = bias[outputIndex];
    }
    for (var i = 0; i < inSize; i += 32) {
        let localIndex = i + (tid.x % 32);
        var weight: f32 = 0.0;
        var input: f32 = 0.0;
        if (localIndex < inSize) {
            input = inputs[i];
        }
        if (outputIndex < outSize && localIndex < inSize) {
            weight = weight[outputIndex*inSize + localIndex];
        }
        localSum += input * weight;
    }

    // Reduce across each group of 32 threads.
    localSums[tid.x] = localSum;
    workgroupBarrier();
    for (var i = 1; i < 32; i *= 2) {
        let otherValue = localSums[tid.x ^ i];
        workgroupBarrier();
        localSum += otherValue;
        localSums[tid.x] = localSum;
        workgroupBarrier();
    }

    if (tid.x % 32 == 0) {
        outputs[outputIndex] = localSums[tid.x];
    }
}
