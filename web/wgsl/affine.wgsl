@group(0) @binding(0) var<storage, read> numInputs: u32;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> sum: f32;
@group(0) @binding(6) var<storage, read> sqSum: f32;

// Perform an affine transformation and normalization.
@compute @workgroup_size(256)
fn affine(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    var localValue = 0.0;
    var localWeight = 0.0;
    var localBias = 0.0;
    let globalIndex = tid + 256 * ctaId.x;
    if (globalIndex < numInputs) {
        localValue = inputs[globalIndex];
        localWeight = weight[globalIndex];
        localBias = bias[globalIndex];
    }

    let mean = sum / f32(numInputs);
    let sqMean = sqSum / f32(numInputs);
    let stddev = sqrt(sqMean - mean*mean + 1e-5);

    let output = ((localValue - mean) / stddev) * localWeight + localBias;
    if (globalIndex < numInputs) {
        outputs[globalIndex] = output;
    }
}
