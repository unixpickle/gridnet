@group(0) @binding(0) var<storage, read> numInputs: u32;
@group(0) @binding(1) var<storage, read_write> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> sum: f32;
@group(0) @binding(6) var<storage, read_write> sqSum: f32;

// Perform an affine transformation and normalization.
@compute @workgroup_size(256)
fn affine(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    var localValue: f32 = 0.0;
    var localWeight: f32 = 0.0;
    var localBias: f32 = 0.0;
    let globalIndex = tid + ctaId.x*256;
    if (globalIndex < numInputs) {
        localValue = inputs[globalIndex];
        localWeight = weight[globalIndex];
        localBias = bias[globalIndex];
    }

    let mean: f32 = sum / f32(numInputs);
    let sqMean: f32 = sqSum / f32(numInputs);
    let stddev = sqrt(max(0, sqMean - mean*mean + 1e-5));

    let output = (localValue-mean)/stddev * localWeight + localBias;
    if (globalIndex < numInputs) {
        outputs[globalIndex] = output;
    }
}
