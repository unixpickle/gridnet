@group(0) @binding(0) var<storage, read> isFirst: u32;
@group(0) @binding(1) var<storage, read> numInputs: u32;
@group(0) @binding(2) var<storage, read_write> inputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> sumOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> sqSumOut: array<f32>;

var<workgroup> reductionBuffer: array<f32, 512>;

// Compute the mean and second moment of blocks of data.
@compute @workgroup_size(256)
fn reduceMoments(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    var localValue: f32 = 0.0;
    let globalIndex = tid + ctaId.x*256;
    if (globalIndex < numInputs) {
        localValue = inputs[globalIndex];
    }

    let sum: f32 = reduceShared(localValue, tid);
    if (tid == 0) {
        sumOut[ctaId.x] = sum;
    }

    if (isFirst == 1) {
        let sqSum: f32 = reduceShared(localValue*localValue, tid);
        if (tid == 0) {
            sqSumOut[ctaId.x] = sqSum;
        }
    }
}

fn reduceShared(input: f32, localIndex: u32) -> f32 {
    reductionBuffer[localIndex] = input;
    workgroupBarrier();
    for (var i: u32 = 0; i < 8; i++) {
        workgroupBarrier();
        let shift: u32 = u32(1) << i;
        let otherIndex = localIndex ^ shift;
        let ourValue = reductionBuffer[localIndex];
        let otherValue = reductionBuffer[otherIndex];
        workgroupBarrier();
        reductionBuffer[localIndex] = ourValue + otherValue;
        workgroupBarrier();
    }
    return reductionBuffer[localIndex];
}