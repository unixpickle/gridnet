// [in_ch x height x width]
@group(0) @binding(0) var<storage, read> inputs: array<f32>;

// [out_grid x out_grid x out_grid]
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;

// [out_ch x in_ch x kernel_size x kernel_size]
@group(0) @binding(2) var<storage, read> weight: array<f32>;

// [out_ch]
@group(0) @binding(3) var<storage, read> bias: array<f32>;

// Apply a 4x4 patch embedding to go from a 3x256x256 to the first
// 8 channels along the z axis of a 64x64x64 grid.
@compute @workgroup_size(32)
fn patchEmbedStandard4x4(
    @builtin(workgroup_id) ctaId: vec3u,
    @builtin(local_invocation_index) tid: u32,
) {
    let outChannel = tid % 8;
    let localBias: f32 = bias[outChannel];
    var localWeights: array<f32, 4*4*3> = array<f32, 4*4*3>();
    for (var i: u32 = 0; i < 4*4*3; i++) {
        localWeights[i] = weight[outChannel*4*4*3 + i];
    }

    let row = ctaId.x;
    let firstCol: u32 = tid / 8;
    for (var col = firstCol; col < 64; col += 4) {
        var sum: f32 = localBias;
        for (var ch: u32 = 0; ch < 3; ch++) {
            for (var i: u32 = 0; i < 4; i++) {
                for (var j: u32 = 0; j < 4; j++) {
                    let input = inputs[(ch*256 + (row*4+i))*256 + col*4 + j];
                    let w = localWeights[(ch*4 + i)*4 + j];
                    sum += input * w;
                }
            }
        }
        outputs[(row*64 + col)*64 + outChannel] = sum;
    }
}