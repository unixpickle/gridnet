var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
function testWebGPUPatchEmbed() {
    return __awaiter(this, void 0, void 0, function* () {
        const patchEmbCode = yield (yield fetch('/glsl/patch_embed.glsl')).text();
        const weight = Tensor.zeros(new Shape(8, 3, 4, 4));
        const bias = Tensor.zeros(new Shape(8));
        const input = Tensor.zeros(new Shape(3, 256, 256));
        randomize(weight);
        randomize(bias);
        randomize(input);
        const cpuLayer = new PatchEmbed(weight, bias);
        const expectedOutput = cpuLayer.forward(input);
        const output = Tensor.zeros(new Shape(64, 64, 64));
        const sequence = new KernelSequence([
            new ComputePass(patchEmbCode, 'patchEmbedStandard4x4', [
                new Buffer(input.data),
                new Buffer(output.data, output.data),
                new Buffer(weight.data),
                new Buffer(bias.data),
            ], [64]),
        ]);
        yield sequence.execute();
        let maxError = 0.0;
        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                for (let ch = 0; ch < 8; ch++) {
                    const actual = output.get(y, x, ch);
                    const expected = expectedOutput.get(ch, y, x);
                    maxError = Math.max(maxError, Math.abs(actual - expected));
                }
            }
        }
        console.log('patch embed MAE:', maxError);
    });
}
function testWebGPULayerNorm() {
    return __awaiter(this, void 0, void 0, function* () {
        const statsCode = yield (yield fetch('/glsl/moments.glsl')).text();
        const affineCode = yield (yield fetch('/glsl/affine.glsl')).text();
        const input = Tensor.zeros(new Shape(64, 64, 64));
        const weight = Tensor.zeros(new Shape(64, 64, 64));
        const bias = Tensor.zeros(new Shape(64, 64, 64));
        randomize(input);
        randomize(weight);
        randomize(bias);
        const cpuLayer = new LayerNorm(weight, bias);
        const expectedOutput = cpuLayer.forward(input);
        const output = Tensor.zeros(new Shape(64, 64, 64));
        const sizeBuffer = new Buffer(new Uint32Array([64 * 64 * 64]));
        const inBuffer = new Buffer(input.data, null, true);
        const tmp1 = new Buffer(new Float32Array(1024), new Float32Array(1024));
        const tmp2 = new Buffer(new Float32Array(1024), new Float32Array(1024));
        const tmp3 = new Buffer(new Float32Array(4), null, true);
        const tmp4 = new Buffer(new Float32Array(4), null, true);
        const unused = new Buffer(new Float32Array(1), null, true);
        const sequence = new KernelSequence([
            new ComputePass(statsCode, 'reduceMoments', [
                new Buffer(new Uint32Array([1])),
                sizeBuffer,
                inBuffer,
                tmp1,
                tmp2,
            ], [1024]),
            new ComputePass(statsCode, 'reduceMoments', [
                new Buffer(new Uint32Array([0])),
                new Buffer(new Uint32Array([1024])),
                tmp1,
                tmp3,
                unused,
            ], [4]),
            new ComputePass(statsCode, 'reduceMoments', [
                new Buffer(new Uint32Array([0])),
                new Buffer(new Uint32Array([1024])),
                tmp2,
                tmp4,
                unused,
            ], [4]),
            new ComputePass(statsCode, 'reduceMoments', [
                new Buffer(new Uint32Array([0])),
                new Buffer(new Uint32Array([4])),
                tmp3,
                tmp1,
                unused,
            ], [1]),
            new ComputePass(statsCode, 'reduceMoments', [
                new Buffer(new Uint32Array([0])),
                new Buffer(new Uint32Array([4])),
                tmp4,
                tmp2,
                unused,
            ], [1]),
            new ComputePass(affineCode, 'affine', [
                sizeBuffer,
                inBuffer,
                new Buffer(output.data, output.data),
                new Buffer(weight.data),
                new Buffer(bias.data),
                tmp1,
                tmp2,
            ], [1024]),
        ]);
        yield sequence.execute();
        let maxError = 0.0;
        for (let z = 0; z < 64; z++) {
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const actual = output.get(z, y, x);
                    const expected = expectedOutput.get(z, y, x);
                    maxError = Math.max(maxError, Math.abs(actual - expected));
                }
            }
        }
        console.log('LayerNorm embed MAE:', maxError);
    });
}
function randomize(t) {
    for (let i = 0; i < t.data.length; i++) {
        t.data[i] = Math.random() * 2 - 1;
    }
}
//# sourceMappingURL=tests.js.map