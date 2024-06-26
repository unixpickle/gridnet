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
        const patchEmbCode = yield fetchKernel('patch_embed.wgsl');
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
        const input = Tensor.zeros(new Shape(64, 64, 64));
        const weight = Tensor.zeros(new Shape(64, 64, 64));
        const bias = Tensor.zeros(new Shape(64, 64, 64));
        randomize(input);
        randomize(weight);
        randomize(bias);
        const cpuLayer = new LayerNorm(weight, bias);
        const expectedOutput = cpuLayer.forward(input);
        const output = Tensor.zeros(new Shape(64, 64, 64));
        const sequence = new KernelSequence(yield webgpuLayerNorm(new Buffer(input.data), new Buffer(output.data, output.data), new Buffer(weight.data), new Buffer(bias.data)));
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
        console.log('LayerNorm MAE:', maxError);
    });
}
function testWebGPUGridnet() {
    return __awaiter(this, void 0, void 0, function* () {
        const input = Tensor.zeros(new Shape(64, 64, 64));
        const weight = Tensor.zeros(new Shape(27, 64, 64, 64));
        const bias = Tensor.zeros(new Shape(64, 64, 64));
        const scale = Tensor.zeros(new Shape(64, 64, 64));
        randomize(input);
        randomize(weight);
        randomize(bias);
        randomize(scale);
        const iters = 6;
        const cpuLayer = new Gridnet(weight, bias, scale, iters, 8, 'leaky_relu');
        const expectedOutput = cpuLayer.forward(input);
        const output = input.clone();
        const sequence = new KernelSequence([
            new ComputePass(yield fetchKernel('gridnet.wgsl'), 'gridnet8x8x8', [
                new Buffer(new Uint32Array([iters])),
                new Buffer(new Uint32Array([64])),
                new Buffer(input.data),
                new Buffer(output.data, output.data),
                new Buffer(weight.data),
                new Buffer(bias.data),
                new Buffer(scale.data),
            ], [8 * 8 * 8])
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
        console.log('Gridnet MAE:', maxError);
    });
}
function testWebGPUReadout() {
    return __awaiter(this, void 0, void 0, function* () {
        const input = Tensor.zeros(new Shape(64, 64, 64));
        const weight = Tensor.zeros(new Shape(1000, 64 * 64));
        const bias = Tensor.zeros(new Shape(1000));
        const normWeight = Tensor.zeros(new Shape(64 * 64));
        const normBias = Tensor.zeros(new Shape(64 * 64));
        randomize(input);
        randomize(weight);
        randomize(bias);
        randomize(normWeight);
        randomize(normBias);
        const cpuLayer = new Readout(new LayerNorm(normWeight, normBias), new Linear(weight, bias));
        const expectedOutput = cpuLayer.forward(input);
        const output = Tensor.zeros(new Shape(1000));
        const normInput = new Buffer(new Float32Array(64 * 64), null, true);
        const normOutputData = new Float32Array(64 * 64);
        const normOutput = new Buffer(normOutputData, normOutputData);
        const sequence = new KernelSequence([
            new ComputePass(yield fetchKernel('slice_output.wgsl'), 'sliceOutput', [
                new Buffer(new Uint32Array([64])),
                new Buffer(new Uint32Array([1])),
                new Buffer(input.data),
                normInput,
            ], [Math.ceil((64 * 64) / 256)]),
            ...yield webgpuLayerNorm(normInput.readOnly(), normOutput, new Buffer(normWeight.data), new Buffer(normBias.data)),
            new ComputePass(yield fetchKernel('unembed.wgsl'), 'unembed', [
                new Buffer(new Uint32Array([64 * 64])),
                new Buffer(new Uint32Array([1000])),
                normOutput.readOnly(),
                new Buffer(weight.data),
                new Buffer(bias.data),
                new Buffer(output.data, output.data),
            ], [Math.ceil(1000 / 8)]),
        ]);
        yield sequence.execute();
        let maxError = 0.0;
        for (let x = 0; x < 1000; x++) {
            const actual = output.get(x);
            const expected = expectedOutput.get(x);
            maxError = Math.max(maxError, Math.abs(actual - expected));
        }
        console.log('unembed MAE:', maxError);
    });
}
function randomize(t) {
    for (let i = 0; i < t.data.length; i++) {
        t.data[i] = Math.random() * 2 - 1;
    }
}
//# sourceMappingURL=tests.js.map