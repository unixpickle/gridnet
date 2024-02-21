async function testWebGPUPatchEmbed() {
    const patchEmbCode = await (await fetch('/glsl/patch_embed.glsl')).text();

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
        new ComputePass(
            patchEmbCode,
            'patchEmbedStandard4x4',
            [
                new Buffer(input.data),
                new Buffer(output.data, output.data),
                new Buffer(weight.data),
                new Buffer(bias.data),
            ],
            [64],
        ),
    ]);
    await sequence.execute();

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
}

function randomize(t: Tensor) {
    for (let i = 0; i < t.data.length; i++) {
        t.data[i] = Math.random() * 2 - 1;
    }
}