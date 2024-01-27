// Loosely adapted from flatten:
// https://github.com/unixpickle/flatten/blob/e8ed7791fdc64597d309ac0400afd82ae7104150/src/model.ts

interface ModelConfig {
    activation: string;
    inner_iters: number;
    outer_iters: number;
    outer_residual: boolean;
}

interface Metadata {
    params: [string, number[]][];
    config: ModelConfig;
}

interface Checkpoint {
    params: Map<string, Tensor>;
    config: ModelConfig;
}

async function loadCheckpoint(url: string): Promise<Checkpoint> {
    const buf = await (await fetch(url)).arrayBuffer();
    const bytes = new Uint8Array(buf);
    const metadataSize = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    const metadata = JSON.parse(
        String.fromCharCode.apply(null, bytes.slice(4, 4 + metadataSize)),
    ) as Metadata;

    let allData = new Float32Array(flipToLittleEndian(buf.slice(4 + metadataSize)));
    const params = new Map();
    metadata["params"].forEach((info: [string, number[]]) => {
        const [name, rawShape] = info;
        const shape = new Shape(...rawShape);
        const param = new Tensor(shape, allData.slice(0, shape.numel()));
        allData = allData.slice(shape.numel());
        params.set(name, param);
    });
    return { params: params, config: metadata.config };
}

function flipToLittleEndian(input: ArrayBuffer): ArrayBuffer {
    if (!isBigEndian()) {
        return input;
    }
    let arr = new Uint8Array(input);
    const output = new ArrayBuffer(arr.length);
    const out = new Uint8Array(output);
    for (let i = 0; i < arr.length; i += 4) {
        const w = arr[i];
        const x = arr[i + 1];
        const y = arr[i + 2];
        const z = arr[i + 3];
        out[i] = z;
        out[i + 1] = y;
        out[i + 2] = x;
        out[i + 3] = w;
    }
    return output;
}

function isBigEndian() {
    const x = new ArrayBuffer(4);
    new Float32Array(x)[0] = 1;
    return new Uint8Array(x)[0] != 0;
}
