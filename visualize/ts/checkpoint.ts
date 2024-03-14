// Loosely adapted from flatten:
// https://github.com/unixpickle/flatten/blob/e8ed7791fdc64597d309ac0400afd82ae7104150/src/model.ts

interface ModelConfig {
    activation: string;
    innerIters: number;
    outerIters: number;
    outerResidual: boolean;
}

type CheckpointPrecision = 16 | 32;

interface Metadata {
    params: [string, number[]][];
    precision: CheckpointPrecision;
    config: ModelConfig;
}

interface Checkpoint {
    params: Map<string, Tensor>;
    config: ModelConfig;
}

async function loadCheckpointData(url: string): Promise<ArrayBuffer> {
    return await (await fetch(url)).arrayBuffer();
}

function decodeCheckpoint(buf: ArrayBuffer): Checkpoint {
    const bytes = new Uint8Array(buf);
    const metadataSize = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    const metadata = JSON.parse(
        String.fromCharCode.apply(null, bytes.slice(4, 4 + metadataSize)),
    ) as Metadata;

    let allData = loadFloats(metadata.precision || 32, buf.slice(4 + metadataSize));
    const params = new Map();
    metadata["params"].forEach((info: [string, number[]]) => {
        const [name, rawShape] = info;
        const shape = new Shape(...rawShape);
        const param = Tensor.from(shape, allData.slice(0, shape.numel()));
        allData = allData.slice(shape.numel());
        params.set(name, param);
    });
    return { params: params, config: metadata.config };
}

function flipToLittleEndian(precision: CheckpointPrecision, input: ArrayBuffer): ArrayBuffer {
    if (!isBigEndian()) {
        return input;
    }
    let arr = new Uint8Array(input);
    const output = new ArrayBuffer(arr.length);
    const out = new Uint8Array(output);
    if (precision == 32) {
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
    } else {
        for (let i = 0; i < arr.length; i += 2) {
            const x = arr[i];
            const y = arr[i + 1];
            out[i] = y;
            out[i + 1] = x;
        }
    }
    return output;
}

function loadFloats(precision: CheckpointPrecision, input: ArrayBuffer): Float32Array {
    const nativeEndianBuf = flipToLittleEndian(precision, input);
    if (precision == 32) {
        return new Float32Array(nativeEndianBuf);
    } else {
        const halves = new Uint16Array(nativeEndianBuf);
        const buf = new ArrayBuffer(halves.length * 4);
        const words = new Uint32Array(buf);
        for (let i = 0; i < halves.length; i++) {
            const half = halves[i];
            const sign = (half >> 15) & 1;
            const exp = ((half >> 10) & ((1 << 5) - 1)) - 15;
            const frac = half & ((1 << 10) - 1);
            words[i] = (sign << 31) | ((exp + 127) << 23) | (frac << 13);
        }
        const result = new Float32Array(buf);
        for (let i = 0; i < result.length; i++) {
            if (!isFinite(result[i])) {
                console.log('uhoh', result[i]);
            }
        }
        return result;
    }
}

function isBigEndian() {
    const x = new ArrayBuffer(4);
    new Float32Array(x)[0] = 1;
    return new Uint8Array(x)[0] != 0;
}
