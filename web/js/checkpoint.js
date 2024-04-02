var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
function loadCheckpointData(url) {
    return __awaiter(this, void 0, void 0, function* () {
        return yield (yield fetch(url)).arrayBuffer();
    });
}
function decodeCheckpoint(buf) {
    const bytes = new Uint8Array(buf);
    const metadataSize = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    const metadata = JSON.parse(String.fromCharCode.apply(null, bytes.slice(4, 4 + metadataSize)));
    let allData = loadFloats(metadata.precision || 32, buf.slice(4 + metadataSize));
    const params = new Map();
    metadata["params"].forEach((info) => {
        const [name, rawShape] = info;
        const shape = new Shape(...rawShape);
        const param = Tensor.from(shape, allData.slice(0, shape.numel()));
        allData = allData.slice(shape.numel());
        params.set(name, param);
    });
    return { params: params, config: metadata.config };
}
function flipToLittleEndian(precision, input) {
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
    }
    else {
        for (let i = 0; i < arr.length; i += 2) {
            const x = arr[i];
            const y = arr[i + 1];
            out[i] = y;
            out[i + 1] = x;
        }
    }
    return output;
}
function loadFloats(precision, input) {
    const nativeEndianBuf = flipToLittleEndian(precision, input);
    if (precision == 32) {
        return new Float32Array(nativeEndianBuf);
    }
    else {
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
//# sourceMappingURL=checkpoint.js.map