var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
function loadCheckpoint(url) {
    return __awaiter(this, void 0, void 0, function* () {
        const buf = yield (yield fetch(url)).arrayBuffer();
        const bytes = new Uint8Array(buf);
        const metadataSize = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
        const metadata = JSON.parse(String.fromCharCode.apply(null, bytes.slice(4, 4 + metadataSize)));
        let allData = new Float32Array(flipToLittleEndian(buf.slice(4 + metadataSize)));
        const params = new Map();
        metadata["params"].forEach((info) => {
            const [name, rawShape] = info;
            const shape = new Shape(...rawShape);
            const param = Tensor.from(shape, allData.slice(0, shape.numel()));
            allData = allData.slice(shape.numel());
            params.set(name, param);
        });
        return { params: params, config: metadata.config };
    });
}
function flipToLittleEndian(input) {
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
//# sourceMappingURL=checkpoint.js.map