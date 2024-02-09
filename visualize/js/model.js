class Shape extends Array {
    constructor(...args) {
        if (args.length == 1) {
            super();
            this.push(args[0]);
        }
        else {
            super(...args);
        }
    }
    toString() {
        return 'Shape(' + this.join(', ') + ')';
    }
    numel() {
        let result = 1;
        this.forEach((x) => result *= x);
        return result;
    }
    equal(other) {
        if (this.length !== other.length) {
            return false;
        }
        for (let i = 0; i < this.length; i++) {
            if (this[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
}
class Tensor {
    constructor(shape, data) {
        this.shape = shape;
        this.data = data;
    }
    static from(shape, data) {
        if (shape.length === 1) {
            return new Tensor1(shape, data);
        }
        else if (shape.length === 2) {
            return new Tensor2(shape, data);
        }
        else if (shape.length === 3) {
            return new Tensor3(shape, data);
        }
        else if (shape.length === 4) {
            return new Tensor4(shape, data);
        }
        else {
            throw new Error(`unsupported shape: ${shape}`);
        }
    }
    static zeros(shape) {
        const data = new Float32Array(shape.numel());
        return Tensor.from(shape, data);
    }
    slice(start, end) {
        assert(start.length == this.shape.length);
        assert(end.length == this.shape.length);
        const newShape = start.map((x, i) => {
            return end[i] - x;
        });
        const result = Tensor.zeros(new Shape(...newShape));
        for (let i = 0; i < result.data.length; i++) {
            const sourceIndex = [];
            let globalIdx = i;
            let outOfBounds = false;
            for (let j = result.shape.length - 1; j >= 0; j--) {
                const idx = (globalIdx % result.shape[j]) + start[j];
                sourceIndex.push(idx);
                if (idx < 0 || idx >= this.shape[j]) {
                    outOfBounds = true;
                    break;
                }
                globalIdx = Math.floor(globalIdx / result.shape[j]);
            }
            if (!outOfBounds) {
                sourceIndex.reverse();
                result.data[i] = this.get(...sourceIndex);
            }
        }
        return result;
    }
    add(other) {
        assert(this.shape.equal(other.shape));
        const newData = this.data.slice();
        for (let i = 0; i < other.data.length; i++) {
            newData[i] += other.data[i];
        }
        return Tensor.from(this.shape, newData);
    }
    sub(other) {
        assert(this.shape.equal(other.shape));
        const newData = this.data.slice();
        for (let i = 0; i < other.data.length; i++) {
            newData[i] -= other.data[i];
        }
        return Tensor.from(this.shape, newData);
    }
}
class Tensor1 extends Tensor {
    constructor(shape, data) {
        super(shape, data);
        assert(shape.length == 1);
    }
    get(i) {
        return this.data[i];
    }
    set(x, i) {
        this.data[i] = x;
    }
    clone() {
        return new Tensor1(this.shape, this.data.slice());
    }
}
class Tensor2 extends Tensor {
    constructor(shape, data) {
        super(shape, data);
        assert(shape.length == 2);
    }
    get(i, j) {
        return this.data[i * this.shape[1] + j];
    }
    set(x, i, j) {
        this.data[i * this.shape[1] + j] = x;
    }
    clone() {
        return new Tensor2(this.shape, this.data.slice());
    }
}
class Tensor3 extends Tensor {
    constructor(shape, data) {
        super(shape, data);
        assert(shape.length == 3);
    }
    get(i, j, k) {
        return this.data[(i * this.shape[1] + j) * this.shape[2] + k];
    }
    set(x, i, j, k) {
        this.data[(i * this.shape[1] + j) * this.shape[2] + k] = x;
    }
    slice(start, end) {
        assert(start.length == this.shape.length);
        assert(end.length == this.shape.length);
        const newShape = start.map((x, i) => {
            return end[i] - x;
        });
        const result = Tensor.zeros(new Shape(...newShape));
        let outIndex = 0;
        for (let i = start[0]; i < end[0]; i++) {
            for (let j = start[1]; j < end[1]; j++) {
                for (let k = start[2]; k < end[2]; k++) {
                    result.data[outIndex++] = this.get(i, j, k);
                }
            }
        }
        return result;
    }
    clone() {
        return new Tensor3(this.shape, this.data.slice());
    }
}
class Tensor4 extends Tensor {
    constructor(shape, data) {
        super(shape, data);
        assert(shape.length == 4);
    }
    get(i, j, k, l) {
        return this.data[((i * this.shape[1] + j) * this.shape[2] + k) * this.shape[3] + l];
    }
    set(x, i, j, k, l) {
        this.data[((i * this.shape[1] + j) * this.shape[2] + k) * this.shape[3] + l] = x;
    }
    slice(start, end) {
        assert(start.length == this.shape.length);
        assert(end.length == this.shape.length);
        const newShape = start.map((x, i) => {
            return end[i] - x;
        });
        const result = Tensor.zeros(new Shape(...newShape));
        let outIndex = 0;
        for (let i = start[0]; i < end[0]; i++) {
            for (let j = start[1]; j < end[1]; j++) {
                for (let k = start[2]; k < end[2]; k++) {
                    for (let l = start[3]; l < end[3]; l++) {
                        result.data[outIndex++] = this.get(i, j, k, l);
                    }
                }
            }
        }
        return result;
    }
    clone() {
        return new Tensor3(this.shape, this.data.slice());
    }
}
function sigmoid(x) {
    if (x < 0) {
        return 1 - sigmoid(-x);
    }
    const exp = Math.exp(x);
    return exp / (1 + exp);
}
function activationImpl(act) {
    if (act == "silu") {
        return (x) => x * sigmoid(x);
    }
    else if (act == "relu") {
        return (x) => Math.max(0, x);
    }
    else if (act == "leaky_relu") {
        return (x) => x < 0 ? 0.01 * x : x;
    }
}
class Layer {
}
class PatchEmbed extends Layer {
    constructor(weight, bias) {
        super();
        this.weight = weight;
        this.bias = bias;
        assert(this.weight.shape.length == 4, this.weight.shape);
        assert(this.weight.shape[2] == this.weight.shape[3], this.weight.shape);
        assert(this.bias.shape.length == 1, this.bias.shape);
        assert(this.bias.shape[0] == this.weight.shape[0], this.bias.shape);
        this.outChannels = this.weight.shape[0];
        this.inChannels = this.weight.shape[1];
        this.kernelSize = this.weight.shape[2];
    }
    forward(x) {
        assert(x.shape.length == 3);
        assert(x.shape[0] == this.inChannels);
        assert(x.shape[1] == x.shape[2]);
        assert(x.shape[1] % this.kernelSize == 0);
        const outSize = x.shape[1] / this.kernelSize;
        const out = Tensor.zeros(new Shape(this.outChannels, outSize, outSize));
        for (let outCh = 0; outCh < this.outChannels; outCh++) {
            let bias = this.bias.get(outCh);
            for (let outY = 0; outY < outSize; outY++) {
                for (let outX = 0; outX < outSize; outX++) {
                    let accum = bias;
                    for (let in_ch = 0; in_ch < this.inChannels; in_ch++) {
                        for (let i = 0; i < this.kernelSize; i++) {
                            for (let j = 0; j < this.kernelSize; j++) {
                                const weight = this.weight.get(outCh, in_ch, i, j);
                                const value = x.get(in_ch, outY * this.kernelSize + i, outX * this.kernelSize + j);
                                accum += weight * value;
                            }
                        }
                    }
                    out.set(accum, outCh, outY, outX);
                }
            }
        }
        return out;
    }
}
class LayerNorm extends Layer {
    constructor(weight, bias) {
        super();
        this.weight = weight;
        this.bias = bias;
        assert(bias.shape.equal(weight.shape));
    }
    forward(x) {
        assert(x.shape.equal(this.weight.shape));
        let mean = 0;
        for (let i = 0; i < x.data.length; i++) {
            mean += x.data[i];
        }
        mean /= x.data.length;
        let variance = 0;
        for (let i = 0; i < x.data.length; i++) {
            variance += Math.pow(x.data[i] - mean, 2);
        }
        variance /= x.data.length;
        let std = Math.sqrt(variance + 1e-5);
        const result = Tensor.zeros(x.shape);
        for (let i = 0; i < x.data.length; i++) {
            result.data[i] = (x.data[i] - mean) / std * this.weight.data[i] + this.bias.data[i];
        }
        return result;
    }
}
class Linear extends Layer {
    constructor(weight, bias) {
        super();
        this.weight = weight;
        this.bias = bias;
        assert(weight.shape.length == 2);
        assert(bias.shape.length == 1);
        assert(bias.shape[0] == weight.shape[0]);
        this.inChannels = weight.shape[1];
        this.outChannels = weight.shape[0];
    }
    forward(x) {
        assert(x.shape.length == 1);
        assert(x.shape[0] == this.inChannels);
        const result = Tensor.zeros(new Shape(this.outChannels));
        for (let i = 0; i < this.outChannels; i++) {
            let acc = this.bias.data[i];
            const offset = i * this.inChannels;
            for (let j = 0; j < this.inChannels; j++) {
                acc += x.data[j] * this.weight.data[offset + j];
            }
            result.data[i] = acc;
        }
        return result;
    }
}
class Readout extends Layer {
    constructor(norm, proj) {
        super();
        this.norm = norm;
        this.proj = proj;
        this.inChannels = proj.inChannels;
    }
    forward(x) {
        const plane_size = x.shape[0] * x.shape[1];
        assert(this.inChannels % plane_size == 0);
        const zLayers = this.inChannels / plane_size;
        const flatOut = Tensor.zeros(new Shape(this.inChannels));
        let out_idx = 0;
        for (let i = 0; i < x.shape[0]; i++) {
            for (let j = 0; j < x.shape[1]; j++) {
                for (let k = x.shape[2] - zLayers; k < x.shape[2]; k++) {
                    flatOut.data[out_idx++] = x.get(i, j, k);
                }
            }
        }
        const h = this.norm.forward(flatOut);
        return this.proj.forward(h);
    }
}
class Gridnet extends Layer {
    constructor(weight, bias, residualScale, innerActivations, blockSize, activation) {
        super();
        this.weight = weight;
        this.bias = bias;
        this.residualScale = residualScale;
        this.innerActivations = innerActivations;
        this.blockSize = blockSize;
        this.activation = activationImpl(activation);
    }
    forward(x) {
        const inputIndices = this.blockInputIndices();
        const output = x.clone();
        for (let i = 0; i < x.shape[0]; i += this.blockSize) {
            for (let j = 0; j < x.shape[0]; j += this.blockSize) {
                for (let k = 0; k < x.shape[0]; k += this.blockSize) {
                    const inActs = x.slice([i - 1, j - 1, k - 1], [i + this.blockSize + 1, j + this.blockSize + 1, k + this.blockSize + 1]);
                    const weight = this.weight.slice([0, i, j, k], [this.weight.shape[0], i + this.blockSize, j + this.blockSize, k + this.blockSize]);
                    const bias = this.bias.slice([i, j, k], [i + this.blockSize, j + this.blockSize, k + this.blockSize]);
                    const residualScale = this.residualScale.slice([i, j, k], [i + this.blockSize, j + this.blockSize, k + this.blockSize]);
                    const block_out = this.applyBlock(inputIndices, inActs, weight, bias, residualScale);
                    for (let a = 0; a < this.blockSize; a++) {
                        for (let b = 0; b < this.blockSize; b++) {
                            for (let c = 0; c < this.blockSize; c++) {
                                const val = block_out.get(a + 1, b + 1, c + 1);
                                output.set(val, a + i, b + j, c + k);
                            }
                        }
                    }
                }
            }
        }
        return output;
    }
    applyBlock(indices, inActs, weight, bias, residualScale) {
        let input = inActs;
        let output = inActs;
        for (let step = 0; step < this.innerActivations; step++) {
            output = input.clone();
            let unroll_idx = 0;
            for (let a = 0; a < this.blockSize; a++) {
                for (let b = 0; b < this.blockSize; b++) {
                    for (let c = 0; c < this.blockSize; c++) {
                        let acc = bias.get(a, b, c);
                        for (let i = 0; i < 3 * 3 * 3; i++) {
                            const source_idx = indices[unroll_idx++];
                            acc += input.data[source_idx] * weight.get(i, a, b, c);
                        }
                        const result = (output.get(a + 1, b + 1, c + 1) +
                            this.activation(acc) * residualScale.get(a, b, c));
                        output.set(result, a + 1, b + 1, c + 1);
                    }
                }
            }
            input = output;
        }
        return output;
    }
    blockInputIndices() {
        const idxInBlock = (i, j, k) => {
            return k + (this.blockSize + 2) * (j + (this.blockSize + 2) * i);
        };
        let result = [];
        for (let i = 0; i < this.blockSize; i++) {
            for (let j = 0; j < this.blockSize; j++) {
                for (let k = 0; k < this.blockSize; k++) {
                    const row = [];
                    for (let a = 0; a < 3; a++) {
                        for (let b = 0; b < 3; b++) {
                            for (let c = 0; c < 3; c++) {
                                row.push(idxInBlock(i + a, j + b, k + c));
                            }
                        }
                    }
                    result = result.concat(row);
                }
            }
        }
        return result;
    }
}
class ImagenetClassifier extends Layer {
    constructor(ckpt) {
        super();
        this.config = ckpt.config;
        this.initIn = ckpt.params.get('init_in');
        this.norm = new LayerNorm(ckpt.params.get('norm.weight'), ckpt.params.get('norm.bias'));
        this.network = new Gridnet(ckpt.params.get('network.weight'), ckpt.params.get('network.bias'), ckpt.params.get('network.residual_scale'), ckpt.config.innerIters, 8, ckpt.config.activation);
        this.patchEmb = new PatchEmbed(ckpt.params.get('patch_emb.weight'), ckpt.params.get('patch_emb.bias'));
        this.readout = new Readout(new LayerNorm(ckpt.params.get('readout.norm.weight'), ckpt.params.get('readout.norm.bias')), new Linear(ckpt.params.get('readout.proj.weight'), ckpt.params.get('readout.proj.bias')));
    }
    forward(x) {
        const emb = this.patchEmb.forward(x);
        let h = this.initIn.clone();
        for (let ch = 0; ch < emb.shape[0]; ch++) {
            for (let y = 0; y < emb.shape[1]; y++) {
                for (let x = 0; x < emb.shape[2]; x++) {
                    h.set(emb.get(ch, y, x), y, x, ch);
                }
            }
        }
        for (let i = 0; i < this.config.outerIters; i++) {
            if (this.config.outerIters) {
                const norm_h = this.norm.forward(h);
                h = h.add(this.network.forward(norm_h).sub(norm_h));
            }
            else {
                h = this.network.forward(h);
                h = this.norm.forward(h);
            }
        }
        return this.readout.forward(h);
    }
}
function assert(x, ...data) {
    if (!x) {
        if (data.length) {
            console.error(...data);
        }
        throw new Error('assertion failed');
    }
}
//# sourceMappingURL=model.js.map