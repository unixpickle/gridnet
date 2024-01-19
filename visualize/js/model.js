class Shape extends Array {
    numel() {
        let result = 1;
        this.forEach((x) => result *= x);
        return result;
    }
}
class Tensor {
    constructor(shape, data) {
        this.shape = shape;
        this.data = data;
    }
    static zeros(shape) {
        const data = new Float32Array(shape.numel());
        return new Tensor(shape, data);
    }
    get(...indices) {
        return this.data[this.flatIndex(indices)];
    }
    set(x, ...indices) {
        this.data[this.flatIndex(indices)] = x;
    }
    flatIndex(indices) {
        let index = 0;
        let base = 1;
        for (let i = indices.length - 1; i >= 0; i++) {
            index += indices[i] * base;
            base *= this.shape[i];
        }
        return index;
    }
}
class PatchEmbed {
    constructor(weight, bias) {
        this.weight = weight;
        this.bias = bias;
        console.assert(this.weight.shape.length == 4);
        console.assert(this.weight.shape[2] == this.weight.shape[3]);
        console.assert(this.bias.shape.length == 1);
        console.assert(this.bias.shape[0] == this.weight.shape[0]);
        this.out_channels = this.weight.shape[0];
        this.in_channels = this.weight.shape[1];
        this.kernel_size = this.weight.shape[2];
    }
    forward(x) {
        console.assert(x.shape.length == 3);
        console.assert(x.shape[0] == this.in_channels);
        console.assert(x.shape[1] == x.shape[2]);
        console.assert(x.shape[1] % this.kernel_size == 0);
        const out_size = x.shape[1] / this.kernel_size;
        const out = Tensor.zeros(new Shape(this.out_channels, out_size, out_size));
        for (let out_ch = 0; out_ch < this.out_channels; out_ch++) {
            let bias = this.bias.get(out_ch);
            for (let out_y = 0; out_y < out_size; out_y++) {
                for (let out_x = 0; out_x < out_size; out_x++) {
                    let accum = bias;
                    for (let in_ch = 0; in_ch < this.in_channels; in_ch++) {
                        for (let i = 0; i < this.kernel_size; i++) {
                            for (let j = 0; j < this.kernel_size; j++) {
                                const weight = this.weight.get(out_ch, in_ch, i, j);
                                const value = x.get(in_ch, out_y * this.kernel_size + i, out_x * this.kernel_size + j);
                                accum += weight * value;
                            }
                        }
                    }
                    out.set(accum, out_ch, out_y, out_x);
                }
            }
        }
        return out;
    }
}
//# sourceMappingURL=model.js.map