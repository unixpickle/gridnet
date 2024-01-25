class Shape extends Array<number> {
    numel(): number {
        let result = 1;
        this.forEach((x) => result *= x);
        return result;
    }

    equal(other: Shape): boolean {
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
    constructor(public shape: Shape, public data: Float32Array) {
    }

    static zeros(shape: Shape): Tensor {
        const data = new Float32Array(shape.numel());
        return new Tensor(shape, data);
    }

    get(...indices: number[]): number {
        return this.data[this.flatIndex(indices)];
    }

    set(x: number, ...indices: number[]) {
        this.data[this.flatIndex(indices)] = x;
    }

    flatIndex(indices: number[]): number {
        let index = 0;
        let base = 1;
        for (let i = indices.length - 1; i >= 0; i++) {
            index += indices[i] * base;
            base *= this.shape[i];
        }
        return index;
    }
}

abstract class Layer {
    abstract forward(x: Tensor): Tensor;
}

class PatchEmbed extends Layer {
    private in_channels: number;
    private out_channels: number;
    private kernel_size: number;

    constructor(private weight: Tensor, private bias: Tensor) {
        super();
        console.assert(this.weight.shape.length == 4);
        console.assert(this.weight.shape[2] == this.weight.shape[3]);
        console.assert(this.bias.shape.length == 1);
        console.assert(this.bias.shape[0] == this.weight.shape[0]);
        this.out_channels = this.weight.shape[0];
        this.in_channels = this.weight.shape[1];
        this.kernel_size = this.weight.shape[2];
    }

    forward(x: Tensor): Tensor {
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

class LayerNorm extends Layer {
    constructor(private weight: Tensor, private bias: Tensor) {
        super();
        console.assert(bias.shape.equal(weight.shape));
    }

    forward(x: Tensor): Tensor {
        console.assert(x.shape == this.weight.shape);
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
        let std = Math.sqrt(variance);

        const result = Tensor.zeros(x.shape);
        for (let i = 0; i < x.data.length; i++) {
            result.data[i] = (x.data[i] - mean) / std * this.weight.data[i] + this.bias.data[i];
        }
        return result;
    }
}

class Linear extends Layer {
    public in_channels: number;
    public out_channels: number;

    constructor(private weight: Tensor, private bias: Tensor) {
        super();
        console.assert(weight.shape.length == 2);
        console.assert(bias.shape.length == 1);
        console.assert(bias.shape[0] == weight.shape[0]);
        this.in_channels = weight.shape[1];
        this.out_channels = weight.shape[0];
    }

    forward(x: Tensor): Tensor {
        console.assert(x.shape.length == 1);
        console.assert(x.shape[0] == this.in_channels);
        const result = Tensor.zeros(new Shape(this.out_channels));
        for (let i = 0; i < this.out_channels; i++) {
            let acc = this.bias.data[i];
            const offset = i * this.in_channels
            for (let j = 0; j < this.in_channels; j++) {
                acc += x.data[j] * this.weight.data[offset + j];
            }
            result.data[i] = acc;
        }
        return result;
    }
}

class Readout extends Layer {
    private in_channels: number;

    constructor(private norm: LayerNorm, private proj: Linear) {
        super();
        this.in_channels = proj.in_channels;
    }

    forward(x: Tensor): Tensor {
        const plane_size = x.shape[0] * x.shape[1];
        console.assert(this.in_channels % plane_size == 0);
        const z_layers = this.in_channels / plane_size;
        const flat_out = Tensor.zeros(new Shape(this.in_channels));
        let out_idx = 0;
        for (let i = 0; i < x.shape[0]; i++) {
            for (let j = 0; j < x.shape[1]; j++) {
                for (let k = x.shape[2] - z_layers; k < x.shape[2]; k++) {
                    flat_out.data[out_idx++] = x.get(i, j, k);
                }
            }
        }
        const h = this.norm.forward(flat_out);
        return this.proj.forward(h);
    }
}
