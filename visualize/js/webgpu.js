var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class Buffer {
    constructor(input, output = null, writable = null) {
        this.input = input;
        this.output = output;
        this.writable = writable;
        this.deviceBuffer = null;
        this.resultBuffer = null;
        if (this.writable == null) {
            this.writable = (this.output == null ? false : true);
        }
    }
    createDeviceBuffer(device) {
        this.deviceBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: this.input.byteLength,
            usage: GPUBufferUsage.STORAGE | (this.output != null ? GPUBufferUsage.COPY_SRC : 0),
        });
        const arrayBuffer = this.deviceBuffer.getMappedRange();
        const ctr = this.input.constructor;
        new ctr(arrayBuffer).set(this.input);
        this.deviceBuffer.unmap();
    }
    layout() {
        return {
            type: this.writable ? 'storage' : 'read-only-storage',
        };
    }
    buffer() {
        return this;
    }
    readOnly() {
        return new ReadOnlyBuffer(this);
    }
    size() {
        return this.input.length;
    }
}
class ReadOnlyBuffer {
    constructor(_buffer) {
        this._buffer = _buffer;
    }
    layout() {
        return {
            type: 'read-only-storage',
        };
    }
    buffer() {
        return this._buffer;
    }
    readOnly() {
        return this;
    }
    size() {
        return this._buffer.size();
    }
}
class ShaderModuleCache {
    constructor() {
        this.items = [];
    }
    createOrReuse(device, code) {
        for (let i = 0; i < this.items.length; i++) {
            if (this.items[i].device == device && this.items[i].code == code) {
                return this.items[i].module;
            }
        }
        const module = device.createShaderModule({
            code: code,
        });
        this.items.push({
            code: code,
            device: device,
            module: module,
        });
        return module;
    }
}
ShaderModuleCache.Global = new ShaderModuleCache();
class ComputePass {
    constructor(code, entrypoint, bindings, gridSize, constants = {}) {
        this.code = code;
        this.entrypoint = entrypoint;
        this.bindings = bindings;
        this.gridSize = gridSize;
        this.constants = constants;
    }
    encode(device, encoder) {
        return __awaiter(this, void 0, void 0, function* () {
            const bindGroupLayout = device.createBindGroupLayout(this.bindGroupLayout());
            const bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: this.bindGroup(),
            });
            const shaderModule = ShaderModuleCache.Global.createOrReuse(device, this.code);
            const pipeline = yield device.createComputePipelineAsync({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout]
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: this.entrypoint,
                    constants: this.constants,
                },
            });
            const passEncoder = encoder.beginComputePass();
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(this.gridSize[0], this.gridSize[1], this.gridSize[2]);
            passEncoder.end();
        });
    }
    bindGroupLayout() {
        return {
            entries: this.bindings.map((buf, i) => {
                return {
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: buf.layout(),
                };
            })
        };
    }
    bindGroup() {
        return this.bindings.map((buf, i) => {
            return {
                binding: i,
                resource: {
                    buffer: buf.buffer().deviceBuffer,
                },
            };
        });
    }
}
class KernelSequence {
    constructor(passes) {
        this.passes = passes;
    }
    execute(device = null) {
        return __awaiter(this, void 0, void 0, function* () {
            if (device == null) {
                const adapter = yield navigator.gpu.requestAdapter();
                if (!adapter) {
                    throw new Error('failed to get WebGPU adapter');
                }
                device = yield adapter.requestDevice();
            }
            device.pushErrorScope('validation');
            device.pushErrorScope('internal');
            device.pushErrorScope('out-of-memory');
            this.createDeviceBuffers(device);
            const encoder = device.createCommandEncoder();
            for (let i = 0; i < this.passes.length; i++) {
                yield this.passes[i].encode(device, encoder);
            }
            this.encodeResultCopies(device, encoder);
            const gpuCommands = encoder.finish();
            device.queue.submit([gpuCommands]);
            for (let i = 0; i < 3; i++) {
                const error = yield device.popErrorScope();
                if (error) {
                    throw error;
                }
            }
            yield this.copyResults();
        });
    }
    createDeviceBuffers(device) {
        this.buffers().forEach((buf) => buf.createDeviceBuffer(device));
    }
    encodeResultCopies(device, encoder) {
        this.buffers().filter((x) => x.output != null).forEach((buf) => {
            buf.resultBuffer = device.createBuffer({
                size: buf.output.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            encoder.copyBufferToBuffer(buf.deviceBuffer, 0, buf.resultBuffer, 0, buf.output.byteLength);
        });
    }
    copyResults() {
        return __awaiter(this, void 0, void 0, function* () {
            const b = this.buffers();
            for (let i = 0; i < b.length; i++) {
                const buf = b[i];
                if (buf.output == null) {
                    continue;
                }
                yield buf.resultBuffer.mapAsync(GPUMapMode.READ);
                const arrayBuffer = buf.resultBuffer.getMappedRange();
                const ctr = buf.output.constructor;
                buf.output.set(new ctr(arrayBuffer));
                buf.resultBuffer.unmap();
            }
        });
    }
    buffers() {
        const results = [];
        this.passes.forEach((pass) => {
            pass.bindings.forEach((buf) => {
                if (!results.includes(buf.buffer())) {
                    results.push(buf.buffer());
                }
            });
        });
        return results;
    }
}
function fetchKernel(name) {
    return __awaiter(this, void 0, void 0, function* () {
        return yield (yield fetch(`/wgsl/${name}`)).text();
    });
}
function webgpuLayerNorm(input, output, weight, bias) {
    return __awaiter(this, void 0, void 0, function* () {
        const statsCode = yield fetchKernel('moments.wgsl');
        const affineCode = yield fetchKernel('affine.wgsl');
        const inBuffer = input.readOnly();
        let moment1 = new Buffer(new Float32Array(1024), null, true);
        let moment2 = new Buffer(new Float32Array(1024), null, true);
        let moment1Tmp = new Buffer(new Float32Array(1024), null, true);
        let moment2Tmp = new Buffer(new Float32Array(1024), null, true);
        const unused = new Buffer(new Float32Array(1), null, true);
        const inputSize = input.size();
        const sizeBuffer = new Buffer(new Uint32Array([inputSize]));
        const isFirstTrue = new Buffer(new Uint32Array([1]));
        const isFirstFalse = new Buffer(new Uint32Array([0]));
        let numBlocks = Math.ceil(inputSize / 256);
        const passes = [
            new ComputePass(statsCode, 'reduceMoments', [
                isFirstTrue,
                sizeBuffer,
                inBuffer,
                moment1,
                moment2,
            ], [numBlocks]),
        ];
        while (numBlocks > 1) {
            const newNumBlocks = Math.ceil(numBlocks / 256);
            const countBuf = new Buffer(new Uint32Array([numBlocks]));
            passes.push(new ComputePass(statsCode, 'reduceMoments', [
                isFirstFalse,
                countBuf,
                moment1.readOnly(),
                moment1Tmp,
                unused,
            ], [newNumBlocks]));
            passes.push(new ComputePass(statsCode, 'reduceMoments', [
                isFirstFalse,
                countBuf,
                moment2.readOnly(),
                moment2Tmp,
                unused,
            ], [newNumBlocks]));
            numBlocks = newNumBlocks;
            let tmp = moment1;
            moment1 = moment1Tmp;
            moment1Tmp = tmp;
            tmp = moment2;
            moment2 = moment2Tmp;
            moment2Tmp = tmp;
        }
        passes.push(new ComputePass(affineCode, 'affine', [
            sizeBuffer,
            inBuffer,
            output,
            weight,
            bias,
            moment1.readOnly(),
            moment2.readOnly(),
        ], [Math.ceil(inputSize / 256)]));
        return passes;
    });
}
//# sourceMappingURL=webgpu.js.map