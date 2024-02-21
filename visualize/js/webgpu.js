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
}
class ComputePass {
    constructor(code, entrypoint, bindings, gridSize) {
        this.code = code;
        this.entrypoint = entrypoint;
        this.bindings = bindings;
        this.gridSize = gridSize;
    }
    encode(device, encoder) {
        const bindGroupLayout = device.createBindGroupLayout(this.bindGroupLayout());
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: this.bindGroup(),
        });
        const shaderModule = device.createShaderModule({
            code: this.code,
        });
        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            }),
            compute: {
                module: shaderModule,
                entryPoint: this.entrypoint,
            },
        });
        const passEncoder = encoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(this.gridSize[0], this.gridSize[1], this.gridSize[2]);
        passEncoder.end();
    }
    bindGroupLayout() {
        return {
            entries: this.bindings.map((buf, i) => {
                return {
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: buf.writable ? 'storage' : 'read-only-storage',
                    },
                };
            })
        };
    }
    bindGroup() {
        return this.bindings.map((buf, i) => {
            return {
                binding: i,
                resource: {
                    buffer: buf.deviceBuffer,
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
            this.passes.forEach((pass) => {
                pass.encode(device, encoder);
            });
            this.encodeResultCopies(device, encoder);
            const gpuCommands = encoder.finish();
            device.queue.submit([gpuCommands]);
            for (let i = 0; i < 3; i++) {
                const error = yield device.popErrorScope();
                if (error) {
                    throw new Error('error from kernel call: ' + error);
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
                if (!results.includes(buf)) {
                    results.push(buf);
                }
            });
        });
        return results;
    }
}
//# sourceMappingURL=webgpu.js.map