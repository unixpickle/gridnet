type CPUArray = Float32Array | Uint32Array;
type CPUArrayConstructor = Float32ArrayConstructor | Uint32ArrayConstructor;

class Buffer {
    // Created when we execute the kernel.
    public deviceBuffer: GPUBuffer = null;

    // Created when mapping results copied back from the buffer.
    public resultBuffer: GPUBuffer = null;

    constructor(
        public input: CPUArray,
        public output: CPUArray = null,
        public writable: boolean = null,
    ) {
        if (this.writable == null) {
            this.writable = (this.output == null ? false : true);
        }
    }

    createDeviceBuffer(device: GPUDevice) {
        this.deviceBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: this.input.byteLength,
            usage: GPUBufferUsage.STORAGE | (this.output != null ? GPUBufferUsage.COPY_SRC : 0),
        });
        const arrayBuffer = this.deviceBuffer.getMappedRange();
        const ctr = this.input.constructor as CPUArrayConstructor;
        new ctr(arrayBuffer).set(this.input);
        this.deviceBuffer.unmap();
    }
}

class ComputePass {
    constructor(
        public code: string,
        public entrypoint: string,
        public bindings: Buffer[],
        public gridSize: [number] | [number, number] | [number, number, number],
    ) {
    }

    encode(device: GPUDevice, encoder: GPUCommandEncoder) {
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

    private bindGroupLayout(): GPUBindGroupLayoutDescriptor {
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

    private bindGroup(): GPUBindGroupEntry[] {
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
    constructor(public passes: ComputePass[]) {
    }

    async execute(device: GPUDevice = null) {
        if (device == null) {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error('failed to get WebGPU adapter');
            }
            device = await adapter.requestDevice();
        }

        device.pushErrorScope('validation');
        device.pushErrorScope('internal');
        device.pushErrorScope('out-of-memory');

        this.createDeviceBuffers(device)

        const encoder = device.createCommandEncoder();
        this.passes.forEach((pass) => {
            pass.encode(device, encoder);
        });
        this.encodeResultCopies(device, encoder);

        const gpuCommands = encoder.finish();
        device.queue.submit([gpuCommands]);

        for (let i = 0; i < 3; i++) {
            const error = await device.popErrorScope();
            if (error) {
                throw error;
            }
        }

        await this.copyResults();
    }

    private createDeviceBuffers(device: GPUDevice) {
        this.buffers().forEach((buf) => buf.createDeviceBuffer(device));
    }

    private encodeResultCopies(device: GPUDevice, encoder: GPUCommandEncoder) {
        this.buffers().filter((x) => x.output != null).forEach((buf) => {
            buf.resultBuffer = device.createBuffer({
                size: buf.output.byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            encoder.copyBufferToBuffer(
                buf.deviceBuffer,
                0,
                buf.resultBuffer,
                0,
                buf.output.byteLength,
            );
        });
    }

    private async copyResults() {
        const b = this.buffers();
        for (let i = 0; i < b.length; i++) {
            const buf = b[i];
            if (buf.output == null) {
                continue;
            }
            await buf.resultBuffer.mapAsync(GPUMapMode.READ);
            const arrayBuffer = buf.resultBuffer.getMappedRange();
            const ctr = buf.output.constructor as CPUArrayConstructor;
            buf.output.set(new ctr(arrayBuffer));
            buf.resultBuffer.unmap();
        }
    }

    private buffers(): Buffer[] {
        const results: Buffer[] = [];
        this.passes.forEach((pass) => {
            pass.bindings.forEach((buf) => {
                if (!results.includes(buf)) {
                    results.push(buf);
                }
            })
        });
        return results;
    }
}
