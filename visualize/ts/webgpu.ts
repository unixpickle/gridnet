type CPUArray = Float32Array | Uint32Array;
type CPUArrayConstructor = Float32ArrayConstructor | Uint32ArrayConstructor;

interface BindingArg {
    layout(): GPUBufferBindingLayout;
    buffer(): Buffer;
}

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

    layout(): GPUBufferBindingLayout {
        return {
            type: this.writable ? 'storage' : 'read-only-storage',
        }
    }

    buffer(): Buffer {
        return this;
    }

    readOnly(): ReadOnlyBuffer {
        return new ReadOnlyBuffer(this);
    }
}

class ReadOnlyBuffer {
    constructor(private _buffer: Buffer) {
    }

    layout(): GPUBufferBindingLayout {
        return {
            type: 'read-only-storage',
        }
    }

    buffer(): Buffer {
        return this._buffer;
    }
}

interface ShaderModuleCacheItem {
    code: string;
    device: GPUDevice;
    module: GPUShaderModule;
}

class ShaderModuleCache {
    private items: ShaderModuleCacheItem[] = [];
    static Global = new ShaderModuleCache();

    createOrReuse(device: GPUDevice, code: string): GPUShaderModule {
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

class ComputePass {
    constructor(
        public code: string,
        public entrypoint: string,
        public bindings: BindingArg[],
        public gridSize: [number] | [number, number] | [number, number, number],
    ) {
    }

    async encode(device: GPUDevice, encoder: GPUCommandEncoder) {
        const bindGroupLayout = device.createBindGroupLayout(this.bindGroupLayout());
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: this.bindGroup(),
        });
        const shaderModule = ShaderModuleCache.Global.createOrReuse(device, this.code);

        const pipeline = await device.createComputePipelineAsync({
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
                    buffer: buf.layout(),
                };
            })
        };
    }

    private bindGroup(): GPUBindGroupEntry[] {
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
        for (let i = 0; i < this.passes.length; i++) {
            await this.passes[i].encode(device, encoder);
        }
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
                if (!results.includes(buf.buffer())) {
                    results.push(buf.buffer());
                }
            })
        });
        return results;
    }
}

async function fetchKernel(name: string): Promise<string> {
    return await (await fetch(`/wgsl/${name}`)).text();
}

async function webgpuLayerNorm(
    input: Buffer,
    output: Buffer,
    weight: Buffer,
    bias: Buffer,
): Promise<ComputePass[]> {
    const statsCode = await fetchKernel('moments.wgsl');
    const affineCode = await fetchKernel('affine.wgsl');

    const inBuffer = input.readOnly();

    let moment1 = new Buffer(new Float32Array(1024), null, true);
    let moment2 = new Buffer(new Float32Array(1024), null, true);
    let moment1Tmp = new Buffer(new Float32Array(1024), null, true);
    let moment2Tmp = new Buffer(new Float32Array(1024), null, true);
    const unused = new Buffer(new Float32Array(1), null, true);

    const inputSize = input.input.length;
    const sizeBuffer = new Buffer(new Uint32Array([inputSize]));

    const isFirstTrue = new Buffer(new Uint32Array([1]));
    const isFirstFalse = new Buffer(new Uint32Array([0]));

    let numBlocks = Math.ceil(inputSize / 256);
    const passes: ComputePass[] = [
        new ComputePass(
            statsCode,
            'reduceMoments',
            [
                isFirstTrue,
                sizeBuffer,
                inBuffer,
                moment1,
                moment2,
            ],
            [numBlocks],
        ),
    ];

    while (numBlocks > 1) {
        const newNumBlocks = Math.ceil(numBlocks / 256);
        const countBuf = new Buffer(new Uint32Array([numBlocks]));
        passes.push(new ComputePass(
            statsCode,
            'reduceMoments',
            [
                isFirstFalse,
                countBuf,
                moment1.readOnly(),
                moment1Tmp,
                unused,
            ],
            [newNumBlocks],
        ));
        passes.push(new ComputePass(
            statsCode,
            'reduceMoments',
            [
                isFirstFalse,
                countBuf,
                moment2.readOnly(),
                moment2Tmp,
                unused,
            ],
            [newNumBlocks],
        ));

        numBlocks = newNumBlocks;
        let tmp = moment1;
        moment1 = moment1Tmp;
        moment1Tmp = tmp;
        tmp = moment2;
        moment2 = moment2Tmp;
        moment2Tmp = tmp;

        // Not sure why this doesn't type check:
        // moment1, moment1Tmp = [moment1Tmp, moment1];
        // moment2, moment2Tmp = [moment2Tmp, moment2];
    }

    passes.push(new ComputePass(
        affineCode,
        'affine',
        [
            sizeBuffer,
            inBuffer,
            output,
            weight,
            bias,
            moment1.readOnly(),
            moment2.readOnly(),
        ],
        [Math.ceil(inputSize / 256)],
    ));

    return passes;
}
