// Based on https://github.com/unixpickle/flatten/blob/e8ed7791fdc64597d309ac0400afd82ae7104150/src/client.ts

interface ModelClientCallback {
    resolve: (_: any) => void;
    reject: (_: any) => void;
}

class WorkerClient {
    private worker: Worker;
    private callbacks: { [key: string]: ModelClientCallback };
    private curRequestId: number;

    constructor() {
        this.worker = new Worker("js/webworker.js");
        this.callbacks = {};
        this.curRequestId = 0;
        this.worker.onmessage = (e) => {
            const msg = e.data;
            const cb = this.callbacks[msg.id];
            if (msg["error"]) {
                cb.reject(new Error(msg.error));
                delete this.callbacks[msg.id];
            } else {
                cb.resolve(msg.data);
                delete this.callbacks[msg.id];
            }
        };
    }

    private async call(method: string, args: any[]): Promise<any> {
        const reqId = this.curRequestId++;
        const promise = new Promise((resolve, reject) => {
            this.callbacks[reqId] = { resolve: resolve, reject: reject };
        });
        this.worker.postMessage({
            id: reqId,
            method: method,
            args: args,
        });
        return promise;
    }

    async putCheckpoint(data: ArrayBuffer): Promise<void> {
        await this.call('putCheckpoint', [data]);
    }

    async predict(image: Tensor): Promise<Tensor> {
        assert(image.shape.equal(new Shape(3, 256, 256)));
        return new Tensor1(new Shape(1000), await this.call('predict', [image.data]));
    }
}
