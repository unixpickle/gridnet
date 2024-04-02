var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class WorkerClient {
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
            }
            else {
                cb.resolve(msg.data);
                delete this.callbacks[msg.id];
            }
        };
    }
    call(method, args) {
        return __awaiter(this, void 0, void 0, function* () {
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
        });
    }
    putCheckpoint(data) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.call('putCheckpoint', [data]);
        });
    }
    predict(image) {
        return __awaiter(this, void 0, void 0, function* () {
            assert(image.shape.equal(new Shape(3, 256, 256)));
            return new Tensor1(new Shape(1000), yield this.call('predict', [image.data]));
        });
    }
}
//# sourceMappingURL=worker_client.js.map