/// <reference lib="webworker" />

const ctx: WorkerGlobalScope = (self as any)

importScripts(
    "model.js",
    "checkpoint.js",
);

let model: ImagenetClassifier;

onmessage = (event) => {
    const methods: { [key: string]: (...args: any[]) => Promise<any> } = {
        "putCheckpoint": putCheckpoint,
        "predict": predict,
    }
    const msg = event.data;
    if (!methods.hasOwnProperty(msg.method)) {
        postMessage({ id: msg.id, error: "no such method: " + msg.method });
        return;
    }
    methods[msg.method].apply(null, msg.args).then((x: any) => {
        postMessage({ id: msg.id, data: x });
    }).catch((e: any) => {
        postMessage({ id: msg.id, error: "" + e });
    });
}

async function putCheckpoint(data: ArrayBuffer) {
    const ckpt = decodeCheckpoint(data);
    model = new ImagenetClassifier(ckpt);
}

async function predict(imageData: Float32Array) {
    const inputTensor = new Tensor3(new Shape(3, 256, 256), imageData);
    return model.forward(inputTensor).data;
}
