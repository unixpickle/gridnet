var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const ctx = self;
importScripts("model.js", "checkpoint.js");
let model;
onmessage = (event) => {
    const methods = {
        "putCheckpoint": putCheckpoint,
        "predict": predict,
    };
    const msg = event.data;
    if (!methods.hasOwnProperty(msg.method)) {
        postMessage({ id: msg.id, error: "no such method: " + msg.method });
        return;
    }
    methods[msg.method].apply(null, msg.args).then((x) => {
        postMessage({ id: msg.id, data: x });
    }).catch((e) => {
        postMessage({ id: msg.id, error: "" + e });
    });
};
function putCheckpoint(data) {
    return __awaiter(this, void 0, void 0, function* () {
        const ckpt = decodeCheckpoint(data);
        model = new ImagenetClassifier(ckpt);
    });
}
function predict(imageData) {
    return __awaiter(this, void 0, void 0, function* () {
        const inputTensor = new Tensor3(new Shape(3, 256, 256), imageData);
        return model.forward(inputTensor).data;
    });
}
//# sourceMappingURL=webworker.js.map