var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
function loadModel() {
    return __awaiter(this, void 0, void 0, function* () {
        const ckpt = yield loadCheckpoint('/checkpoints/imagenet_64x64');
        const model = new ImagenetClassifier(ckpt);
        const input = Tensor.zeros(new Shape(3, 256, 256));
        for (let i = 0; i < input.data.length; i++) {
            input.data[i] = Math.sin(i);
        }
        console.log(model.forward(input));
    });
}
window.addEventListener('load', () => {
    const button = document.createElement('button');
    button.onclick = () => {
        loadModel();
        button.textContent = 'Done';
    };
    button.textContent = 'Run forward';
    document.body.appendChild(button);
});
//# sourceMappingURL=app.js.map