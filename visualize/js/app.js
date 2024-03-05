var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class Predictions {
    constructor() {
        this.element = document.getElementById('predictions');
    }
    showPredictions(probs) {
        const classes = [];
        for (let i = 0; i < probs.data.length; i++) {
            classes.push([probs.data[i], window.ImagenetClasses[i]]);
        }
        classes.sort((a, b) => b[0] - a[0]);
        this.element.innerHTML = '';
        classes.slice(0, 10).forEach((probAndCls) => {
            const name = document.createElement('label');
            name.textContent = probAndCls[1];
            name.className = 'predictions-row-label';
            const percentText = (probAndCls[0] * 100).toFixed(2) + '%';
            const probBar = document.createElement('div');
            probBar.className = 'predictions-row-prob-bar';
            const probBarInner = document.createElement('div');
            probBarInner.className = 'predictions-row-prob-bar-inner';
            probBarInner.style.width = percentText;
            probBar.appendChild(probBarInner);
            const probLabel = document.createElement('label');
            probLabel.textContent = percentText;
            probLabel.className = 'predictions-row-prob-label';
            probBar.appendChild(probLabel);
            const row = document.createElement('div');
            row.className = 'predictions-row';
            row.appendChild(name);
            row.appendChild(probBar);
            this.element.appendChild(row);
        });
    }
}
class App {
    constructor() {
        this.readyToClassify = false;
        this.imagePicker = new ImagePicker();
        this.classifyButton = document.getElementById('classify-button');
        this.backend = document.getElementById('backend-select');
        this.predictions = new Predictions();
        loadModel().then((model) => {
            this.model = model;
            if (this.readyToClassify) {
                this.classifyButton.style.display = 'block';
            }
        });
        this.imagePicker.onReadyToClassify = () => {
            this.readyToClassify = true;
            if (this.model) {
                this.classifyButton.style.display = 'block';
            }
        };
        this.classifyButton.addEventListener('click', () => __awaiter(this, void 0, void 0, function* () {
            const img = this.imagePicker.getImage();
            const t1 = new Date().getTime();
            const pred = yield this.predict(img);
            const probs = softmax(pred);
            const t2 = new Date().getTime();
            console.log('predicted in', t2 - t1, 'ms');
            this.predictions.showPredictions(probs);
        }));
    }
    predict(image) {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.backend.value == 'CPU') {
                return this.model.forward(image);
            }
            else {
                const output = Tensor.zeros(new Shape(1000));
                const sequence = new KernelSequence(yield webgpuImageNetClassifier(new Buffer(image.data), new Buffer(this.model.initIn.data, null, true), new Buffer(this.model.patchEmb.weight.data), new Buffer(this.model.patchEmb.bias.data), this.model.patchEmb.outChannels, new Buffer(this.model.network.weight.data), new Buffer(this.model.network.bias.data), new Buffer(this.model.network.residualScale.data), 64, this.model.network.innerIterations, this.model.config.outerIters, new Buffer(this.model.norm.weight.data), new Buffer(this.model.norm.bias.data), new Buffer(this.model.readout.norm.weight.data), new Buffer(this.model.readout.norm.bias.data), new Buffer(this.model.readout.proj.weight.data), new Buffer(this.model.readout.proj.bias.data), this.model.readout.inChannels / (64 * 64), new Buffer(output.data, output.data)));
                yield sequence.execute();
                return output;
            }
        });
    }
}
function loadModel() {
    return __awaiter(this, void 0, void 0, function* () {
        const ckpt = yield loadCheckpoint('/checkpoints/imagenet_64x64');
        return new ImagenetClassifier(ckpt);
    });
}
window.addEventListener('load', () => {
    new App();
});
//# sourceMappingURL=app.js.map