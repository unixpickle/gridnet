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
        this.classifyButton.addEventListener('click', () => {
            const img = this.imagePicker.getImage();
            const pred = this.model.forward(img);
            const probs = softmax(pred);
            this.predictions.showPredictions(probs);
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