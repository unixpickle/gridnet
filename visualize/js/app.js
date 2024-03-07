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
        this.loader = document.getElementById('loader');
        this.error = document.getElementById('error');
        this.predictions = new Predictions();
        loadModel().then(([client, model]) => {
            this.client = client;
            this.model = model;
            if (this.readyToClassify) {
                this.classifyButton.style.display = 'block';
            }
            this.clearLoader();
        }).catch((err) => {
            this.showError(`Error loading model: ${err}`);
        });
        this.imagePicker.onReadyToClassify = () => {
            this.readyToClassify = true;
            if (this.model) {
                this.classifyButton.style.display = 'block';
            }
        };
        this.classifyButton.addEventListener('click', () => __awaiter(this, void 0, void 0, function* () {
            this.classifyButton.classList.add('disabled');
            this.showLoader('Running classifier...');
            const img = this.imagePicker.getImage();
            const t1 = new Date().getTime();
            let pred;
            try {
                pred = yield this.predict(img);
            }
            catch (err) {
                this.showError(`Failed to run classifier: ${err}`);
                return;
            }
            finally {
                this.classifyButton.classList.remove('disabled');
            }
            const probs = softmax(pred);
            const t2 = new Date().getTime();
            console.log('predicted in', t2 - t1, 'ms');
            this.clearLoader();
            this.predictions.showPredictions(probs);
        }));
    }
    predict(image) {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.backend.value == 'CPU') {
                return yield this.client.predict(image);
            }
            else {
                const grid = this.model.network.bias.shape[0];
                const output = Tensor.zeros(new Shape(1000));
                const sequence = new KernelSequence(yield webgpuImageNetClassifier(new Buffer(image.data), new Buffer(this.model.initIn.data, null, true), new Buffer(this.model.patchEmb.weight.data), new Buffer(this.model.patchEmb.bias.data), this.model.patchEmb.outChannels, new Buffer(this.model.network.weight.data), new Buffer(this.model.network.bias.data), new Buffer(this.model.network.residualScale.data), grid, this.model.network.innerIterations, this.model.config.outerIters, new Buffer(this.model.norm.weight.data), new Buffer(this.model.norm.bias.data), new Buffer(this.model.readout.norm.weight.data), new Buffer(this.model.readout.norm.bias.data), new Buffer(this.model.readout.proj.weight.data), new Buffer(this.model.readout.proj.bias.data), this.model.readout.inChannels / (grid * grid), new Buffer(output.data, output.data)));
                yield sequence.execute();
                return output;
            }
        });
    }
    clearLoader() {
        this.loader.style.display = 'none';
        this.error.style.display = 'none';
    }
    showLoader(msg) {
        this.error.style.display = 'none';
        this.loader.style.display = 'block';
        this.loader.textContent = msg;
    }
    showError(msg) {
        this.error.style.display = 'inline-block';
        this.error.textContent = msg;
        this.loader.style.display = 'none';
    }
}
function loadModel() {
    return __awaiter(this, void 0, void 0, function* () {
        const data = yield loadCheckpointData('/checkpoints/imagenet_64x64');
        const client = new WorkerClient();
        yield client.putCheckpoint(data);
        return [client, new ImagenetClassifier(decodeCheckpoint(data))];
    });
}
window.addEventListener('load', () => {
    new App();
});
//# sourceMappingURL=app.js.map