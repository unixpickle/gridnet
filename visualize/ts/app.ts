class Predictions {
    private element: HTMLElement;

    constructor() {
        this.element = document.getElementById('predictions');
    }

    showPredictions(probs: Tensor) {
        const classes: [number, string][] = [];
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
            probBar.className = 'predictions-row-prob-bar'
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
    private backend: HTMLSelectElement;
    private imagePicker: ImagePicker;
    private classifyButton: HTMLButtonElement;
    private model: ImagenetClassifier;
    private predictions: Predictions;
    private readyToClassify: boolean = false;

    constructor() {
        this.imagePicker = new ImagePicker();
        this.classifyButton = document.getElementById('classify-button') as HTMLButtonElement;
        this.backend = document.getElementById('backend-select') as HTMLSelectElement;
        this.predictions = new Predictions();
        loadModel().then((model) => {
            this.model = model;
            if (this.readyToClassify) {
                // We loaded the model _after_ an image was picked.
                this.classifyButton.style.display = 'block';
            }
        });

        this.imagePicker.onReadyToClassify = () => {
            this.readyToClassify = true;
            if (this.model) {
                // We picked an image after the model was loaded.
                this.classifyButton.style.display = 'block';
            }
        };

        this.classifyButton.addEventListener('click', async () => {
            const img = this.imagePicker.getImage();
            const t1 = new Date().getTime();
            const pred = await this.predict(img);
            const probs = softmax(pred);
            const t2 = new Date().getTime();
            console.log('predicted in', t2 - t1, 'ms');
            this.predictions.showPredictions(probs);
        });
    }

    async predict(image: Tensor3): Promise<Tensor1> {
        if (this.backend.value == 'CPU') {
            return this.model.forward(image);
        } else {
            const output = Tensor.zeros(new Shape(1000));
            const sequence = new KernelSequence(await webgpuImageNetClassifier(
                new Buffer(image.data),
                new Buffer(this.model.initIn.data, null, true),
                new Buffer(this.model.patchEmb.weight.data),
                new Buffer(this.model.patchEmb.bias.data),
                this.model.patchEmb.outChannels,
                new Buffer(this.model.network.weight.data),
                new Buffer(this.model.network.bias.data),
                new Buffer(this.model.network.residualScale.data),
                64,
                this.model.network.innerIterations,
                this.model.config.outerIters,
                new Buffer(this.model.norm.weight.data),
                new Buffer(this.model.norm.bias.data),
                new Buffer(this.model.readout.norm.weight.data),
                new Buffer(this.model.readout.norm.bias.data),
                new Buffer(this.model.readout.proj.weight.data),
                new Buffer(this.model.readout.proj.bias.data),
                this.model.readout.inChannels / (64 * 64),
                new Buffer(output.data, output.data),
            ));
            await sequence.execute();
            return output;
        }
    }
}

async function loadModel(): Promise<ImagenetClassifier> {
    const ckpt = await loadCheckpoint('/checkpoints/imagenet_64x64');
    return new ImagenetClassifier(ckpt);
}

window.addEventListener('load', () => {
    // const button = document.createElement('button');
    // button.onclick = () => {
    //     loadModel().then(() => {
    //         button.textContent = 'Done';
    //     });
    // };
    // button.textContent = 'Run forward';
    // document.body.appendChild(button);

    new App();
});