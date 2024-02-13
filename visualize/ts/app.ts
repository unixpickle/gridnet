class ImagePicker {
    public onReadyToClassify: () => void;

    private uploadButton: HTMLButtonElement;
    private canvas: HTMLCanvasElement;
    private input: HTMLInputElement;

    // State of the picker.
    private image: HTMLImageElement = null;
    private scale: number = 1.0;
    private offset: [number, number] = null;

    constructor() {
        this.uploadButton = document.getElementById('upload-button') as HTMLButtonElement;
        this.canvas = document.getElementById('image-resizer') as HTMLCanvasElement;
        this.input = document.createElement('input');
        this.input.style.visibility = 'hidden';
        this.input.style.position = 'fixed';
        this.input.type = 'file';
        document.body.appendChild(this.input);
        this.uploadButton.addEventListener('click', () => this.input.click());
        this.input.addEventListener('input', () => this.handleUpload());
        this.setupPointerEvents();
    }

    public getImage(): Tensor3 {
        const dst = document.createElement('canvas');
        dst.width = 256;
        dst.height = 256;

        const imgScale = this.scale * (256 / Math.min(this.image.width, this.image.height));

        const ctx = dst.getContext('2d');

        ctx.clearRect(0, 0, 256, 256);
        ctx.translate(-this.offset[0], -this.offset[1]);
        ctx.scale(imgScale, imgScale);
        ctx.drawImage(this.image, 0, 0);

        const data = ctx.getImageData(0, 0, 256, 256);
        const output = Tensor3.zeros(new Shape(3, 256, 256));

        let offset = 0;
        for (let y = 0; y < 256; y++) {
            for (let x = 0; x < 256; x++) {
                const r = data.data[offset++] / 255;
                const g = data.data[offset++] / 255;
                const b = data.data[offset++] / 255;
                offset++;
                output.set((r - 0.485) / 0.229, 0, y, x);
                output.set((g - 0.456) / 0.224, 1, y, x);
                output.set((b - 0.406) / 0.225, 2, y, x);
            }
        }

        return output;
    }

    private handleUpload() {
        if (this.input.files && this.input.files[0]) {
            var reader = new FileReader();
            reader.addEventListener('load', () => {
                const img = document.createElement('img');
                img.addEventListener('load', () => {
                    this.handleImage(img);
                });
                img.src = reader.result as string;
            });
            reader.readAsDataURL(this.input.files[0]);
        }
    }

    private handleImage(img: HTMLImageElement) {
        this.canvas.style.display = 'block';
        this.onReadyToClassify();

        this.image = img;
        if (img.width > img.height) {
            this.offset = [(img.width - img.height) / 2, 0];
        } else {
            this.offset = [0, (img.height - img.width) / 2];
        }
        this.draw();
    }

    private draw() {
        if (this.image == null) {
            const ctx = this.canvas.getContext('2d');
            ctx.fillStyle = '#555';
            ctx.fillText('Please select an image', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }

        const windowSize = this.canvas.width;
        const imgScale = this.scale * (windowSize / Math.min(this.image.width, this.image.height));

        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, windowSize, windowSize);
        ctx.save();
        ctx.scale(imgScale, imgScale);
        ctx.translate(-this.offset[0], -this.offset[1]);
        ctx.drawImage(this.image, 0, 0);
        ctx.restore();
    }

    private setupPointerEvents() {
        this.canvas.addEventListener('mousedown', (startEvent: MouseEvent) => {
            const startOffset = this.offset;
            const moveEvent = (moveEvent: MouseEvent) => {
                const deltaX = moveEvent.clientX - startEvent.clientX;
                const deltaY = moveEvent.clientY - startEvent.clientY;
                const rectSize = this.canvas.getBoundingClientRect();
                const relScale = Math.min(this.image.width, this.image.height) / (
                    rectSize.width * this.scale
                );
                this.offset = [
                    startOffset[0] - relScale * deltaX,
                    startOffset[1] - relScale * deltaY,
                ];
                this.constrainOffset();
                this.draw();
                moveEvent.preventDefault();
                moveEvent.stopPropagation();
            };
            window.addEventListener('mousemove', moveEvent);
            window.addEventListener('mouseup', () => {
                window.removeEventListener('mousemove', moveEvent);
            });
        });
        this.canvas.addEventListener('mousewheel', (e: WheelEvent) => {
            const minSize = Math.min(this.image.width, this.image.height);
            const oldViewportSize = minSize / this.scale;

            this.scale *= Math.exp(-e.deltaY / 500);
            if (this.scale < 1) {
                this.scale = 1.0;
            }
            const newViewportSize = minSize / this.scale;

            const centerDelta = (oldViewportSize - newViewportSize) / 2;
            this.offset[0] += centerDelta;
            this.offset[1] += centerDelta;
            this.constrainOffset();

            this.draw();
        });
    }

    private constrainOffset() {
        const maxOffset = this.maxOffset();
        this.offset = [
            Math.min(maxOffset[0], Math.max(0, this.offset[0])),
            Math.min(maxOffset[1], Math.max(0, this.offset[1])),
        ];
    }

    private maxOffset(): [number, number] {
        const minSize = Math.min(this.image.width, this.image.height);
        const viewportSize = minSize / this.scale;
        return [this.image.width - viewportSize, this.image.height - viewportSize];
    }
}

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
    private imagePicker: ImagePicker;
    private classifyButton: HTMLButtonElement;
    private model: ImagenetClassifier;
    private predictions: Predictions;
    private readyToClassify: boolean = false;

    constructor() {
        this.imagePicker = new ImagePicker();
        this.classifyButton = document.getElementById('classify-button') as HTMLButtonElement;
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

        this.classifyButton.addEventListener('click', () => {
            const img = this.imagePicker.getImage();
            const pred = this.model.forward(img);
            const probs = softmax(pred);
            this.predictions.showPredictions(probs);
        });
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