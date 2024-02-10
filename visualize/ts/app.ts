class ImagePicker {
    public onReadyToClassify: () => void;

    private static PADDING: number = 20;

    private uploadButton: HTMLButtonElement;
    private canvas: HTMLCanvasElement;
    private input: HTMLInputElement;

    // State of the picker.
    private image: HTMLImageElement = null;
    private offset: [number, number] = null;
    private maxOffset: [number, number] = null;

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

        const imgScale = 256 / Math.min(this.image.width, this.image.height);

        const ctx = dst.getContext('2d');

        ctx.clearRect(0, 0, 256, 256);
        ctx.scale(imgScale, imgScale);
        ctx.translate(-this.offset[0], -this.offset[1]);
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
        const minSize = Math.min(img.width, img.height);
        this.maxOffset = [img.width - minSize, img.height - minSize];
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
        const imgScale = (windowSize - ImagePicker.PADDING * 2) / Math.min(this.image.width, this.image.height);

        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, windowSize, windowSize);
        ctx.save();
        ctx.translate(ImagePicker.PADDING, ImagePicker.PADDING);
        ctx.scale(imgScale, imgScale);
        ctx.translate(-this.offset[0], -this.offset[1]);
        ctx.drawImage(this.image, 0, 0);
        ctx.restore();

        ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.fillRect(ImagePicker.PADDING, 0, windowSize - ImagePicker.PADDING * 2, ImagePicker.PADDING);
        ctx.fillRect(0, 0, ImagePicker.PADDING, windowSize);
        ctx.fillRect(windowSize - ImagePicker.PADDING, 0, ImagePicker.PADDING, windowSize);
        ctx.fillRect(ImagePicker.PADDING, windowSize - ImagePicker.PADDING, windowSize - ImagePicker.PADDING * 2, ImagePicker.PADDING);
    }

    private setupPointerEvents() {
        this.canvas.addEventListener('mousedown', (startEvent: MouseEvent) => {
            const start_offset = this.offset;
            const moveEvent = (moveEvent: MouseEvent) => {
                const deltaX = moveEvent.clientX - startEvent.clientX;
                const deltaY = moveEvent.clientY - startEvent.clientY;
                this.offset = [
                    Math.min(this.maxOffset[0], Math.max(0, start_offset[0] - deltaX)),
                    Math.min(this.maxOffset[1], Math.max(0, start_offset[1] - deltaY)),
                ];
                this.draw();
            };
            window.addEventListener('mousemove', moveEvent);
            window.addEventListener('mouseup', () => {
                window.removeEventListener('mousemove', moveEvent);
            });
        });
    }
}

class App {
    private imagePicker: ImagePicker;
    private classifyButton: HTMLButtonElement;
    private model: ImagenetClassifier;
    private predictions: HTMLElement;
    private readyToClassify: boolean = false;

    constructor() {
        this.imagePicker = new ImagePicker();
        this.classifyButton = document.getElementById('classify-button') as HTMLButtonElement;
        this.predictions = document.getElementById('predictions');
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
            const classes: [number, string][] = [];
            for (let i = 0; i < probs.data.length; i++) {
                classes.push([probs.data[i], window.ImagenetClasses[i]]);
            }
            classes.sort((a, b) => b[0] - a[0]);
            this.predictions.innerHTML = '';
            classes.slice(0, 10).forEach((probAndCls) => {
                const item = document.createElement('div');
                const name = document.createElement('label');
                name.textContent = probAndCls[1];
                const probLabel = document.createElement('label');
                probLabel.textContent = '' + probAndCls[0];
                item.appendChild(name);
                item.appendChild(probLabel);
                this.predictions.appendChild(item);
            });
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