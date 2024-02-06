class ImagePicker {
    private upload_button: HTMLButtonElement;
    private canvas: HTMLCanvasElement;
    private input: HTMLInputElement;

    constructor() {
        this.upload_button = document.getElementById('upload-button') as HTMLButtonElement;
        this.canvas = document.getElementById('image-resizer') as HTMLCanvasElement;
        this.input = document.createElement('input');
        this.input.style.visibility = 'hidden';
        this.input.style.position = 'fixed';
        document.body.appendChild(this.input);
        this.upload_button.addEventListener('click', () => this.input.click());
        this.input.addEventListener('input', () => this.handleUpload());
    }

    handleUpload() {
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

    handleImage(img: HTMLImageElement) {
        // TODO: setup picker here.
    }
}

async function loadModel() {
    const ckpt = await loadCheckpoint('/checkpoints/imagenet_64x64');
    const model = new ImagenetClassifier(ckpt);
    const input = Tensor.zeros(new Shape(3, 256, 256));
    for (let i = 0; i < input.data.length; i++) {
        input.data[i] = Math.sin(i);
    }
    console.log(model.forward(input));
}

window.addEventListener('load', () => {
    const button = document.createElement('button');
    button.onclick = () => {
        loadModel().then(() => {
            button.textContent = 'Done';
        });
    };
    button.textContent = 'Run forward';
    document.body.appendChild(button);
});