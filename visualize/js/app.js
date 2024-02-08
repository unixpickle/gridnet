var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class ImagePicker {
    constructor() {
        this.image = null;
        this.offset = null;
        this.max_offset = null;
        this.upload_button = document.getElementById('upload-button');
        this.canvas = document.getElementById('image-resizer');
        this.input = document.createElement('input');
        this.input.style.visibility = 'hidden';
        this.input.style.position = 'fixed';
        this.input.type = 'file';
        document.body.appendChild(this.input);
        this.upload_button.addEventListener('click', () => this.input.click());
        this.input.addEventListener('input', () => this.handleUpload());
        this.setupPointerEvents();
    }
    getImage() {
        const dst = document.createElement('canvas');
        dst.width = 256;
        dst.height = 256;
        const img_scale = (256 - ImagePicker.PADDING * 2) / Math.min(this.image.width, this.image.height);
        const ctx = dst.getContext('2d');
        ctx.clearRect(0, 0, 256, 256);
        ctx.scale(img_scale, img_scale);
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
    handleUpload() {
        if (this.input.files && this.input.files[0]) {
            var reader = new FileReader();
            reader.addEventListener('load', () => {
                const img = document.createElement('img');
                img.addEventListener('load', () => {
                    this.handleImage(img);
                });
                img.src = reader.result;
            });
            reader.readAsDataURL(this.input.files[0]);
        }
    }
    handleImage(img) {
        this.image = img;
        if (img.width > img.height) {
            this.offset = [(img.width - img.height) / 2, 0];
        }
        else {
            this.offset = [0, (img.height - img.width) / 2];
        }
        const min_size = Math.min(img.width, img.height);
        this.max_offset = [img.width - min_size, img.height - min_size];
        this.draw();
    }
    draw() {
        if (this.image == null) {
            const ctx = this.canvas.getContext('2d');
            ctx.fillStyle = '#555';
            ctx.fillText('Please select an image', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }
        const window_size = this.canvas.width;
        const img_scale = (window_size - ImagePicker.PADDING * 2) / Math.min(this.image.width, this.image.height);
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, window_size, window_size);
        ctx.save();
        ctx.translate(ImagePicker.PADDING, ImagePicker.PADDING);
        ctx.scale(img_scale, img_scale);
        ctx.translate(-this.offset[0], -this.offset[1]);
        ctx.drawImage(this.image, 0, 0);
        ctx.restore();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.fillRect(ImagePicker.PADDING, 0, window_size - ImagePicker.PADDING * 2, ImagePicker.PADDING);
        ctx.fillRect(0, 0, ImagePicker.PADDING, window_size);
        ctx.fillRect(window_size - ImagePicker.PADDING, 0, ImagePicker.PADDING, window_size);
        ctx.fillRect(ImagePicker.PADDING, window_size - ImagePicker.PADDING, window_size - ImagePicker.PADDING * 2, ImagePicker.PADDING);
    }
    setupPointerEvents() {
        this.canvas.addEventListener('mousedown', (startEvent) => {
            const start_offset = this.offset;
            const moveEvent = (moveEvent) => {
                const delta_x = moveEvent.clientX - startEvent.clientX;
                const delta_y = moveEvent.clientY - startEvent.clientY;
                this.offset = [
                    Math.min(this.max_offset[0], Math.max(0, start_offset[0] - delta_x)),
                    Math.min(this.max_offset[1], Math.max(0, start_offset[1] - delta_y)),
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
ImagePicker.PADDING = 20;
class App {
    constructor() {
        this.image_picker = new ImagePicker();
        this.classify_button = document.getElementById('classify-button');
        loadModel().then((model) => {
            this.model = model;
        });
        this.classify_button.addEventListener('click', () => {
            const img = this.image_picker.getImage();
            const pred = this.model.forward(img);
            console.log(pred);
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