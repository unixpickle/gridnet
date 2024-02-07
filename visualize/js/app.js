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
    }
}
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
    new App();
});
//# sourceMappingURL=app.js.map