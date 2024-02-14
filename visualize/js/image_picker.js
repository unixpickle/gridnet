class ImagePicker {
    constructor() {
        this.image = null;
        this.scale = 1.0;
        this.offset = null;
        this.uploadButton = document.getElementById('upload-button');
        this.canvas = document.getElementById('image-resizer');
        this.input = document.createElement('input');
        this.input.style.visibility = 'hidden';
        this.input.style.position = 'fixed';
        this.input.type = 'file';
        document.body.appendChild(this.input);
        this.uploadButton.addEventListener('click', () => this.input.click());
        this.input.addEventListener('input', () => this.handleUpload());
        this.setupPointerEvents();
    }
    getImage() {
        const dst = document.createElement('canvas');
        dst.width = ImagePicker.OUTPUT_IMAGE_SIZE;
        dst.height = ImagePicker.OUTPUT_IMAGE_SIZE;
        const imgScale = dst.width / this.viewportSize();
        const ctx = dst.getContext('2d');
        ctx.clearRect(0, 0, dst.width, dst.height);
        ctx.translate(-this.offset[0], -this.offset[1]);
        ctx.scale(imgScale, imgScale);
        ctx.drawImage(this.image, 0, 0);
        const data = ctx.getImageData(0, 0, dst.width, dst.height);
        const output = Tensor3.zeros(new Shape(3, dst.height, dst.width));
        let offset = 0;
        for (let y = 0; y < dst.height; y++) {
            for (let x = 0; x < dst.width; x++) {
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
        this.canvas.style.display = 'block';
        this.onReadyToClassify();
        this.image = img;
        if (img.width > img.height) {
            this.offset = [(img.width - img.height) / 2, 0];
        }
        else {
            this.offset = [0, (img.height - img.width) / 2];
        }
        this.draw();
    }
    draw() {
        if (this.image == null) {
            const ctx = this.canvas.getContext('2d');
            ctx.fillStyle = '#555';
            ctx.fillText('Please select an image', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }
        const imgScale = this.canvas.width / this.viewportSize();
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.save();
        ctx.scale(imgScale, imgScale);
        ctx.translate(-this.offset[0], -this.offset[1]);
        ctx.drawImage(this.image, 0, 0);
        ctx.restore();
    }
    setupPointerEvents() {
        this.canvas.addEventListener('mousedown', (startEvent) => {
            const startOffset = this.offset;
            const moveEvent = (moveEvent) => {
                const deltaX = moveEvent.clientX - startEvent.clientX;
                const deltaY = moveEvent.clientY - startEvent.clientY;
                this.offset = [
                    startOffset[0] - this.scaleClientToImage(deltaX),
                    startOffset[1] - this.scaleClientToImage(deltaY),
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
        this.canvas.addEventListener('mousewheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const oldCursorX = this.scaleClientToImage(e.offsetX);
            const oldCursorY = this.scaleClientToImage(e.offsetY);
            this.scale *= Math.exp(-e.deltaY / 500);
            if (this.scale < 1) {
                this.scale = 1.0;
            }
            const newCursorX = this.scaleClientToImage(e.offsetX);
            const newCursorY = this.scaleClientToImage(e.offsetY);
            this.offset[0] += oldCursorX - newCursorX;
            this.offset[1] += oldCursorY - newCursorY;
            this.constrainOffset();
            this.draw();
        });
    }
    constrainOffset() {
        const maxOffset = this.maxOffset();
        this.offset = [
            Math.min(maxOffset[0], Math.max(0, this.offset[0])),
            Math.min(maxOffset[1], Math.max(0, this.offset[1])),
        ];
    }
    maxOffset() {
        const viewportSize = this.viewportSize();
        return [this.image.width - viewportSize, this.image.height - viewportSize];
    }
    scaleClientToImage(x) {
        const rectSize = this.canvas.getBoundingClientRect();
        const scale = this.viewportSize() / rectSize.width;
        return x * scale;
    }
    viewportSize() {
        const minSize = Math.min(this.image.width, this.image.height);
        return minSize / this.scale;
    }
}
ImagePicker.OUTPUT_IMAGE_SIZE = 256;
//# sourceMappingURL=image_picker.js.map