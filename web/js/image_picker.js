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
        this.setupTouchEvents();
    }
    setEnabled(enabled) {
        if (enabled) {
            this.canvas.classList.remove('disabled');
            this.uploadButton.classList.remove('disabled');
        }
        else {
            this.canvas.classList.add('disabled');
            this.uploadButton.classList.add('disabled');
        }
    }
    getImage() {
        const dst = document.createElement('canvas');
        dst.width = ImagePicker.OUTPUT_IMAGE_SIZE;
        dst.height = ImagePicker.OUTPUT_IMAGE_SIZE;
        const imgScale = dst.width / this.viewportSize();
        const ctx = dst.getContext('2d');
        ctx.clearRect(0, 0, dst.width, dst.height);
        ctx.scale(imgScale, imgScale);
        ctx.translate(-this.offset[0], -this.offset[1]);
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
                img.addEventListener('error', (e) => {
                    if (e.error) {
                        alert(`Error loading image: ${e.error}`);
                    }
                    else {
                        alert('Failed to load image.');
                    }
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
        this.scale = 1.0;
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
            this.offset = [
                this.offset[0] + oldCursorX - newCursorX,
                this.offset[1] + oldCursorY - newCursorY,
            ];
            this.constrainOffset();
            this.draw();
        });
    }
    setupTouchEvents() {
        const prevPositions = new Map();
        const updatePositions = (e) => {
            for (let i = 0; i < e.changedTouches.length; i++) {
                const touch = e.changedTouches[i];
                prevPositions.set(touch.identifier, [touch.clientX, touch.clientY]);
            }
        };
        this.canvas.addEventListener('touchstart', (startEvent) => {
            startEvent.preventDefault();
            startEvent.stopPropagation();
            updatePositions(startEvent);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            if (e.touches.length == 1) {
                const [prevX, prevY] = prevPositions.get(e.touches[0].identifier);
                const deltaX = e.touches[0].clientX - prevX;
                const deltaY = e.touches[0].clientY - prevY;
                this.offset = [
                    this.offset[0] - this.scaleClientToImage(deltaX),
                    this.offset[1] - this.scaleClientToImage(deltaY),
                ];
                this.constrainOffset();
                this.draw();
            }
            else if (e.touches.length == 2) {
                const oldP1 = prevPositions.get(e.touches[0].identifier);
                const oldP2 = prevPositions.get(e.touches[1].identifier);
                const oldCenter = [(oldP2[0] + oldP1[0]) / 2, (oldP2[1] + oldP1[1]) / 2];
                const oldDist = Math.sqrt(Math.pow(oldP2[0] - oldP1[0], 2) + Math.pow(oldP2[1] - oldP1[1], 2));
                const newCenter = [
                    (e.touches[0].clientX + e.touches[1].clientX) / 2,
                    (e.touches[0].clientY + e.touches[1].clientY) / 2,
                ];
                const newDist = Math.sqrt(Math.pow(e.touches[0].clientX - e.touches[1].clientX, 2) +
                    Math.pow(e.touches[0].clientY - e.touches[1].clientY, 2));
                const oldImageCenter = [
                    this.scaleClientToImage(oldCenter[0] - rect.left),
                    this.scaleClientToImage(oldCenter[1] - rect.top),
                ];
                this.scale *= (newDist / oldDist);
                if (this.scale < 1) {
                    this.scale = 1;
                }
                const newImageCenter = [
                    this.scaleClientToImage(newCenter[0] - rect.left),
                    this.scaleClientToImage(newCenter[1] - rect.top),
                ];
                this.offset = [
                    this.offset[0] + oldImageCenter[0] - newImageCenter[0],
                    this.offset[1] + oldImageCenter[1] - newImageCenter[1],
                ];
                this.constrainOffset();
                this.draw();
            }
            updatePositions(e);
        });
        const deleteTouch = (e) => {
            for (let i = 0; i < e.changedTouches.length; i++) {
                prevPositions.delete(e.changedTouches[i].identifier);
            }
        };
        this.canvas.addEventListener('touchend', deleteTouch);
        this.canvas.addEventListener('touchcancel', deleteTouch);
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