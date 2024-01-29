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