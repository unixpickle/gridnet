async function loadModel() {
    const ckpt = await loadCheckpoint('/checkpoints/imagenet_64x64');
    console.log('ckpt', ckpt);
    const model = new ImagenetClassifier(ckpt);
    const input = Tensor.zeros(new Shape(3, 256, 256));
    console.log(model.forward(input));
}

loadModel();