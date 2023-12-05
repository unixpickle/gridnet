import argparse
from itertools import count
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import datasets, transforms

from gridnet import Gridnet, Readout


class Model(nn.Module):
    def __init__(
        self,
        *,
        inner_iters: int,
        outer_iters: int,
        init_scale: float,
        device: torch.device,
    ):
        super().__init__()
        self.outer_iters = outer_iters
        self.device = device
        self.network = Gridnet(
            (32, 32, 32), inner_iters, 8, init_scale=init_scale, device=device
        )
        self.readout = Readout((32, 32, 32), out_channels=10, device=device)

    def forward(self, batch: torch.Tensor):
        batch = batch.reshape(-1, 28, 28)
        init_acts = torch.zeros(batch.shape[0], 32, 32, 32, device=self.device)
        init_acts[:, 2:-2, 2:-2, 0] = batch
        h = init_acts
        for _ in range(self.outer_iters):
            h = self.network(h)
        return self.readout(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--inner_iters", type=int, default=7)
    parser.add_argument("--outer_iters", type=int, default=6)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Model(
        inner_iters=args.inner_iters,
        outer_iters=args.outer_iters,
        init_scale=args.init_scale,
        device=device,
    )
    opt = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_iter = iterate_data(args.batch_size, train=True)
    test_iter = iterate_data(args.batch_size, train=False)
    for i in range(args.max_iters):
        inputs, targets = next(train_iter)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        train_loss = F.cross_entropy(logits, targets)
        train_acc = accuracy(logits, targets)

        # Empty the graph before using more memory.
        opt.zero_grad()
        train_loss.backward()

        with torch.no_grad():
            inputs, targets = next(test_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            test_loss = F.cross_entropy(logits, targets)
            test_acc = accuracy(logits, targets)

        opt.step()

        print(
            f"step {i}: test_loss={test_loss.item()} test_acc={test_acc} "
            f"train_loss={train_loss} train_acc={train_acc}"
        )


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(-1) == targets).float().mean().item()


def iterate_data(batch_size: int, train: bool) -> Iterator[torch.Tensor]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST(
        "../data", train=train, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    while True:
        yield from loader


if __name__ == "__main__":
    main()
