import argparse
import sys
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import datasets, transforms

from gridnet import GatedGridnet, Gridnet, Readout
from gridnet.backend_torch import ActivationFn


class Model(nn.Module):
    def __init__(
        self,
        *,
        inner_iters: int,
        outer_iters: int,
        init_scale: float,
        residual_scale: float,
        activation: ActivationFn,
        gated: bool,
        remember_bias: float,
        device: torch.device,
    ):
        super().__init__()
        self.outer_iters = outer_iters
        self.device = device
        self.init_in = nn.Parameter(torch.randn(32, 32, 32, device=device))
        if gated:
            self.network = GatedGridnet(
                (32, 32, 32),
                inner_iters,
                8,
                init_scale=init_scale,
                device=device,
                activation=activation,
            )
            with torch.no_grad():
                self.network.bias[1].fill_(remember_bias)
            self.norm = nn.Identity()
        else:
            self.network = Gridnet(
                (32, 32, 32),
                inner_iters,
                8,
                init_scale=init_scale,
                residual_scale=residual_scale,
                device=device,
                normalize=False,
                activation=activation,
            )
            self.norm = nn.LayerNorm((32,) * 3, device=device)
        self.readout = Readout((32, 32, 32), out_channels=10, device=device)

    def forward(self, batch: torch.Tensor):
        batch = batch.reshape(-1, 28, 28)
        init_acts = self.init_in[None].repeat(batch.shape[0], 1, 1, 1)
        init_acts[:, 2:-2, 2:-2, 0] = batch
        h = init_acts
        for _ in range(self.outer_iters):
            h = self.network(h)
            h = self.norm(h)
        return self.readout(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--inner_iters", type=int, default=1)
    parser.add_argument("--outer_iters", type=int, default=64)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--remember_bias", type=float, default=2.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("arguments:", sys.argv)

    model = Model(
        inner_iters=args.inner_iters,
        outer_iters=args.outer_iters,
        init_scale=args.init_scale,
        residual_scale=args.residual_scale,
        activation=args.activation,
        gated=args.gated,
        remember_bias=args.remember_bias,
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
