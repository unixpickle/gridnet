"""
Simple test to see if a model can learn to copy its input.

This _should_ be an easy task, but disappearing gradients might
make it surprisingly difficult.
"""

import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW

from gridnet import GatedGridnet, Gridnet, Readout
from gridnet.backend_torch import ActivationFn
from gridnet.scripts.train_mnist import iterate_data


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
        self.readout = Readout((32, 32, 32), out_channels=28 * 28, device=device)

    def forward(self, batch: torch.Tensor):
        chunk = batch.reshape(-1, 28, 28)
        init_acts = self.init_in[None].repeat(batch.shape[0], 1, 1, 1)
        init_acts[:, 2:-2, 2:-2, 0] = chunk
        h = init_acts
        for _ in range(self.outer_iters):
            h = self.network(h)
            h = self.norm(h)
        h = self.readout(h)
        return h.reshape(batch.shape)


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
        inputs, _ = next(train_iter)
        inputs = inputs.to(device)
        outputs = model(inputs)
        train_loss = (inputs - outputs).pow(2).mean()

        # Empty the graph before using more memory.
        opt.zero_grad()
        train_loss.backward()

        with torch.no_grad():
            inputs, targets = next(test_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            test_loss = (inputs - outputs).pow(2).mean()

        opt.step()

        print(f"step {i}: test_loss={test_loss.item()} train_loss={train_loss}")


if __name__ == "__main__":
    main()
