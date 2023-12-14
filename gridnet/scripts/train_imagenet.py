import argparse
import hashlib
import os
import sys
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import datasets, transforms

from gridnet import Gridnet, Readout
from gridnet.backend_torch import ActivationFn


class Model(nn.Module):
    def __init__(
        self,
        *,
        inner_iters: int,
        outer_iters: int,
        outer_residual: bool,
        init_scale: float,
        residual_scale: float,
        emb_channels: int,
        activation: ActivationFn,
        device: torch.device,
    ):
        super().__init__()
        self.outer_iters = outer_iters
        self.outer_residual = outer_residual
        self.emb_channels = emb_channels
        self.device = device
        self.init_in = nn.Parameter(torch.randn(64, 64, 64, device=device))
        self.network = Gridnet(
            (64, 64, 64),
            inner_iters,
            8,
            init_scale=init_scale,
            residual_scale=residual_scale,
            device=device,
            normalize=False,
            activation=activation,
        )
        self.patch_emb = nn.Conv2d(
            3, emb_channels, kernel_size=4, stride=4, device=device
        )
        self.norm = nn.LayerNorm((64,) * 3, device=device)
        self.readout = Readout((64, 64, 64), out_channels=1000, device=device)

    def forward(self, images: torch.Tensor):
        init_acts = self.init_in[None].repeat(images.shape[0], 1, 1, 1)
        init_acts[:, :, :, : self.emb_channels] = self.patch_emb(images).permute(
            0, 2, 3, 1
        )
        h = init_acts
        for _ in range(self.outer_iters):
            if self.outer_residual:
                norm_h = self.norm(h)
                h = h + (self.network(norm_h) - norm_h)
            else:
                h = self.network(h)
                h = self.norm(h)
        return self.readout(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="leaky_relu")
    parser.add_argument("--emb_channels", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--inner_iters", type=int, default=8)
    parser.add_argument("--outer_iters", type=int, default=12)
    parser.add_argument("--outer_residual", action="store_true")
    args = parser.parse_args()

    print("Command-line args:", sys.argv[1:])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Model(
        inner_iters=args.inner_iters,
        outer_iters=args.outer_iters,
        outer_residual=args.outer_residual,
        init_scale=args.init_scale,
        residual_scale=args.residual_scale,
        emb_channels=args.emb_channels,
        activation=args.activation,
        device=device,
    )
    opt = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if os.path.exists(args.save_path):
        print(f"loading from {args.save_path}...")
        state = torch.load(args.save_path, map_location="cpu")
        model.load_state_dict(state["model"])
        start_step = state["step"]
    else:
        start_step = 0

    train_iter = iterate_data(args.batch_size, args.dataset_dir, train=True)
    test_iter = iterate_data(args.batch_size, args.dataset_dir, train=False)
    for i in range(start_step, 2**40):
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

        if (i + 1) % args.save_interval == 0:
            print(f"saving to {args.save_path}...")
            torch.save(
                dict(
                    step=i + 1,
                    model=model.state_dict(),
                ),
                args.save_path,
            )


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(-1) == targets).float().mean().item()


def iterate_data(batch_size: int, dir_path: str, train: bool) -> Iterator[torch.Tensor]:
    def file_checker(path: str) -> bool:
        hash = hashlib.md5(path.encode("utf-8")).digest()
        # 1/16 of the data is in the validation set.
        if train:
            return hash[0] >= 0x10
        else:
            return hash[0] < 0x10

    xf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(dir_path, is_valid_file=file_checker, transform=xf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    while True:
        yield from loader


if __name__ == "__main__":
    main()
