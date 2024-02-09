"""
Convert a checkpoint from train_imagenet.py to a checkpoint for loading in the
web backend.
"""

import argparse
import json
import struct

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--activation", type=str, default="leaky_relu")
    parser.add_argument("--emb_channels", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--inner_iters", type=int, default=8)
    parser.add_argument("--outer_iters", type=int, default=12)
    parser.add_argument("--outer_residual", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")

    with open(args.input_path, "rb") as f:
        state = torch.load(f, map_location=device)["model"]

    metadata = dict(
        params=[(k, v.shape) for k, v in state.items()],
        config={
            "activation": args.activation,
            "innerIters": args.inner_iters,
            "outerIters": args.outer_iters,
            "outerResidual": args.outer_residual,
        },
    )
    metadata_bytes = bytes(json.dumps(metadata), "utf-8")

    with open(args.output_path, "wb") as f:
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)
        for v in state.values():
            data = v.reshape(-1).float().tolist()
            f.write(struct.pack(f"<{len(data)}f", *data))


if __name__ == "__main__":
    main()
