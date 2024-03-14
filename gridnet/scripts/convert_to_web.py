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
    parser.add_argument("--precision", type=int, default=32)
    args = parser.parse_args()

    assert args.precision in {16, 32}

    device = torch.device("cpu")

    with open(args.input_path, "rb") as f:
        state = torch.load(f, map_location=device)["model"]

    metadata = dict(
        params=[(k, v.shape) for k, v in state.items()],
        precision=args.precision,
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
            float_data = struct.pack(f"<{len(data)}f", *data)
            if args.precision == 32:
                f.write(float_data)
            else:
                # Convert float32 to float16
                words = struct.unpack(f"<{len(data)}I", float_data)
                # float32: sign(1) + exp(8) + frac(23)
                # float16: sign(1) + exp(5) + frac(10)
                signs = [(x >> 31) & 1 for x in words]
                exps = [((x >> 23) & 0b11111111) - 127 for x in words]
                fracs = [x & ((1 << 23) - 1) for x in words]
                for i, x in enumerate(exps.copy()):
                    if x < -15:
                        exps[i] = -15
                    elif x > 15:
                        assert f"number too large {data[i]}"
                float16s = [
                    (sign << 15) | (((exp + 15) & 0b11111) << 10) | (frac >> 13)
                    for sign, exp, frac in zip(signs, exps, fracs)
                ]
                data = struct.pack(f"<{len(data)}H", *float16s)
                f.write(data)


if __name__ == "__main__":
    main()
