import argparse
import os
import re
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="plot.png")
    parser.add_argument("--smoothing", type=float, default=0.99)
    parser.add_argument("logs", type=str, nargs="+")
    args = parser.parse_args()

    datas = [read_log(path) for path in args.logs]
    for data in datas:
        plt.plot(
            data.steps,
            smooth_curve(data.train_loss, rate=args.smoothing),
            label=data.name,
        )
    plt.xlabel("step")
    plt.ylabel("train_loss")
    plt.legend()
    plt.savefig(args.output)


@dataclass
class LogData:
    name: str
    steps: List[int]
    train_loss: List[float]
    test_loss: List[float]
    train_acc: List[float]
    test_acc: List[float]


def read_log(log_path: str) -> LogData:
    data = LogData(
        name=os.path.splitext(os.path.basename(log_path))[0],
        steps=[],
        train_loss=[],
        test_loss=[],
        train_acc=[],
        test_acc=[],
    )
    expr = re.compile(
        ".*step ([0-9]*): test_loss=([0-9\\.]*) test_acc=([0-9\\.]*)"
        " train_loss=([0-9\\.]*) train_acc=([0-9\\.]*).*"
    )
    with open(log_path, "r") as f:
        for line in f:
            results = expr.match(line)
            if results:
                data.steps.append(int(results[1]))
                data.test_loss.append(float(results[2]))
                data.test_acc.append(float(results[3]))
                data.train_loss.append(float(results[4]))
                data.train_acc.append(float(results[5]))
    if not len(data.steps):
        raise RuntimeError(f"no lines found in file: {log_path}")
    return data


def smooth_curve(curve: List[float], rate: float = 0.99) -> List[float]:
    avg = curve[0]
    results = []
    for x in curve:
        avg *= rate
        avg += (1 - rate) * x
        results.append(avg)
    return results


if __name__ == "__main__":
    main()
