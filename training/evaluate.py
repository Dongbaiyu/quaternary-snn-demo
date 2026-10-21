from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predictor import Predictor
from qsn_model.constants import Config
from qsn_model.dataset import get_cifar10_testset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained quaternary SNN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_path=args.checkpoint,
        mode="evaluate",
        data_dir=args.data_dir,
    )
    predictor = Predictor(config)
    predictor.load_checkpoint(args.checkpoint)
    dataset = get_cifar10_testset(config.resolved_data_dir(), download=False, transform=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    started = __import__("time").perf_counter()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = predictor.model(inputs.to(predictor.device))
            total += targets.size(0)
            correct += outputs.argmax(1).cpu().eq(targets).sum().item()
    elapsed = __import__("time").perf_counter() - started
    print(f"checkpoint: {Path(args.checkpoint).name}")
    print(f"device: {predictor.device}")
    print(f"accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
    print(f"elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
