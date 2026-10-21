from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsn_model.constants import Config
from qsn_model.dataset import get_dataloaders
from qsn_model.model import SResNet20
from qsn_model.neurons import QuaternaryLIF


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    correct = total = 0
    fire_rate_sum = 0.0
    spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
    started = time.perf_counter()
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        fire_rates = []
        for module in model.modules():
            if isinstance(module, QuaternaryLIF):
                fire_rates.append(module.fire_rate)
                for value in spike_counts:
                    spike_counts[value] += module.batch_spike_counts[value]
        fire_rate_sum += sum(fire_rates) / len(fire_rates) if fire_rates else 0.0
        if step % 200 == 0:
            print(
                f"[Quaternary Model] Step [{step}], Loss: {loss.item():.4f}, "
                f"Acc: {100 * correct / total:.1f}%, LR: {scheduler.get_last_lr()[0]:.5f}"
            )
    accuracy = 100 * correct / total
    avg_fire_rate = 100 * fire_rate_sum / len(loader)
    print(
        f"[Quaternary Model] Epoch {epoch} Time: {time.perf_counter()-started:.2f}s, "
        f"Final Train Acc: {accuracy:.2f}%, Avg Fire: {avg_fire_rate:.2f}%"
    )
    return accuracy, avg_fire_rate, spike_counts


def evaluate_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total += targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
    return 100 * correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train the qua_v10 SResNet20 model")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.875)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        tau=args.tau,
        device=args.device,
        checkpoint_path=args.checkpoint,
        mode="train",
        data_dir=args.data_dir,
    )
    device = config.resolved_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    trainloader, testloader = get_dataloaders(config, download=not args.no_download)
    model = SResNet20(config.time_steps, config.tau).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(trainloader),
        epochs=config.num_epochs,
        pct_start=0.15,
    )
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else (
        PROJECT_ROOT / "checkpoints" / f"qsn_sresnet20_t{config.time_steps}_best.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, config.num_epochs + 1):
        train_one_epoch(model, trainloader, optimizer, criterion, scheduler, device, epoch)
        accuracy = evaluate_model(model, testloader, device)
        print(f"Epoch {epoch}: Test Acc: {accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
    metadata = {
        "model": "SResNet20",
        "spike_type": "quaternary",
        "spike_values": [-1, 0, 1, 2],
        "time_steps": config.time_steps,
        "tau": config.tau,
        "dataset": "CIFAR-10",
        "test_accuracy": round(best_accuracy, 4),
        "epoch": best_epoch,
        "training_date": time.strftime("%Y-%m-%d"),
        "quantization": "FP32",
        "source": "qua_v10",
    }
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved best checkpoint: {checkpoint_path} ({best_accuracy:.2f}%)")


if __name__ == "__main__":
    main()
