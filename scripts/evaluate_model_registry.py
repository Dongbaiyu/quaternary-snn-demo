from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from inference.model_registry import MODEL_REGISTRY, ModelSpec, sha256_file
from qsn_model.constants import CIFAR10_CLASSES, SPIKE_VALUES, Config
from qsn_model.dataset import get_cifar10_testset
from qsn_model.model import SResNet20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every registered Time-step checkpoint on CIFAR-10"
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        nargs="+",
        default=sorted(MODEL_REGISTRY),
        choices=sorted(MODEL_REGISTRY),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="")
    return parser.parse_args()


def extract_state_dict(payload) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a state_dict mapping")
    for key in ("state_dict", "model_state_dict"):
        if key in payload:
            payload = payload[key]
            break
    if not isinstance(payload, dict) or not payload:
        raise TypeError("checkpoint state_dict is empty or invalid")
    if all(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return payload


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_confusion_matrix(matrix: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8), dpi=160)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(CIFAR10_CLASSES)),
        yticks=np.arange(len(CIFAR10_CLASSES)),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=7,
            )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def previous_metadata(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def evaluate_one(
    spec: ModelSpec,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    checkpoint_hash = sha256_file(spec.checkpoint)
    old_metadata = previous_metadata(spec.metadata_path)
    print(
        f"[T={spec.time_steps}] loading {spec.checkpoint.name} "
        f"tau={spec.tau} sha256={checkpoint_hash}",
        flush=True,
    )
    model = SResNet20(time_steps=spec.time_steps, tau=spec.tau).to(device)
    payload = torch.load(spec.checkpoint, map_location=device, weights_only=True)
    state_dict = extract_state_dict(payload)
    strict_result = model.load_state_dict(state_dict, strict=True)
    model.eval()
    missing_keys = list(strict_result.missing_keys)
    unexpected_keys = list(strict_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"strict load failed: missing={missing_keys}, unexpected={unexpected_keys}"
        )

    matrix = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.int64)
    correct = 0
    total = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            outputs = model(inputs.to(device, non_blocking=True), time_steps=spec.time_steps)
            predictions = outputs.argmax(dim=1).cpu()
            targets_cpu = targets.cpu()
            correct += int(predictions.eq(targets_cpu).sum().item())
            total += int(targets_cpu.numel())
            np.add.at(matrix, (targets_cpu.numpy(), predictions.numpy()), 1)
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"[T={spec.time_steps}] {total}/{len(loader.dataset)} images",
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if total != 10_000 or int(matrix.sum()) != total:
        raise RuntimeError(f"expected 10000 test samples, evaluated {total}")

    accuracy = 100.0 * correct / total
    save_confusion_matrix(
        matrix,
        spec.confusion_matrix,
        f"SResNet20 Quaternary T={spec.time_steps} "
        f"(Verified Test Accuracy: {accuracy:.2f}%)",
    )
    existing_hash = (
        sha256_file(spec.existing_confusion_matrix)
        if spec.existing_confusion_matrix.is_file()
        else None
    )
    metadata = {
        "model": "SResNet20",
        "spike_type": "quaternary",
        "spike_values": list(SPIKE_VALUES),
        "time_steps": spec.time_steps,
        "tau": spec.tau,
        "dataset": "CIFAR-10",
        "test_accuracy": accuracy,
        "correct_count": correct,
        "test_count": total,
        "epoch": spec.epoch,
        "training_date": old_metadata.get("training_date"),
        "quantization": "FP32",
        "source": spec.source,
        "checkpoint_sha256": checkpoint_hash,
        "verified_confusion_matrix": str(
            spec.confusion_matrix.relative_to(PROJECT_ROOT)
        ),
    }
    write_json_atomic(spec.metadata_path, metadata)
    result = {
        **metadata,
        "checkpoint": str(spec.checkpoint.relative_to(PROJECT_ROOT)),
        "evaluation_seconds": elapsed,
        "strict_load": {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "existing_confusion_matrix": {
            "path": str(spec.existing_confusion_matrix.relative_to(PROJECT_ROOT)),
            "sha256": existing_hash,
            "comparison": (
                "source raster has no recoverable numeric matrix; regenerated "
                "from this evaluation's 10000 predictions"
            ),
        },
        "verified_confusion_matrix_sha256": sha256_file(spec.confusion_matrix),
        "previous_test_accuracy": old_metadata.get("test_accuracy"),
        "accuracy_difference": (
            accuracy - float(old_metadata["test_accuracy"])
            if "test_accuracy" in old_metadata
            else None
        ),
    }
    print(
        f"[T={spec.time_steps}] accuracy={accuracy:.2f}% "
        f"correct={correct}/{total} elapsed={elapsed:.3f}s",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    args = parse_args()
    config = Config(device=args.device, data_dir=args.data_dir)
    device = config.resolved_device()
    dataset = get_cifar10_testset(
        config.resolved_data_dir(), download=False, transform=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"dataset={config.resolved_data_dir()} samples={len(dataset)} "
        f"device={device} batch_size={args.batch_size}",
        flush=True,
    )
    results = [
        evaluate_one(MODEL_REGISTRY[time_steps], loader, device)
        for time_steps in args.time_steps
    ]
    output = PROJECT_ROOT / "results" / "model_registry_evaluation.json"
    write_json_atomic(output, {"models": results})
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
