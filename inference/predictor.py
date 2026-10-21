from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from qsn_model.constants import (
    CIFAR10_CLASSES,
    PROJECT_ROOT,
    SPIKE_VALUES,
    Config,
)
from qsn_model.dataset import get_cifar10_testset, test_transform
from qsn_model.model import SResNet20, normalize_time_steps

from .model_registry import get_model_spec, sha256_file
from .result_types import PredictionResult, TopClass


REQUIRED_METADATA = {
    "model",
    "spike_type",
    "spike_values",
    "time_steps",
    "tau",
    "dataset",
    "test_accuracy",
    "correct_count",
    "test_count",
    "epoch",
    "quantization",
    "source",
    "checkpoint_sha256",
}


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def display_name(self) -> str:
        return (
            f"{self.metadata['model']} | T={self.metadata['time_steps']} | "
            f"Acc={self.metadata['test_accuracy']:.2f}%"
        )


def metadata_path_for(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_suffix(".json")


def validate_metadata(metadata: dict[str, Any]) -> None:
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"metadata missing fields: {', '.join(sorted(missing))}")
    if metadata["model"] != "SResNet20":
        raise ValueError("only SResNet20 checkpoints are supported")
    if metadata["spike_type"] != "quaternary":
        raise ValueError("checkpoint is not quaternary")
    if tuple(metadata["spike_values"]) != SPIKE_VALUES:
        raise ValueError("checkpoint spike_values must be [-1, 0, 1, 2]")
    if metadata["dataset"] != "CIFAR-10":
        raise ValueError("checkpoint dataset must be CIFAR-10")
    training_time_steps = normalize_time_steps(metadata["time_steps"])
    spec = get_model_spec(training_time_steps)
    if metadata["source"] != spec.source:
        raise ValueError(
            f"metadata source must be {spec.source} for T={training_time_steps}"
        )
    if metadata["quantization"] not in {"FP32", "INT8"}:
        raise ValueError("checkpoint quantization must be FP32 or INT8")
    if float(metadata["tau"]) != spec.tau:
        raise ValueError(
            f"metadata tau must be {spec.tau} for T={training_time_steps}"
        )


def scan_checkpoints(directory: str | Path = PROJECT_ROOT / "checkpoints") -> list[CheckpointInfo]:
    available = []
    for path in sorted(Path(directory).glob("*.pth")):
        metadata_path = metadata_path_for(path)
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_metadata(metadata)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        available.append(CheckpointInfo(path.resolve(), metadata_path.resolve(), metadata))
    return available


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
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


class Predictor:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.device = self.config.resolved_device()
        self.model: SResNet20 | None = None
        self.checkpoint: CheckpointInfo | None = None
        self._testset = None

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        metadata_path = metadata_path_for(path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_metadata(metadata)
        expected_hash = str(metadata["checkpoint_sha256"]).lower()
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        model = SResNet20(
            time_steps=int(metadata["time_steps"]), tau=float(metadata["tau"])
        ).to(self.device)
        payload = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(_extract_state_dict(payload), strict=True)
        model.eval()
        self.model = model
        self.checkpoint = CheckpointInfo(path, metadata_path, metadata)

    def _ensure_loaded(self) -> tuple[SResNet20, CheckpointInfo]:
        if self.model is None or self.checkpoint is None:
            raise RuntimeError("load_checkpoint() must be called before inference")
        return self.model, self.checkpoint

    def predict_image(self, image, time_steps: int | None = None) -> PredictionResult:
        model, checkpoint = self._ensure_loaded()
        current_time_steps = (
            normalize_time_steps(checkpoint.metadata["time_steps"])
            if time_steps is None
            else normalize_time_steps(time_steps)
        )
        if isinstance(image, Image.Image):
            tensor = test_transform()(image.convert("RGB"))
        elif torch.is_tensor(image):
            tensor = image.detach()
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError("image tensor must have shape [C,H,W]")
        else:
            raise TypeError("image must be a PIL image or torch tensor")
        tensor = tensor.unsqueeze(0).to(self.device)
        model.reset_spike_statistics()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor, time_steps=current_time_steps)
            probabilities = torch.softmax(logits, dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        probabilities = probabilities.detach().cpu()
        top_scores, top_indices = torch.topk(probabilities, k=3)
        top3 = tuple(
            TopClass(CIFAR10_CLASSES[int(index)], float(score))
            for score, index in zip(top_scores, top_indices)
        )
        return PredictionResult(
            predicted_class=top3[0].class_name,
            top3_classes=top3,
            class_scores={
                name: float(probabilities[index]) for index, name in enumerate(CIFAR10_CLASSES)
            },
            time_steps=current_time_steps,
            inference_time=elapsed,
            checkpoint_name=checkpoint.path.name,
            model_name=str(checkpoint.metadata["model"]),
            spike_type=str(checkpoint.metadata["spike_type"]),
            spike_statistics=model.spike_statistics(),
            training_time_steps=normalize_time_steps(checkpoint.metadata["time_steps"]),
            tau=float(checkpoint.metadata["tau"]),
            source=str(checkpoint.metadata["source"]),
            checkpoint_sha256=str(checkpoint.metadata["checkpoint_sha256"]),
        )

    def predict_cifar_index(
        self, index: int, time_steps: int | None = None
    ) -> PredictionResult:
        if self._testset is None:
            self._testset = get_cifar10_testset(
                self.config.resolved_data_dir(), download=False, transform=True
            )
        if not 0 <= index < len(self._testset):
            raise IndexError(f"CIFAR-10 test index must be in [0, {len(self._testset) - 1}]")
        image, _ = self._testset[index]
        return self.predict_image(image, time_steps=time_steps)
