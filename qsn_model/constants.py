from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
SPIKE_VALUES = (-1, 0, 1, 2)


@dataclass(frozen=True, slots=True)
class DemoSettings:
    """Display-only settings; algorithm parameters deliberately do not live here."""

    default_checkpoint: str = "checkpoints/qsn_sresnet20_t4_best.pth"
    default_image_index: int = 0
    demo_mode: str = "showcase"
    result_directory: str = "results"
    window_width: int = 1366
    window_height: int = 768

    def result_path(self) -> Path:
        path = Path(self.result_directory).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_demo_settings(
    path: str | Path = PROJECT_ROOT / "configs" / "default.json",
) -> DemoSettings:
    """Load only presentation settings and fall back safely when the file is absent."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return DemoSettings()
    defaults = DemoSettings()
    values = {
        field: payload.get(field, getattr(defaults, field))
        for field in DemoSettings.__dataclass_fields__
    }
    try:
        settings = DemoSettings(
            default_checkpoint=str(values["default_checkpoint"]),
            default_image_index=max(0, min(9999, int(values["default_image_index"]))),
            demo_mode=str(values["demo_mode"]),
            result_directory=str(values["result_directory"]),
            window_width=max(1100, int(values["window_width"])),
            window_height=max(700, int(values["window_height"])),
        )
    except (TypeError, ValueError):
        return defaults
    if settings.demo_mode not in {"showcase", "research"}:
        return DemoSettings(
            default_checkpoint=settings.default_checkpoint,
            default_image_index=settings.default_image_index,
            result_directory=settings.result_directory,
            window_width=settings.window_width,
            window_height=settings.window_height,
        )
    return settings


@dataclass(slots=True)
class Config:
    """Runtime settings; defaults preserve qua_v10.py behavior."""

    time_steps: int = 4
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 3e-3
    weight_decay: float = 5e-4
    tau: float = 0.875
    device: str = "auto"
    checkpoint_path: str = ""
    mode: str = "inference"
    data_dir: str = ""
    num_workers: int = 2

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def resolved_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir).expanduser().resolve()
        candidates = (PROJECT_ROOT / "data", PROJECT_ROOT.parent / "Demo" / "data")
        return next((path for path in candidates if path.exists()), candidates[0])

    def to_dict(self) -> dict:
        return asdict(self)
