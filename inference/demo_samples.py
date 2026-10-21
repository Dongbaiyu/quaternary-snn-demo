from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qsn_model.constants import CIFAR10_CLASSES, PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class DemoSample:
    index: int
    true_label: str
    expected_prediction: str
    verified: bool

    @property
    def display_name(self) -> str:
        return (
            f"Index {self.index} — {self.true_label} → "
            f"{self.expected_prediction}"
        )


def load_demo_samples(
    path: str | Path = PROJECT_ROOT / "configs" / "demo_samples.json",
) -> list[DemoSample]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return []
    rows = payload.get("samples", []) if isinstance(payload, dict) else []
    available = []
    for row in rows:
        try:
            sample = DemoSample(
                index=int(row["index"]),
                true_label=str(row["true_label"]),
                expected_prediction=str(row["expected_prediction"]),
                verified=row["verified"] is True,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            0 <= sample.index < 10000
            and sample.true_label in CIFAR10_CLASSES
            and sample.expected_prediction in CIFAR10_CLASSES
            and sample.verified
        ):
            available.append(sample)
    return available
