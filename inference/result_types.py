from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TopClass:
    class_name: str
    score: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    predicted_class: str
    top3_classes: tuple[TopClass, ...]
    class_scores: dict[str, float]
    time_steps: int
    inference_time: float
    checkpoint_name: str
    model_name: str
    spike_type: str
    spike_statistics: dict[int, int]
    training_time_steps: int = 4
    tau: float = 0.875
    source: str = "qua_v10.py"
    checkpoint_sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
