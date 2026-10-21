from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from qsn_model.constants import PROJECT_ROOT
from qsn_model.model import normalize_time_steps


@dataclass(frozen=True, slots=True)
class ModelSpec:
    time_steps: int
    tau: float
    checkpoint: Path
    source: str
    confusion_matrix: Path
    existing_confusion_matrix: Path
    training_result: Path
    recorded_train_accuracy: float
    recorded_test_accuracy: float
    epoch: int = 100

    @property
    def metadata_path(self) -> Path:
        return self.checkpoint.with_suffix(".json")


VERIFIED_RESULTS_DIR = PROJECT_ROOT / "results" / "verified_models"
ALL_TIME_STEP_RESULTS_DIR = PROJECT_ROOT / "results" / "all_time_step_results"

MODEL_REGISTRY: dict[int, ModelSpec] = {
    1: ModelSpec(
        time_steps=1,
        tau=0.875,
        checkpoint=PROJECT_ROOT
        / "original"
        / "quaternary_sresnet20_qua_t_1_best_weights.pth",
        source="qua_timestep_1.py",
        confusion_matrix=VERIFIED_RESULTS_DIR / "confusion_matrix_t1_verified.png",
        existing_confusion_matrix=PROJECT_ROOT
        / "original"
        / "confusion_matrix_qua_t_1.png",
        training_result=ALL_TIME_STEP_RESULTS_DIR / "quaternary_result_qua_t_1.png",
        recorded_train_accuracy=98.96,
        recorded_test_accuracy=91.18,
    ),
    2: ModelSpec(
        time_steps=2,
        tau=0.875,
        checkpoint=PROJECT_ROOT
        / "original"
        / "quaternary_sresnet20_qua_t_2_best_weights.pth",
        source="qua_timestep_2.py",
        confusion_matrix=VERIFIED_RESULTS_DIR / "confusion_matrix_t2_verified.png",
        existing_confusion_matrix=PROJECT_ROOT
        / "original"
        / "confusion_matrix_qua_t_2.png",
        training_result=ALL_TIME_STEP_RESULTS_DIR / "quaternary_result_qua_t_2.png",
        recorded_train_accuracy=99.57,
        recorded_test_accuracy=92.04,
    ),
    4: ModelSpec(
        time_steps=4,
        tau=0.875,
        checkpoint=PROJECT_ROOT / "checkpoints" / "qsn_sresnet20_t4_best.pth",
        source="qua_v10.py",
        confusion_matrix=VERIFIED_RESULTS_DIR / "confusion_matrix_t4_verified.png",
        existing_confusion_matrix=PROJECT_ROOT / "results" / "confusion_matrix_v10.png",
        training_result=ALL_TIME_STEP_RESULTS_DIR / "quaternary_result_qua_t_4.png",
        recorded_train_accuracy=99.84,
        recorded_test_accuracy=92.03,
    ),
    6: ModelSpec(
        time_steps=6,
        tau=0.9,
        checkpoint=PROJECT_ROOT
        / "original"
        / "quaternary_sresnet20_qua_t_6_best_weights.pth",
        source="qua_timestep_6.py",
        confusion_matrix=VERIFIED_RESULTS_DIR / "confusion_matrix_t6_verified.png",
        existing_confusion_matrix=PROJECT_ROOT
        / "original"
        / "confusion_matrix_qua_t_6.png",
        training_result=ALL_TIME_STEP_RESULTS_DIR / "quaternary_result_qua_t_6.png",
        recorded_train_accuracy=99.84,
        recorded_test_accuracy=92.30,
    ),
}


def get_model_spec(time_steps) -> ModelSpec:
    normalized = normalize_time_steps(time_steps)
    try:
        return MODEL_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(MODEL_REGISTRY))
        raise ValueError(
            f"unsupported Time-step {normalized}; expected one of {supported}"
        ) from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_asset_errors(require_verified_outputs: bool = True) -> list[str]:
    errors: list[str] = []
    for time_steps, spec in MODEL_REGISTRY.items():
        paths = [("checkpoint", spec.checkpoint)]
        if require_verified_outputs:
            paths.extend(
                (
                    ("metadata", spec.metadata_path),
                    ("confusion matrix", spec.confusion_matrix),
                    ("training result", spec.training_result),
                )
            )
        for label, path in paths:
            if not path.is_file():
                errors.append(f"T={time_steps} {label} missing: {path}")
    return errors
