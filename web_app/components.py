from __future__ import annotations

import base64
import csv
import html
import io
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import MutableMapping

from PIL import Image, ImageOps

from inference.demo_samples import load_demo_samples
from inference.model_registry import MODEL_REGISTRY, registry_asset_errors, sha256_file
from inference.predictor import Predictor, validate_metadata
from qsn_model.model import normalize_time_steps
from inference.result_types import PredictionResult
from qsn_model.constants import PROJECT_ROOT, Config


DEFAULT_MODEL_SPEC = MODEL_REGISTRY[4]
CHECKPOINT_PATH = DEFAULT_MODEL_SPEC.checkpoint
METADATA_PATH = DEFAULT_MODEL_SPEC.metadata_path
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_samples.json"
SAMPLE_ASSET_DIR = PROJECT_ROOT / "web_app" / "assets" / "demo_samples"
TRAINING_CURVE_PATH = PROJECT_ROOT / "results" / "quaternary_result_v10.png"
CONFUSION_MATRIX_PATH = DEFAULT_MODEL_SPEC.confusion_matrix
HISTORY_FIELDS = (
    "timestamp",
    "source",
    "image_index",
    "true_label",
    "prediction",
    "correct",
    "top3",
    "checkpoint",
    "checkpoint_sha256",
    "training_time_steps",
    "time_steps",
    "tau",
    "inference_time_seconds",
)


def is_verified_time_steps(current_time_steps, checkpoint_time_steps) -> bool:
    return normalize_time_steps(current_time_steps) == normalize_time_steps(
        checkpoint_time_steps
    )


def inference_result_key(
    image_identifier: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_time_steps,
    inference_time_steps,
    tau: float,
) -> str:
    return (
        f"{image_identifier}|{Path(checkpoint_path).resolve()}|"
        f"sha256={checkpoint_sha256}|"
        f"training_t={normalize_time_steps(training_time_steps)}|"
        f"inference_t={normalize_time_steps(inference_time_steps)}|tau={float(tau):.12g}"
    )


def implementation_source_label(source: str) -> str:
    normalized = str(source).strip()
    return normalized if normalized.endswith(".py") else f"{normalized}.py"

@dataclass(frozen=True, slots=True)
class WebDemoSample:
    index: int
    true_label: str
    expected_prediction: str
    verified: bool
    image_path: Path

    @property
    def display_name(self) -> str:
        return f"案例 {self.index} — {self.true_label} → {self.expected_prediction}"


@dataclass(slots=True)
class LockedPredictor:
    """One cached Predictor protected across concurrent Streamlit sessions."""

    predictor: Predictor
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def device_label(self) -> str:
        return str(self.predictor.device).upper()

    def predict_image(
        self, image: Image.Image, time_steps: int | None = None
    ) -> PredictionResult:
        normalized_time_steps = None if time_steps is None else normalize_time_steps(time_steps)
        with self.lock:
            return self.predictor.predict_image(image, time_steps=normalized_time_steps)


def create_cpu_service(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> LockedPredictor:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    predictor = Predictor(Config(device="cpu", mode="inference"))
    predictor.load_checkpoint(str(path))
    return LockedPredictor(predictor)


def load_metadata(path: str | Path = METADATA_PATH) -> dict:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(payload)
    return payload


def sample_image_path(index: int, label: str, asset_dir: str | Path = SAMPLE_ASSET_DIR) -> Path:
    return Path(asset_dir) / f"index_{index:04d}_{label}.png"


def load_web_samples(
    config_path: str | Path = SAMPLE_CONFIG_PATH,
    asset_dir: str | Path = SAMPLE_ASSET_DIR,
) -> list[WebDemoSample]:
    samples = load_demo_samples(config_path)
    return [
        WebDemoSample(
            index=sample.index,
            true_label=sample.true_label,
            expected_prediction=sample.expected_prediction,
            verified=sample.verified,
            image_path=sample_image_path(sample.index, sample.true_label, asset_dir),
        )
        for sample in samples
    ]


def validate_web_assets(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    metadata_path: str | Path = METADATA_PATH,
    config_path: str | Path = SAMPLE_CONFIG_PATH,
    asset_dir: str | Path = SAMPLE_ASSET_DIR,
) -> list[str]:
    errors: list[str] = registry_asset_errors(require_verified_outputs=True)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        errors.append(f"Checkpoint missing: {checkpoint}")
    try:
        load_metadata(metadata_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Metadata error: {exc}")
    samples = load_web_samples(config_path, asset_dir)
    if len(samples) != 10:
        errors.append(f"Demo samples error: expected 10, found {len(samples)}")
    for sample in samples:
        if not sample.image_path.is_file():
            errors.append(f"Demo image missing: {sample.image_path}")
    return errors


def load_sample_image(sample: WebDemoSample) -> Image.Image:
    if not sample.image_path.is_file():
        raise FileNotFoundError(f"demo image not found: {sample.image_path}")
    with Image.open(sample.image_path) as image:
        return image.convert("RGB").copy()


def prepare_uploaded_image(image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"),
        (32, 32),
        method=Image.Resampling.BILINEAR,
    )


def image_data_uri(image: Image.Image) -> str:
    """Encode a small image inline so cloud rendering does not depend on media URLs."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def pixel_image_html(image: Image.Image, alt_text: str) -> str:
    resized = image.resize((320, 320), Image.Resampling.NEAREST)
    return (
        '<div class="pixel-image">'
        f'<img src="{image_data_uri(resized)}" alt="{html.escape(alt_text)}">'
        "</div>"
    )


def report_image_html(image: Image.Image, alt_text: str) -> str:
    """Render a report image inline without Streamlit's temporary media endpoint."""
    return (
        '<div class="report-image">'
        f'<img src="{image_data_uri(image)}" alt="{html.escape(alt_text)}">'
        "</div>"
    )


def ensure_session_history(state: MutableMapping) -> list[dict]:
    if "history" not in state:
        state["history"] = []
    return state["history"]


def append_session_record(state: MutableMapping, record: dict) -> None:
    ensure_session_history(state).append(dict(record))


def prediction_record(
    result: PredictionResult,
    source: str,
    true_label: str | None,
    image_index: int | None,
    timestamp: str,
) -> dict:
    correct = (
        result.predicted_class == true_label if true_label is not None else None
    )
    return {
        "timestamp": timestamp,
        "source": source,
        "image_index": "" if image_index is None else image_index,
        "true_label": true_label or "unknown",
        "prediction": result.predicted_class,
        "correct": "" if correct is None else correct,
        "top3": "; ".join(item.class_name for item in result.top3_classes),
        "checkpoint": result.checkpoint_name,
        "checkpoint_sha256": result.checkpoint_sha256,
        "training_time_steps": result.training_time_steps,
        "time_steps": result.time_steps,
        "tau": result.tau,
        "inference_time_seconds": f"{result.inference_time:.6f}",
    }


def history_csv_bytes(records: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8-sig")


def metric_card_html(label: str, value: str, accent: bool = False) -> str:
    class_name = "metric-card accent" if accent else "metric-card"
    return (
        f'<div class="{class_name}"><div class="metric-label">'
        f'{html.escape(label)}</div><div class="metric-value">'
        f"{html.escape(value)}</div></div>"
    )


def hero_html(time_steps: int, tau: float, accuracy: float, device: str) -> str:
    """Formal light dashboard hero with no institutional branding."""
    return (
        '<section class="editorial-hero">'
        '<div class="hero-layout"><div class="hero-copy">'
        '<div class="eyebrow">系統總覽</div>'
        '<h1>四元脈衝神經網路<br><em>推論與視覺化系統</em></h1>'
        '<div class="hero-tags">'
        f'<span>SResNet20</span><span>T={int(time_steps)}</span>'
        f'<span>TAU={float(tau):g}</span><span>{html.escape(device)} · FP32</span>'
        '</div></div>'
        '<div class="hero-evidence">'
        '<div class="signal-map" aria-hidden="true">'
        '<span class="signal-ring r1"></span><span class="signal-ring r2"></span>'
        '<span class="signal-axis x"></span><span class="signal-axis y"></span>'
        '<b class="pulse p1">-1</b><b class="pulse p2">0</b>'
        '<b class="pulse p3">1</b><b class="pulse p4">2</b>'
        '<div class="signal-core"><small>脈衝值</small><strong>{-1, 0, 1, 2}</strong></div>'
        '</div>'
        '<div class="hero-facts">'
        f'<div class="featured"><small>最佳測試準確率</small><strong>{accuracy:.2f}%</strong></div>'
        '<div><small>資料集</small><strong>CIFAR-10</strong></div>'
        '<div><small>模型架構</small><strong>SResNet20</strong></div>'
        '<div><small>輸出類別</small><strong>10 類</strong></div>'
        '</div></div></div>'
        '</section>'
    )


def inference_empty_html() -> str:
    return (
        '<div class="result-placeholder">'
        '<div class="placeholder-head"><span>OUTPUT PREVIEW</span><b>WAITING</b></div>'
        '<div class="placeholder-target" aria-hidden="true">'
        '<i></i><i></i><i></i><i></i><strong>10</strong><small>CLASSES</small></div>'
        '<h3>準備進行單張推論</h3>'
        '<p>完成推論後，此區會同步呈現 Top-1、Top-3、'
        'Softmax 機率與四元脈衝分布。</p>'
        '<div class="placeholder-grid"><span>TOP-1</span><span>TOP-3</span>'
        '<span>SOFTMAX</span><span>SPIKES</span></div></div>'
    )


def section_header_html(kicker: str, title: str, description: str = "") -> str:
    description_html = (
        f'<p>{html.escape(description)}</p>' if description else ""
    )
    return (
        '<div class="section-heading">'
        f'<span>{html.escape(kicker)}</span><h2>{html.escape(title)}</h2>'
        f'{description_html}</div>'
    )


def accuracy_strip_html(selected_time_steps: int) -> str:
    values = ((1, 91.18), (2, 92.04), (4, 92.03), (6, 92.30))
    cards = []
    for time_steps, accuracy in values:
        selected = " selected" if time_steps == int(selected_time_steps) else ""
        cards.append(
            f'<div class="accuracy-item{selected}"><span>T={time_steps}</span>'
            f'<strong>{accuracy:.2f}%</strong><small>最佳測試準確率</small></div>'
        )
    return '<div class="accuracy-strip">' + "".join(cards) + "</div>"


def recorded_results_table_html() -> str:
    rows = [
        ("三元脈衝（文獻）", 6, None, 92.07, False),
        ("四元脈衝", 6, 99.84, 92.30, True),
        ("", 4, 99.84, 92.03, True),
        ("", 2, 99.57, 92.04, True),
        ("", 1, 98.96, 91.18, False),
    ]
    body = []
    for model_label, time_steps, train_accuracy, test_accuracy, accent in rows:
        test_class = "recorded-test accent" if accent else "recorded-test"
        train_text = "—" if train_accuracy is None else f"{train_accuracy:.2f}%"
        body.append(
            "<tr>"
            f"<td>{html.escape(model_label)}</td>"
            f"<td>T={time_steps}</td>"
            f"<td>{train_text}</td>"
            f'<td class="{test_class}">{test_accuracy:.2f}%</td>'
            "</tr>"
        )
    return (
        '<div class="recorded-results-wrap"><table class="recorded-results">'
        "<thead><tr><th>模型</th><th>時間步</th><th>訓練準確率</th>"
        "<th>測試準確率</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def pipeline_html() -> str:
    stages = (
        ("01", "CIFAR-10", "32×32 RGB"),
        ("02", "正規化", "無降採樣"),
        ("03", "時間步輸入", "重複輸入 T 次"),
        ("04", "SResNet20", "三個殘差階段"),
        ("05", "四元 LIF", "{-1, 0, 1, 2}"),
        ("06", "平均輸出分數", "FC 128→10"),
        ("07", "Softmax", "第一名／前三名"),
    )
    cards = []
    for number, title, detail in stages:
        cards.append(
            '<div class="pipeline-stage">'
            f'<span>{number}</span><strong>{html.escape(title)}</strong>'
            f'<small>{html.escape(detail)}</small></div>'
        )
    return '<div class="pipeline">' + '<span class="arrow">→</span>'.join(cards) + "</div>"


def probability_bars_html(scores: dict[str, float]) -> str:
    rows = []
    for class_name, score in scores.items():
        percentage = max(0.0, min(100.0, float(score) * 100.0))
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-name">{html.escape(class_name)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill probability" style="width:{percentage:.4f}%"></div>'
            "</div>"
            f'<div class="bar-value">{percentage:.2f}%</div>'
            "</div>"
        )
    return '<div class="chart-title">分類機率</div>' + "".join(rows)


def spike_bars_html(counts: dict[int, int]) -> str:
    colors = {-1: "negative", 0: "zero", 1: "positive-one", 2: "positive-two"}
    total = sum(max(0, int(counts.get(value, 0))) for value in (-1, 0, 1, 2))
    rows = []
    for value in (-1, 0, 1, 2):
        count = max(0, int(counts.get(value, 0)))
        percentage = 100.0 * count / total if total else 0.0
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-name">脈衝 {value}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill {colors[value]}" style="width:{percentage:.4f}%"></div>'
            "</div>"
            f'<div class="bar-value spike-value"><span>{count:,}</span>'
            f'<small>{percentage:.2f}%</small></div>'
            "</div>"
        )
    return '<div class="chart-title">四元脈衝分布</div>' + "".join(rows)
