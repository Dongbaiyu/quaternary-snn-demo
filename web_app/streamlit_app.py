from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_registry import MODEL_REGISTRY, ModelSpec, get_model_spec, sha256_file

from web_app.components import (
    CHECKPOINT_PATH,
    CONFUSION_MATRIX_PATH,
    METADATA_PATH,
    TRAINING_CURVE_PATH,
    LockedPredictor,
    WebDemoSample,
    append_session_record,
    create_cpu_service,
    ensure_session_history,
    accuracy_strip_html,
    hero_html,
    history_csv_bytes,
    inference_empty_html,
    load_metadata,
    implementation_source_label,
    inference_result_key,
    is_verified_time_steps,
    load_sample_image,
    load_web_samples,
    metric_card_html,
    normalize_time_steps,
    pipeline_html,
    pixel_image_html,
    prediction_record,
    prepare_uploaded_image,
    probability_bars_html,
    recorded_results_table_html,
    section_header_html,
    spike_bars_html,
    validate_web_assets,
)


PREDICTOR_CACHE_VERSION = "dedicated-model-registry-v1"
PAGES = ("系統總覽", "推論", "模型資訊", "執行紀錄")


def report_image_html(image: Image.Image, alt_text: str) -> str:
    """Embed report graphics without a cross-module hot-reload dependency."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<div class="report-image">'
        f'<img src="data:image/png;base64,{payload}" alt="{html.escape(alt_text)}">'
        "</div>"
    )


@st.cache_resource(show_spinner=False)
def cached_service(
    checkpoint_path: str,
    checkpoint_sha256: str,
    training_time_steps: int,
    tau: float,
    interface_version: str,
) -> LockedPredictor:
    del interface_version
    service = create_cpu_service(checkpoint_path)
    checkpoint = service.predictor.checkpoint
    if checkpoint is None:
        raise RuntimeError("cached Predictor has no checkpoint identity")
    metadata = checkpoint.metadata
    if (
        str(metadata["checkpoint_sha256"]) != checkpoint_sha256
        or normalize_time_steps(metadata["time_steps"]) != normalize_time_steps(training_time_steps)
        or float(metadata["tau"]) != float(tau)
    ):
        raise ValueError("cached Predictor identity does not match the model registry")
    return service


@st.cache_data(show_spinner=False)
def cached_metadata(path: str, modified_ns: int) -> dict:
    del modified_ns
    return load_metadata(path)


@st.cache_data(show_spinner=False)
def cached_samples(config_path: str, modified_ns: int) -> list[WebDemoSample]:
    del modified_ns
    return load_web_samples(config_path)


@st.cache_data(show_spinner=False)
def cached_image(path: str, modified_ns: int) -> Image.Image:
    del modified_ns
    sample_path = Path(path)
    with Image.open(sample_path) as image:
        return image.convert("RGB").copy()


def load_css() -> None:
    css_path = PROJECT_ROOT / "web_app" / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def initialize_state(samples: list[WebDemoSample]) -> None:
    ensure_session_history(st.session_state)
    defaults = {
        "page": "系統總覽",
        "selected_sample_index": samples[0].index if samples else 0,
        "last_result": None,
        "selected_time_steps": 4,
        "last_context": None,
        "auto_run": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_result() -> None:
    st.session_state.last_result = None
    st.session_state.last_context = None


def quick_demo(samples: list[WebDemoSample], checkpoint_time_steps: int) -> None:
    if samples:
        st.session_state.selected_sample_index = samples[0].index
    st.session_state.selected_time_steps = normalize_time_steps(checkpoint_time_steps)
    st.session_state.auto_run = True
    st.session_state.page = "推論"


def shift_sample(indices: list[int], delta: int) -> None:
    if not indices:
        return
    current = st.session_state.selected_sample_index
    position = indices.index(current) if current in indices else 0
    st.session_state.selected_sample_index = indices[(position + delta) % len(indices)]
    clear_result()


def render_overview(
    metadata: dict,
    service: LockedPredictor,
    samples: list[WebDemoSample],
    spec: ModelSpec,
) -> None:
    st.markdown(
        hero_html(
            spec.time_steps,
            spec.tau,
            spec.recorded_test_accuracy,
            service.device_label,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        section_header_html(
            "MODEL PROFILE",
            "實驗模型總覽",
            "專用 checkpoint、時間步與 TAU 形成一組不可拆分的模型設定。",
        ),
        unsafe_allow_html=True,
    )
    cards = (
        metric_card_html("Model", str(metadata["model"]))
        + metric_card_html("Dataset", str(metadata["dataset"]))
        + metric_card_html("Spike", "{-1, 0, 1, 2}")
        + metric_card_html("Checkpoint training T", f"T={metadata['time_steps']}", True)
        + metric_card_html("TAU", str(metadata["tau"]))
        + metric_card_html(
            "Best Test Accuracy", f"{spec.recorded_test_accuracy:.2f}%", True
        )
        + metric_card_html("Precision", str(metadata["quantization"]))
        + metric_card_html("Backend", "PyTorch Inference")
        + metric_card_html("Device", service.device_label)
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)
    st.markdown(accuracy_strip_html(spec.time_steps), unsafe_allow_html=True)
    st.markdown(
        section_header_html(
            "INFERENCE PIPELINE",
            "軟體推論資料流",
            "無獨立 Poisson、rate 或 latency encoder；正規化影像於 T 個時間步重複輸入。",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(pipeline_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="trust-note"><span>◈</span><div><strong>Inference-only deployment</strong>'
        '<p>使用預先訓練完成的 FP32 checkpoint；不重新訓練、'
        '不下載 CIFAR-10、不執行 INT8 conversion。</p></div></div>',
        unsafe_allow_html=True,
    )
    st.button(
        "開始快速展示",
        type="primary",
        use_container_width=True,
        on_click=quick_demo,
        args=(samples, int(metadata["time_steps"])),
    )


def execute_inference(
    service: LockedPredictor,
    image: Image.Image,
    true_label: str | None,
    image_index: int | None,
    source: str,
    time_steps: int,
) -> None:
    selected_time_steps = normalize_time_steps(time_steps)
    try:
        with st.spinner("CPU inference…"):
            result = service.predict_image(image, time_steps=selected_time_steps)
    except Exception as exc:
        st.error(f"推論失敗：{exc}")
        return
    result_time_steps = normalize_time_steps(result.time_steps)
    if result_time_steps != selected_time_steps:
        st.error("推論結果的 Time-step 與目前選擇不一致，結果已捨棄。")
        clear_result()
        return
    st.session_state.last_result = result
    st.session_state.last_context = {
        "true_label": true_label,
        "image_index": image_index,
        "source": source,
        "result_key": inference_result_key(
            source,
            service.predictor.checkpoint.path,
            result.checkpoint_sha256,
            result.training_time_steps,
            result_time_steps,
            result.tau,
        ),
    }
    append_session_record(
        st.session_state,
        prediction_record(
            result,
            source=source,
            true_label=true_label,
            image_index=image_index,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        ),
    )


def render_result(metadata: dict, expected_result_key: str) -> None:
    result = st.session_state.last_result
    context = st.session_state.last_context
    if result is None or context is None:
        st.markdown(inference_empty_html(), unsafe_allow_html=True)
        return
    true_label = context["true_label"]
    if context.get("result_key") != expected_result_key:
        st.info("圖片、checkpoint 或 Time-step 已變更，請重新執行推論。")
        return

    is_correct = result.predicted_class == true_label if true_label is not None else None
    correctness = (
        '<span class="correct">Correct ✓</span>'
        if is_correct is True
        else '<span class="incorrect">Incorrect ✕</span>'
        if is_correct is False
        else '<span class="small-muted">自訂圖片沒有 CIFAR-10 true label</span>'
    )
    top3 = "<br>".join(
        f"{rank}. {item.class_name} — {item.score * 100:.2f}%"
        for rank, item in enumerate(result.top3_classes, start=1)
    )
    st.markdown(
        '<div class="result-panel">'
        '<div class="small-muted">Top-1 Prediction</div>'
        f'<div class="top-one">{result.predicted_class}</div>'
        f"<div>{correctness}</div><hr>"
        f"<strong>Top-3</strong><br>{top3}<hr>"
        f"<strong>Inference Time:</strong> {result.inference_time * 1000:.2f} ms<br>"
        f"<strong>Checkpoint training T:</strong> {normalize_time_steps(metadata['time_steps'])}<br>"
        f"<strong>Current inference T:</strong> {normalize_time_steps(result.time_steps)}<br>"
        f"<strong>Checkpoint:</strong> {result.checkpoint_name}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="chart-panel">{probability_bars_html(result.class_scores)}</div>',
        unsafe_allow_html=True,
    )


def render_spike_result(expected_result_key: str) -> None:
    """Render the current inference's spike statistics below the source image."""
    result = st.session_state.last_result
    context = st.session_state.last_context
    if result is None or context is None:
        return
    if context.get("result_key") != expected_result_key:
        return
    st.markdown(
        f'<div class="chart-panel spike-panel">'
        f'{spike_bars_html(result.spike_statistics)}</div>',
        unsafe_allow_html=True,
    )


def render_inference(
    metadata: dict,
    service: LockedPredictor,
    samples: list[WebDemoSample],
    spec: ModelSpec,
) -> None:
    st.markdown(
        section_header_html(
            "LIVE INFERENCE",
            "CIFAR-10 四元 SNN 推論",
            "選擇已驗證案例與專用 Time-step 模型，即時觀察分類機率與脈衝分布。",
        ),
        unsafe_allow_html=True,
    )
    indices = [sample.index for sample in samples]
    lookup = {sample.index: sample for sample in samples}
    if st.session_state.selected_sample_index not in lookup:
        st.session_state.selected_sample_index = indices[0]

    checkpoint_training_t = normalize_time_steps(metadata["time_steps"])
    left, middle, right = st.columns([1, 1, 1], gap="large")
    with middle:
        st.selectbox(
            "展示案例",
            options=indices,
            key="selected_sample_index",
            format_func=lambda index: lookup[index].display_name,
            on_change=clear_result,
        )
        previous, next_column = st.columns(2)
        st.selectbox(
            "Inference Time-step",
            options=(1, 2, 4, 6),
            key="selected_time_steps",
            format_func=lambda value: f"T = {value}",
            on_change=clear_result,
        )
        current_time_steps = normalize_time_steps(st.session_state.selected_time_steps)
        previous.button(
            "上一張",
            use_container_width=True,
            on_click=shift_sample,
            args=(indices, -1),
        )
        next_column.button(
            "下一張",
            use_container_width=True,
            on_click=shift_sample,
            args=(indices, 1),
        )
        uploaded = st.file_uploader(
            "自訂圖片（選填）",
            type=("png", "jpg", "jpeg", "webp"),
            on_change=clear_result,
        )
        if uploaded is not None:
            try:
                image = prepare_uploaded_image(Image.open(uploaded))
            except Exception as exc:
                st.error(f"圖片無法讀取：{exc}")
                return
            true_label = None
            image_index = None
            upload_digest = hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]
            source = f"upload:{uploaded.name}:{upload_digest}"
            st.markdown(
                '<div class="notice">模型針對 CIFAR-10 32×32 影像訓練，'
                "自訂圖片僅供實驗。</div>",
                unsafe_allow_html=True,
            )
        else:
            sample = lookup[st.session_state.selected_sample_index]
            image = cached_image(
                str(sample.image_path),
                sample.image_path.stat().st_mtime_ns,
            )
            true_label = sample.true_label
            image_index = sample.index
            source = f"verified_sample:{sample.index}"

        st.markdown(
            '<div class="info-panel">'
            f"<strong>Model:</strong> {metadata['model']}<br>"
            "<strong>Spike:</strong> {-1, 0, 1, 2}<br>"
            f"<strong>Checkpoint training T:</strong> {checkpoint_training_t}<br>"
            f"<strong>Current inference T:</strong> {current_time_steps}<br>"
            f"<strong>TAU:</strong> {metadata['tau']}<br>"
            f"<strong>Device:</strong> {service.device_label}<br>"
            f"<strong>Model implementation source:</strong> {implementation_source_label(metadata['source'])}<br>"
            f"<strong>Checkpoint:</strong> {spec.checkpoint.name}"
            "</div>",
            unsafe_allow_html=True,
        )
        is_verified_setting = is_verified_time_steps(
            current_time_steps, checkpoint_training_t
        )
        if is_verified_setting:
            st.success(
                f"Best Test Accuracy：{spec.recorded_test_accuracy:.2f}%"
            )
            st.caption(
                "此數值來自 all_time_step_results 原始訓練成果記錄的 Best Test Accuracy。"
            )
        else:
            st.warning(
                f"此結果使用 T={checkpoint_training_t}-trained checkpoint "
                f"進行 T={current_time_steps} 實驗性推論，尚未完成此組合的完整測試。"
            )
            st.caption(
                f"T={checkpoint_training_t} checkpoint recorded Best Test Accuracy："
                f"{spec.recorded_test_accuracy:.2f}%"
            )
        run_clicked = st.button(
            "開始推論",
            type="primary",
            use_container_width=True,
        )

    with left:
        image_alt = (
            f"CIFAR-10 {true_label}, index {image_index}"
            if true_label is not None
            else "Uploaded 32 by 32 image"
        )
        st.markdown(
            pixel_image_html(image, image_alt),
            unsafe_allow_html=True,
        )
        if true_label is not None:
            st.markdown(
                f"**Image Index:** {image_index}  \n"
                f"**True Label:** {true_label}  \n"
                "**Image Size:** 32×32 RGB"
            )
        else:
            st.markdown("**Image Size:** 32×32 RGB（由上傳圖片轉換）")

    should_run = run_clicked or bool(st.session_state.auto_run)
    st.session_state.auto_run = False
    if should_run:
        execute_inference(
            service,
            image,
            true_label,
            image_index,
            source,
            current_time_steps,
        )

    expected_result_key = inference_result_key(
        source,
        spec.checkpoint,
        str(metadata["checkpoint_sha256"]),
        metadata["time_steps"],
        current_time_steps,
        metadata["tau"],
    )
    with left:
        render_spike_result(expected_result_key)
    with right:
        render_result(metadata, expected_result_key)


def render_model_info(metadata: dict, spec: ModelSpec) -> None:
    st.markdown(
        section_header_html(
            "MODEL EVIDENCE",
            "模型成果與可追溯資料",
            "主要成果採用訓練過程記錄的 Best Test Accuracy，並保留 checkpoint 重評資料供實驗追溯。",
        ),
        unsafe_allow_html=True,
    )
    reevaluated_accuracy = float(metadata["test_accuracy"])
    recorded_accuracy = float(spec.recorded_test_accuracy)
    difference = reevaluated_accuracy - recorded_accuracy
    cards = (
        metric_card_html("Recorded Best Test Accuracy", f"{recorded_accuracy:.2f}%", True)
        + metric_card_html("Checkpoint Re-evaluation", f"{reevaluated_accuracy:.2f}%")
        + metric_card_html("Re-evaluated Correct", f"{metadata['correct_count']}/{metadata['test_count']}")
        + metric_card_html("Selected Time-step", f"T={spec.time_steps}")
    )
    st.markdown(f'<div class="metric-grid four">{cards}</div>', unsafe_allow_html=True)

    st.subheader("ResNet-20 All Time-step Results")
    st.markdown(recorded_results_table_html(), unsafe_allow_html=True)
    st.caption(
        "原始實驗紀錄來源：results/all_time_step_results。"
        "表中的 Test accuracy 是訓練期間記錄的 best test accuracy。"
    )

    comparison_rows = []
    for time_steps in (6, 4, 2, 1):
        item = MODEL_REGISTRY[time_steps]
        item_metadata = load_metadata(item.metadata_path)
        current_accuracy = float(item_metadata["test_accuracy"])
        comparison_rows.append(
            {
                "Time-step": f"T={time_steps}",
                "Recorded train": f"{item.recorded_train_accuracy:.2f}%",
                "Recorded best test": f"{item.recorded_test_accuracy:.2f}%",
                "Checkpoint re-evaluation": f"{current_accuracy:.2f}%",
                "Difference": f"{current_accuracy - item.recorded_test_accuracy:+.2f} pp",
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
    if abs(difference) > 0.000001:
        st.warning(
            f"T={spec.time_steps} 原始圖表記錄 {recorded_accuracy:.2f}%，"
            f"目前 checkpoint 完整重評為 {reevaluated_accuracy:.2f}%"
            f"（差異 {difference:+.2f} 個百分點）。兩者不相同。"
        )
    else:
        st.success(
            f"T={spec.time_steps} 原始圖表與目前 checkpoint 完整重評一致："
            f"{reevaluated_accuracy:.2f}%。"
        )

    curve, matrix = st.columns(2, gap="large")
    with curve:
        st.subheader(f"T={spec.time_steps} Original Training Result")
        if spec.training_result.is_file():
            training_image = cached_image(
                str(spec.training_result), spec.training_result.stat().st_mtime_ns
            )
            st.markdown(
                report_image_html(
                    training_image,
                    f"T={spec.time_steps} original training result",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Training result 尚未提供：{spec.training_result.name}")
    with matrix:
        st.subheader(f"T={spec.time_steps} Re-evaluated Confusion Matrix")
        if spec.confusion_matrix.is_file():
            confusion_image = cached_image(
                str(spec.confusion_matrix), spec.confusion_matrix.stat().st_mtime_ns
            )
            st.markdown(
                report_image_html(
                    confusion_image,
                    f"T={spec.time_steps} re-evaluated confusion matrix",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.error("Confusion matrix 尚未提供")

    st.subheader("Checkpoint Information")
    metadata_rows = (
        ("Model", metadata["model"]),
        ("Spike values", "{-1, 0, 1, 2}"),
        ("Checkpoint training T", normalize_time_steps(metadata["time_steps"])),
        ("TAU", metadata["tau"]),
        ("Dataset", metadata["dataset"]),
        ("Recorded final train accuracy", f"{spec.recorded_train_accuracy:.2f}%"),
        ("Recorded best test accuracy", f"{recorded_accuracy:.2f}%"),
        ("Checkpoint re-evaluated accuracy", f"{reevaluated_accuracy:.2f}%"),
        ("Re-evaluated correct", f"{metadata['correct_count']}/{metadata['test_count']}"),
        ("Epochs", metadata["epoch"]),
        ("Precision", metadata["quantization"]),
        ("Model implementation source", implementation_source_label(metadata["source"])),
    )
    st.dataframe(
        pd.DataFrame(
            [(field, str(value)) for field, value in metadata_rows],
            columns=("Field", "Value"),
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("進階資訊：Checkpoint Metadata", expanded=False):
        st.json(metadata)


def render_history() -> None:
    st.markdown(
        section_header_html(
            "SESSION LOG",
            "本次推論執行紀錄",
            "記錄僅保存於目前瀏覽器 session，可下載 CSV，不寫入共用伺服器檔案。",
        ),
        unsafe_allow_html=True,
    )
    records = ensure_session_history(st.session_state)
    if not records:
        st.info("此瀏覽器 session 尚無推論紀錄。")
        return
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    st.download_button(
        "下載目前 Session CSV",
        data=history_csv_bytes(records),
        file_name="quaternary_snn_session_history.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Quaternary SNN Demo",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_css()

    errors = validate_web_assets()
    if errors:
        st.error("Web Demo 尚未就緒：")
        for error in errors:
            st.error(error)
        return

    samples = cached_samples(
        str(PROJECT_ROOT / "configs" / "demo_samples.json"),
        (PROJECT_ROOT / "configs" / "demo_samples.json").stat().st_mtime_ns,
    )
    initialize_state(samples)
    try:
        selected_time_steps = normalize_time_steps(st.session_state.selected_time_steps)
        spec = get_model_spec(selected_time_steps)
        metadata = cached_metadata(
            str(spec.metadata_path), spec.metadata_path.stat().st_mtime_ns
        )
        service = cached_service(
            str(spec.checkpoint),
            sha256_file(spec.checkpoint),
            spec.time_steps,
            spec.tau,
            PREDICTOR_CACHE_VERSION,
        )
    except Exception as exc:
        st.error(f"模型載入失敗：{exc}")
        return

    st.markdown(
        '<div class="app-topbar">'
        '<div class="topbar-brand"><div class="brand-mark">Q</div>'
        '<div><strong>QUATERNARY SNN</strong><small>Inference System</small></div></div>'
        '<div class="topbar-status"><span class="live-dot"></span>'
        f'<div><small>MODEL ONLINE</small><strong>T={spec.time_steps} · TAU={spec.tau}</strong>'
        f'<em>{service.device_label} · FP32</em></div></div></div>',
        unsafe_allow_html=True,
    )
    st.segmented_control(
        "Navigation",
        PAGES,
        key="page",
        required=True,
        width="stretch",
        label_visibility="collapsed",
    )

    if st.session_state.page == "系統總覽":
        render_overview(metadata, service, samples, spec)
    elif st.session_state.page == "推論":
        render_inference(metadata, service, samples, spec)
    elif st.session_state.page == "模型資訊":
        render_model_info(metadata, spec)
    else:
        render_history()


if __name__ == "__main__":
    main()
