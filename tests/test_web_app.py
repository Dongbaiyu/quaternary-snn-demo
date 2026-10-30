from __future__ import annotations

import base64
import importlib
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from streamlit.testing.v1 import AppTest

from inference.predictor import Predictor
from qsn_model.constants import CIFAR10_CLASSES, Config, PROJECT_ROOT
from qsn_model.model import normalize_time_steps
from qsn_model.dataset import test_transform as cifar_test_transform
from web_app.components import (
    CHECKPOINT_PATH,
    METADATA_PATH,
    SAMPLE_CONFIG_PATH,
    LockedPredictor,
    append_session_record,
    create_cpu_service,
    ensure_session_history,
    history_csv_bytes,
    load_sample_image,
    implementation_source_label,
    inference_result_key,
    is_verified_time_steps,
    load_web_samples,
    validate_web_assets,
    load_metadata,
    image_data_uri,
    pixel_image_html,
    report_image_html,
    recorded_results_table_html,
)


@pytest.fixture(scope="module")
def service() -> LockedPredictor:
    return create_cpu_service()


@pytest.mark.parametrize(
    ("value", "expected"),
    ((1, 1), ("1", 1), ("T=2", 2), ("T = 4", 4), (6, 6)),
)
def test_time_step_values_are_normalized_before_comparison(value, expected):
    assert normalize_time_steps(value) == expected
    assert is_verified_time_steps(value, str(expected))


@pytest.mark.parametrize("value", (True, 0, "T=0", "T=four", ""))
def test_invalid_time_step_values_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_time_steps(value)


def test_locked_predictor_passes_normalized_int_to_predictor():
    predictor = Mock()
    predictor.predict_image.return_value = object()
    service = LockedPredictor(predictor)
    image = object()
    service.predict_image(image, time_steps="T = 2")
    predictor.predict_image.assert_called_once_with(image, time_steps=2)


def test_result_identity_contains_sample_checkpoint_and_inference_t(tmp_path):
    checkpoint = tmp_path / "model.pth"
    first = inference_result_key("sample:143", checkpoint, "abc123", "4", "T=4", 0.875)
    assert "sample:143" in first
    assert str(checkpoint) in first
    assert "inference_t=4" in first
    assert first != inference_result_key("sample:143", checkpoint, "abc123", 4, 1, 0.875)
    assert first != inference_result_key("sample:144", checkpoint, "abc123", 4, 4, 0.875)



def test_recorded_result_table_matches_user_provided_summary():
    table = recorded_results_table_html()
    for value in ("91.18%", "92.04%", "92.03%", "92.30%", "92.07%"):
        assert value in table
    for value in ("98.96%", "99.57%", "99.84%"):
        assert value in table
    assert "三元脈衝（文獻）" in table
    assert "四元脈衝" in table

def test_metadata_source_is_implementation_not_inference_time_step():
    metadata = load_metadata()
    assert metadata["source"] == "qua_v10.py"
    assert implementation_source_label(metadata["source"]) == "qua_v10.py"
    assert is_verified_time_steps("T = 4", metadata["time_steps"])
    assert not is_verified_time_steps("T = 1", metadata["time_steps"])

@pytest.fixture(scope="module")
def samples():
    return load_web_samples()


def hook_count(model) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def test_web_app_imports_without_running_server():
    module = importlib.import_module("web_app.streamlit_app")
    assert callable(module.main)


def test_checkpoint_loads_on_cpu(service):
    assert service.predictor.device.type == "cpu"
    assert service.predictor.model is not None
    assert service.predictor.model.training is False


def test_ten_exported_samples_cover_every_cifar10_class(samples):
    assert len(samples) == 10
    assert {sample.true_label for sample in samples} == set(CIFAR10_CLASSES)
    for sample in samples:
        assert sample.image_path.is_file()
        image = load_sample_image(sample)
        assert image.mode == "RGB"
        assert image.size == (32, 32)


def test_demo_image_is_embedded_as_valid_png_data_uri(samples):
    image = load_sample_image(samples[0])
    uri = image_data_uri(image)
    prefix = "data:image/png;base64,"
    assert uri.startswith(prefix)
    decoded = base64.b64decode(uri.removeprefix(prefix), validate=True)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
    markup = pixel_image_html(image, 'sample <image>')
    assert uri[:32] in markup
    assert 'alt="sample &lt;image&gt;"' in markup


def test_all_model_report_images_are_embedded_as_valid_png_data_uris():
    from PIL import Image
    from inference.model_registry import MODEL_REGISTRY

    for time_steps, spec in MODEL_REGISTRY.items():
        for label, path in (
            ("original training result", spec.training_result),
            ("re-evaluated confusion matrix", spec.confusion_matrix),
        ):
            assert path.is_file(), f"T={time_steps} missing {label}: {path.name}"
            with Image.open(path) as source:
                source.verify()
            with Image.open(path) as source:
                markup = report_image_html(
                    source.convert("RGB"), f'T={time_steps} {label} <verified>'
                )
            assert 'class="report-image"' in markup
            assert 'src="data:image/png;base64,' in markup
            assert "&lt;verified&gt;" in markup


def test_all_demo_predictions_are_unchanged_on_cpu(service, samples):
    for sample in samples:
        result = service.predict_image(load_sample_image(sample))
        assert result.predicted_class == sample.expected_prediction
        assert result.predicted_class == sample.true_label
        assert all(count >= 0 for count in result.spike_statistics.values())


def test_web_logits_match_existing_predictor(service, samples):
    reference = Predictor(Config(device="cpu", mode="inference"))
    reference.load_checkpoint(str(CHECKPOINT_PATH))
    image = load_sample_image(samples[0])
    inputs = cifar_test_transform()(image).unsqueeze(0)

    web_model = service.predictor.model
    reference_model = reference.model
    assert web_model is not None and reference_model is not None
    with torch.no_grad():
        web_logits = web_model(inputs)
        reference_logits = reference_model(inputs)
    torch.testing.assert_close(web_logits, reference_logits, rtol=0.0, atol=0.0)


def test_consecutive_web_inference_leaves_no_hooks(service, samples):
    model = service.predictor.model
    assert model is not None
    before = hook_count(model)
    first = service.predict_image(load_sample_image(samples[0]))
    second = service.predict_image(load_sample_image(samples[0]))
    assert first.spike_statistics == second.spike_statistics
    assert hook_count(model) == before == 0


@pytest.mark.parametrize("time_steps", (1, 2, 4, 6))
def test_locked_web_predictor_supports_dynamic_time_steps(service, samples, time_steps):
    result = service.predict_image(load_sample_image(samples[0]), time_steps=time_steps)
    assert result.time_steps == time_steps
    assert sum(result.class_scores.values()) == pytest.approx(1.0)



def test_session_histories_are_isolated_and_download_is_in_memory():
    first_state: dict = {}
    second_state: dict = {}
    append_session_record(first_state, {"timestamp": "one", "prediction": "cat"})
    assert len(ensure_session_history(first_state)) == 1
    assert ensure_session_history(second_state) == []
    append_session_record(second_state, {"timestamp": "two", "prediction": "ship"})
    assert len(first_state["history"]) == 1
    assert len(second_state["history"]) == 1
    csv_bytes = history_csv_bytes(first_state["history"])
    assert b"prediction" in csv_bytes
    assert b"cat" in csv_bytes


def test_missing_checkpoint_or_images_are_reported_without_crash(tmp_path):
    missing_checkpoint = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError):
        create_cpu_service(missing_checkpoint)

    checkpoint_errors = validate_web_assets(
        checkpoint_path=missing_checkpoint,
        metadata_path=METADATA_PATH,
        config_path=SAMPLE_CONFIG_PATH,
    )
    assert any("Checkpoint missing" in error for error in checkpoint_errors)

    image_errors = validate_web_assets(
        checkpoint_path=CHECKPOINT_PATH,
        metadata_path=METADATA_PATH,
        config_path=SAMPLE_CONFIG_PATH,
        asset_dir=tmp_path,
    )
    assert sum("Demo image missing" in error for error in image_errors) == 10


def test_streamlit_overview_smoke_run():
    app_path = PROJECT_ROOT / "web_app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=120)
    assert not app.exception
    assert app.title or app.markdown
    assert not app.sidebar.radio
    assert app.segmented_control[0].value == "系統總覽"
    overview_markup = "\n".join(item.value for item in app.markdown)
    assert "app-topbar" not in overview_markup
    assert "QUATERNARY SNN" not in overview_markup
    assert "MODEL ONLINE" not in overview_markup
    assert "Inference System" not in overview_markup
    assert "四元脈衝神經網路" in overview_markup
    assert "實驗模型總覽" in overview_markup
    assert "軟體推論資料流" in overview_markup
    assert "模型設定" not in overview_markup
    assert "推論流程" not in overview_markup
    app.segmented_control[0].set_value("推論").run(timeout=120)
    assert not app.exception
    labels = {widget.label for widget in app.selectbox}
    assert "推論時間步" in labels
    assert not any("模型來源" in item.value or "權重檔案" in item.value for item in app.markdown)

    expected_models = {
        1: ("91.18%", "quaternary_sresnet20_qua_t_1_best_weights.pth", "qua_timestep_1.py", "0.875"),
        2: ("92.04%", "quaternary_sresnet20_qua_t_2_best_weights.pth", "qua_timestep_2.py", "0.875"),
        6: ("92.30%", "quaternary_sresnet20_qua_t_6_best_weights.pth", "qua_timestep_6.py", "0.9"),
    }
    for selected_t, (accuracy, checkpoint, source, tau) in expected_models.items():
        time_step_select = next(
            widget for widget in app.selectbox if widget.label == "推論時間步"
        )
        time_step_select.set_value(selected_t).run(timeout=120)
        assert not app.exception
        assert not app.warning
        assert any(
            f"最佳測試準確率：{accuracy}" in success.value
            for success in app.success
        )
        info_html = "\n".join(item.value for item in app.markdown)
        assert checkpoint not in info_html
        assert source not in info_html
        assert f"<strong>TAU／裝置</strong><span>{tau}／" in info_html

    time_step_select = next(
        widget for widget in app.selectbox if widget.label == "推論時間步"
    )
    time_step_select.set_value(4).run(timeout=120)
    assert not app.exception
    assert not any(
        "T=4-trained checkpoint" in warning.value for warning in app.warning
    )
    assert any(
        "最佳測試準確率：92.03%" in success.value for success in app.success
    )
    assert not any("此數值來自 all_time_step_results" in caption.value for caption in app.caption)

    run_button = next(button for button in app.button if button.label == "開始推論")
    run_button.click().run(timeout=120)
    assert not app.exception
    assert any("第一名預測" in item.value for item in app.markdown)
    assert not any("推論時間：" in item.value or "統計範圍：" in item.value for item in app.markdown)

    time_step_select = next(
        widget for widget in app.selectbox if widget.label == "推論時間步"
    )
    time_step_select.set_value(1).run(timeout=120)
    assert not any("第一名預測" in item.value for item in app.markdown)
    assert any("result-panel-empty" in item.value for item in app.markdown)


def test_public_streamlit_config_hides_developer_details():
    config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config["client"]["toolbarMode"] == "viewer"
    assert config["client"]["showErrorDetails"] == "none"
    assert config["client"]["showErrorLinks"] is False
    assert config["client"]["showSidebarNavigation"] is False
    assert config["theme"]["base"] == "light"
    assert config["theme"]["backgroundColor"] == "#f4f5f7"
    assert config["theme"]["textColor"] == "#15171a"
