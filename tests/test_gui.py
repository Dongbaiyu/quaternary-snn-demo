from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QScrollArea

from gui.main_window import MainWindow
from gui.pages.history_page import HistoryPage
from gui.pages.inference_page import InferencePage
from gui.pages.model_info_page import ModelInfoPage
from inference.predictor import CheckpointInfo, scan_checkpoints
from qsn_model.constants import PROJECT_ROOT


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture(scope="module")
def checkpoints():
    return scan_checkpoints(PROJECT_ROOT / "checkpoints")


def wait_for_worker(app, worker, timeout_ms=30000):
    loop = QEventLoop()
    timed_out = []
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: (timed_out.append(True), loop.quit()))
    worker.finished.connect(loop.quit)
    timer.start(timeout_ms)
    if worker.isRunning():
        loop.exec()
    worker.wait()
    app.processEvents()
    assert not timed_out


def test_main_window_has_four_pages_and_mode_switch(app):
    window = MainWindow()
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "系統總覽",
        "推論",
        "模型資訊",
        "執行紀錄",
    ]
    assert not window.tabs.isTabVisible(3)
    window.set_mode("research")
    assert window.tabs.isTabVisible(3)
    window.close()


def test_checkpoint_scan_exposes_only_t4(checkpoints):
    assert [(item.metadata["time_steps"], item.path.name) for item in checkpoints] == [
        (4, "qsn_sresnet20_t4_best.pth")
    ]


def test_demo_sample_dropdown_uses_verified_first_case(app, checkpoints):
    page = InferencePage(checkpoints, lambda record: None)
    assert page.sample_combo.count() == 10
    assert page.sample_combo.itemData(0) == 143
    assert page.index.value() == 143
    page.sample_combo.setCurrentIndex(1)
    assert page.index.value() == 108
    assert page.true_label.text() == "ship"
    page.close()


def test_qthread_inference_updates_result_and_charts(app, checkpoints):
    records = []
    page = InferencePage(checkpoints, records.append)
    page.index.setValue(143)
    page.start_inference()
    assert page.worker is not None
    wait_for_worker(app, page.worker)

    assert page.top1.text() == "cat"
    assert page.top3.text() != "-"
    assert page.checkpoint_name.text() == "qsn_sresnet20_t4_best.pth"
    assert any(row.bar.value() > 0 for row in page.score_chart.rows.values())
    assert any(row.bar.value() > 0 for row in page.spike_chart.rows.values())
    assert records
    page.close()


def test_double_click_mode_tab_switch_and_close_are_worker_safe(app):
    window = MainWindow()
    window.inference_page.record_callback = lambda record: None
    window.inference_page.index.setValue(143)
    window.inference_page.start_inference()
    first_worker = window.inference_page.worker
    assert first_worker is not None

    window.inference_page.start_inference()
    assert window.inference_page.worker is first_worker
    window.set_mode("research")
    window.set_mode("showcase")
    window.tabs.setCurrentWidget(window.info_page)
    window.tabs.setCurrentWidget(window.inference_page)
    assert window.inference_page.worker is first_worker

    window.close()
    assert not first_worker.isRunning()


def test_no_checkpoint_is_an_inline_error(app):
    page = InferencePage([], lambda record: None)
    assert not page.start_button.isEnabled()
    page.start_inference()
    assert "找不到有效 checkpoint" in page.status.text()
    page.close()


def test_missing_checkpoint_and_invalid_metadata_are_safe(
    app, checkpoints, tmp_path
):
    bad_checkpoint = tmp_path / "bad.pth"
    bad_checkpoint.write_bytes(b"not a checkpoint")
    bad_checkpoint.with_suffix(".json").write_text(
        json.dumps({"model": "SResNet20", "time_steps": 1}),
        encoding="utf-8",
    )
    assert scan_checkpoints(tmp_path) == []

    source = checkpoints[0]
    missing = CheckpointInfo(
        tmp_path / "missing.pth",
        tmp_path / "missing.json",
        source.metadata,
    )
    page = InferencePage([missing], lambda record: None)
    page.start_inference()
    assert page.worker is None
    assert "checkpoint 遺失" in page.status.text()
    page.close()


def test_missing_dataset_reports_error_without_download(
    app, checkpoints, monkeypatch
):
    calls = []

    def missing_dataset(data_dir, download, transform):
        calls.append(download)
        raise RuntimeError("dataset missing")

    monkeypatch.setattr(
        "gui.pages.inference_page.get_cifar10_testset",
        missing_dataset,
    )
    page = InferencePage(checkpoints, lambda record: None, data_dir="/missing")
    assert calls == [False]
    assert page.dataset is None
    assert "dataset missing" in page.status.text()
    assert not page.start_button.isEnabled()
    page.close()


def test_missing_result_images_do_not_break_page(app, checkpoints, tmp_path):
    page = ModelInfoPage(checkpoints, result_directory=tmp_path)
    messages = [
        label.text()
        for label in page.findChildren(QLabel)
        if label.text() == "結果檔案尚未提供"
    ]
    assert len(messages) == 2
    page.close()


def test_csv_export_permission_error_is_reported(app, monkeypatch):
    page = HistoryPage()
    page._append_row(
        {
            "timestamp": "2026-08-03T00:00:00",
            "image_index": 143,
            "true_label": "cat",
            "prediction": "cat",
            "top3": "cat; dog; frog",
            "checkpoint": "qsn_sresnet20_t4_best.pth",
            "time_steps": 4,
            "inference_time": "0.1",
            "correct": True,
        }
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("/sys/snn_demo_forbidden.csv", "CSV (*.csv)"),
    )
    page.export_csv()
    assert "匯出失敗" in page.status_label.text()
    page.close()


def test_index_out_of_range_is_clamped_or_ignored(app, checkpoints):
    page = InferencePage(checkpoints, lambda record: None)
    page.show_image(-1)
    page.show_image(10000)
    page.index.setValue(-100)
    assert page.index.value() == 0
    page.index.setValue(20000)
    assert page.index.value() == 9999
    page.close()


@pytest.mark.parametrize("width,height", [(1366, 768), (1920, 1080)])
def test_supported_display_sizes_have_no_horizontal_scroll(
    app, checkpoints, width, height
):
    window = MainWindow()
    window.resize(width, height)
    window.show()
    app.processEvents()
    assert window.width() == width
    assert window.height() == height

    for page in (window.inference_page, window.info_page):
        window.tabs.setCurrentWidget(page)
        app.processEvents()
        scrolls = page.findChildren(QScrollArea)
        assert scrolls
        assert all(scroll.horizontalScrollBar().maximum() == 0 for scroll in scrolls)

    assert window.inference_page.top1.objectName() == "predictionValue"
    assert any(
        label.text() == "92.03%" for label in window.info_page.findChildren(QLabel)
    )
    window.close()
