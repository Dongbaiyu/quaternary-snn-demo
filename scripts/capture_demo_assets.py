from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QScrollArea

from gui.main_window import MainWindow


SCREENSHOT_DIR = PROJECT_ROOT / "demo_assets" / "screenshots"
EXPORT_DIR = PROJECT_ROOT / "demo_assets" / "exported_results"


def save_grab(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not widget.grab().save(str(path)):
        raise RuntimeError(f"failed to save screenshot: {path}")
    print(f"saved: {path.relative_to(PROJECT_ROOT)}")


def wait_for_worker(app: QApplication, worker, timeout_ms: int = 30000) -> None:
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
    if timed_out:
        raise TimeoutError("GUI inference worker timed out")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.inference_page.record_callback = lambda record: None

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    window.tabs.setCurrentWidget(window.overview_page)
    app.processEvents()
    save_grab(
        window,
        SCREENSHOT_DIR / "system_overview_1366x768.png",
    )

    first_sample = window.demo_samples[0]
    window.start_quick_demo(first_sample.index)
    app.processEvents()
    worker = window.inference_page.worker
    if worker is None:
        raise RuntimeError("quick demo did not create an inference worker")
    wait_for_worker(app, worker)
    result = window.inference_page.last_result
    if result is None:
        raise RuntimeError("quick demo did not produce a result")
    if result.predicted_class != first_sample.expected_prediction:
        raise RuntimeError(
            f"verified sample changed: expected {first_sample.expected_prediction}, "
            f"got {result.predicted_class}"
        )

    window.tabs.setCurrentWidget(window.inference_page)
    app.processEvents()
    save_grab(
        window,
        SCREENSHOT_DIR / "verified_inference_1366x768.png",
    )
    save_grab(
        window.inference_page.spike_chart,
        SCREENSHOT_DIR / "spike_distribution_actual.png",
    )

    window.resize(1920, 1080)
    window.tabs.setCurrentWidget(window.info_page)
    app.processEvents()
    scroll = window.info_page.findChild(QScrollArea)
    if scroll is not None:
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    app.processEvents()
    save_grab(
        window,
        SCREENSHOT_DIR / "model_results_1920x1080.png",
    )

    export = {
        "image_index": first_sample.index,
        "true_label": first_sample.true_label,
        "expected_prediction": first_sample.expected_prediction,
        "verified": result.predicted_class == first_sample.true_label,
        "prediction_result": result.to_dict(),
    }
    export_path = EXPORT_DIR / "verified_demo_inference.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps(export, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {export_path.relative_to(PROJECT_ROOT)}")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
