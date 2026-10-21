from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMainWindow,
    QTabWidget,
    QToolBar,
)

from inference.demo_samples import load_demo_samples
from inference.predictor import scan_checkpoints
from qsn_model.constants import PROJECT_ROOT, load_demo_settings

from .pages import HistoryPage, InferencePage, ModelInfoPage, OverviewPage


STYLE_SHEET = """
QWidget {
    font-family: "Noto Sans CJK TC", "Noto Sans", sans-serif;
    font-size: 14px;
    color: #e7edf4;
    background: #111923;
}
QMainWindow, QTabWidget::pane { background: #111923; }
QToolBar {
    background: #172331;
    border: 0;
    border-bottom: 1px solid #314356;
    spacing: 10px;
    padding: 6px 14px;
}
QTabBar::tab {
    background: #172331;
    color: #aebdca;
    padding: 10px 22px;
    border: 0;
}
QTabBar::tab:selected { background: #213449; color: white; }
QGroupBox {
    border: 1px solid #314356;
    border-radius: 9px;
    margin-top: 12px;
    padding: 12px;
    background: #172331;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QLabel#pageTitle { font-size: 26px; font-weight: 700; color: #f5f8fb; }
QLabel#sectionTitle { font-size: 19px; font-weight: 650; color: #d9e8f5; }
QLabel#metricCaption, QLabel#mutedText { color: #91a4b5; font-size: 12px; }
QLabel#metricValue { font-size: 21px; font-weight: 700; color: #f5f8fb; }
QLabel#predictionValue { font-size: 30px; font-weight: 800; color: #56c8ff; }
QFrame#metricCard, QFrame#metricCardAccent {
    background: #1a2938;
    border: 1px solid #314356;
    border-radius: 8px;
}
QFrame#metricCardAccent { border: 2px solid #39aee8; }
QLabel#pipelineStage {
    background: #1c3042;
    border: 1px solid #3b5870;
    border-radius: 7px;
    padding: 7px;
    font-weight: 600;
}
QLabel#pipelineArrow { color: #56c8ff; font-size: 22px; font-weight: bold; }
QLabel#noticeText {
    background: #182c38;
    border-left: 4px solid #39aee8;
    padding: 12px;
    color: #cfe5f4;
}
QLabel#imagePreview, QLabel#resultFigure {
    background: #0d141c;
    border: 1px solid #314356;
    border-radius: 8px;
}
QLabel#statusBusy { color: #ffca58; font-weight: 700; }
QLabel#statusReady, QLabel#correctResult { color: #68d391; font-weight: 700; }
QLabel#statusError, QLabel#incorrectResult { color: #ff6b6b; font-weight: 700; }
QPushButton {
    background: #24384a;
    border: 1px solid #3b5870;
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton:hover { background: #2d4a61; }
QPushButton:disabled { color: #647482; background: #1a2530; }
QPushButton#primaryButton { background: #087db5; font-size: 16px; font-weight: 700; }
QPushButton#primaryButton:hover { background: #109bdc; }
QComboBox, QSpinBox {
    background: #0f1923;
    border: 1px solid #3b5870;
    border-radius: 5px;
    padding: 6px;
}
QTableWidget {
    background: #101923;
    alternate-background-color: #162431;
    gridline-color: #314356;
}
QHeaderView::section { background: #213449; padding: 7px; border: 0; }
QProgressBar {
    background: #0d141c;
    border: 0;
    border-radius: 4px;
    min-height: 9px;
    max-height: 9px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_demo_settings()
        self.setWindowTitle("四元 SNN 推論與視覺化系統")
        self.resize(self.settings.window_width, self.settings.window_height)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLE_SHEET)

        self.checkpoints = scan_checkpoints(PROJECT_ROOT / "checkpoints")
        self.demo_samples = load_demo_samples()
        self.history_page = HistoryPage()
        self.info_page = ModelInfoPage(
            self.checkpoints,
            result_directory=self.settings.result_path(),
            settings=self.settings,
        )
        self.inference_page = InferencePage(
            self.checkpoints,
            self.history_page.add_record,
            settings=self.settings,
            samples=self.demo_samples,
        )
        self.overview_page = OverviewPage(
            self.checkpoints,
            self.settings,
            samples=self.demo_samples,
        )
        self.overview_page.quick_demo_requested.connect(self.start_quick_demo)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.overview_page, "系統總覽")
        self.tabs.addTab(self.inference_page, "推論")
        self.tabs.addTab(self.info_page, "模型資訊")
        self.tabs.addTab(self.history_page, "執行紀錄")
        self.setCentralWidget(self.tabs)
        self._build_mode_toolbar()
        self.set_mode(self.settings.demo_mode)

    def _build_mode_toolbar(self) -> None:
        toolbar = QToolBar("Mode")
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("介面模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("展示模式", "showcase")
        self.mode_combo.addItem("研究模式", "research")
        self.mode_combo.currentIndexChanged.connect(
            lambda: self.set_mode(self.mode_combo.currentData())
        )
        toolbar.addWidget(self.mode_combo)
        self.addToolBar(toolbar)

    def set_mode(self, mode: str) -> None:
        normalized = "research" if mode == "research" else "showcase"
        index = self.mode_combo.findData(normalized)
        if index >= 0 and self.mode_combo.currentIndex() != index:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)
        self.tabs.setTabVisible(3, normalized == "research")
        if normalized == "showcase" and self.tabs.currentIndex() == 3:
            self.tabs.setCurrentIndex(0)

    def start_quick_demo(self, image_index: int) -> None:
        self.tabs.setCurrentWidget(self.inference_page)
        self.inference_page.run_quick_demo(image_index)

    def closeEvent(self, event) -> None:
        worker = self.inference_page.worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait()
        event.accept()
