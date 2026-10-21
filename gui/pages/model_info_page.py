from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.widgets import MetricCard
from inference.predictor import CheckpointInfo
from qsn_model.constants import DemoSettings, load_demo_settings


IMPLEMENTATION_NOTICE = (
    "本模型依據相關三元脈衝神經網路文獻之方法描述自行實作。\n"
    "由於原文未提供官方程式碼，本系統使用相同研究概念與本研究\n"
    "建立的訓練流程進行三元及四元模型比較，不宣稱完全復現原論文程式碼。"
)


class ModelInfoPage(QWidget):
    def __init__(
        self,
        checkpoints: list[CheckpointInfo],
        result_directory: str | Path | None = None,
        settings: DemoSettings | None = None,
    ):
        super().__init__()
        self.checkpoints = checkpoints
        self.settings = settings or load_demo_settings()
        self.result_directory = (
            Path(result_directory).resolve()
            if result_directory is not None
            else self.settings.result_path()
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Model Information & Verified Results")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        layout.addLayout(self._build_metrics())
        layout.addWidget(self._build_checkpoint_table())

        result_title = QLabel("模型成果（讀取既有 results 檔案）")
        result_title.setObjectName("sectionTitle")
        layout.addWidget(result_title)
        figures = QHBoxLayout()
        figures.addWidget(
            self._result_figure(
                "Training Accuracy Curve",
                self.result_directory / "quaternary_result_v10.png",
            ),
            1,
        )
        figures.addWidget(
            self._result_figure(
                "Confusion Matrix",
                self.result_directory / "confusion_matrix_v10.png",
            ),
            1,
        )
        layout.addLayout(figures)

        notice_box = QGroupBox("研究聲明")
        notice_layout = QVBoxLayout(notice_box)
        notice = QLabel(IMPLEMENTATION_NOTICE)
        notice.setWordWrap(True)
        notice.setObjectName("noticeText")
        notice_layout.addWidget(notice)
        layout.addWidget(notice_box)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _primary_metadata(self) -> dict:
        return self.checkpoints[0].metadata if self.checkpoints else {}

    def _training_epochs(self, metadata: dict) -> str:
        csv_path = self.result_directory / "quaternary_training_log_v10.csv"
        if csv_path.is_file():
            try:
                with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                    epochs = [
                        int(float(row["Epoch"]))
                        for row in csv.DictReader(handle)
                        if row.get("Epoch")
                    ]
                if epochs:
                    return str(max(epochs))
            except (OSError, ValueError, KeyError):
                pass
        epoch = metadata.get("epoch")
        return str(epoch) if epoch is not None else "結果檔案尚未提供"

    def _build_metrics(self) -> QGridLayout:
        metadata = self._primary_metadata()
        accuracy = metadata.get("test_accuracy")
        accuracy_text = f"{float(accuracy):.2f}%" if accuracy is not None else "-"
        correct_text = (
            f"{round(float(accuracy) * 100)}/10000" if accuracy is not None else "-"
        )
        values = (
            ("Best Test Accuracy", accuracy_text),
            ("Test Correct", correct_text),
            ("Time-step", f"T={metadata.get('time_steps', '-')}"),
            ("Training Epochs", self._training_epochs(metadata)),
        )
        grid = QGridLayout()
        for index, (label, value) in enumerate(values):
            grid.addWidget(MetricCard(label, value, label == "Best Test Accuracy"), 0, index)
        return grid

    def _build_checkpoint_table(self) -> QGroupBox:
        box = QGroupBox("Available Verified Checkpoints")
        layout = QVBoxLayout(box)
        table = QTableWidget(len(self.checkpoints), 6)
        table.setHorizontalHeaderLabels(
            ["Checkpoint", "Time-step", "TAU", "Accuracy", "訓練日期", "Precision"]
        )
        for row, info in enumerate(self.checkpoints):
            metadata = info.metadata
            values = (
                info.path.name,
                str(metadata["time_steps"]),
                str(metadata["tau"]),
                f"{metadata['test_accuracy']:.2f}%",
                str(metadata.get("training_date", "未記錄")),
                str(metadata["quantization"]),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(150)
        layout.addWidget(table)
        if not self.checkpoints:
            message = QLabel("找不到有效 checkpoint 與 metadata")
            message.setObjectName("statusError")
            layout.addWidget(message)
        return box

    def _result_figure(self, title: str, path: Path) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(300)
        label.setObjectName("resultFigure")
        if path.is_file():
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                label.setText("結果檔案無法讀取")
            else:
                label.setPixmap(
                    pixmap.scaled(560, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                label.setToolTip(str(path))
        else:
            label.setText("結果檔案尚未提供")
        layout.addWidget(label)
        return box
