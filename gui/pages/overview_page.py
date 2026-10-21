from __future__ import annotations

import torch
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.widgets import MetricCard, ModelPipelineWidget
from inference.demo_samples import DemoSample
from inference.predictor import CheckpointInfo
from qsn_model.constants import DemoSettings


def runtime_device_name() -> str:
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(torch.cuda.current_device())}"
    return "CPU"


class OverviewPage(QWidget):
    quick_demo_requested = Signal(int)

    def __init__(
        self,
        checkpoints: list[CheckpointInfo],
        settings: DemoSettings,
        samples: list[DemoSample] | None = None,
    ):
        super().__init__()
        self.checkpoints = checkpoints
        self.settings = settings
        self.samples = samples or []
        metadata = checkpoints[0].metadata if checkpoints else {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(16)

        title = QLabel(
            "四元脈衝神經網路推論與視覺化系統\n"
            "Quaternary SNN Inference and Visualization System"
        )
        title.setObjectName("pageTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        cards = QGridLayout()
        values = (
            ("Model", str(metadata.get("model", "SResNet20"))),
            ("Dataset", str(metadata.get("dataset", "CIFAR-10"))),
            ("Spike", "{-1, 0, 1, 2}"),
            ("Time-step", f"T={metadata.get('time_steps', 4)}"),
            ("TAU", str(metadata.get("tau", 0.875))),
            ("Test Accuracy", f"{metadata['test_accuracy']:.2f}%" if metadata else "-"),
            ("Precision", str(metadata.get("quantization", "-"))),
            ("Backend", "PyTorch Inference"),
            ("Device", runtime_device_name()),
        )
        for index, (label, value) in enumerate(values):
            cards.addWidget(
                MetricCard(label, value, label in {"Time-step", "Test Accuracy"}),
                index // 3,
                index % 3,
            )
        layout.addLayout(cards)

        flow_title = QLabel("System Inference Flow")
        flow_title.setObjectName("sectionTitle")
        layout.addWidget(flow_title)
        layout.addWidget(ModelPipelineWidget())

        note = QLabel(
            "使用預先訓練完成的 FP32 checkpoint 進行推論；"
            "展示過程不訓練、不下載資料、不執行量化轉換。"
        )
        note.setObjectName("noticeText")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.quick_button = QPushButton("開始快速展示")
        self.quick_button.setObjectName("primaryButton")
        self.quick_button.setMinimumHeight(48)
        self.quick_button.setEnabled(bool(checkpoints))
        default_index = (
            self.samples[0].index if self.samples else settings.default_image_index
        )
        self.quick_button.clicked.connect(
            lambda: self.quick_demo_requested.emit(default_index)
        )
        if not checkpoints:
            self.quick_button.setToolTip("找不到有效 checkpoint 與 metadata")
        layout.addWidget(self.quick_button)
