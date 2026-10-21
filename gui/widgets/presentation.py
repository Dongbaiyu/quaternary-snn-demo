from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class MetricCard(QFrame):
    def __init__(self, label: str, value: str, accent: bool = False):
        super().__init__()
        self.setObjectName("metricCardAccent" if accent else "metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        caption = QLabel(label)
        caption.setObjectName("metricCaption")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.value_label.setWordWrap(True)
        layout.addWidget(caption)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ModelPipelineWidget(QWidget):
    STAGES = (
        "CIFAR-10\nImage",
        "Convolution",
        "Quaternary\nLIF",
        "Residual\nBlocks",
        "Global Average\nPooling",
        "Fully Connected\nLayer",
        "Classification\nResult",
    )

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        for index, stage in enumerate(self.STAGES):
            box = QLabel(stage)
            box.setAlignment(Qt.AlignCenter)
            box.setWordWrap(True)
            box.setMinimumHeight(64)
            box.setObjectName("pipelineStage")
            layout.addWidget(box, 1)
            if index < len(self.STAGES) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("pipelineArrow")
                arrow.setAlignment(Qt.AlignCenter)
                layout.addWidget(arrow)
