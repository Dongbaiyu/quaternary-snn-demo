from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import torch
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.widgets import ClassScoreChart, SpikeDistributionWidget
from inference.demo_samples import DemoSample, load_demo_samples
from inference.predictor import CheckpointInfo, Predictor
from qsn_model.constants import (
    CIFAR10_CLASSES,
    Config,
    DemoSettings,
    load_demo_settings,
)
from qsn_model.dataset import get_cifar10_testset


class InferenceWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, checkpoint: str, index: int, data_dir: str):
        super().__init__()
        self.checkpoint = checkpoint
        self.index = index
        self.data_dir = data_dir

    def run(self):
        try:
            predictor = Predictor(Config(data_dir=self.data_dir, mode="inference"))
            predictor.load_checkpoint(self.checkpoint)
            self.succeeded.emit(predictor.predict_cifar_index(self.index))
        except Exception as exc:
            self.failed.emit(str(exc))


def runtime_device_name() -> str:
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(torch.cuda.current_device())}"
    return "CPU"


class InferencePage(QWidget):
    def __init__(
        self,
        checkpoints: list[CheckpointInfo],
        record_callback,
        settings: DemoSettings | None = None,
        samples: list[DemoSample] | None = None,
        data_dir: str = "",
    ):
        super().__init__()
        self.checkpoints = checkpoints
        self.record_callback = record_callback
        self.settings = settings or load_demo_settings()
        self.demo_samples = samples if samples is not None else load_demo_samples()
        self.worker = None
        self.last_result = None
        self.config = Config(mode="inference", data_dir=data_dir)
        self.dataset = None
        self._build_ui()
        self._select_default_checkpoint()
        self._load_dataset()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(22, 18, 22, 22)
        root.setSpacing(14)

        title = QLabel("CIFAR-10 Quaternary SNN Inference")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        columns.addWidget(self._build_input_panel(), 3)
        columns.addWidget(self._build_control_panel(), 3)
        columns.addWidget(self._build_result_panel(), 5)
        root.addLayout(columns)
        self.spike_chart = SpikeDistributionWidget()
        root.addWidget(self.spike_chart)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_input_panel(self) -> QGroupBox:
        panel = QGroupBox("Input Image")
        layout = QVBoxLayout(panel)
        self.image_label = QLabel("載入中…")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("imagePreview")
        layout.addWidget(self.image_label, 0, Qt.AlignHCenter)

        index_form = QFormLayout()
        self.index = QSpinBox()
        self.index.setRange(0, 9999)
        self.index.setValue(self.settings.default_image_index)
        self.index.valueChanged.connect(self.show_image)
        self.true_label = QLabel("-")
        index_form.addRow("Image Index", self.index)
        index_form.addRow("True Label", self.true_label)
        index_form.addRow("Image Size", QLabel("32×32 RGB"))
        layout.addLayout(index_form)
        return panel

    def _build_control_panel(self) -> QGroupBox:
        panel = QGroupBox("Model & Controls")
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.sample_combo = QComboBox()
        for sample in self.demo_samples:
            self.sample_combo.addItem(sample.display_name, sample.index)
        self.sample_combo.currentIndexChanged.connect(self._select_demo_sample)
        self.checkpoint_combo = QComboBox()
        for info in self.checkpoints:
            self.checkpoint_combo.addItem(info.display_name, str(info.path))
        self.checkpoint_combo.currentIndexChanged.connect(self._update_checkpoint_details)
        self.model_value = QLabel("-")
        self.spike_value = QLabel("{-1, 0, 1, 2}")
        self.time_step_value = QLabel("-")
        self.tau_value = QLabel("-")
        self.device_value = QLabel(runtime_device_name())
        form.addRow("展示案例", self.sample_combo)
        form.addRow("Checkpoint", self.checkpoint_combo)
        form.addRow("Model", self.model_value)
        form.addRow("Spike Type", self.spike_value)
        form.addRow("Time-step", self.time_step_value)
        form.addRow("TAU", self.tau_value)
        form.addRow("Device", self.device_value)
        layout.addLayout(form)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("上一張")
        self.previous_button.clicked.connect(self.previous_image)
        self.random_button = QPushButton("隨機圖片")
        self.random_button.clicked.connect(
            lambda: self.index.setValue(random.randrange(10000))
        )
        self.next_button = QPushButton("下一張")
        self.next_button.clicked.connect(self.next_image)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.random_button)
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)

        self.start_button = QPushButton("開始推論")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumHeight(46)
        self.start_button.clicked.connect(self.start_inference)
        self.start_button.setEnabled(bool(self.checkpoints))
        layout.addWidget(self.start_button)
        layout.addStretch()
        return panel

    def _build_result_panel(self) -> QGroupBox:
        panel = QGroupBox("Inference Result")
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.status = QLabel("就緒")
        self.status.setObjectName("statusReady")
        self.top1 = QLabel("-")
        self.top1.setObjectName("predictionValue")
        self.correctness = QLabel("-")
        self.top3 = QLabel("-")
        self.top3.setWordWrap(True)
        self.elapsed = QLabel("-")
        self.checkpoint_name = QLabel("-")
        self.checkpoint_name.setWordWrap(True)
        form.addRow("Status", self.status)
        form.addRow("Top-1 Prediction", self.top1)
        form.addRow("Result", self.correctness)
        form.addRow("Top-3 (Softmax)", self.top3)
        form.addRow("Inference Time", self.elapsed)
        form.addRow("Checkpoint", self.checkpoint_name)
        layout.addLayout(form)
        self.score_chart = ClassScoreChart()
        layout.addWidget(self.score_chart)
        return panel

    def _select_demo_sample(self) -> None:
        selected = self.sample_combo.currentData()
        if selected is not None:
            self.index.setValue(int(selected))

    def _select_default_checkpoint(self) -> None:
        if not self.checkpoints:
            self._update_checkpoint_details()
            return
        configured = self.settings.default_checkpoint
        for index, info in enumerate(self.checkpoints):
            if configured in {str(info.path), info.path.name, str(info.path.relative_to(info.path.parents[1]))}:
                self.checkpoint_combo.setCurrentIndex(index)
                break
        self._update_checkpoint_details()

    def _update_checkpoint_details(self) -> None:
        index = self.checkpoint_combo.currentIndex()
        if not 0 <= index < len(self.checkpoints):
            for label in (self.model_value, self.time_step_value, self.tau_value):
                label.setText("-")
            return
        metadata = self.checkpoints[index].metadata
        self.model_value.setText(str(metadata["model"]))
        self.time_step_value.setText(f"T={metadata['time_steps']}")
        self.tau_value.setText(str(metadata["tau"]))

    def _load_dataset(self):
        try:
            self.dataset = get_cifar10_testset(
                self.config.resolved_data_dir(), download=False, transform=False
            )
            self.show_image(self.settings.default_image_index)
        except Exception as exc:
            self.image_label.setText(f"CIFAR-10 載入失敗\n{exc}")
            self._show_error(str(exc))

    def show_image(self, index: int):
        if self.dataset is None:
            return
        if not 0 <= index < len(self.dataset):
            return
        image, label = self.dataset[index]
        pixmap = QPixmap.fromImage(ImageQt(image.convert("RGB"))).scaled(
            300,
            300,
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.true_label.setText(CIFAR10_CLASSES[label])

    def previous_image(self) -> None:
        self.index.setValue((self.index.value() - 1) % 10000)

    def next_image(self) -> None:
        self.index.setValue((self.index.value() + 1) % 10000)

    def run_quick_demo(self, index: int) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.index.setValue(max(0, min(9999, index)))
        if self.checkpoints:
            self.checkpoint_combo.setCurrentIndex(0)
        QTimer.singleShot(0, self.start_inference)

    def _set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(
            not busy and bool(self.checkpoints) and self.dataset is not None
        )
        self.checkpoint_combo.setEnabled(not busy)
        self.sample_combo.setEnabled(not busy)
        self.index.setEnabled(not busy)
        self.previous_button.setEnabled(not busy)
        self.random_button.setEnabled(not busy)
        self.next_button.setEnabled(not busy)
        if busy:
            self.status.setText("Busy — Loading checkpoint / Inference…")
            self.status.setObjectName("statusBusy")
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)

    def start_inference(self):
        checkpoint = self.checkpoint_combo.currentData()
        if not checkpoint:
            self._show_error("找不到有效 checkpoint 與 metadata。")
            return
        if not Path(str(checkpoint)).is_file():
            self._show_error(f"checkpoint 遺失：{checkpoint}")
            return
        if self.dataset is None:
            self._show_error("找不到本機 CIFAR-10 test set；Demo 不會自動下載資料。")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._set_busy(True)
        self.worker = InferenceWorker(
            checkpoint,
            self.index.value(),
            str(self.config.resolved_data_dir()),
        )
        self.worker.succeeded.connect(self._show_result)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _show_result(self, result):
        self.last_result = result
        self.status.setText("完成")
        self.status.setObjectName("statusReady")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.top1.setText(result.predicted_class)
        self.top3.setText(
            "\n".join(
                f"{rank}. {item.class_name} — {item.score * 100:.2f}%"
                for rank, item in enumerate(result.top3_classes, start=1)
            )
        )
        self.elapsed.setText(f"{result.inference_time * 1000:.2f} ms")
        self.checkpoint_name.setText(result.checkpoint_name)
        self.score_chart.set_scores(result.class_scores)
        self.spike_chart.set_statistics(result.spike_statistics)

        _, true_index = self.dataset[self.index.value()]
        true_label = CIFAR10_CLASSES[true_index]
        is_correct = result.predicted_class == true_label
        self.correctness.setText("Correct ✓" if is_correct else "Incorrect ✕")
        self.correctness.setObjectName("correctResult" if is_correct else "incorrectResult")
        self.correctness.style().unpolish(self.correctness)
        self.correctness.style().polish(self.correctness)
        self.record_callback(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image_index": self.index.value(),
                "true_label": true_label,
                "prediction": result.predicted_class,
                "top3": "; ".join(item.class_name for item in result.top3_classes),
                "checkpoint": result.checkpoint_name,
                "time_steps": result.time_steps,
                "inference_time": f"{result.inference_time:.6f}",
                "correct": is_correct,
            }
        )

    def _show_error(self, message: str):
        self.status.setText(f"錯誤：{message}")
        self.status.setWordWrap(True)
        self.status.setObjectName("statusError")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self._set_busy(False)
