from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from qsn_model.constants import CIFAR10_CLASSES


class _BarRow(QWidget):
    def __init__(self, label: str, color: str = "#3daee9"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        self.name.setMinimumWidth(88)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
        )
        self.value = QLabel("0.00%")
        self.value.setMinimumWidth(132)
        layout.addWidget(self.name)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.value)

    def set_fraction(self, fraction: float, suffix: str | None = None) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        self.bar.setValue(round(fraction * 1000))
        self.value.setText(suffix if suffix is not None else f"{fraction * 100:.2f}%")


class ClassScoreChart(QGroupBox):
    """Ten-class softmax probabilities; the model logits are left untouched."""

    def __init__(self):
        super().__init__("Softmax Probability — CIFAR-10 Classes")
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        self.rows = {}
        for class_name in CIFAR10_CLASSES:
            row = _BarRow(class_name)
            self.rows[class_name] = row
            layout.addWidget(row)

    def set_scores(self, scores: dict[str, float]) -> None:
        for class_name, row in self.rows.items():
            row.set_fraction(scores.get(class_name, 0.0))


class SpikeDistributionWidget(QGroupBox):
    COLORS = {-1: "#ef5350", 0: "#78909c", 1: "#66bb6a", 2: "#ab47bc"}

    def __init__(self):
        super().__init__("Quaternary Spike Distribution")
        layout = QVBoxLayout(self)
        self.scope = QLabel("統計範圍：單次 inference 的所有 Time-step 與所有 LIF 層累計")
        self.scope.setWordWrap(True)
        self.scope.setObjectName("mutedText")
        layout.addWidget(self.scope)
        self.rows = {}
        for value in (-1, 0, 1, 2):
            row = _BarRow(f"Spike {value}", self.COLORS[value])
            self.rows[value] = row
            layout.addWidget(row)
        rates = QFormLayout()
        self.nonzero_rate = QLabel("-")
        self.zero_rate = QLabel("-")
        rates.addRow("Non-zero Spike Rate", self.nonzero_rate)
        rates.addRow("Zero Spike Rate", self.zero_rate)
        layout.addLayout(rates)

    def set_statistics(self, counts: dict[int, int]) -> None:
        total = sum(int(counts.get(value, 0)) for value in (-1, 0, 1, 2))
        for value, row in self.rows.items():
            count = int(counts.get(value, 0))
            fraction = count / total if total else 0.0
            row.set_fraction(fraction, f"{count:,}  {fraction * 100:.2f}%")
        zero = int(counts.get(0, 0))
        zero_fraction = zero / total if total else 0.0
        self.zero_rate.setText(f"{zero_fraction * 100:.2f}%")
        self.nonzero_rate.setText(
            f"{(1.0 - zero_fraction) * 100:.2f}%" if total else "0.00%"
        )
