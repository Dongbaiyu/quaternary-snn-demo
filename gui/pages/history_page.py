from __future__ import annotations

import csv

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qsn_model.constants import PROJECT_ROOT

FIELDS = (
    "timestamp",
    "image_index",
    "true_label",
    "prediction",
    "top3",
    "checkpoint",
    "time_steps",
    "inference_time",
    "correct",
)


class HistoryPage(QWidget):
    def __init__(self):
        super().__init__()
        self.records: list[dict] = []
        self.history_path = PROJECT_ROOT / "results" / "inference_history.csv"
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        export_button = QPushButton("匯出 CSV")
        export_button.clicked.connect(self.export_csv)
        buttons.addWidget(export_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.table = QTableWidget(0, len(FIELDS))
        self.table.setHorizontalHeaderLabels(FIELDS)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self._load_history()

    def _load_history(self) -> None:
        if not self.history_path.is_file():
            return
        with self.history_path.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                self._append_row(record)

    def _append_row(self, record: dict) -> None:
        self.records.append(record)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, field in enumerate(FIELDS):
            self.table.setItem(row, column, QTableWidgetItem(str(record.get(field, ""))))

    def add_record(self, record: dict) -> None:
        self._append_row(record)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.history_path.exists() or self.history_path.stat().st_size == 0
        with self.history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def export_csv(self) -> None:
        if not self.records:
            QMessageBox.information(self, "無資料", "尚無推論紀錄可匯出。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出推論紀錄", "inference_log.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._write_csv(path)
        except OSError as exc:
            self.status_label.setText(f"匯出失敗：{exc}")
            self.status_label.setObjectName("statusError")
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
            return
        self.status_label.setText(f"匯出完成：{path}")
        self.status_label.setObjectName("statusReady")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        QMessageBox.information(self, "匯出完成", f"已儲存至：{path}")

    def _write_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.records)
