"""
StagingPanel — a QTableView widget for staging data rows and committing to storage+VCS.

Provides a checkbox column for row selection, a "Staging" label with checked count,
and a "Commit" button that persists checked rows via DuckDBStorageEngine and
registers the commit via MerkleDAGVersionControl.

Can be run standalone with ``python -m gui.staging_panel``.
"""

import hashlib
import sys
from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt, QModelIndex, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from storage import DuckDBStorageEngine, MerkleDAGVersionControl


# ---------------------------------------------------------------------------
# staging model / proxy helpers
# ---------------------------------------------------------------------------


def _sha256_of_dataframe(df: pd.DataFrame) -> str:
    """Compute SHA-256 of a DataFrame serialised as Parquet bytes."""
    raw = df.to_parquet(index=False)
    return hashlib.sha256(raw).hexdigest()


def _build_model(df: pd.DataFrame) -> QStandardItemModel:
    """Build a QStandardItemModel with a checkbox column on the left.

    Each row starts unchecked.
    """
    nrows, ncols = df.shape
    model = QStandardItemModel(nrows, ncols + 1)

    # headers: checkbox column + original column names
    headers = [""] + list(df.columns)
    model.setHorizontalHeaderLabels(headers)

    for r in range(nrows):
        # checkbox item
        check_item = QStandardItem()
        check_item.setCheckState(Qt.CheckState.Unchecked)
        check_item.setCheckable(True)
        model.setItem(r, 0, check_item)

        for c in range(ncols):
            val = df.iat[r, c]
            item = QStandardItem()
            # display as string; non-string cells get repr
            if isinstance(val, (int, float)):
                item.setData(val, Qt.ItemDataRole.DisplayRole)
            else:
                item.setText(str(val))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            model.setItem(r, c + 1, item)

    return model


# ---------------------------------------------------------------------------
# StagingPanel
# ---------------------------------------------------------------------------


class StagingPanel(QWidget):
    """A QTableView with checkbox-based staging and a Commit button.

    Signals
    -------
    commit_success(commit_id: str)
        Emitted after a successful commit, carrying the commit hash.
    """

    commit_success = Signal(str)

    def __init__(
        self,
        storage: Optional[DuckDBStorageEngine] = None,
        vcs: Optional[MerkleDAGVersionControl] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._current_branch = "main"

        # Allow caller to inject dependencies; fall back to defaults.
        self._storage = storage if storage is not None else DuckDBStorageEngine()
        self._vcs = vcs if vcs is not None else MerkleDAGVersionControl()

        # ── widgets ──────────────────────────────────────────────
        self._label = QLabel("Staging: 0")
        self._table = QTableView()
        self._commit_btn = QPushButton("🚀 Commit")
        self._commit_btn.setStyleSheet("background-color: #2b579a; color: white; font-weight: bold;")
        self._branch_label = QLabel("🌿 main")
        self._branch_label.setStyleSheet("font-family: Consolas; background: #f0f0f0; padding: 2px 6px; border-radius: 3px;")

        # ── layout ────────────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.addWidget(self._label)
        top_bar.addStretch()
        top_bar.addWidget(self._branch_label)
        top_bar.addWidget(self._commit_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._table)

        # ── connections ────────────────────────────────────────────
        self._commit_btn.clicked.connect(self._on_commit)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def set_data(self, df: pd.DataFrame) -> None:
        """Load a DataFrame into the table view.

        Replaces any previously loaded data.
        """
        self._df = df.copy()
        model = _build_model(df)
        self._table.setModel(model)

        # visual polish
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)

        # connect checkbox change tracking
        model.dataChanged.connect(self._on_data_changed)
        self._update_label()

    def get_staged_data(self) -> pd.DataFrame:
        """Return the subset of rows that are currently checked."""
        if self._df is None:
            return pd.DataFrame()
        model = self._table.model()
        if model is None:
            return pd.DataFrame()
        if not isinstance(model, QStandardItemModel):
            return pd.DataFrame()
        mask = self._checked_mask(model)
        return self._df.loc[mask].reset_index(drop=True)

    @property
    def current_branch(self) -> str:
        """The branch name used for the last (or next) commit."""
        return self._current_branch

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _checked_mask(self, model: QStandardItemModel) -> list[bool]:
        return [
            bool(model.item(r, 0).checkState() == Qt.CheckState.Checked)
            for r in range(model.rowCount())
        ]

    def _update_label(self) -> None:
        model = self._table.model()
        if model is None:
            self._label.setText("Staging: 0")
            return
        if not isinstance(model, QStandardItemModel):
            self._label.setText("Staging: 0")
            return
        count = sum(self._checked_mask(model))
        self._label.setText(f"Staging: {count}")

    def _on_data_changed(self, _top_left: QModelIndex, _bottom_right: QModelIndex) -> None:
        self._update_label()

    def _on_commit(self) -> None:
        """Collect checked rows, show commit dialog, save to storage, commit to VCS."""
        df = self.get_staged_data()
        if df.empty:
            return

        # ── 提交对话框 ────────────────────────────────────────
        dialog = QDialog(self)
        dialog.setWindowTitle("提交暂存区")
        dialog.setMinimumWidth(420)
        form = QFormLayout(dialog)

        author_edit = QLineEdit()
        author_edit.setPlaceholderText("操作人姓名")
        form.addRow("作者:", author_edit)

        message_edit = QTextEdit()
        message_edit.setPlaceholderText("描述本次变更...")
        message_edit.setMaximumHeight(100)
        form.addRow("提交消息:", message_edit)

        equipment_edit = QLineEdit()
        equipment_edit.setPlaceholderText("如 SEM-2 / 炉A")
        form.addRow("设备:", equipment_edit)

        branch_edit = QLineEdit()
        branch_edit.setText(self._current_branch)
        form.addRow("分支:", branch_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        author = author_edit.text().strip()
        message = message_edit.toPlainText().strip()
        equipment = equipment_edit.text().strip()
        branch = branch_edit.text().strip() or "main"
        self._current_branch = branch
        self._branch_label.setText(f"🌿 {branch}")

        data_hash = self._storage.save_dataframe(df)

        commit_id = self._vcs.commit_version(
            parent_ids=None,
            sop_sequence=["staging_ingest"],
            parameters={"n_rows": len(df), "columns": list(df.columns)},
            data_file_hash=data_hash,
            author=author,
            message=message,
            equipment=equipment,
            branch=branch,
        )

        self.commit_success.emit(commit_id)


# ---------------------------------------------------------------------------
# standalone test harness
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone test: create a window with a StagingPanel and sample data."""
    app = QApplication(sys.argv)

    # Sample DataFrame
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "composition": ["A-0.3/B-0.7", "A-0.5/B-0.5", "A-0.8/B-0.2", "A-0.1/B-0.9"],
        "temp_K": [300, 350, 400, 450],
    })

    panel = StagingPanel()
    panel.set_data(df)

    def _on_commit(commit_id: str) -> None:
        sys.stdout.write(f"Commit succeeded: {commit_id}\n")

    panel.commit_success.connect(_on_commit)

    window = QMainWindow()
    window.setWindowTitle("Staging Panel — Standalone Test")
    window.setCentralWidget(panel)
    window.resize(600, 400)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
