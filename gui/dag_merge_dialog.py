"""
gui/dag_merge_dialog.py — 合并冲突解决向导
==============================================

交互流程
--------
1. 用户选择源节点 (Source) 和目标节点 (Target)
2. 系统读取两个节点的 parameters
3. 显示参数穿梭框：
   - 左侧：Source 参数（勾选要继承的）
   - 右侧：Target 参数（勾选要继承的）
   - 冲突行（同一参数两者都改了且数值不同）：爆红，强制二选一或手动输入
4. 确认后生成双亲 commit

用法
----
    dialog = DagMergeDialog(vcs, source_id, target_id, parent=window)
    if dialog.exec():
        new_commit_id = dialog.merged_commit_id()
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use("QtAgg")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

COLOR_CONFLICT = QColor(255, 200, 200)       # 冲突爆红
COLOR_CONFLICT_SELECTED = QColor(255, 180, 180)
COLOR_SELECTED = QColor(200, 240, 200)        # 勾选绿色
COLOR_UNSELECTED = QColor(245, 245, 245)      # 未勾选灰


# =====================================================================
# 合并对话框
# =====================================================================

class DagMergeDialog(QDialog):
    """参数穿梭框 + 冲突高亮 + 生成双亲节点。

    Attributes
    ----------
    merged_commit_id : str or None
        合并成功后设置的 commit ID。
    """

    def __init__(
        self,
        vcs: Any,
        source_id: str,
        target_id: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("🔗 合并 & 参数穿梭")
        self.setMinimumSize(800, 550)
        self.resize(900, 600)

        self._vcs = vcs
        self._source_id = source_id
        self._target_id = target_id
        self.merged_commit_id: Optional[str] = None

        # 加载元数据
        self._meta_src = vcs.get_commit(source_id)
        self._meta_tgt = vcs.get_commit(target_id)
        self._params_src = self._meta_src.get("parameters", {})
        self._params_tgt = self._meta_tgt.get("parameters", {})

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 标题 ──────────────────────────────────────────────
        title = QLabel(
            f"🔗 合并: {self._source_id[:16]}...  →  {self._target_id[:16]}..."
        )
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        desc = QLabel(
            "勾选要继承到合并结果的参数。\n"
            "红色行 = 冲突 — 两个分支修改了同一参数且数值不同，请二选一或手动输入。"
        )
        desc.setStyleSheet("color: #555; padding-bottom: 6px;")
        layout.addWidget(desc)

        # ── 参数穿梭表 ────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "继承", "参数名", "Source 值", "Target 值", "合并后值"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._populate_table()
        layout.addWidget(self._table, 1)

        # ── 作者 & 消息 ───────────────────────────────────────
        form = QFormLayout()
        self._author_edit = QLineEdit()
        self._author_edit.setPlaceholderText("操作人")
        form.addRow("作者:", self._author_edit)

        self._message_edit = QTextEdit()
        self._message_edit.setPlaceholderText("合并消息...")
        self._message_edit.setMaximumHeight(60)
        form.addRow("合并消息:", self._message_edit)
        layout.addLayout(form)

        # ── 按钮 ──────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_table(self) -> None:
        """Fill the parameter table with source/target values and conflict markers."""
        all_keys = sorted(set(self._params_src) | set(self._params_tgt))
        self._table.setRowCount(len(all_keys))
        self._checkbox_items: List[QCheckBox] = []
        self._conflict_rows: List[int] = []

        for i, k in enumerate(all_keys):
            vs = self._params_src.get(k, "")
            vt = self._params_tgt.get(k, "")

            # ── 继承复选框 ────────────────────────────────────
            cb = QCheckBox()
            cb.setChecked(True)
            self._checkbox_items.append(cb)
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(i, 0, cb_widget)
            cb.toggled.connect(lambda _, row=i: self._on_check_toggle(row))

            # ── 参数名 ────────────────────────────────────────
            name_item = QTableWidgetItem(k)
            name_item.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            self._table.setItem(i, 1, name_item)

            # ── Source 值 ─────────────────────────────────────
            src_item = QTableWidgetItem(str(vs))
            src_item.setFont(QFont("Consolas", 9))
            self._table.setItem(i, 2, src_item)

            # ── Target 值 ────────────────────────────────────
            tgt_item = QTableWidgetItem(str(vt))
            tgt_item.setFont(QFont("Consolas", 9))
            self._table.setItem(i, 3, tgt_item)

            # ── 合并后值（可编辑） ─────────────────────────────
            final_val = vt if vt != "" else vs
            final_item = QTableWidgetItem(str(final_val))
            final_item.setFont(QFont("Consolas", 9))
            final_item.setFlags(final_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 4, final_item)

            # ── 冲突检测 ──────────────────────────────────────
            if vs and vt and str(vs) != str(vt):
                self._conflict_rows.append(i)
                name_item.setBackground(COLOR_CONFLICT)
                src_item.setBackground(COLOR_CONFLICT)
                tgt_item.setBackground(COLOR_CONFLICT)
                final_item.setBackground(COLOR_CONFLICT)

        self._table.resizeColumnsToContents()

        # Show conflict count
        if self._conflict_rows:
            QMessageBox.information(
                self, "冲突检测",
                f"发现 {len(self._conflict_rows)} 个参数冲突，已标红，请处理。",
            )

    # -----------------------------------------------------------------
    # handlers
    # -----------------------------------------------------------------

    def _on_check_toggle(self, row: int) -> None:
        """Gray out un-checked rows."""
        for col in range(1, 5):
            item = self._table.item(row, col)
            if item:
                item.setBackground(
                    COLOR_UNSELECTED if not self._checkbox_items[row].isChecked()
                    else QColor(255, 255, 255)
                )

    def _on_accept(self) -> None:
        """Validate, create merge commit, set self.merged_commit_id."""
        # Collect merged parameters
        merged_params: Dict[str, Any] = {}
        for i in range(self._table.rowCount()):
            cb_widget = self._table.cellWidget(i, 0)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                key = self._table.item(i, 1).text()
                val = self._table.item(i, 4).text()
                # Try to preserve numeric types
                try:
                    if "." in val:
                        merged_params[key] = float(val)
                    else:
                        merged_params[key] = int(val)
                except ValueError:
                    merged_params[key] = val

        message = self._message_edit.toPlainText().strip() or \
            f"Merge: {self._source_id[:12]} ←→ {self._target_id[:12]}"
        author = self._author_edit.text().strip() or "merge-bot"

        # Merge SOP sequences
        sop_a = self._meta_src.get("sop_sequence", [])
        sop_b = self._meta_tgt.get("sop_sequence", [])
        merged_sop = self._vcs.calculate_sop_diff(sop_a, sop_b)
        sop_result = merged_sop.get("common", []) + \
            merged_sop.get("only_in_a", []) + \
            merged_sop.get("only_in_b", [])

        # Create merge commit with two parents
        # Use data_file_hash from target (or source as fallback)
        data_hash = self._meta_tgt.get("data_file_hash") or \
            self._meta_src.get("data_file_hash", "merge-placeholder")

        self.merged_commit_id = self._vcs.commit_version(
            parent_ids=[self._source_id, self._target_id],
            sop_sequence=sop_result,
            parameters=merged_params,
            data_file_hash=data_hash,
            author=author,
            message=message,
            equipment="merge",
            branch=self._meta_tgt.get("branch", "main"),
        )

        self.accept()


# =====================================================================
# 入口
# =====================================================================

def main() -> None:
    """Standalone test."""
    app = QApplication(sys.argv)

    from storage.vcs import MerkleDAGVersionControl
    vcs = MerkleDAGVersionControl()

    # Seed branches
    vcs.commit_version(None, ["init"], {"T": 100, "t": 10, "P": 50}, "hash0",
                       author="alice", message="root", branch="main")
    head = vcs.list_commits(limit=1)[0]["commit_id"]

    # Branch A
    vcs.commit_version([head], ["step_a"], {"T": 150, "t": 30, "P": 55}, "hash1",
                       author="bob", message="branch A", branch="branch-A")
    a_head = vcs.list_commits(branch="branch-A", limit=1)[0]["commit_id"]

    # Branch B (conflict on T)
    vcs.commit_version([head], ["step_b"], {"T": 200, "t": 45, "Q": 80}, "hash2",
                       author="carol", message="branch B", branch="branch-B")
    b_head = vcs.list_commits(branch="branch-B", limit=1)[0]["commit_id"]

    win = QMainWindow()
    win.setWindowTitle("Merge Dialog Test")
    dialog = DagMergeDialog(vcs, a_head, b_head, parent=win)
    if dialog.exec():
        print(f"Merged commit: {dialog.merged_commit_id[:16]}...")
        print(vcs.get_commit(dialog.merged_commit_id))


if __name__ == "__main__":
    main()