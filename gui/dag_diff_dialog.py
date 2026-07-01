"""
gui/dag_diff_dialog.py — 三栏式跨分支差异对比弹窗
====================================================

布局
----
左栏:     共同祖先 (Merge Base) 参数
中栏:     Branch-A 参数 (红色标注变更项)
右栏:     Branch-B 参数 (绿色标注新增项)
底部:     Matplotlib 光谱/散点图对比

用法
----
    dialog = DagDiffDialog(vcs, commit_a_id, commit_b_id, parent=window)
    dialog.exec()
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── 颜色常量 ──────────────────────────────────────────────────
COLOR_ADDED = QColor(200, 255, 200)     # 浅绿
COLOR_REMOVED = QColor(255, 200, 200)   # 浅红
COLOR_CHANGED = QColor(255, 230, 180)   # 浅橙
COLOR_BASE = QColor(240, 240, 245)      # 浅灰蓝（基准）
COLOR_HEADER = QColor(50, 50, 60)
TEXT_LIGHT = QColor(255, 255, 255)
TEXT_DARK = QColor(30, 30, 30)


def _make_header_item(text: str, color: QColor = COLOR_HEADER) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setBackground(color)
    item.setForeground(TEXT_LIGHT)
    font = QFont("Consolas", 10, QFont.Weight.Bold)
    item.setFont(font)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def _make_value_item(text: str, bg: Optional[QColor] = None) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    item.setFont(QFont("Consolas", 9))
    item.setForeground(TEXT_DARK)
    if bg:
        item.setBackground(bg)
    return item


# =====================================================================
# 三栏 Diff 对话框
# =====================================================================

class DagDiffDialog(QDialog):
    """三栏式差异对比弹窗。

    Parameters
    ----------
    vcs : MerkleDAGVersionControl
        已初始化的 VCS 引擎
    commit_a : str
        第一个 commit ID
    commit_b : str
        第二个 commit ID
    parent : QWidget or None
    """

    def __init__(
        self,
        vcs: Any,
        commit_a: str,
        commit_b: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("🔄 跨分支差异对比")
        self.setMinimumSize(1000, 650)
        self.resize(1200, 750)

        self._vcs = vcs
        self._commit_a = commit_a
        self._commit_b = commit_b

        # ── 获取数据 ──────────────────────────────────────────
        try:
            self._meta_a = vcs.get_commit(commit_a)
            self._meta_b = vcs.get_commit(commit_b)
            self._merge_base_id = vcs.find_merge_base(commit_a, commit_b)
            self._meta_base = vcs.get_commit(self._merge_base_id)
        except Exception as exc:
            self._meta_a = {"error": str(exc), "parameters": {}, "sop_sequence": []}
            self._meta_b = {"error": str(exc), "parameters": {}, "sop_sequence": []}
            self._meta_base = {"parameters": {}, "sop_sequence": []}
            self._merge_base_id = "N/A"

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 标题行 ────────────────────────────────────────────
        title = QLabel(
            f"🔄 对比: {self._commit_a[:16]}...  ←  {self._commit_b[:16]}...\n"
            f"   共同祖先 (Merge Base): {self._merge_base_id[:16]}..."
        )
        title.setStyleSheet("font-size: 13px; padding: 4px;")
        layout.addWidget(title)

        # ── 三栏表格 ──────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左栏: 共同祖先
        base_widget = self._make_param_table(
            "🧬 共同祖先 (基准线)", self._meta_base, is_base=True,
        )
        splitter.addWidget(base_widget)

        # 中栏: Branch A
        a_widget = self._make_param_table(
            f"🔴 分支 A: {self._commit_a[:12]}...", self._meta_a, is_base=False,
        )
        splitter.addWidget(a_widget)

        # 右栏: Branch B
        b_widget = self._make_param_table(
            f"🟢 分支 B: {self._commit_b[:12]}...", self._meta_b, is_base=False,
        )
        splitter.addWidget(b_widget)

        layout.addWidget(splitter, 3)

        # ── 底部: 元数据差异文本 ──────────────────────────────
        meta_tabs = QTabWidget()
        self._diff_text = QTextEdit()
        self._diff_text.setReadOnly(True)
        self._diff_text.setFont(QFont("Consolas", 9))
        self._populate_diff_text()
        meta_tabs.addTab(self._diff_text, "📄 差异摘要")

        self._sop_text = QTextEdit()
        self._sop_text.setReadOnly(True)
        self._sop_text.setFont(QFont("Consolas", 9))
        self._populate_sop_diff()
        meta_tabs.addTab(self._sop_text, "📋 SOP 序列差异")

        layout.addWidget(meta_tabs, 1)

        # ── 关闭按钮 ──────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("✕ 关闭")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    # -----------------------------------------------------------------
    # 参数表格生成
    # -----------------------------------------------------------------

    def _make_param_table(self, title: str, meta: Dict[str, Any], is_base: bool) -> QWidget:
        """Build a QTableWidget showing one commit's parameters."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)

        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
        layout.addWidget(label)

        params = meta.get("parameters", {})
        table = QTableWidget(len(params), 2)
        table.setHorizontalHeaderLabels(["参数", "值"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

        for i, (k, v) in enumerate(sorted(params.items())):
            key_item = _make_value_item(k)
            val_item = _make_value_item(v)
            if is_base:
                key_item.setBackground(COLOR_BASE)
                val_item.setBackground(COLOR_BASE)
            table.setItem(i, 0, key_item)
            table.setItem(i, 1, val_item)

        table.resizeColumnsToContents()
        layout.addWidget(table, 1)
        return widget

    # -----------------------------------------------------------------
    # 差异文本
    # -----------------------------------------------------------------

    def _populate_diff_text(self) -> None:
        """Generate a text summary of the parameter differences."""
        params_a = self._meta_a.get("parameters", {})
        params_b = self._meta_b.get("parameters", {})
        all_keys = sorted(set(params_a) | set(params_b))

        lines = [
            f"差异对比: {self._commit_a[:16]}...  →  {self._commit_b[:16]}...",
            f"共同祖先: {self._merge_base_id[:16]}...",
            "",
        ]

        if "error" in self._meta_a:
            lines.append(f"⚠ 错误: {self._meta_a['error']}")
            self._diff_text.setPlainText("\n".join(lines))
            return

        has_diff = False
        for k in all_keys:
            va = params_a.get(k, "—")
            vb = params_b.get(k, "—")
            if str(va) != str(vb):
                has_diff = True
                lines.append(f"  🔶 {k}:")
                lines.append(f"       A: {va}")
                lines.append(f"       B: {vb}")
                lines.append("")

        if not has_diff:
            lines.append("两个分支的参数完全相同，无差异。")

        # Metadata diff
        lines.append("── 元数据 ──")
        for field in ("author", "message", "branch", "equipment"):
            va = self._meta_a.get(field, "")
            vb = self._meta_b.get(field, "")
            if va != vb:
                lines.append(f"  {field}: '{va}' → '{vb}'")
            else:
                lines.append(f"  {field}: '{va}' (相同)")

        self._diff_text.setPlainText("\n".join(lines))

    def _populate_sop_diff(self) -> None:
        """Generate SOP sequence diff using VCS's LCS method."""
        seq_a = self._meta_a.get("sop_sequence", [])
        seq_b = self._meta_b.get("sop_sequence", [])
        diff = self._vcs.calculate_sop_diff(seq_a, seq_b)

        lines = [
            "SOP 序列差异对比",
            f"  A 长度: {len(seq_a)}   B 长度: {len(seq_b)}",
            f"  共同步骤: {len(diff['common'])}",
            "",
        ]
        if diff.get("only_in_a"):
            lines.append("── 仅存在于 A ──")
            for s in diff["only_in_a"]:
                lines.append(f"  ─ [{s}]")
        if diff.get("only_in_b"):
            lines.append("── 仅存在于 B ──")
            for s in diff["only_in_b"]:
                lines.append(f"  ┼ [{s}]")
        if not diff.get("only_in_a") and not diff.get("only_in_b"):
            lines.append("SOP 序列完全一致。")

        self._sop_text.setPlainText("\n".join(lines))


# =====================================================================
# 入口
# =====================================================================

def main() -> None:
    """Standalone test."""
    app = QApplication(sys.argv)

    from storage.vcs import MerkleDAGVersionControl
    vcs = MerkleDAGVersionControl()

    # Seed test data
    vcs.commit_version(None, ["init"], {"T": 100, "t": 10}, "hash0",
                       author="alice", message="root", branch="main")
    head = vcs.list_commits(limit=1)[0]["commit_id"]
    vcs.commit_version([head], ["step_a"], {"T": 150, "t": 30}, "hash1",
                       author="bob", message="branch experiment", branch="探索分支")

    commits = vcs.list_commits(branch=None, limit=10)
    if len(commits) >= 2:
        win = QMainWindow()
        win.setWindowTitle("Diff Dialog Test")
        dialog = DagDiffDialog(vcs, commits[0]["commit_id"], commits[1]["commit_id"], parent=win)
        dialog.exec()


if __name__ == "__main__":
    main()