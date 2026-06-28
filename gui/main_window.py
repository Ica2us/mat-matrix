"""
mat_matrix_gui — 实验员决策仪表盘

4 个 Tab:
  1. 条件格式表格 — 红/绿渐变 + 排序 + 行选中联动
  2. 散点图 — 悬停 Tooltip + 矩形框选
  3. Pareto 前沿 — 快速非支配排序 + 前沿面绘制
  4. 趋势 + 推荐 — 历史走势 + 系统推荐下一组实验

左侧: 条件格式表格（数据总览）
右侧: 智能详情面板（选中行元数据 + 异常诊断）
底部: 暂存区（勾选 → 提交归档）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.analytics import RobustAnalyticsEngine
from engine.bo import MaterialBayesianOptimizer
from engine.pareto_fast import fast_non_dominated_sort
from storage.db_engine import DuckDBStorageEngine
from storage.vcs import MerkleDAGVersionControl
from gui.markdown_render import md_to_html
from gui.worker import Worker, WorkerPool
from gui.table_view import ConditionFormatTable
from gui.scatter_canvas import HoverScatterCanvas
from gui.pareto_canvas import ParetoCanvas
from gui.staging_panel import StagingPanel


# =====================================================================
# 实验数据持有对象（跨线程传递）
# =====================================================================

class ExperimentData:
    """封装一份实验数据及其列角色映射。"""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.comp_cols: List[str] = []
        self.param_cols: List[str] = []
        self.target_cols: List[str] = []
        self.target_directions: List[str] = []

    @property
    def all_var_cols(self) -> List[str]:
        return self.comp_cols + self.param_cols

    def guess_column_roles(self) -> None:
        """启发式猜测列角色（基于列名和数据类型）。"""
        cols = list(self.df.columns)
        skip = {"exp_id", "expid", "id", "sample", "batch", "date", "备注", "note"}
        comp, param, target = [], [], []

        for c in cols:
            cl = c.lower().strip()
            if cl in skip or "id" in cl:
                continue
            if any(k in cl for k in ("resin", "filler", "pct_", "comp", "ratio")):
                comp.append(c)
            elif any(k in cl for k in ("tensile", "strength", "cost", "curing", "target", "强度", "成本")):
                target.append(c)
            elif pd.api.types.is_float_dtype(self.df[c]):
                param.append(c)

        self.comp_cols = comp
        self.param_cols = param
        self.target_cols = target
        self.target_directions = ["maximize"] * len(target)


# =====================================================================
# 异常诊断水平条形图
# =====================================================================

class AnomalyBarChart(FigureCanvasQTAgg):
    """水平条形图：各维度对异常得分的贡献度。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self.fig = Figure(figsize=(2.8, 2.2), dpi=90)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)  # type: ignore[no-untyped-call]
        self.setParent(parent)

    def plot_contributions(self, scores: Dict[str, float]) -> None:
        """绘制水平条形图。scores: {'温度': 0.68, '压力': 0.22, ...}"""
        self.axes.clear()

        labels = list(scores.keys())
        values = list(scores.values())
        total = sum(values) or 1
        pcts = [v / total * 100 for v in values]

        bars = self.axes.barh(labels, pcts, color="#d9534f", height=0.6)
        for bar, pct in zip(bars, pcts):
            self.axes.text(
                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%",
                va="center", fontsize=8,
            )

        self.axes.set_xlim(0, max(pcts) * 1.3 if pcts else 100)
        self.axes.tick_params(labelsize=8)
        self.axes.set_title("维度贡献度", fontsize=9)
        self.fig.tight_layout()
        self.draw_idle()  # type: ignore[no-untyped-call]


# =====================================================================
# 右侧详情面板
# =====================================================================

class RightDetailPanel(QWidget):
    """右侧智能详情与诊断面板：元数据 + 异常诊断 + 实验注释。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._analytics = RobustAnalyticsEngine()
        self._note_text = ""
        self._current_commit_id: Optional[str] = None
        self._current_parent_ids: List[str] = []
        self.setFixedWidth(300)
        self._build_ui()

    # ── UI 构建 ──

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ---- 元数据区 ----
        meta_group = QGroupBox("📋 元数据")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setSpacing(2)
        self._meta_labels: Dict[str, QLabel] = {}
        for key, default in [
            ("实验 ID", "#1024"),
            ("操作人", "张三"),
            ("时间", "2026-06-28"),
            ("设备", "SEM-2"),
            ("Commit", "—"),
            ("父版本", "—"),
        ]:
            lbl = QLabel(f"<b>{key}:</b> {default}")
            lbl.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
            if key in ("Commit", "父版本"):
                lbl.setStyleSheet("font-family: Consolas; background: #f0f0f0; padding: 2px 4px;")
            self._meta_labels[key] = lbl
            meta_layout.addWidget(lbl)
        meta_group.setMaximumHeight(180)
        layout.addWidget(meta_group)

        # ---- 异常诊断区 ----
        diag_group = QGroupBox("🔴 异常诊断")
        diag_layout = QVBoxLayout(diag_group)
        diag_layout.setSpacing(2)
        self._diag_status = QLabel("MCD 距离: — (无异常)")
        self._diag_status.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        diag_layout.addWidget(self._diag_status)
        self._anomaly_chart = AnomalyBarChart()
        diag_layout.addWidget(self._anomaly_chart)
        self._anomaly_chart.setVisible(False)
        diag_group.setMaximumHeight(260)
        layout.addWidget(diag_group)

        # ---- 实验注释区 ----
        note_group = QGroupBox("✏️ 实验注释")
        note_layout = QVBoxLayout(note_group)
        note_layout.setSpacing(4)

        self._note_tabs = QTabWidget()
        self._note_edit = QTextEdit()
        self._note_edit.setPlaceholderText(
            "支持 Markdown 语法:\n"
            "# 标题\n"
            "**粗体** *斜体* `代码`\n"
            "- 列表项\n"
            "[链接](url) ![图片](url)"
        )
        self._note_preview = QTextEdit()
        self._note_preview.setReadOnly(True)

        self._note_tabs.addTab(self._note_edit, "编辑")
        self._note_tabs.addTab(self._note_preview, "预览")
        self._note_tabs.currentChanged.connect(self._on_tab_changed)
        note_layout.addWidget(self._note_tabs)

        self._btn_save_note = QPushButton("💾 保存注释")
        self._btn_save_note.clicked.connect(self._on_save_note)
        note_layout.addWidget(self._btn_save_note)

        layout.addWidget(note_group, 1)

    # ── 元数据更新 ──

    def update_metadata(
        self,
        commit_id: str,
        parent_ids: Optional[List[str]] = None,
        data: Optional[Dict[str, str]] = None,
    ) -> None:
        """更新元数据标签。"""
        self._current_commit_id = commit_id
        self._current_parent_ids = parent_ids or []

        commit_short = commit_id[:8] if commit_id else "—"
        parent_short = self._current_parent_ids[0][:8] if self._current_parent_ids else "(root)"

        self._meta_labels["Commit"].setText(f"<b>Commit:</b> {commit_short}")
        self._meta_labels["父版本"].setText(f"<b>父版本:</b> {parent_short}")

        if data:
            for key, val in data.items():
                if key in self._meta_labels:
                    self._meta_labels[key].setText(f"<b>{key}:</b> {val}")

    # ── 异常诊断更新 ──

    def update_anomaly(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """计算异常检测并更新诊断区。"""
        if df.empty or not numeric_cols:
            self._diag_status.setText("MCD 距离: — (无数据)")
            self._anomaly_chart.setVisible(False)
            return

        analytics = RobustAnalyticsEngine()
        mask, distances = analytics.robust_anomaly_detection(df, [], numeric_cols)
        has_anomaly = bool(mask.any())

        if has_anomaly:
            max_dist = float(distances.max())
            self._diag_status.setText(
                f'<span style="color:red">MCD 距离: {max_dist:.2f} (离群)</span>'
            )

            # 计算各维度贡献度（Z-score 均值）
            numeric_data = df[numeric_cols].values
            means = np.nanmean(numeric_data, axis=0)
            stds = np.nanstd(numeric_data, axis=0)
            stds = np.where(stds == 0, 1e-12, stds)
            z_scores = np.abs((numeric_data - means) / stds)
            avg_z = z_scores.mean(axis=0)
            scores: Dict[str, float] = {col: float(z) for col, z in zip(numeric_cols, avg_z)}

            self._anomaly_chart.plot_contributions(scores)
            self._anomaly_chart.setVisible(True)
        else:
            avg_dist = float(distances.mean())
            self._diag_status.setText(f"MCD 距离: {avg_dist:.2f} (正常)")
            self._anomaly_chart.setVisible(False)

    # ── 注释功能 ──

    def _on_tab_changed(self, index: int) -> None:
        """切换编辑/预览标签时更新预览内容。"""
        if index == 1:
            self._note_text = self._note_edit.toPlainText()
            html = md_to_html(self._note_text)
            self._note_preview.setHtml(html)

    def _on_save_note(self) -> None:
        """保存注释（直接更新，简化版本）。"""
        self._note_text = self._note_edit.toPlainText()
        if not self._note_text.strip():
            return
        self._btn_save_note.setEnabled(False)
        self._btn_save_note.setText("✅ 已保存")
        self._btn_save_note.setEnabled(True)

    def load_note(self, note_text: str) -> None:
        """加载历史注释。"""
        self._note_text = note_text
        self._note_edit.setPlainText(note_text)

class MainWindow(QMainWindow):
    """实验员决策仪表盘主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("⚛ 材料研发数据系统")
        self.resize(1400, 900)

        # ── 数据 ──
        self._data: Optional[ExperimentData] = None

        # ── 后端引擎（惰性初始化） ──
        self._analytics: Optional[RobustAnalyticsEngine] = None
        self._optimizer: Optional[MaterialBayesianOptimizer] = None
        self._storage: Optional[DuckDBStorageEngine] = None
        self._vcs: Optional[MerkleDAGVersionControl] = None

        # ── UI ──
        self._build_ui()

    # -----------------------------------------------------------------
    # 引擎惰性初始化
    # -----------------------------------------------------------------

    def _get_analytics(self) -> RobustAnalyticsEngine:
        if self._analytics is None:
            self._analytics = RobustAnalyticsEngine()
        return self._analytics

    def _get_optimizer(self) -> MaterialBayesianOptimizer:
        if self._optimizer is None:
            self._optimizer = MaterialBayesianOptimizer()
        return self._optimizer

    def _get_storage(self) -> DuckDBStorageEngine:
        if self._storage is None:
            self._storage = DuckDBStorageEngine(data_dir="v/data")
        return self._storage

    def _get_vcs(self) -> MerkleDAGVersionControl:
        if self._vcs is None:
            self._vcs = MerkleDAGVersionControl(index_dir="v")
        return self._vcs

    # -----------------------------------------------------------------
    # UI 构建
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── 工具栏 ──
        toolbar = QHBoxLayout()
        self.btn_import = QPushButton("📁 导入 CSV")
        self.btn_import.setMinimumHeight(36)
        self.btn_import.clicked.connect(self._on_import)
        toolbar.addWidget(self.btn_import)

        self.btn_run = QPushButton("▶ 运行全管线")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_pipeline)
        toolbar.addWidget(self.btn_run)

        toolbar.addStretch()

        # 列配置下拉
        toolbar.addWidget(QLabel("组分:"))
        self.comp_combo = QComboBox()
        self.comp_combo.setMinimumWidth(120)
        self.comp_combo.setEnabled(False)
        toolbar.addWidget(self.comp_combo)
        toolbar.addWidget(QLabel("目标:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(120)
        self.target_combo.setEnabled(False)
        toolbar.addWidget(self.target_combo)

        main_layout.addLayout(toolbar)

        # ── 水平分割: 左侧表格 | 中央 Tab + 右侧详情 ──
        h_split = QSplitter(Qt.Orientation.Horizontal)

        # 左侧: 条件格式表格
        self.data_table = ConditionFormatTable()
        h_split.addWidget(self.data_table)

        # 中央: Tab 视图
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        center_layout.addWidget(self.tabs, 1)

        # Tab 1: 散点图
        self.scatter_tab = QWidget()
        self._build_scatter_tab()
        self.tabs.addTab(self.scatter_tab, "📉 散点图")

        # Tab 2: Pareto 前沿
        self.pareto_tab = QWidget()
        self._build_pareto_tab()
        self.tabs.addTab(self.pareto_tab, "🎯 Pareto 权衡")

        # Tab 3: 历史趋势
        self.trend_tab = QWidget()
        self._build_trend_tab()
        self.tabs.addTab(self.trend_tab, "📈 历史趋势")

        # Tab 4: 推荐 + 异常
        self.recommend_tab = QWidget()
        self._build_recommend_tab()
        self.tabs.addTab(self.recommend_tab, "🧠 下一步推荐")

        h_split.addWidget(center_widget)

        # 右侧: 智能详情面板
        self.right_panel = RightDetailPanel()
        h_split.addWidget(self.right_panel)

        h_split.setSizes([400, 600, 300])
        main_layout.addWidget(h_split, 1)

        # ── 底部: 暂存区 ──
        self.staging_panel = StagingPanel()
        self.staging_panel.setMaximumHeight(200)
        self.staging_panel.commit_success.connect(self._on_staging_commit)
        main_layout.addWidget(self.staging_panel)

        # ── 状态栏 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)

        self._update_status("请导入 CSV 数据文件")

    # ---------------------------------------------------------
    # 右侧详情面板（移除旧的简单实现）
    # ---------------------------------------------------------
    # Tab 1: 散点图
    # ---------------------------------------------------------

    def _build_scatter_tab(self) -> None:
        layout = QVBoxLayout(self.scatter_tab)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("X 轴:"))
        self.scatter_x = QComboBox()
        self.scatter_x.setMinimumWidth(160)
        controls.addWidget(self.scatter_x)
        controls.addWidget(QLabel("Y 轴:"))
        self.scatter_y = QComboBox()
        self.scatter_y.setMinimumWidth(160)
        controls.addWidget(self.scatter_y)
        controls.addStretch()

        self.scatter_x.currentIndexChanged.connect(self._on_scatter_change)
        self.scatter_y.currentIndexChanged.connect(self._on_scatter_change)

        layout.addLayout(controls)
        self.scatter_canvas = HoverScatterCanvas()
        nav = NavigationToolbar2QT(self.scatter_canvas, parent=None)  # type: ignore[no-untyped-call]
        layout.addWidget(nav)
        layout.addWidget(self.scatter_canvas, 1)

        # 框选联动 → 表格高亮
        self.scatter_canvas.selectionChanged.connect(self._on_scatter_selection)

    def _on_scatter_selection(self, indices: List[int]) -> None:
        """散点框选后，在右侧详情面板显示选中点信息。"""
        if self._data is None or not indices:
            self.right_panel.load_note("")
            return
        df = self._data.df
        text = f"选中 {len(indices)} 个点\n\n"
        for idx in indices[:5]:
            row = df.iloc[idx]
            text += f"  #{idx}: {row.to_dict()}\n"
        if len(indices) > 5:
            text += f"  ... 还有 {len(indices)-5} 个\n"
        self.right_panel.load_note(text)

    def _on_scatter_change(self) -> None:
        if self._data is None:
            return
        x_col = self.scatter_x.currentText()
        y_col = self.scatter_y.currentText()
        if not x_col or not y_col:
            return
        self.scatter_canvas.set_data(self._data.df, x_col, y_col)

    # ---------------------------------------------------------
    # Tab 2: Pareto 前沿
    # ---------------------------------------------------------

    def _build_pareto_tab(self) -> None:
        layout = QVBoxLayout(self.pareto_tab)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("目标 1:"))
        self.pareto_obj1 = QComboBox()
        self.pareto_obj1.setMinimumWidth(150)
        controls.addWidget(self.pareto_obj1)
        self.pareto_dir1 = QComboBox()
        self.pareto_dir1.addItems(["maximize", "minimize"])
        controls.addWidget(self.pareto_dir1)

        controls.addWidget(QLabel("目标 2:"))
        self.pareto_obj2 = QComboBox()
        self.pareto_obj2.setMinimumWidth(150)
        controls.addWidget(self.pareto_obj2)
        self.pareto_dir2 = QComboBox()
        self.pareto_dir2.addItems(["maximize", "minimize"])
        controls.addWidget(self.pareto_dir2)

        self.btn_plot_pareto = QPushButton("绘制 Pareto")
        self.btn_plot_pareto.clicked.connect(self._on_plot_pareto)
        controls.addWidget(self.btn_plot_pareto)
        controls.addStretch()

        layout.addLayout(controls)
        self.pareto_canvas = ParetoCanvas()
        nav = NavigationToolbar2QT(self.pareto_canvas, parent=None)  # type: ignore[no-untyped-call]
        layout.addWidget(nav)
        layout.addWidget(self.pareto_canvas, 1)

        self.pareto_canvas.point_clicked.connect(self._on_pareto_point_clicked)

    def _on_pareto_point_clicked(self, idx: int) -> None:
        """点击 Pareto 点 → 右侧详情显示该实验数据。"""
        if self._data is None:
            return
        df = self._data.df
        if idx < len(df):
            row = df.iloc[idx]
            text = f"Pareto 点 #{idx}\n\n"
            for k, v in row.items():
                text += f"  {k}: {v}\n"
            self.right_panel.load_note(text)

    def _on_plot_pareto(self) -> None:
        if self._data is None:
            return
        o1 = self.pareto_obj1.currentText()
        o2 = self.pareto_obj2.currentText()
        d1 = self.pareto_dir1.currentText()
        d2 = self.pareto_dir2.currentText()
        if not o1 or not o2:
            return

        metrics = self._data.df[[o1, o2]].values.astype(np.float64)
        directions = [d1, d2]
        labels = self._data.df.index.astype(str).tolist()

        self.pareto_canvas.set_data(metrics, directions, labels)

    # ---------------------------------------------------------
    # Tab 3: 历史趋势
    # ---------------------------------------------------------

    def _build_trend_tab(self) -> None:
        layout = QVBoxLayout(self.trend_tab)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("目标值:"))
        self.trend_target = QComboBox()
        self.trend_target.setMinimumWidth(150)
        controls.addWidget(self.trend_target)
        controls.addWidget(QLabel("最近 N 次:"))
        self.trend_n = QComboBox()
        self.trend_n.addItems(["全部", "10", "20", "30", "50"])
        self.trend_n.setCurrentText("20")
        controls.addWidget(self.trend_n)
        controls.addStretch()

        self.trend_target.currentIndexChanged.connect(self._on_trend_change)
        self.trend_n.currentIndexChanged.connect(self._on_trend_change)

        layout.addLayout(controls)
        self.trend_canvas = ParetoCanvas()
        nav = NavigationToolbar2QT(self.trend_canvas, parent=None)  # type: ignore[no-untyped-call]
        layout.addWidget(nav)
        layout.addWidget(self.trend_canvas, 1)

    def _on_trend_change(self) -> None:
        if self._data is None:
            return
        col = self.trend_target.currentText()
        n_text = self.trend_n.currentText()
        if not col:
            return

        df = self._data.df
        if n_text == "全部":
            values = df[col].values
        else:
            n = int(n_text)
            values = df[col].values[-n:]

        # 用 matplotlib 直接画折线
        canvas = self.trend_canvas
        canvas.ax.clear()
        x_arr = np.arange(len(values))
        vals_float: np.ndarray = np.asarray(values, dtype=np.float64)
        canvas.ax.plot(x_arr, vals_float, "b-o", markersize=4, linewidth=1.5)
        mean_val = float(np.mean(vals_float))
        canvas.ax.axhline(y=mean_val, color="r", linestyle="--", alpha=0.5, label="均值")
        canvas.ax.set_xlabel("实验序号")
        canvas.ax.set_ylabel(col)
        canvas.ax.set_title(f"{col} — 历史趋势（最近 {len(values)} 次）")
        canvas.ax.legend()
        canvas.ax.grid(True, alpha=0.3)
        canvas.draw_idle()  # type: ignore[no-untyped-call]

    # ---------------------------------------------------------
    # Tab 4: 推荐 + 异常
    # ---------------------------------------------------------

    def _build_recommend_tab(self) -> None:
        layout = QVBoxLayout(self.recommend_tab)

        controls = QHBoxLayout()
        self.btn_recommend = QPushButton("🎯 推荐下一实验")
        self.btn_recommend.setMinimumHeight(40)
        self.btn_recommend.clicked.connect(self._on_recommend)
        controls.addWidget(self.btn_recommend)
        controls.addStretch()
        layout.addLayout(controls)

        result_area = QHBoxLayout()

        left_panel = QVBoxLayout()
        self.rec_text = QTextEdit()
        self.rec_text.setReadOnly(True)
        self.rec_text.setMaximumHeight(250)
        font = QFont("Consolas", 11)
        self.rec_text.setFont(font)
        left_panel.addWidget(QLabel("推荐参数:"))
        left_panel.addWidget(self.rec_text)

        self.anomaly_text = QTextEdit()
        self.anomaly_text.setReadOnly(True)
        self.anomaly_text.setMaximumHeight(250)
        left_panel.addWidget(QLabel("异常警告:"))
        left_panel.addWidget(self.anomaly_text)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        result_area.addWidget(left_widget, 1)

        self.rec_canvas = ParetoCanvas()
        result_area.addWidget(self.rec_canvas, 1)

        layout.addLayout(result_area, 1)

        self.vcs_text = QTextEdit()
        self.vcs_text.setReadOnly(True)
        self.vcs_text.setMaximumHeight(60)
        layout.addWidget(QLabel("版本记录:"))
        layout.addWidget(self.vcs_text)

    def _on_recommend(self) -> None:
        if self._data is None:
            return
        self._set_buttons_enabled(False)
        self.statusBar().showMessage("正在运行推荐管线...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        WorkerPool.submit(
            fn=self._do_recommend,
            on_result=self._on_recommend_result,
            on_error=self._on_error,
            on_progress=self.progress_bar.setValue,
            on_finished=self._on_recommend_finished,
        )

    def _do_recommend(self, progress_callback: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        data = cast(ExperimentData, self._data)
        df = data.df
        comp_cols = data.comp_cols
        param_cols = data.param_cols
        target_cols = data.target_cols
        directions = data.target_directions

        if not comp_cols and not param_cols:
            raise ValueError("请先配置组分列和工艺参数列")
        if not target_cols:
            raise ValueError("请先配置目标列")

        def _notify(v: int) -> None:
            if progress_callback:
                progress_callback(v)

        _notify(10)

        analytics = self._get_analytics()
        anomaly_mask, mahal_sq = analytics.robust_anomaly_detection(
            df, comp_cols=comp_cols, numeric_cols=target_cols,
        )

        _notify(30)

        optimizer = self._get_optimizer()
        metrics = df[target_cols].values
        pareto_mask = optimizer.calculate_pareto_front(metrics, directions)

        _notify(50)

        comp_bounds = {c: (float(df[c].min()), float(df[c].max())) for c in comp_cols}
        process_bounds = {p: (float(df[p].min()), float(df[p].max())) for p in param_cols}
        recommendation = optimizer.recommend_next_experiment(
            historical_df=df,
            comp_cols=comp_cols,
            param_cols=param_cols,
            target_cols=target_cols,
            comp_bounds=comp_bounds,
            process_bounds=process_bounds,
            target_directions=directions,
        )

        _notify(70)

        storage = self._get_storage()
        data_hash = storage.save_dataframe(df)

        vcs = self._get_vcs()
        sop_seq = comp_cols + param_cols + target_cols
        commit_id = vcs.commit_version(
            parent_ids=None,
            sop_sequence=sop_seq,
            parameters={
                "comp_cols": comp_cols,
                "param_cols": param_cols,
                "target_cols": target_cols,
                "target_directions": directions,
            },
            data_file_hash=data_hash,
        )

        _notify(85)

        anomaly_info: Dict[int, Dict[str, Any]] = {}
        if anomaly_mask.sum() > 0:
            try:
                anomaly_info = analytics.anomaly_explain(
                    df, comp_cols=comp_cols, numeric_cols=target_cols, top_k=3,
                )
            except Exception:
                anomaly_info = {}

        vcs.commit_version(
            parent_ids=[commit_id],
            sop_sequence=sop_seq,
            parameters={"recommendation": {k: round(v, 4) for k, v in recommendation.items()}},
            data_file_hash=data_hash,
        )

        _notify(100)

        return {
            "metrics": metrics,
            "pareto_mask": pareto_mask,
            "recommendation": recommendation,
            "target_cols": target_cols,
            "anomaly_mask": anomaly_mask,
            "anomaly_info": anomaly_info,
            "commit_id": commit_id,
            "comp_cols": comp_cols,
            "target_directions": directions,
        }

    def _on_recommend_result(self, data: Dict[str, Any]) -> None:
        rec = data["recommendation"]
        target_cols = data["target_cols"]
        anomaly_mask = data["anomaly_mask"]
        anomaly_info = data["anomaly_info"]
        commit_id = data["commit_id"]

        lines = []
        for k, v in rec.items():
            lines.append(f"  {k}: {v:.4f}")
        self.rec_text.setPlainText("\n".join(lines))

        n_anom = int(anomaly_mask.sum())
        if n_anom > 0:
            lines = [f"⚠ 发现 {n_anom} 个异常样本:\n"]
            for idx, info in anomaly_info.items():
                top = info.get("top_features", [])
                lines.append(f"  样本 #{idx}: {', '.join(top[:3])}")
            self.anomaly_text.setPlainText("\n".join(lines))
        else:
            self.anomaly_text.setPlainText("✅ 未发现异常样本")

        # ── 右侧面板更新 ──
        if self._data is not None:
            self.right_panel.update_metadata(
                commit_id=commit_id,
                data={"实验 ID": f"#{len(self._data.df)}", "操作人": "系统"},
            )
            if target_cols:
                self.right_panel.update_anomaly(self._data.df, target_cols)

        metrics = data["metrics"]
        mask = data["pareto_mask"]
        labels = [str(i) for i in range(metrics.shape[0])]
        if target_cols:
            # 只取前两个目标画 Pareto（ParetoCanvas 要求 2D）
            if metrics.shape[1] >= 2:
                d1 = data.get("target_directions", ["maximize"] * 2)[0]
                d2 = data.get("target_directions", ["maximize"] * 2)[1]
                self.rec_canvas.set_data(metrics[:, :2], [d1, d2], labels)
                self.rec_canvas.setVisible(True)
            else:
                self.rec_canvas.setVisible(False)

        self.vcs_text.setPlainText(
            f"Commit: {commit_id[:16]}... | "
            f"异常: {n_anom} 个 | "
            f"数据已保存到 v/data/"
        )
        self.statusBar().showMessage(f"推荐完成 | Commit {commit_id[:12]}...", 8000)

    def _on_recommend_finished(self) -> None:
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self._set_buttons_enabled(True)

    # ---------------------------------------------------------
    # 数据导入
    # ---------------------------------------------------------

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 数据文件", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError("CSV 文件为空")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"无法读取文件:\n{e}")
            return

        self._data = ExperimentData(df)
        self._data.guess_column_roles()
        self._on_data_loaded()

    def _on_data_loaded(self) -> None:
        data = self._data
        if data is None:
            return
        df = data.df
        all_cols = list(df.columns)

        # ── 左侧表格 ──
        self.data_table.set_data(df)

        # ── 下拉框 ──
        for combo in [self.scatter_x, self.scatter_y, self.trend_target,
                       self.pareto_obj1, self.pareto_obj2]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(all_cols)
            combo.blockSignals(False)

        # 列配置快捷键
        self.comp_combo.blockSignals(True)
        self.comp_combo.clear()
        self.comp_combo.addItems(all_cols)
        self.comp_combo.setEnabled(True)
        self.comp_combo.blockSignals(False)

        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(all_cols)
        self.target_combo.setEnabled(True)
        self.target_combo.blockSignals(False)

        # 智能默认选择
        if data.target_cols:
            if data.comp_cols:
                self.scatter_x.setCurrentText(data.comp_cols[0])
            self.scatter_y.setCurrentText(data.target_cols[0])
            self.trend_target.setCurrentText(data.target_cols[0])
            if len(data.target_cols) >= 2:
                self.pareto_obj1.setCurrentText(data.target_cols[0])
                self.pareto_obj2.setCurrentText(data.target_cols[1])
        elif len(all_cols) >= 2:
            self.scatter_x.setCurrentIndex(0)
            self.scatter_y.setCurrentIndex(1)
            self.trend_target.setCurrentIndex(0)

        self._on_scatter_change()

        self.btn_run.setEnabled(True)

        # ── 暂存区 ──
        self.staging_panel.set_data(df)

        # ── 右侧面板: 元数据 ──
        if data.target_cols:
            self.right_panel.update_anomaly(df, data.target_cols)

        self._update_status(
            f"已加载 {len(df)} 条实验 | "
            f"组分: {len(data.comp_cols)}列 | "
            f"工艺: {len(data.param_cols)}列 | "
            f"目标: {len(data.target_cols)}列"
        )

    # ---------------------------------------------------------
    # 暂存区提交回调
    # ---------------------------------------------------------

    def _on_staging_commit(self, commit_id: str) -> None:
        self.statusBar().showMessage(f"暂存提交成功: {commit_id[:16]}...", 5000)

    # ---------------------------------------------------------
    # 全管线运行
    # ---------------------------------------------------------

    def _on_run_pipeline(self) -> None:
        self.tabs.setCurrentIndex(3)
        self._on_recommend()

    # ---------------------------------------------------------
    # 通用
    # ---------------------------------------------------------

    def _on_error(self, exc_info: Tuple[Any, ...]) -> None:
        _type, msg, tb = exc_info
        self.statusBar().clearMessage()
        self.progress_bar.hide()
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "错误", str(msg))

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.btn_import.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)

    def _update_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def closeEvent(self, event: Any) -> None:
        pool = QThreadPool.globalInstance()
        pool.clear()
        if not pool.waitForDone(3000):
            print("警告: 线程池未在 3 秒内完成，强制退出")
        event.accept()


# =====================================================================
# 入口
# =====================================================================

def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()