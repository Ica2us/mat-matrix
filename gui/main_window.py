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
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # type: ignore[attr-defined]
from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
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
from storage.db_engine import DuckDBStorageEngine
from storage.vcs import MerkleDAGVersionControl
from gui.markdown_render import md_to_html
from gui.worker import Worker, WorkerPool
from gui.table_view import ConditionFormatTable
from gui.scatter_canvas import HoverScatterCanvas
from gui.pareto_canvas import ParetoCanvas
from gui.staging_panel import StagingPanel

from gui.in_situ_widget import InSituVideoCharacterizationWidget


# =====================================================================
# 核心融入：基于 Pandas + 启发式规则的智能列类型推理与数据治理引擎
# =====================================================================
class ImportTypeInferenceEngine:
    """
    工业级物料数据特征空间多维启发式推理引擎。
    不看表头，依据数值区间自动推断成分占比与工艺/响应参数，并执行守恒约束校验。
    """
    @staticmethod
    def infer_and_validate(df: pd.DataFrame) -> Dict[str, Any]:
        # 仅提取数值型列进行统计解构
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        skip = {"exp_id", "expid", "id", "sample", "batch", "pareto_front"}
        numeric_cols = [c for c in numeric_cols if c.lower().strip() not in skip and "id" not in c.lower()]

        inferred_comps = []
        inferred_others = []

        # ---- 阶段 1: 扫描数值特征矩与有界空间边界 ----
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            
            c_min, c_max = series.min(), series.max()
            col_lower = col.lower()

            # 强暗示判定：若列名强暗示为温度/时间/压力等物理工艺量，优先划入非成分类
            if any(k in col_lower for k in ["temp", "time", "press", "load", "curing", "param"]):
                inferred_others.append(col)
                continue

            # 规则 A: 小数制成分判定 —— 数值严格落在 [0.0, 1.005] 内，且具备连续分布特征
            if 0.0 <= c_min and c_max <= 1.005:
                if series.nunique() > 2 or (c_min != c_max):  # 排除全 0/1 的常数或布尔控制列
                    inferred_comps.append(col)
                else:
                    inferred_others.append(col)
            
            # 规则 B: 百分制成分判定 —— 数值落在 [0.0, 100.005] 内
            elif 0.0 <= c_min and c_max <= 100.005:
                # 排除可能在该区间波动的高危工艺/物理响应参数（如强度、成本等）
                if any(k in col_lower for k in ["target", "strength", "elong", "yield", "强度", "成本", "cost", "tensile"]):
                    inferred_others.append(col)
                else:
                    inferred_comps.append(col)
            else:
                inferred_others.append(col)

        # ---- 阶段 2: 验证单纯形守恒空间（Simplex Closure Check） ----
        is_percentage = False
        needs_normalization = False
        scale_factor = 1.0

        if inferred_comps:
            row_sums = df[inferred_comps].sum(axis=1)
            mean_sum = row_sums.mean()
            
            # 动态判别当前体系是标准小数制(和为1)还是工业百分制(和为100%)
            if 80.0 <= mean_sum <= 120.0:
                is_percentage = True
                scale_factor = 100.0
            else:
                is_percentage = False
                scale_factor = 1.0

            # 计算各样本行偏离物理守恒界线的绝对误差，若任一行偏离绝对值 > 1e-3 则判定需要归一化
            deviations = np.abs(row_sums - scale_factor)
            if (deviations > 1e-3).any():
                needs_normalization = True

        return {
            "comps": inferred_comps,
            "others": inferred_others,
            "is_percentage": is_percentage,
            "needs_normalization": needs_normalization,
            "scale_factor": scale_factor
        }


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
        """基础保底猜测逻辑（高级智能识别已由导入区引擎接管）。"""
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

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self._note_text = self._note_edit.toPlainText()
            html = md_to_html(self._note_text)
            self._note_preview.setHtml(html)

    def _on_save_note(self) -> None:
        self._note_text = self._note_edit.toPlainText()
        if not self._note_text.strip():
            return
        self._btn_save_note.setEnabled(False)
        self._btn_save_note.setText("✅ 已保存")
        self._btn_save_note.setEnabled(True)

    def load_note(self, note_text: str) -> None:
        self._note_text = note_text
        self._note_edit.setPlainText(note_text)

    def update_metadata(self, commit_id: str, data: Dict[str, str]) -> None:
        """更新元数据区域显示内容。"""
        if "实验 ID" in data:
            self._meta_labels["实验 ID"].setText(f"<b>实验 ID:</b> {data['实验 ID']}")
        if "操作人" in data:
            self._meta_labels["操作人"].setText(f"<b>操作人:</b> {data['操作人']}")
        if commit_id:
            self._meta_labels["Commit"].setText(
                f"<b>Commit:</b> <code>{commit_id[:16]}...</code>"
            )

    def update_anomaly(
        self, df: pd.DataFrame, target_cols: List[str]
    ) -> None:
        """运行异常检测并更新诊断区域显示。"""
        try:
            numeric_cols = [c for c in target_cols if c in df.columns]
            if not numeric_cols:
                return
            _, mahal_sq = self._analytics.robust_anomaly_detection(
                df, comp_cols=[], numeric_cols=numeric_cols,
            )
            mean_md = float(np.mean(mahal_sq))
            max_md = float(np.max(mahal_sq))
            md95 = float(np.percentile(mahal_sq, 95))
            self._diag_status.setText(
                f"MCD 距离: 均值={mean_md:.2f}, "
                f"P95={md95:.2f}, "
                f"最大={max_md:.2f} "
                f"{'(⚠ 含异常)' if max_md > 10 else '(正常)'}"
            )
        except Exception:
            self._diag_status.setText("MCD 距离: — (计算失败)")


# =====================================================================
# 主窗口大闸调度器
# =====================================================================

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
        self._insitu_widget: Optional[InSituVideoCharacterizationWidget] = None

        # ── 菜单 ──
        self._setup_menu()

        # ── UI ──
        self._build_ui()

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

    def _setup_menu(self) -> None:
        mb = self.menuBar()

        # 📁 数据
        data_menu = mb.addMenu("📁 数据")
        commit_action = QAction("🚀 提交暂存区 (Commit)", self)
        commit_action.setShortcut(QKeySequence("Ctrl+Return"))
        commit_action.triggered.connect(self._on_menu_commit)
        data_menu.addAction(commit_action)

        calibrate_action = QAction("📊 批次偏移校准", self)
        calibrate_action.triggered.connect(self._on_menu_calibrate)
        data_menu.addAction(calibrate_action)

        # 👁 视图
        view_menu = mb.addMenu("👁 视图")
        view_menu.addAction("表格视图 (聚焦左侧)")
        view_menu.addAction("重置布局")

    def _on_menu_commit(self) -> None:
        self.staging_panel._on_commit()

    def _on_menu_calibrate(self) -> None:
        QMessageBox.information(
            self, "批次偏移校准",
            "校准功能正在开发中",
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── 工具栏 ──
        toolbar = QHBoxLayout()
        self.btn_import = QPushButton("📁 导入并推理 CSV")
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

        # Tab 5: 原位表征视频
        self._insitu_widget = InSituVideoCharacterizationWidget(
            table_view_reference=self.data_table.table_view,
            main_dataframe=pd.DataFrame(),
        )
        self.tabs.addTab(self._insitu_widget, "📹 原位表征视频")

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

        self.scatter_canvas.selectionChanged.connect(self._on_scatter_selection)

    def _on_scatter_selection(self, indices: List[int]) -> None:
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

        if self._data is not None:
            self.right_panel.update_metadata(
                commit_id=commit_id,
                data={"实验 ID": f"#{len(self._data.df)}", "操作人": "系统"},
            )
            if target_cols:
                self.right_panel.update_anomaly(self._data.df, target_cols)

        metrics = data["metrics"]
        labels = [str(i) for i in range(metrics.shape[0])]
        if target_cols:
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
    # 核心深度融合修改点：智能化启发式导入与治理
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

        # 🛠️ 唤醒核心启发式推理机
        inference = ImportTypeInferenceEngine.infer_and_validate(df)
        comps = inference["comps"]
        others = inference["others"]
        is_pct = inference["is_percentage"]
        needs_norm = inference["needs_normalization"]
        scale = inference["scale_factor"]

        # ⚖️ 组分单纯形不守恒条件弹窗拦截
        if comps and needs_norm:
            unit_str = "100.0%" if is_pct else "1.0"
            reply = QMessageBox.question(
                self, 
                "⚖️ 组分单纯形不封闭弹窗核验", 
                f"检测到自动推理出的组分列 {comps} 在物理样本行中的均值加和不等于标准的物料守恒常数 ({unit_str})。\n\n"
                f"是否启动系统内置的 [单纯形规整化投影算法] 自动进行全自动按行归一化对齐？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                row_sums = df[comps].sum(axis=1).replace(0, 1.0)
                for col in comps:
                    df[col] = (df[col] / row_sums) * scale

        # 🔄 百分制自动优雅向下兼容降维转换（0~100% 映射至 0~1）
        if comps and is_pct:
            reply_conv = QMessageBox.question(
                self,
                "🔄 进制归一化自动转换",
                "当前导入的成分列判定为百分制形式（0~100%）。为了完美兼容内核主动学习优化器，"
                "是否自动将其平移规整为小数占比制（0~1）？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply_conv == QMessageBox.StandardButton.Yes:
                for col in comps:
                    df[col] = df[col] / 100.0

        # 装载进数据容器，并强行对齐列特征映射
        self._data = ExperimentData(df)
        self._data.comp_cols = comps

        # 细化工艺参数类与响应目标类的自适应判定划分
        param_cols = []
        target_cols = []
        for c in others:
            cl = c.lower().strip()
            if any(k in cl for k in ("param", "temp", "time", "press", "curing", "load")):
                param_cols.append(c)
            elif any(k in cl for k in ("tensile", "strength", "cost", "target", "强度", "成本", "yield", "elong")):
                target_cols.append(c)
            else:
                # 默认保底判定
                if pd.api.types.is_float_dtype(df[c]):
                    param_cols.append(c)
                else:
                    target_cols.append(c)

        # 极端不均衡状况下的保底均分策略
        if not param_cols and others:
            param_cols = others[:max(1, len(others)//2)]
            target_cols = others[len(param_cols):]
        elif not target_cols and others:
            target_cols = others

        self._data.param_cols = param_cols
        self._data.target_cols = target_cols
        self._data.target_directions = ["maximize"] * len(target_cols)

        # 激活 downstream UI 画布渲染生命周期
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

        # ── 原位表征视频 ──
        if self._insitu_widget is not None:
            self._insitu_widget.set_dataframe(df)

        # ── 右侧面板: 元数据 ──
        if data.target_cols:
            self.right_panel.update_anomaly(df, data.target_cols)

        self._update_status(
            f"已加载 {len(df)} 条实验 | "
            f"智能识别组分: {len(data.comp_cols)}列 | "
            f"工艺工艺: {len(data.param_cols)}列 | "
            f"目标指标: {len(data.target_cols)}列"
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
        if self._insitu_widget is not None:
            self._insitu_widget.cleanup()
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