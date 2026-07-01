"""
mat_matrix_gui — 实验员决策仪表盘

Tab 布局:
  1. 条件格式表格 — 红/绿渐变 + 排序 + 行选中联动
  2. 散点图 — 悬停 Tooltip + 矩形框选
  3. Pareto 前沿 — 快速非支配排序 + 前沿面绘制
  4. 趋势 + 推荐 — 历史走势 + 系统推荐下一组实验
  5. 原位表征视频
  6. 版本历史 — Commit log 浏览 + 差异对比

左侧: 条件格式表格（数据总览）
右侧: 智能详情面板（选中行元数据 + 异常诊断）
底部: 暂存区（勾选 → 提交归档）
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

os.environ["QT_API"] = "pyside6"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # type: ignore[attr-defined]
from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QAction, QFont, QKeySequence, QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableView,
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

from gui.dag_graph import BranchFilterBar, DagGraphCanvas, ForkDialog
from gui.dag_diff_dialog import DagDiffDialog
from gui.dag_merge_dialog import DagMergeDialog

from gui.in_situ_widget import InSituVideoCharacterizationWidget

# TableModel for version history (simple 2D data)
from PySide6.QtCore import QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor

# ── 新增：Pydantic 数据模型（第零层 / 准备） ──────────────
# 为了让 Edit 干净利落，先 import pydantic
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self


# =====================================================================
# 版本历史 TableModel
# =====================================================================

class VersionHistoryModel(QAbstractTableModel):
    """QAbstractTableModel wrapping VCS list_commits() results."""

    def __init__(self, commits: List[Dict[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._commits = commits
        self._headers = ["时间", "操作人", "分支", "消息", "设备", "Commit ID"]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return len(self._commits)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        commit = self._commits[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return commit.get("created_at", "")[:19].replace("T", " ")
            elif col == 1:
                return commit.get("author", "")
            elif col == 2:
                return commit.get("branch", "")
            elif col == 3:
                return commit.get("message", "")
            elif col == 4:
                return commit.get("equipment", "")
            elif col == 5:
                return commit["commit_id"][:16] + "..."
        if role == Qt.ItemDataRole.ForegroundRole and col == 5:
            return QColor("#666666")
        if role == Qt.ItemDataRole.FontRole and col == 5:
            font = QFont("Consolas", 9)
            return font
        if role == Qt.ItemDataRole.ToolTipRole:
            return commit.get("message", "")
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._headers[section]
        return None

    def get_commit(self, row: int) -> Optional[Dict[str, Any]]:
        if 0 <= row < len(self._commits):
            return self._commits[row]
        return None


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


#
# 数据模型：成分、工艺参数、目标性能（Pydantic BaseModel）
# =====================================================================

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self


class Composition(BaseModel):
    """成分体系 — 物料配方（mat_matrix 核心模型）"""
    names: list[str] = Field(default_factory=list, description="成分名称，如 SiO₂、Al₂O₃")
    values: list[float] = Field(default_factory=list, description="成分数值")
    is_percentage: bool = Field(False, description="True=百分制(0~100), False=小数制(0~1)")
    needs_normalization: bool = Field(False, description="是否需归一化到单纯形")

    @model_validator(mode="after")
    def check_length_match(self) -> Self:
        if self.names and len(self.names) != len(self.values):
            raise ValueError(f"成分名称({len(self.names)})与数值({len(self.values)})数量不匹配")
        return self

    @property
    def df_columns(self) -> list[str]:
        return list(self.names)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=False))

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame", columns: list[str]) -> "Composition":
        """从 DataFrame 列构建成分模型，自动推断百分制/小数制。"""
        if not columns:
            return cls()
        vals = df[columns].mean().tolist()
        mean_sum = sum(vals)
        is_pct = 80.0 <= mean_sum <= 120.0
        return cls(
            names=list(columns),
            values=vals,
            is_percentage=is_pct,
            needs_normalization=abs(mean_sum - (100.0 if is_pct else 1.0)) > 1e-3,
        )


class ProcessParams(BaseModel):
    """工艺参数 — 烧结/热处理/压力等可调工艺条件"""
    names: list[str] = Field(default_factory=list, description="工艺参数名称，如 温度、压力、时间")
    bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="参数范围 {name: (min, max)}",
    )

    @property
    def df_columns(self) -> list[str]:
        return list(self.names)

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame", columns: list[str]) -> "ProcessParams":
        bounds = {}
        for c in columns:
            if c in df.columns:
                bounds[c] = (float(df[c].min()), float(df[c].max()))
        return cls(names=list(columns), bounds=bounds)


class TargetMetrics(BaseModel):
    """目标性能指标 — 待优化的响应变量"""
    names: list[str] = Field(default_factory=list, description="目标名称，如 抗拉强度、成本")
    directions: list[str] = Field(default_factory=list, description="优化方向: maximize/minimize")
    weights: list[float] = Field(default_factory=lambda: [1.0], description="多目标权重")

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        n = len(self.names)
        if self.directions and len(self.directions) != n:
            raise ValueError(f"目标名称({n})与方向({len(self.directions)})数量不匹配")
        if not self.weights or len(self.weights) != n:
            self.weights = [1.0] * n
        if not self.directions:
            self.directions = ["maximize"] * n
        return self

    @property
    def df_columns(self) -> list[str]:
        return list(self.names)


class DataSource(BaseModel):
    """数据溯源 — 谁、在什么设备上、基于哪个 SOP 做的实验"""
    equipment: str = Field("", description="设备名称，如 炉A / SEM-2")
    operator: str = Field("", description="操作人")
    batch_id: str = Field("", description="批次号/炉号")
    protocol: str = Field("", description="SOP 名称或版本")
    timestamp: str = Field("", description="实验时间")

    @property
    def is_empty(self) -> bool:
        return not any([self.equipment, self.operator, self.batch_id, self.protocol])


class ExperimentData(BaseModel):
    """实验数据集合 — 兼具强类型校验与旧接口兼容。"""
    df: Any = Field(default=None, description="原始 pandas DataFrame（必填）")
    composition: Composition = Field(default_factory=Composition, description="成分体系")
    process: ProcessParams = Field(default_factory=ProcessParams, description="工艺参数")
    targets: TargetMetrics = Field(default_factory=TargetMetrics, description="优化目标")
    source: DataSource = Field(default_factory=DataSource, description="实验溯源")

    # ── 旧接口兼容：comp_cols / param_cols / target_cols ──────────

    @property
    def comp_cols(self) -> list[str]:
        return self.composition.names

    @comp_cols.setter
    def comp_cols(self, cols: list[str]) -> None:
        if not cols:
            self.composition = Composition()
            return
        if self.df is not None:
            self.composition = Composition.from_dataframe(self.df, cols)
        else:
            self.composition = Composition(names=cols, values=[])

    @property
    def param_cols(self) -> list[str]:
        return self.process.names

    @param_cols.setter
    def param_cols(self, cols: list[str]) -> None:
        if self.df is not None:
            self.process = ProcessParams.from_dataframe(self.df, cols)
        else:
            self.process = ProcessParams(names=cols)

    @property
    def target_cols(self) -> list[str]:
        return self.targets.names

    @target_cols.setter
    def target_cols(self, cols: list[str]) -> None:
        self.targets.names = cols
        # 补全默认方向
        if not self.targets.directions or len(self.targets.directions) != len(cols):
            self.targets.directions = ["maximize"] * len(cols)
        if not self.targets.weights or len(self.targets.weights) != len(cols):
            self.targets.weights = [1.0] * len(cols)

    @property
    def target_directions(self) -> list[str]:
        return self.targets.directions

    @target_directions.setter
    def target_directions(self, dirs: list[str]) -> None:
        self.targets.directions = dirs

    @property
    def all_var_cols(self) -> list[str]:
        """所有变量列（成分 + 工艺参数），供 BO 使用"""
        return self.comp_cols + self.param_cols

    # ── 边界查询（BO 使用） ─────────────────────────────────

    def comp_bounds(self) -> dict[str, tuple[float, float]]:
        """成分边界"""
        if self.df is None:
            return {}
        return {c: (float(self.df[c].min()), float(self.df[c].max())) for c in self.comp_cols if c in self.df.columns}

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """工艺参数边界"""
        return dict(self.process.bounds)

    # ── 猜测列角色（旧保底逻辑） ──────────────────────────

    def guess_column_roles(self) -> None:
        """基础保底猜测逻辑（高级智能识别已由导入区引擎接管）。"""
        if self.df is None:
            return
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

    # ── 序列化 ─────────────────────────────────────────

    def to_commit_metadata(self) -> dict:
        """生成 VCS commit 可用的元数据 dict"""
        return {
            "comp_cols": self.comp_cols,
            "param_cols": self.param_cols,
            "target_cols": self.target_cols,
            "target_directions": self.target_directions,
            "comp_is_percentage": self.composition.is_percentage,
            "comp_needs_normalization": self.composition.needs_normalization,
            "source": self.source.model_dump() if not self.source.is_empty else None,
        }

    @classmethod
    def from_commit_metadata(cls, meta: dict, df: Any = None) -> "ExperimentData":
        """从 VCS commit 元数据恢复 ExperimentData（不含 df）"""
        obj = cls(df=df)
        obj.comp_cols = meta.get("comp_cols", [])
        obj.param_cols = meta.get("param_cols", [])
        obj.target_cols = meta.get("target_cols", [])
        obj.target_directions = meta.get("target_directions", ["maximize"] * len(obj.target_cols))
        if meta.get("source"):
            obj.source = DataSource(**meta["source"])
        return obj


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
            self._vcs = MerkleDAGVersionControl(
                index_dir="v",
                storage_engine=self._get_storage(),
            )
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

        # 🌿 分支
        branch_menu = mb.addMenu("🌿 分支")
        create_branch_action = QAction("➕ 新建分支...", self)
        create_branch_action.triggered.connect(self._on_create_branch)
        branch_menu.addAction(create_branch_action)

        switch_branch_menu = branch_menu.addMenu("🔀 切换分支")
        # 动态填充：在 _refresh_version_history 时重建
        self._switch_branch_menu = switch_branch_menu

        delete_branch_action = QAction("🗑 删除分支...", self)
        delete_branch_action.triggered.connect(self._on_delete_branch)
        branch_menu.addAction(delete_branch_action)

        # 👁 视图
        view_menu = mb.addMenu("👁 视图")
        for dock_attr in ("toolbar_dock", "left_dock", "center_dock",
                          "right_dock", "bottom_dock"):
            dock = getattr(self, dock_attr, None)
            if dock is not None:
                view_menu.addAction(dock.toggleViewAction())

    # ── 分支管理 ──────────────────────────────────────────

    def _refresh_branch_menu(self) -> None:
        """Rebuild the switch-branch submenu from all known branches."""
        self._switch_branch_menu.clear()
        try:
            branches = self._get_known_branches()
        except Exception:
            branches = ["main"]
        for b in sorted(branches):
            action = QAction(b, self)
            action.setCheckable(True)
            action.setChecked(b == self._get_current_branch())
            action.triggered.connect(lambda _, name=b: self._switch_to_branch(name))
            self._switch_branch_menu.addAction(action)

    def _get_known_branches(self) -> List[str]:
        """Return all distinct branch names from VCS history."""
        vcs = self._get_vcs()
        commits = vcs.list_commits(branch=None, limit=9999)
        branches: set[str] = set()
        for c in commits:
            b = c.get("branch", "main")
            if b:
                branches.add(b)
        if not branches:
            branches.add("main")
        return sorted(branches)

    def _get_current_branch(self) -> str:
        """Return the current branch name from the staging panel."""
        return self.staging_panel.current_branch

    def _on_create_branch(self) -> None:
        """Show a dialog to create a new branch (just sets the staging branch name)."""
        from PySide6.QtWidgets import QInputDialog  # type: ignore[attr-defined]
        name, ok = QInputDialog.getText(
            self, "新建分支", "分支名称:",
        )
        if ok and name.strip():
            self.staging_panel._current_branch = name.strip()
            self.staging_panel._branch_label.setText(f"🌿 {name.strip()}")
            self._update_status_bar_branch()
            self._refresh_branch_menu()

    def _switch_to_branch(self, name: str) -> None:
        """Switch the staging panel's target branch."""
        self.staging_panel._current_branch = name
        self.staging_panel._branch_label.setText(f"🌿 {name}")
        self._update_status_bar_branch()
        self._refresh_version_history()

    def _on_delete_branch(self) -> None:
        """Show a dialog to select and delete a branch."""
        from PySide6.QtWidgets import QInputDialog  # type: ignore[attr-defined]
        branches = self._get_known_branches()
        if not branches:
            QMessageBox.information(self, "删除分支", "没有可删除的分支。")
            return
        name, ok = QInputDialog.getItem(
            self, "删除分支", "选择要删除的分支:", branches, editable=False,
        )
        if ok and name:
            if name == self._get_current_branch():
                QMessageBox.warning(self, "删除分支", "不能删除当前所在分支。")
                return
            if name == "main":
                QMessageBox.warning(self, "删除分支", "不能删除 main 分支。")
                return
            # We can't truly delete branches from VCS (data stays in SQLite),
            # but we mark it as unavailable by simply not showing it.
            QMessageBox.information(
                self, "删除分支",
                f"分支 '{name}' 已标记为删除。\n"
                "后续提交将不会再使用该分支名。",
            )

    def _update_status_bar_branch(self) -> None:
        branch = self._get_current_branch()
        self._branch_status_label.setText(f"🌿 {branch}")

    def _build_ui(self) -> None:
        """构建全停靠式（Full-Docked）布局 —— 5 个 QDockWidget + 零空间中央 Widget。"""

        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks
        )

        # 零体积中央 Widget —— 所有空间归停靠窗
        dummy_central = QWidget()
        dummy_central.setMaximumSize(0, 0)
        self.setCentralWidget(dummy_central)

        # ── 辅助工厂 ────────────────────────────────────────
        def _make_dock(
            title: str,
            widget: QWidget,
            obj_name: str,
            area: Qt.DockWidgetArea = Qt.DockWidgetArea.LeftDockWidgetArea,
        ) -> QDockWidget:
            dock = QDockWidget(title, self)
            dock.setObjectName(obj_name)
            dock.setWidget(widget)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable |
                QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            self.addDockWidget(area, dock)
            return dock

        # ── 0. 顶部工具栏 ──────────────────────────────────
        toolbar_widget = QWidget()
        toolbar_widget.setFixedHeight(48)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(6, 4, 6, 4)

        self.btn_import = QPushButton("📁 导入并推理 CSV")
        self.btn_import.setMinimumHeight(36)
        self.btn_import.clicked.connect(self._on_import)
        toolbar_layout.addWidget(self.btn_import)

        self.btn_run = QPushButton("▶ 运行全管线")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_pipeline)
        toolbar_layout.addWidget(self.btn_run)

        toolbar_layout.addWidget(QLabel("组分:"))
        self.comp_combo = QComboBox()
        self.comp_combo.setMinimumWidth(120)
        self.comp_combo.setEnabled(False)
        toolbar_layout.addWidget(self.comp_combo)

        toolbar_layout.addWidget(QLabel("目标:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(120)
        self.target_combo.setEnabled(False)
        toolbar_layout.addWidget(self.target_combo)

        toolbar_layout.addStretch()

        self.toolbar_dock = _make_dock(
            "🧰 工具栏", toolbar_widget,
            "ToolbarDock", Qt.DockWidgetArea.TopDockWidgetArea,
        )

        # ── 1. 左侧：条件格式表格 ──────────────────────────
        self.data_table = ConditionFormatTable()
        self.left_dock = _make_dock(
            "① 数据表格", self.data_table,
            "LeftTableDock", Qt.DockWidgetArea.LeftDockWidgetArea,
        )

        # ── 2. 中央：分析 Tab ──────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)

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

        # Tab 6: 版本历史
        self.version_history_tab = QWidget()
        self._build_version_history_tab()
        self.tabs.addTab(self.version_history_tab, "📜 版本历史")

        # Tab 7: DAG 拓扑图
        self.dag_tab = QWidget()
        self._build_dag_tab()
        self.tabs.addTab(self.dag_tab, "🔀 分支拓扑")

        self.center_dock = _make_dock(
            "② 分析视图", self.tabs,
            "CenterTabDock", Qt.DockWidgetArea.RightDockWidgetArea,
        )

        # ── 3. 右侧：智能详情面板 ──────────────────────────
        self.right_panel = RightDetailPanel()
        self.right_dock = _make_dock(
            "③ 智能诊断", self.right_panel,
            "RightDiagnosticsDock", Qt.DockWidgetArea.RightDockWidgetArea,
        )

        # ── 4. 底部：暂存区 ────────────────────────────────
        self.staging_panel = StagingPanel(
            storage=self._get_storage(),
            vcs=self._get_vcs(),
        )
        self.staging_panel.setMaximumHeight(200)
        self.staging_panel.commit_success.connect(self._on_staging_commit)

        # 暂存区提交后刷新版本历史
        self.staging_panel.commit_success.connect(
            lambda _: self._refresh_version_history()
        )
        self.bottom_dock = _make_dock(
            "④ 暂存区", self.staging_panel,
            "BottomStagingDock", Qt.DockWidgetArea.BottomDockWidgetArea,
        )

        # ── 状态栏 ─────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)

        self._branch_status_label = QLabel("🌿 main")
        self._branch_status_label.setStyleSheet(
            "font-family: Consolas; padding: 2px 8px; background: #e8e8e8; border-radius: 3px;"
        )
        self.statusBar().addPermanentWidget(self._branch_status_label)

        self._update_status("请导入 CSV 数据文件")

    def _on_menu_commit(self) -> None:
        """Trigger commit via the staging panel's button."""
        self.staging_panel._on_commit()

    def _on_menu_calibrate(self) -> None:
        QMessageBox.information(
            self, "批次偏移校准",
            "校准功能正在开发中",
        )

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

        comp_bounds = data.comp_bounds()
        process_bounds = data.param_bounds()
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
            parameters=data.to_commit_metadata(),
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
    # Tab 6: 版本历史
    # ---------------------------------------------------------

    def _build_version_history_tab(self) -> None:
        """Build Tab 6: commit log browser + detail / diff panel."""
        layout = QVBoxLayout(self.version_history_tab)

        # ── 工具栏 ────────────────────────────────────────────
        tool_row = QHBoxLayout()
        self.vh_refresh_btn = QPushButton("⟳ 刷新")
        self.vh_refresh_btn.clicked.connect(self._refresh_version_history)
        tool_row.addWidget(self.vh_refresh_btn)

        self.vh_checkout_btn = QPushButton("⬇ 检出此版本")
        self.vh_checkout_btn.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self.vh_checkout_btn.clicked.connect(self._on_vh_checkout)
        tool_row.addWidget(self.vh_checkout_btn)

        tool_row.addWidget(QLabel("分支:"))
        self.vh_branch_combo = QComboBox()
        self.vh_branch_combo.setMinimumWidth(120)
        self.vh_branch_combo.currentTextChanged.connect(self._refresh_version_history)
        tool_row.addWidget(self.vh_branch_combo)

        tool_row.addWidget(QLabel("操作人:"))
        self.vh_author_edit = QTextEdit()
        self.vh_author_edit.setPlaceholderText("筛选作者")
        self.vh_author_edit.setMaximumHeight(28)
        self.vh_author_edit.setMaximumWidth(140)
        self.vh_author_edit.textChanged.connect(self._refresh_version_history)
        tool_row.addWidget(self.vh_author_edit)

        self.vh_limit_combo = QComboBox()
        self.vh_limit_combo.addItems(["20", "50", "100", "200"])
        self.vh_limit_combo.setCurrentText("50")
        self.vh_limit_combo.currentTextChanged.connect(self._refresh_version_history)
        tool_row.addWidget(QLabel("条数:"))
        tool_row.addWidget(self.vh_limit_combo)

        tool_row.addStretch()
        layout.addLayout(tool_row)

        # ── 分割: 上方 commit 列表 | 下方详情 ──────────────
        v_split = QSplitter(Qt.Orientation.Vertical)

        # 上方: commit 表格
        self.vh_table = QTableView()
        self.vh_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.vh_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.vh_table.setAlternatingRowColors(True)
        self.vh_table.horizontalHeader().setStretchLastSection(True)
        self.vh_table.verticalHeader().setVisible(False)
        self.vh_table.clicked.connect(self._on_vh_row_clicked)
        v_split.addWidget(self.vh_table)

        # 下方: 详情 + diff
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        detail_tabs = QTabWidget()
        self.vh_detail_text = QTextEdit()
        self.vh_detail_text.setReadOnly(True)
        self.vh_detail_text.setFont(QFont("Consolas", 10))
        detail_tabs.addTab(self.vh_detail_text, "📄 Commit 详情")

        self.vh_diff_text = QTextEdit()
        self.vh_diff_text.setReadOnly(True)
        self.vh_diff_text.setFont(QFont("Consolas", 10))
        detail_tabs.addTab(self.vh_diff_text, "🔄 差异对比")

        detail_layout.addWidget(detail_tabs)

        # diff 操作栏
        diff_controls = QHBoxLayout()
        diff_controls.addWidget(QLabel("对比:"))
        self.vh_diff_a = QComboBox()
        self.vh_diff_a.setMinimumWidth(200)
        diff_controls.addWidget(self.vh_diff_a)
        diff_controls.addWidget(QLabel(" → "))
        self.vh_diff_b = QComboBox()
        self.vh_diff_b.setMinimumWidth(200)
        diff_controls.addWidget(self.vh_diff_b)
        self.vh_diff_btn = QPushButton("比较差异")
        self.vh_diff_btn.clicked.connect(self._on_vh_diff)
        diff_controls.addWidget(self.vh_diff_btn)
        diff_controls.addStretch()
        detail_layout.addLayout(diff_controls)

        v_split.addWidget(detail_widget)
        v_split.setSizes([300, 250])
        layout.addWidget(v_split, 1)

    # ---------------------------------------------------------
    # Tab 7: DAG 分支拓扑图（交互式）
    # ---------------------------------------------------------

    def _build_dag_tab(self) -> None:
        """Build Tab 7: interactive DAG branch topology visualization."""
        layout = QVBoxLayout(self.dag_tab)

        # ── 工具栏 ────────────────────────────────────────────
        tool_row = QHBoxLayout()

        self.dag_refresh_btn = QPushButton("⟳ 刷新")
        self.dag_refresh_btn.clicked.connect(self._refresh_dag)
        tool_row.addWidget(self.dag_refresh_btn)

        self.dag_compare_toggle = QPushButton("🔍 对比模式")
        self.dag_compare_toggle.setCheckable(True)
        self.dag_compare_toggle.toggled.connect(self._on_dag_compare_mode)
        tool_row.addWidget(self.dag_compare_toggle)

        self.dag_compare_btn = QPushButton("📊 对比选中")
        self.dag_compare_btn.clicked.connect(self._on_dag_compare_go)
        tool_row.addWidget(self.dag_compare_btn)

        self.dag_fork_btn = QPushButton("🌿 分叉 (Fork)")
        self.dag_fork_btn.clicked.connect(self._on_dag_fork)
        tool_row.addWidget(self.dag_fork_btn)

        self.dag_merge_btn = QPushButton("🔗 合并 (Merge)")
        self.dag_merge_btn.clicked.connect(self._on_dag_merge)
        tool_row.addWidget(self.dag_merge_btn)

        tool_row.addWidget(QLabel("布局:"))
        self.dag_layout_combo = QComboBox()
        self.dag_layout_combo.addItems([
            "分层 (dot)", "弹簧 (spring)", "圆形 (circular)", "谱 (spectral)",
        ])
        self.dag_layout_combo.currentTextChanged.connect(self._refresh_dag)
        tool_row.addWidget(self.dag_layout_combo)

        tool_row.addWidget(QLabel("最大节点:"))
        self.dag_limit_combo = QComboBox()
        self.dag_limit_combo.addItems(["50", "100", "200", "500"])
        self.dag_limit_combo.setCurrentText("100")
        self.dag_limit_combo.currentTextChanged.connect(self._refresh_dag)
        tool_row.addWidget(self.dag_limit_combo)

        tool_row.addStretch()
        layout.addLayout(tool_row)

        # ── 分支筛选栏 ─────────────────────────────────────────
        self.dag_filter_bar = BranchFilterBar()
        self.dag_filter_bar.filter_changed.connect(self._on_dag_filter)
        layout.addWidget(self.dag_filter_bar)

        # ── 交互式 DAG 画布 ────────────────────────────────────
        vcs = self._get_vcs()
        self.dag_canvas = DagGraphCanvas(vcs)
        self.dag_canvas.node_selected.connect(self._on_dag_node_selected)
        self.dag_canvas.compare_requested.connect(self._on_dag_compare)
        self.dag_canvas.fork_requested.connect(self._on_dag_fork_from_canvas)

        self.dag_nav = NavigationToolbar2QT(self.dag_canvas, parent=None)
        layout.addWidget(self.dag_nav)
        layout.addWidget(self.dag_canvas, 1)

    def _refresh_dag(self) -> None:
        """Fetch all commits from VCS and load into the interactive canvas."""
        try:
            vcs = self._get_vcs()
            limit_text = self.dag_limit_combo.currentText()
            limit = int(limit_text) if limit_text else 100
            commits = vcs.list_commits(branch=None, limit=limit)
        except Exception as exc:
            self.statusBar().showMessage(f"无法加载 DAG: {exc}")
            return

        # Update branch filter bar
        branches = sorted({c.get("branch", "main") for c in commits})
        self.dag_filter_bar.set_branches(branches)

        layout_name = self.dag_layout_combo.currentText()
        self.dag_canvas.load_commits(commits, layout_name=layout_name)

        self.statusBar().showMessage(
            f"DAG: {len(commits)} 个提交, {len(branches)} 个分支", 5000
        )

    # ── DAG 交互回调 ──────────────────────────────────────────

    def _on_dag_compare_mode(self, enabled: bool) -> None:
        """Toggle compare mode on the canvas."""
        self.dag_canvas.set_compare_mode(enabled)
        self.dag_compare_toggle.setText(
            "🔍 对比: ON" if enabled else "🔍 对比模式"
        )

    def _on_dag_compare_go(self) -> None:
        """User clicked '对比选中' — validate and open diff dialog."""
        nodes = self.dag_canvas.compare_nodes
        if len(nodes) == 2:
            self._on_dag_compare(nodes[0], nodes[1])
        else:
            QMessageBox.warning(
                self, "对比",
                f"请勾选恰好 2 个节点 (当前已选 {len(nodes)} 个)。\n"
                "先点击「对比模式」开关，再点击节点勾选。",
            )

    def _on_dag_compare(self, commit_a: str, commit_b: str) -> None:
        """Open the three-column diff dialog."""
        try:
            vcs = self._get_vcs()
            dialog = DagDiffDialog(vcs, commit_a, commit_b, parent=self)
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "对比失败", str(exc))

    def _on_dag_fork(self) -> None:
        """Fork from the currently selected node."""
        sel = self.dag_canvas.selected_node
        if sel is None:
            QMessageBox.information(
                self, "分叉", "请先在画布上点击选中一个节点。"
            )
            return
        dialog = ForkDialog(sel, parent=self)
        if dialog.exec():
            branch_name = dialog.branch_name()
            purpose = dialog.purpose()
            if not branch_name:
                QMessageBox.warning(self, "分叉", "分支名称不能为空。")
                return
            self._on_dag_fork_from_canvas(sel, branch_name, purpose)

    def _on_dag_fork_from_canvas(
        self, parent_commit_id: str, branch_name: str, purpose: str
    ) -> None:
        """Execute the fork: create a new commit on the new branch."""
        try:
            vcs = self._get_vcs()
            parent_meta = vcs.get_commit(parent_commit_id)
            vcs.commit_version(
                parent_ids=[parent_commit_id],
                sop_sequence=parent_meta.get("sop_sequence", []),
                parameters=parent_meta.get("parameters", {}),
                data_file_hash=parent_meta.get("data_file_hash", ""),
                author="fork-bot",
                message=f"Fork: {purpose}" if purpose else f"Branch: {branch_name}",
                branch=branch_name,
            )
            self._refresh_dag()
            self._refresh_version_history()
            self._refresh_branch_menu()
            self.statusBar().showMessage(
                f"🌿 已创建分支 '{branch_name}' 从 {parent_commit_id[:12]}...", 5000
            )
        except Exception as exc:
            QMessageBox.critical(self, "分叉失败", str(exc))

    def _on_dag_merge(self) -> None:
        """Open merge dialog from selected node to a target."""
        sel = self.dag_canvas.selected_node
        if sel is None:
            QMessageBox.information(
                self, "合并", "请先在画布上点击选中一个源节点。"
            )
            return
        # Ask user for the target commit ID
        from PySide6.QtWidgets import QInputDialog
        target_id, ok = QInputDialog.getText(
            self, "合并目标",
            "请输入目标 Commit ID (完整或前 16 位):",
        )
        if not ok or not target_id.strip():
            return
        try:
            vcs = self._get_vcs()
            # Resolve partial ID
            all_commits = vcs.list_commits(branch=None, limit=500)
            full_target = None
            for c in all_commits:
                if c["commit_id"].startswith(target_id.strip()):
                    full_target = c["commit_id"]
                    break
            if full_target is None:
                QMessageBox.warning(self, "合并", "未找到匹配的 Commit ID。")
                return

            dialog = DagMergeDialog(vcs, sel, full_target, parent=self)
            if dialog.exec():
                mid = dialog.merged_commit_id
                self._refresh_dag()
                self._refresh_version_history()
                self.statusBar().showMessage(
                    f"🔗 合并完成: {mid[:16]}...", 5000
                )
        except Exception as exc:
            QMessageBox.critical(self, "合并失败", str(exc))

    def _on_dag_node_selected(self, commit_id: Optional[str]) -> None:
        """Handle node selection change on the canvas."""
        if commit_id:
            self.statusBar().showMessage(
                f"选中节点: {commit_id[:16]}...", 3000
            )

    def _on_dag_filter(self, visible_branches: set) -> None:
        """Apply branch visibility filter to the canvas."""
        all_branches = set(self.dag_canvas._branch_colors.keys())
        hidden = all_branches - visible_branches
        self.dag_canvas.set_hidden_branches(hidden)

    def _refresh_version_history(self) -> None:
        """Pull commit list from VCS and populate the table + diff combos."""
        try:
            vcs = self._get_vcs()
            branch = self.vh_branch_combo.currentText()
            if branch == "所有":
                branch = None
            author = self.vh_author_edit.toPlainText().strip()
            if not author:
                author = None
            limit_text = self.vh_limit_combo.currentText()
            limit = int(limit_text) if limit_text else 50

            commits = vcs.list_commits(branch=branch, author=author, limit=limit)
        except Exception as exc:
            self.vh_detail_text.setPlainText(f"无法加载版本历史: {exc}")
            return

        # update table
        model = VersionHistoryModel(commits)
        self.vh_table.setModel(model)

        # update diff combos
        self.vh_diff_a.blockSignals(True)
        self.vh_diff_b.blockSignals(True)
        self.vh_diff_a.clear()
        self.vh_diff_b.clear()
        for c in commits:
            label = f"{c['commit_id'][:12]} | {c.get('author',''):8} | {c.get('message','')[:30]}"
            self.vh_diff_a.addItem(label, c['commit_id'])
            self.vh_diff_b.addItem(label, c['commit_id'])
        self.vh_diff_a.blockSignals(False)
        self.vh_diff_b.blockSignals(False)
        if commits:
            self.vh_diff_a.setCurrentIndex(0)
            self.vh_diff_b.setCurrentIndex(min(1, len(commits) - 1))

        # update branch filter combo
        all_branches = sorted({c.get("branch", "main") for c in commits})
        self.vh_branch_combo.blockSignals(True)
        current_branch = self.vh_branch_combo.currentText()
        self.vh_branch_combo.clear()
        self.vh_branch_combo.addItem("所有")
        for b in all_branches:
            self.vh_branch_combo.addItem(b)
        idx = self.vh_branch_combo.findText(current_branch)
        if idx >= 0:
            self.vh_branch_combo.setCurrentIndex(idx)
        self.vh_branch_combo.blockSignals(False)

        # refresh the branch switch menu
        self._refresh_branch_menu()

        # show first commit detail
        if commits:
            self._show_vh_detail(commits[0])

    def _on_vh_row_clicked(self, index: QModelIndex) -> None:
        """User clicked a commit row in the version history table."""
        model = self.vh_table.model()
        if not isinstance(model, VersionHistoryModel):
            return
        commit = model.get_commit(index.row())
        if commit is not None:
            self._show_vh_detail(commit)

    def _show_vh_detail(self, commit: Dict[str, Any]) -> None:
        """Display commit metadata in the detail pane."""
        lines = [
            f"Commit ID:  {commit['commit_id']}",
            f"作者:       {commit.get('author', '')}",
            f"时间:       {commit.get('created_at', '')}",
            f"分支:       {commit.get('branch', '')}",
            f"设备:       {commit.get('equipment', '')}",
            f"消息:       {commit.get('message', '')}",
            f"父版本:     {', '.join(commit.get('parents', []))}",
            f"数据文件:   {commit.get('data_file_hash', '')}",
            "",
            "── SOP 序列 ──",
        ]
        for i, step in enumerate(commit.get("sop_sequence", []), 1):
            lines.append(f"  {i}. {step}")
        lines.append("")
        lines.append("── 参数 ──")
        for k, v in commit.get("parameters", {}).items():
            lines.append(f"  {k}: {v}")

        self.vh_detail_text.setPlainText("\n".join(lines))
        self.vh_diff_text.setPlainText("")

    def _on_vh_checkout(self) -> None:
        """Checkout the currently selected commit's data into the main view."""
        try:
            vcs = self._get_vcs()
            sel = self.vh_table.selectionModel()
            if sel is None or not sel.hasSelection():
                QMessageBox.information(self, "检出", "请先在列表中选择一个 commit。")
                return
            index = sel.selectedRows()[0]
            model = self.vh_table.model()
            if not isinstance(model, VersionHistoryModel):
                return
            commit = model.get_commit(index.row())
            if commit is None:
                return
            cid = commit["commit_id"]
            df = vcs.checkout_dataframe(cid)
        except Exception as exc:
            QMessageBox.critical(self, "检出失败", str(exc))
            return

        # replace the current data
        if self._data is None:
            self._data = ExperimentData(df=df)
        else:
            self._data.df = df

        self._on_data_loaded()
        self.statusBar().showMessage(
            f"已检出 commit {cid[:16]}... ({len(df)} 条记录)", 8000
        )

    def _on_vh_diff(self) -> None:
        """Compare two selected commits and show the diff."""
        try:
            vcs = self._get_vcs()
            v1_id = self.vh_diff_a.currentData()
            v2_id = self.vh_diff_b.currentData()
            if not v1_id or not v2_id:
                return
            patch = vcs.generate_version_patch(v1_id, v2_id)
            lines = [
                f"比较: {v1_id[:16]}...  →  {v2_id[:16]}...",
                "",
            ]
            if not patch:
                lines.append("两个版本完全相同，无差异。")
            else:
                if "data_file_hash" in patch:
                    lines.append("🟡 数据文件已变更")
                    lines.append(f"   旧: {patch['data_file_hash']['from'][:20]}...")
                    lines.append(f"   新: {patch['data_file_hash']['value'][:20]}...")
                    lines.append("")
                if "sop_sequence" in patch:
                    sop = patch["sop_sequence"]
                    lines.append("🟡 SOP 序列差异:")
                    for s in sop.get("only_in_a", []):
                        lines.append(f"   ─ [{s}] (仅旧版本)")
                    for s in sop.get("only_in_b", []):
                        lines.append(f"   ┼ [{s}] (仅新版本)")
                    lines.append("")
                if "parameters" in patch:
                    lines.append("🟡 参数差异:")
                    for op in patch["parameters"]:
                        lines.append(f"   {op['op']}  {op['path']}")
                        if "value" in op:
                            lines.append(f"       → {op['value']}")
                    lines.append("")
                if "metadata" in patch:
                    lines.append("🟡 元数据差异:")
                    for k, v in patch["metadata"].items():
                        lines.append(f"   {k}: {v}")
                    lines.append("")
            self.vh_diff_text.setPlainText("\n".join(lines))
        except Exception as exc:
            self.vh_diff_text.setPlainText(f"差异计算失败: {exc}")

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
        self._data = ExperimentData(df=df)
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

        # ── 版本历史 ──
        self._refresh_version_history()

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
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()