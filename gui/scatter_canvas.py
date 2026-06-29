"""
HoverScatterCanvas — 可悬停 Tooltip + 鼠标框选散点图
=====================================================

接口符合 prompts/widget_interface_spec.txt 中 ScatterCanvas 规范。

功能
----
1.  Matplotlib 嵌入 PySide6 (FigureCanvasQTAgg)
2.  悬停显示 label + (x, y)
3.  鼠标矩形框选，选中点红色，其余蓝色
4.  selectionChanged 信号
5.  独立 main() 测试
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

os.environ["QT_API"] = "pyside6"

import matplotlib as mpl

mpl.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Annotation
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HoverScatterCanvas(FigureCanvasQTAgg):
    """带悬停 Tooltip 和矩形框选的散点图画布。"""

    selectionChanged = Signal(list)  # List[int]

    def __init__(self, parent: Any = None) -> None:
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)  # type: ignore[no-untyped-call]
        self.setParent(parent)

        # 数据存储
        self._df: Optional[pd.DataFrame] = None
        self._x_col: str = ""
        self._y_col: str = ""
        self._label_col: str = ""
        self._scatter_collection: List[Any] = []  # hold scatter references
        self._annot: Optional[Annotation] = None

        # 选择状态
        self._selected: np.ndarray = np.array([], dtype=bool)
        self._selecting = False
        self._rect_start: Optional[Tuple[float, float]] = None
        self._rect_patch: Optional[Patch] = None

        # 连接事件
        self.mpl_connect("motion_notify_event", self._on_hover)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)

    # -----------------------------------------------------------------
    # 公共接口
    # -----------------------------------------------------------------

    def set_data(self, df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """设置绘图数据。

        Parameters
        ----------
        df : pd.DataFrame
            数据表，需包含 x_col, y_col。
        x_col : str
            X 轴列名。
        y_col : str
            Y 轴列名。
        """
        self._df = df
        self._x_col = x_col
        self._y_col = y_col

        # 猜测标签列
        candidates = ["label", "sample", "id", "exp_id", "name"]
        for c in candidates:
            if c in df.columns:
                self._label_col = c
                break
        else:
            self._label_col = str(df.index.name) if df.index.name is not None else ""

        # 重置选择
        self._selected = np.zeros(len(df), dtype=bool)

        # 清除注释
        if self._annot is not None:
            self._annot.remove()
            self._annot = None

        self._redraw()

    def get_selected_indices(self) -> List[int]:
        """返回选中行的整数索引列表。"""
        if self._df is None:
            return []
        return list(np.where(self._selected)[0])

    def clear_selection(self) -> None:
        """清除所有选中状态。"""
        if self._df is not None:
            self._selected = np.zeros(len(self._df), dtype=bool)
        else:
            self._selected = np.array([], dtype=bool)
        self._redraw()
        self.selectionChanged.emit([])

    # -----------------------------------------------------------------
    # 内部绘图
    # -----------------------------------------------------------------

    def _redraw(self) -> None:
        """清空并重新绘制散点图。"""
        df = self._df
        if df is None:
            return

        self.axes.clear()
        self._scatter_collection.clear()

        x: np.ndarray = np.asarray(df[self._x_col].values)
        y: np.ndarray = np.asarray(df[self._y_col].values)

        sel = self._selected
        not_sel = ~sel

        if not_sel.sum() > 0:
            s = self.axes.scatter(
                x[not_sel], y[not_sel], c="blue", alpha=0.6,
                edgecolors="k", linewidths=0.5, picker=True, label="未选中",
            )
            self._scatter_collection.append(s)

        if sel.sum() > 0:
            s = self.axes.scatter(
                x[sel], y[sel], c="red", alpha=0.8,
                edgecolors="k", linewidths=0.8, s=60, picker=True, label="选中",
            )
            self._scatter_collection.append(s)

        self.axes.set_xlabel(self._x_col)
        self.axes.set_ylabel(self._y_col)
        self.axes.set_title(f"{self._x_col} vs {self._y_col}")
        self.axes.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw_idle()  # type: ignore[no-untyped-call]

    def _get_label(self, idx: int) -> str:
        """获取数据点的标签文本。"""
        if self._label_col and self._df is not None and self._label_col in self._df.columns:
            return str(self._df.iloc[idx][self._label_col])
        return f"#{idx}"

    # -----------------------------------------------------------------
    # 事件处理
    # -----------------------------------------------------------------

    def _on_hover(self, event: MouseEvent) -> None:
        """悬停时显示 Tooltip。"""
        df = self._df
        if df is None or event.inaxes != self.axes:
            if self._annot is not None:
                self._annot.set_visible(False)
                self.draw_idle()  # type: ignore[no-untyped-call]
            return

        x_data: np.ndarray = np.asarray(df[self._x_col].values)
        y_data: np.ndarray = np.asarray(df[self._y_col].values)

        # 找最近点
        distances = np.hypot(x_data - event.xdata, y_data - event.ydata)
        nearest = int(distances.argmin())
        min_dist = distances[nearest]

        # 容差（数据范围的 2%）
        x_range = float(x_data.max() - x_data.min())
        y_range = float(y_data.max() - y_data.min())
        tol = max(x_range, y_range) * 0.02 if max(x_range, y_range) > 0 else 1.0

        if min_dist > tol:
            if self._annot is not None:
                self._annot.set_visible(False)
                self.draw_idle()  # type: ignore[no-untyped-call]
            return

        label = self._get_label(nearest)
        x_val = float(x_data[nearest])
        y_val = float(y_data[nearest])

        if self._annot is None:
            self._annot = self.axes.annotate(
                "", xy=(0, 0), xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle="->", alpha=0.6),
            )

        self._annot.set_position((x_val, y_val))
        self._annot.set_text(f"{label}\n({x_val:.3g}, {y_val:.3g})")
        self._annot.set_visible(True)
        self.draw_idle()  # type: ignore[no-untyped-call]

    def _on_press(self, event: MouseEvent) -> None:
        """开始框选。"""
        if event.button is not MouseButton.LEFT or event.inaxes != self.axes:
            return

        self._selecting = True
        xd = event.xdata
        yd = event.ydata
        if xd is None or yd is None:
            self._selecting = False
            return
        self._rect_start = (xd, yd)

        x0, y0 = self._rect_start
        rect = Rectangle(
            (x0, y0), 0, 0, fill=True, facecolor="gray",
            alpha=0.15, edgecolor="gray", linestyle="--", linewidth=1,
        )
        self._rect_patch = self.axes.add_patch(rect)
        self.draw_idle()  # type: ignore[no-untyped-call]

    def _on_release(self, event: MouseEvent) -> None:
        """结束框选，计算矩形内点。"""
        if not self._selecting or self._df is None:
            return
        self._selecting = False

        if self._rect_start is None:
            self._remove_rect()
            return
        x0, y0 = self._rect_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            self._remove_rect()
            return

        # 确定矩形范围
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)

        # 查找矩形内的点
        x: np.ndarray = np.asarray(self._df[self._x_col].values)
        y: np.ndarray = np.asarray(self._df[self._y_col].values)
        in_rect: np.ndarray = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

        self._selected = in_rect

        self._remove_rect()
        self._redraw()

        # 发射信号
        indices = self.get_selected_indices()
        self.selectionChanged.emit(indices)

    # -----------------------------------------------------------------
    # 矩形辅助
    # -----------------------------------------------------------------

    def _remove_rect(self) -> None:
        """移除选择矩形。"""
        if self._rect_patch is not None:
            try:
                self._rect_patch.remove()
            except Exception:
                pass
            self._rect_patch = None
            self.draw_idle()  # type: ignore[no-untyped-call]


# =====================================================================
# 独立测试入口
# =====================================================================


def _generate_demo_data(n: int = 50) -> pd.DataFrame:
    """生成测试用模拟数据。"""
    np.random.seed(42)
    df = pd.DataFrame({
        "sample": [f"S{i:03d}" for i in range(n)],
        "resin_pct": np.random.uniform(0.1, 0.6, n),
        "tensile_strength": np.random.uniform(20, 80, n),
        "filler_pct": np.random.uniform(0.05, 0.3, n),
    })
    return df


def main() -> None:
    """独立运行测试。"""
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("HoverScatterCanvas — 测试")
    window.resize(900, 700)

    central = QWidget()
    window.setCentralWidget(central)
    layout = QVBoxLayout(central)

    # 画布
    canvas = HoverScatterCanvas()
    layout.addWidget(canvas, 1)

    # 控制区
    controls = QHBoxLayout()
    btn_clear = QPushButton("清除选中")
    info_label = QLabel("未选中任何点")
    controls.addWidget(btn_clear)
    controls.addWidget(info_label)
    controls.addStretch()
    layout.addLayout(controls)

    # 选中列表
    selected_list = QListWidget()
    selected_list.setMaximumHeight(100)
    layout.addWidget(selected_list)

    # 加载数据
    df = _generate_demo_data(50)
    canvas.set_data(df, "resin_pct", "tensile_strength")

    # 信号连接
    def on_selection_changed(indices: List[int]) -> None:
        selected_list.clear()
        for i in indices:
            row = df.iloc[i]
            item_text = (
                f"{row['sample']} "
                f"(resin={row['resin_pct']:.3f}, "
                f"strength={row['tensile_strength']:.1f})"
            )
            selected_list.addItem(item_text)
        info_label.setText(f"选中 {len(indices)} 个点")

    canvas.selectionChanged.connect(on_selection_changed)
    btn_clear.clicked.connect(canvas.clear_selection)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
