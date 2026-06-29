"""
ConditionFormatTable — 条件格式表格视图

提供按列红/绿渐变背景的 QTableView，实现 widget_interface_spec.txt 中
TableView 的接口约定。

用法:
    table = ConditionFormatTable()
    table.set_data(df)
    indices = table.get_selected_rows()
    table.selectionChanged.connect(handler)
"""
from __future__ import annotations

import sys
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractItemModel,
    QPersistentModelIndex,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


# =====================================================================
# 模型
# =====================================================================

class _ColorTableModel(QAbstractItemModel):
    """QTableView 可用的 QAbstractItemModel 子类。

    数据存储为二维 List[Any]，并缓存每列的 min/max 值以支持
    红/绿渐变背景色。单元格为可编辑状态。
    """

    def __init__(self, data: List[List[Any]], columns: List[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._data: List[List[Any]] = data
        self._columns: List[str] = columns
        self._row_count: int = len(data)
        self._col_count: int = len(columns)
        self._outlier_indices: List[int] = []
        self._pareto_mask: Optional[List[bool]] = None

        # 缓存每列的数值范围，用于颜色映射
        self._col_min: List[float] = [0.0] * self._col_count
        self._col_max: List[float] = [1.0] * self._col_count
        self._compute_ranges()

    def _compute_ranges(self) -> None:
        """扫描数据，计算每列数值列的 min/max。"""
        for col in range(self._col_count):
            vals: List[float] = []
            for row in range(self._row_count):
                v = self._data[row][col]
                if isinstance(v, (int, float, np.integer, np.floating)) and not (
                    isinstance(v, float) and np.isnan(v)
                ):
                    vals.append(float(v))
            if vals:
                mn = min(vals)
                mx = max(vals)
                if mx - mn < 1e-15:
                    mn, mx = mn - 0.5, mx + 0.5  # 避免除零
                self._col_min[col] = mn
                self._col_max[col] = mx
            else:
                self._col_min[col] = 0.0
                self._col_max[col] = 1.0

    def set_highlighting(self, outlier_indices: Optional[List[int]] = None, pareto_mask: Optional[List[bool]] = None) -> None:
        """设置异常值和帕累托最优行的高亮。

        Parameters
        ----------
        outlier_indices : List[int] or None
            异常行的索引列表。
        pareto_mask : List[bool] or None
            帕累托最优行的布尔掩码，长度等于行数。
        """
        self._outlier_indices = outlier_indices if outlier_indices is not None else []
        self._pareto_mask = pareto_mask
        self.layoutChanged.emit()

    # ── 纯虚方法实现 ──

    def index(self, row: int, column: int, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> QModelIndex:
        if parent.isValid() or row < 0 or row >= self._row_count or column < 0 or column >= self._col_count:
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, child: QModelIndex | QPersistentModelIndex = QModelIndex()) -> QModelIndex:  # type: ignore[override]
        return QModelIndex()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._row_count

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._col_count

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row >= self._row_count or col >= self._col_count:
            return None
        value = self._data[row][col]

        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(value, (int, np.integer)):
                return str(value)
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    return "NaN"
                return f"{value:.4g}"
            return str(value) if value is not None else ""

        if role == Qt.ItemDataRole.EditRole:
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            return value

        if role == Qt.ItemDataRole.BackgroundRole:
            return self._background(row, col)

        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value) if value is not None else ""

        return None

    def setData(self, index: QModelIndex | QPersistentModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        row, col = index.row(), index.column()
        if row >= self._row_count or col >= self._col_count:
            return False
        try:
            self._data[row][col] = float(value)
        except (ValueError, TypeError):
            self._data[row][col] = value
        self._compute_ranges()
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.BackgroundRole])
        return True

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section] if section < self._col_count else None
        return str(section + 1)

    # ── 颜色 ──

    def _background(self, row: int, col: int) -> Optional[QBrush]:
        """返回高亮或红-绿渐变 QBrush。

        优先级:
          1. 异常值行高亮 (浅红)
          2. 帕累托最优行高亮 (浅绿)
          3. 列内最小值 → 红色 (R=255, G=0, B=0)
             列内最大值 → 绿色 (R=0, G=180, B=0)
          4. 非数值列 → 无背景色。
        """
        # 异常值行高亮
        if row in self._outlier_indices:
            return QBrush(QColor(255, 230, 230))

        # 帕累托最优行高亮
        if self._pareto_mask is not None and row < len(self._pareto_mask) and self._pareto_mask[row]:
            return QBrush(QColor(230, 245, 230))

        v = self._data[row][col]
        if not isinstance(v, (int, float, np.integer, np.floating)):
            return None
        if isinstance(v, float) and np.isnan(v):
            return None

        fv = float(v)
        mn, mx = self._col_min[col], self._col_max[col]
        if mx - mn < 1e-15:
            return QBrush(QColor(128, 128, 128))  # 灰色

        t = (fv - mn) / (mx - mn)  # 0..1
        t = max(0.0, min(1.0, t))

        # 红(0) → 黄(0.5) → 绿(1)
        if t < 0.5:
            r: int = 255
            g: int = int(180 * (t * 2))  # 0 → 180
        else:
            r = int(255 * (1 - (t - 0.5) * 2))  # 255 → 0
            g = 180

        return QBrush(QColor(r, g, 60))


# =====================================================================
# 条件格式表格控件
# =====================================================================

class ConditionFormatTable(QWidget):
    """条件格式表格视图。

    信号:
        selectionChanged(selected_indices: List[int]) — 选中行索引变化时发射。

    方法:
        set_data(df: pd.DataFrame) -> None
        get_selected_rows() -> List[int]
    """

    selectionChanged = Signal(list)  # List[int]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._df: Optional[pd.DataFrame] = None
        self._model: Optional[_ColorTableModel] = None
        self._proxy: Optional[QSortFilterProxyModel] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(False)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_view.verticalHeader().setDefaultSectionSize(28)

        layout.addWidget(self.table_view)

        # 连接选择变化 — 在 set_data 中通过 sel_model 连接
        pass

    def set_data(self, df: pd.DataFrame) -> None:
        """设置表格数据。

        将 DataFrame 转换为二维列表并重建模型。
        """
        self._df = df.copy()
        columns = list(df.columns)
        data: List[List[Any]] = df.values.tolist()

        self._model = _ColorTableModel(data, columns, self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)

        old_sel = self.table_view.selectionModel()
        self.table_view.setModel(self._proxy)
        if old_sel is not None:
            old_sel.deleteLater()

        # 重新连接选择信号
        sel_model = self.table_view.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(self._on_selection_changed)

    def get_selected_rows(self) -> List[int]:
        """返回当前选中行的索引（原始数据行号，不受排序影响）。"""
        if self._proxy is None or self._model is None:
            return []
        sel_model = self.table_view.selectionModel()
        if sel_model is None:
            return []
        indices: List[int] = []
        for idx in sel_model.selectedRows():
            if idx.isValid():
                source_idx = self._proxy.mapToSource(idx)
                indices.append(source_idx.row())
        return sorted(indices)

    @Slot()
    def _on_selection_changed(self) -> None:
        """发射 selectionChanged 信号。"""
        indices = self.get_selected_rows()
        self.selectionChanged.emit(indices)


# =====================================================================
# 独立测试入口
# =====================================================================

def main() -> None:
    """独立运行测试。"""
    app = QApplication(sys.argv)

    # 生成示例数据
    np.random.seed(42)
    n = 30
    df = pd.DataFrame(
        {
            "拉伸强度 (MPa)": np.random.uniform(30, 80, n),
            "成本 (元/kg)": np.random.uniform(5, 25, n),
            "固化时间 (min)": np.random.uniform(10, 60, n),
            "配比": np.random.uniform(0.1, 0.5, n),
        }
    )
    # 添加一些非数值列
    df["批次"] = [f"B{i:03d}" for i in range(n)]

    # 构建窗口
    win = QMainWindow()
    win.setWindowTitle("条件格式表格 — 测试")
    win.resize(800, 500)

    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)

    table = ConditionFormatTable()
    table.set_data(df)

    info_label = QLabel("选中行: (无)")

    # 连接选择信号
    @Slot(list)
    def on_selection(selected: List[int]) -> None:
        info_label.setText(f"选中行: {selected}")

    table.selectionChanged.connect(on_selection)

    # 按钮区域
    btn_layout = QHBoxLayout()
    btn_get = QPushButton("获取选中行")
    btn_get.clicked.connect(lambda: info_label.setText(f"选中行: {table.get_selected_rows()}"))
    btn_layout.addWidget(btn_get)
    btn_layout.addStretch()

    layout.addWidget(table, 1)
    layout.addLayout(btn_layout)
    layout.addWidget(info_label)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()