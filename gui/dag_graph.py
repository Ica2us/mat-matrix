"""
gui/dag_graph.py — 交互式 Merkle DAG 分支拓扑画布
====================================================

功能
----
1. 全景拓扑可视化（泳道 + 分支着色 + 节点微缩卡片）
2. 小地图 (Minimap)
3. 节点 hover 高亮 + 点击选中
4. 分叉 (Fork) 操作：选中节点 → 弹出分支对话框
5. 对比模式：勾选两个节点 → 跨分支三栏 Diff
6. 分支筛选：图例点击 + 工具栏 checkbox

信号
----
- node_selected(commit_id: str | None)
- compare_requested(commit_a: str, commit_b: str)
- fork_requested(parent_commit_id: str, branch_name: str, purpose: str)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, cast

os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# ── 分支调色板（最多 20 种） ──────────────────────────────────
BRANCH_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


# =====================================================================
# Fork 对话框
# =====================================================================

class ForkDialog(QDialog):
    """弹出对话框：输入新分支名称和探索目的。"""

    def __init__(self, parent_commit_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("🌿 新建实验分支")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        # 父节点信息（只读）
        parent_label = QLabel(parent_commit_id[:16] + "...")
        parent_label.setStyleSheet("font-family: Consolas; background: #f0f0f0; padding: 2px 4px;")
        layout.addRow("父节点:", parent_label)

        self.branch_edit = QLineEdit()
        self.branch_edit.setPlaceholderText("例: 高温探索-回火")
        layout.addRow("分支名称:", self.branch_edit)

        self.purpose_edit = QTextEdit()
        self.purpose_edit.setPlaceholderText("描述本次分支的探索目的...")
        self.purpose_edit.setMaximumHeight(80)
        layout.addRow("探索目的:", self.purpose_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def branch_name(self) -> str:
        return self.branch_edit.text().strip()

    def purpose(self) -> str:
        return self.purpose_edit.toPlainText().strip()


# =====================================================================
# 分支筛选面板（内嵌在工具栏下方）
# =====================================================================

class BranchFilterBar(QWidget):
    """Checkbox 组 + 全选/取消，控制画布上分支的显隐。"""

    filter_changed = Signal(set)  # visible branch names

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checkboxes: Dict[str, QCheckBox] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._all_btn = QPushButton("全选")
        self._all_btn.clicked.connect(self._select_all)
        layout.addWidget(self._all_btn)
        self._none_btn = QPushButton("取消")
        self._none_btn.clicked.connect(self._select_none)
        layout.addWidget(self._none_btn)
        layout.addStretch()

    def set_branches(self, branches: List[str]) -> None:
        """Rebuild checkbox group for the given branch list."""
        self._checkboxes.clear()
        layout = self.layout()
        assert layout is not None
        # Remove old checkboxes (keep buttons)
        while layout.count() > 2:
            item = layout.takeAt(layout.count() - 1)
            if item and item.widget():
                item.widget().deleteLater()
        for b in sorted(branches):
            cb = QCheckBox(b)
            cb.setChecked(True)
            cb.stateChanged.connect(self._emit_filter)
            self._checkboxes[b] = cb
            layout.addWidget(cb)

    def visible_branches(self) -> Set[str]:
        return {name for name, cb in self._checkboxes.items() if cb.isChecked()}

    def _select_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _select_none(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def _emit_filter(self) -> None:
        self.filter_changed.emit(self.visible_branches())


# =====================================================================
# 核心 DAG 画布
# =====================================================================

class DagGraphCanvas(FigureCanvasQTAgg):
    """交互式 Merkle DAG 拓扑图画布。

    信号
    ----
    node_selected(commit_id: str | None)
        用户点击了某个节点（或点击空白取消选中）
    compare_requested(commit_a: str, commit_b: str)
        用户勾选了两个节点后点击「对比」
    fork_requested(parent_commit_id: str, branch_name: str, purpose: str)
        用户选中节点后发起分叉操作
    """

    node_selected = Signal(object)  # str | None
    compare_requested = Signal(str, str)
    fork_requested = Signal(str, str, str)

    def __init__(
        self,
        vcs: Any,  # MerkleDAGVersionControl
        parent: Optional[QWidget] = None,
    ) -> None:
        self.fig = Figure(figsize=(10, 6), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)

        self._vcs = vcs
        self._commits: List[Dict[str, Any]] = []
        self._G: nx.DiGraph = nx.DiGraph()
        self._pos: Dict[str, np.ndarray] = {}
        self._branch_colors: Dict[str, str] = {}
        self._selected_node: Optional[str] = None
        self._compare_nodes: Set[str] = set()
        self._compare_mode: bool = False
        self._hidden_branches: Set[str] = set()
        self._layout_name: str = "分层 (dot)"

        # 主图 + 小地图
        self._ax = self.fig.add_subplot(111)
        self._minimap_ax: Optional[Any] = None

        # 交互状态
        self._dragging: bool = False
        self._drag_start: Optional[Tuple[float, float]] = None
        self._pan_start: Optional[Tuple[float, float]] = None
        self._hovered_node: Optional[str] = None

        # ── 连接事件 ───────────────────────────────────────────
        self.mpl_connect("pick_event", self._on_pick)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("axes_enter_event", self._on_axes_enter)

        self.fig.tight_layout()

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def load_commits(
        self,
        commits: List[Dict[str, Any]],
        layout_name: str = "分层 (dot)",
    ) -> None:
        """Load commits into the graph, compute layout, and render."""
        self._commits = commits
        self._layout_name = layout_name
        self._build_graph()
        self._compute_layout()
        self._assign_colors()
        self._render()

    def set_compare_mode(self, enabled: bool) -> None:
        """Toggle compare mode: nodes become selectable for pairwise diff."""
        self._compare_mode = enabled
        if not enabled:
            self._compare_nodes.clear()
        self._render()

    def set_hidden_branches(self, branches: Set[str]) -> None:
        """Set which branches should be hidden from the graph."""
        self._hidden_branches = branches
        self._render()

    @property
    def selected_node(self) -> Optional[str]:
        return self._selected_node

    @property
    def compare_nodes(self) -> List[str]:
        return sorted(self._compare_nodes)

    # -----------------------------------------------------------------
    # internal — graph building
    # -----------------------------------------------------------------

    def _build_graph(self) -> None:
        """Build NetworkX DiGraph from commits."""
        G = nx.DiGraph()
        commit_map = {c["commit_id"]: c for c in self._commits}
        sorted_commits = sorted(self._commits, key=lambda c: c.get("created_at", ""))

        for c in sorted_commits:
            cid = c["commit_id"]
            G.add_node(
                cid,
                label=cid[:8],
                branch=c.get("branch", "main"),
                author=c.get("author", ""),
                created_at=c.get("created_at", ""),
                message=c.get("message", ""),
                parameters=c.get("parameters", {}),
            )
            for pid in c.get("parents", []):
                if pid in commit_map:
                    G.add_edge(pid, cid)

        self._G = G

    def _compute_layout(self) -> None:
        """Compute node positions using the selected layout algorithm."""
        G = self._G
        if len(G.nodes) == 0:
            self._pos = {}
            return

        layout_name = self._layout_name
        if "分层" in layout_name:
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except Exception:
                pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        elif "弹簧" in layout_name:
            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        elif "圆形" in layout_name:
            pos = nx.circular_layout(G)
        elif "谱" in layout_name:
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

        self._pos = pos

    def _assign_colors(self) -> None:
        """Assign a color to each unique branch."""
        branches = sorted({
            self._G.nodes[n].get("branch", "main")
            for n in self._G.nodes()
        })
        self._branch_colors = {}
        for i, b in enumerate(branches):
            self._branch_colors[b] = BRANCH_PALETTE[i % len(BRANCH_PALETTE)]

    # -----------------------------------------------------------------
    # internal — rendering
    # -----------------------------------------------------------------

    def _render(self) -> None:
        """Full redraw of the graph on the main axis + minimap."""
        self.fig.clear()
        self._ax = self.fig.add_subplot(111)

        G = self._G
        pos = self._pos

        if len(G.nodes) == 0:
            self._ax.text(0.5, 0.5, "暂无提交记录",
                          ha="center", va="center", fontsize=14,
                          transform=self._ax.transAxes)
            self._ax.axis("off")
            self.draw_idle()
            return

        # Determine visible nodes
        visible_nodes = [
            n for n in G.nodes()
            if G.nodes[n].get("branch", "main") not in self._hidden_branches
        ]
        if not visible_nodes:
            self._ax.text(0.5, 0.5, "所有分支已隐藏",
                          ha="center", va="center", fontsize=14,
                          transform=self._ax.transAxes)
            self._ax.axis("off")
            self.draw_idle()
            return

        visible_set = set(visible_nodes)
        visible_edges = [(u, v) for u, v in G.edges()
                         if u in visible_set and v in visible_set]

        # ── 节点颜色 ────────────────────────────────────────────
        node_colors = []
        node_edge_colors = []
        node_sizes = []
        for n in visible_nodes:
            branch = G.nodes[n].get("branch", "main")
            color = self._branch_colors.get(branch, "#7f7f7f")
            node_colors.append(color)

            if n == self._selected_node:
                node_edge_colors.append("#000000")
            elif n in self._compare_nodes:
                node_edge_colors.append("#ff6600")
            else:
                node_edge_colors.append("#ffffff")

            node_sizes.append(320 if n == self._selected_node else 280)

        # ── 边 ──────────────────────────────────────────────────
        edge_color = "#555555"

        # ── 绘制 ────────────────────────────────────────────────
        self._ax.set_title("Merkle DAG 分支拓扑", fontsize=14, fontweight="bold")
        self._ax.axis("off")

        # 边
        nx.draw_networkx_edges(
            G, pos, ax=self._ax, edgelist=visible_edges,
            arrows=True, arrowsize=12, arrowstyle="->",
            edge_color=edge_color, alpha=0.5,
            min_source_margin=18, min_target_margin=18,
        )

        # 节点
        nx.draw_networkx_nodes(
            G, pos, ax=self._ax, nodelist=visible_nodes,
            node_size=node_sizes, node_color=node_colors,
            edgecolors=node_edge_colors, linewidths=2.0 if self._compare_mode else 0.8,
            alpha=0.92, picker=8,
        )

        # 标签（截断为 8 位）
        labels = {n: G.nodes[n].get("label", n[:8]) for n in visible_nodes}
        nx.draw_networkx_labels(
            G, pos, ax=self._ax, labels=labels,
            font_size=7, font_family="sans-serif",
        )

        # ── 图例 ────────────────────────────────────────────────
        legend_patches = []
        for b in sorted(self._branch_colors):
            if b in self._hidden_branches:
                continue
            legend_patches.append(Patch(
                color=self._branch_colors[b], label=b, alpha=0.8
            ))
        if legend_patches:
            legend = self._ax.legend(
                handles=legend_patches, loc="upper left",
                fontsize=7, framealpha=0.8, title="分支",
            )
            legend.get_title().set_fontsize(8)

        # ── 选中状态信息 ────────────────────────────────────────
        info_lines = []
        if self._selected_node:
            c = G.nodes[self._selected_node]
            info_lines.append(f"✓ 选中: {self._selected_node[:16]}...")
            info_lines.append(f"  分支: {c.get('branch','')} | 作者: {c.get('author','')}")
        if len(self._compare_nodes) == 1:
            info_lines.append(f"ⓘ 对比: 已选 1 个 (还需 1 个)")
        elif len(self._compare_nodes) == 2:
            a, b = sorted(self._compare_nodes)
            info_lines.append(f"ⓘ 对比: {a[:12]}... ↔ {b[:12]}...")
        if info_lines:
            self._ax.text(
                0.98, 0.02, "\n".join(info_lines),
                transform=self._ax.transAxes,
                fontsize=8, fontfamily="Consolas",
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffdd", alpha=0.85),
            )

        # ── 小地图 (Minimap) ────────────────────────────────────
        self._draw_minimap(visible_nodes)

        self.fig.tight_layout()
        self.draw_idle()

    def _draw_minimap(self, visible_nodes: List[str]) -> None:
        """Draw a small overview map in the bottom-right corner."""
        G = self._G
        pos = self._pos
        if len(G.nodes) < 5:
            return  # too small for minimap

        # Add a sub-axes in the corner
        self._minimap_ax = self.fig.add_axes([0.78, 0.01, 0.20, 0.18])
        self._minimap_ax.axis("off")

        # Draw all nodes as tiny dots
        for n in G.nodes():
            x, y = pos[n]
            branch = G.nodes[n].get("branch", "main")
            color = self._branch_colors.get(branch, "#7f7f7f")
            self._minimap_ax.plot(x, y, "o", color=color, markersize=2,
                                  alpha=0.6, markeredgewidth=0)

        # Draw visible edges
        for u, v in G.edges():
            if u in pos and v in pos:
                xu, yu = pos[u]
                xv, yv = pos[v]
                self._minimap_ax.plot(
                    [xu, xv], [yu, yv], color="#cccccc",
                    linewidth=0.3, alpha=0.4,
                )

        # Viewport rectangle
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        rect = Rectangle(
            (xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
            linewidth=0.8, edgecolor="red", facecolor="none", alpha=0.6,
        )
        self._minimap_ax.add_patch(rect)
        self._minimap_ax.set_xlim(self._ax.get_xlim())
        self._minimap_ax.set_ylim(self._ax.get_ylim())

    # -----------------------------------------------------------------
    # event handlers
    # -----------------------------------------------------------------

    def _on_pick(self, event: PickEvent) -> None:
        """Node was clicked — select it or add to compare set."""
        if not hasattr(event, "artist"):
            return
        artist = event.artist
        if artist not in self._ax.collections:
            return

        # Find the nearest node
        if event.mouseevent is None:
            return
        click_x, click_y = event.mouseevent.xdata, event.mouseevent.ydata
        if click_x is None or click_y is None:
            return

        nearest = self._find_nearest_node(click_x, click_y)
        if nearest is None:
            return

        if self._compare_mode:
            if nearest in self._compare_nodes:
                self._compare_nodes.remove(nearest)
            else:
                self._compare_nodes.add(nearest)
            if len(self._compare_nodes) > 2:
                # Keep the two most recent selections
                self._compare_nodes = set(sorted(self._compare_nodes)[-2:])
        else:
            self._selected_node = nearest if nearest != self._selected_node else None
            self.node_selected.emit(self._selected_node)

        self._render()

    def _on_click(self, event: MouseEvent) -> None:
        """Handle mouse press — start pan or handle toolbar clicks."""
        if event.button == MouseButton.LEFT and event.inaxes == self._ax:
            self._drag_start = (event.xdata, event.ydata) if event.xdata is not None else None
            self._pan_start = (self._ax.get_xlim(), self._ax.get_ylim())
            self._dragging = True

        # Right-click → deselect
        if event.button == MouseButton.RIGHT:
            self._selected_node = None
            self._compare_nodes.clear()
            self.node_selected.emit(None)
            self._render()

    def _on_release(self, event: MouseEvent) -> None:
        self._dragging = False
        self._drag_start = None

    def _on_motion(self, event: MouseEvent) -> None:
        """Handle panning and hover highlighting."""
        if self._dragging and self._drag_start and event.xdata is not None:
            dx = event.xdata - self._drag_start[0]
            dy = event.ydata - self._drag_start[1]
            if self._pan_start:
                xlim, ylim = self._pan_start
                self._ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
                self._ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
                self.draw_idle()
            return

        # Hover detection
        if event.inaxes != self._ax or event.xdata is None:
            return
        nearest = self._find_nearest_node(event.xdata, event.ydata)
        if nearest != self._hovered_node:
            self._hovered_node = nearest
            if nearest:
                self._ax.set_title(
                    f"Merkle DAG 分支拓扑 — {nearest[:16]}... "
                    f"({self._G.nodes[nearest].get('branch','')}) "
                    f"@{self._G.nodes[nearest].get('author','')}",
                    fontsize=13, fontweight="bold",
                )
            else:
                self._ax.set_title("Merkle DAG 分支拓扑", fontsize=14, fontweight="bold")
            self.draw_idle()

    def _on_scroll(self, event: MouseEvent) -> None:
        """Zoom in/out with mouse wheel."""
        if event.inaxes != self._ax:
            return
        scale = 1.15 if event.button == "up" else 0.85
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        self._ax.set_xlim(cx - new_w / 2, cx + new_w / 2)
        self._ax.set_ylim(cy - new_h / 2, cy + new_h / 2)
        self._draw_minimap(list(self._G.nodes()))
        self.draw_idle()

    def _on_axes_enter(self, event: Any) -> None:
        """Focus on the main axes when entering."""
        pass

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _find_nearest_node(self, x: float, y: float, max_dist: float = 0.08) -> Optional[str]:
        """Find the nearest graph node within *max_dist* (normalized)."""
        if not self._pos:
            return None
        # Compute bounding box for normalization
        xs = [p[0] for p in self._pos.values()]
        ys = [p[1] for p in self._pos.values()]
        if not xs or not ys:
            return None
        x_range = max(xs) - min(xs) or 1
        y_range = max(ys) - min(ys) or 1

        best: Optional[str] = None
        best_dist = float("inf")
        for n, (nx, ny) in self._pos.items():
            dn = ((nx - x) / x_range) ** 2 + ((ny - y) / y_range) ** 2
            if dn < best_dist:
                best_dist = dn
                best = n
        return best if best_dist < max_dist ** 2 else None


# =====================================================================
# 独立测试入口
# =====================================================================

def main() -> None:
    """Standalone test: build a mock DAG and show the canvas."""
    app = QApplication(sys.argv)

    from storage.vcs import MerkleDAGVersionControl
    vcs = MerkleDAGVersionControl()

    # Seed some commits
    vcs.commit_version(None, ["init"], {}, "hash0", author="alice", message="root")
    c1 = vcs.commit_version(
        vcs.list_commits(limit=1)[0]["commit_id"],
        ["step_a"], {"T": 150, "t": 30}, "hash1",
        author="alice", message="batch 1", branch="main",
    )
    c2 = vcs.commit_version(
        vcs.list_commits(branch="main", limit=1)[0]["commit_id"],
        ["step_b"], {"T": 180, "t": 45}, "hash2",
        author="bob", message="high temp", branch="高温探索",
    )
    c3 = vcs.commit_version(
        vcs.list_commits(branch="高温探索", limit=1)[0]["commit_id"],
        ["step_c"], {"T": 200, "t": 60}, "hash3",
        author="bob", message="even hotter", branch="高温探索",
    )
    # Create a merge commit
    main_head = vcs.list_commits(branch="main", limit=1)[0]["commit_id"]
    vcs.commit_version(
        [main_head, c2],
        ["step_d"], {"T": 165, "t": 40}, "hash4",
        author="alice", message="merge", branch="main",
    )

    window = QMainWindow()
    window.setWindowTitle("DAG Graph Test")
    window.resize(1000, 700)

    canvas = DagGraphCanvas(vcs)
    commits = vcs.list_commits(branch=None, limit=100)
    canvas.load_commits(commits)

    # Toolbar
    toolbar = QToolBar()
    compare_btn = QPushButton("🔍 对比模式")
    compare_btn.setCheckable(True)

    def _toggle_compare(checked: bool) -> None:
        canvas.set_compare_mode(checked)
        compare_btn.setText("🔍 对比: ON" if checked else "🔍 对比模式")
    compare_btn.toggled.connect(_toggle_compare)
    toolbar.addWidget(compare_btn)

    compare_go_btn = QPushButton("📊 对比选中")
    def _do_compare() -> None:
        nodes = canvas.compare_nodes
        if len(nodes) == 2:
            canvas.compare_requested.emit(nodes[0], nodes[1])
            QMessageBox.information(window, "对比", f"对比: {nodes[0][:12]}... ↔ {nodes[1][:12]}...")
        else:
            QMessageBox.warning(window, "对比", "请勾选恰好 2 个节点")
    compare_go_btn.clicked.connect(_do_compare)
    toolbar.addWidget(compare_go_btn)

    layout_combo = QComboBox()
    layout_combo.addItems(["分层 (dot)", "弹簧 (spring)", "圆形 (circular)", "谱 (spectral)"])
    def _relayout(name: str) -> None:
        commits = vcs.list_commits(branch=None, limit=100)
        canvas.load_commits(commits, layout_name=name)
    layout_combo.currentTextChanged.connect(_relayout)
    toolbar.addWidget(QLabel("  布局:"))
    toolbar.addWidget(layout_combo)

    central = QWidget()
    clayout = QVBoxLayout(central)
    clayout.addWidget(toolbar)
    clayout.addWidget(canvas, 1)
    window.setCentralWidget(central)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()