"""
gui/pareto_canvas.py
====================
ParetoCanvas -- a FigureCanvasQTAgg widget for visualising the Pareto front.

Interface (per widget_interface_spec.txt)
-----------------------------------------
    set_data(metrics, directions, labels) -> None
    get_selected_indices() -> List[int]
    point_clicked(point_index: int)  ... signal emitted on click

Rendering
---------
- All points: gray scatter.
- Pareto-optimal points: red scatter with a stepped Pareto frontier (red line).
- Hover on a Pareto point shows a tooltip with coordinates and label.
- Click on a Pareto point emits ``point_clicked``.
"""

from typing import Any, List, Optional

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from engine.bo import MaterialBayesianOptimizer


class ParetoCanvas(FigureCanvasQTAgg):
    """FigureCanvasQTAgg that draws a 2-D scatter plot with the Pareto front highlighted."""

    point_clicked = Signal(int)  # index of the clicked Pareto point

    def __init__(self, parent: Any = None) -> None:
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)  # type: ignore[no-untyped-call]

        self._metrics: Optional[np.ndarray] = None
        self._directions: List[str] = []
        self._labels: List[str] = []
        self._pareto_mask: Optional[np.ndarray] = None
        self._selected_indices: List[int] = []

        # Internal artist references so we can clear on set_data
        self._scatter_all = None
        self._scatter_pareto = None
        self._step_line = None

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

    # ------------------------------------------------------------------
    # Public interface (widget_interface_spec.txt)
    # ------------------------------------------------------------------

    def set_data(
        self,
        metrics: np.ndarray,
        directions: List[str],
        labels: List[str],
    ) -> None:
        """
        Set the data to display.

        Parameters
        ----------
        metrics : np.ndarray, shape (n_points, 2)
            Two objective columns.
        directions : List[str]
            "maximize" or "minimize" for each objective.
        labels : List[str]
            Labels for each point (used in tooltips).
        """
        self._metrics = metrics
        self._directions = directions
        self._labels = labels
        self._selected_indices = []

        # Compute Pareto mask via MaterialBayesianOptimizer
        self._pareto_mask = MaterialBayesianOptimizer().calculate_pareto_front(metrics, directions)

        self._redraw()

    def get_selected_indices(self) -> List[int]:
        """Return the list of currently selected (clicked) Pareto point indices."""
        return self._selected_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        """Clear axes, re-plot everything."""
        self.ax.clear()

        if self._metrics is None or self._metrics.shape[0] == 0:
            self.ax.set_title("No data")
            self.draw()  # type: ignore[no-untyped-call]
            return

        metrics = self._metrics
        pareto_mask = self._pareto_mask
        if pareto_mask is None:
            pareto_mask = np.zeros(metrics.shape[0], dtype=bool)

        # -- all points (gray) --
        self.ax.scatter(
            metrics[:, 0],
            metrics[:, 1],
            c="lightgray",
            edgecolors="gray",
            alpha=0.7,
            s=40,
            zorder=1,
        )

        # -- Pareto points (red) --
        pareto_idx = np.where(pareto_mask)[0]
        if len(pareto_idx) > 0:
            pareto_pts = metrics[pareto_idx]
            self.ax.scatter(
                pareto_pts[:, 0],
                pareto_pts[:, 1],
                c="red",
                edgecolors="darkred",
                s=60,
                zorder=3,
            )

            # -- Step-line: sort by x then y in the "better" direction --
            # For Pareto visualisation we order front points so the line forms
            # a step from the "best" corner.  Sort by the first objective
            # ascending; if both are "minimize", this is natural.
            sorted_idx = pareto_idx[np.argsort(metrics[pareto_idx, 0])]
            sorted_pts = metrics[sorted_idx]

            # Build step vertices: from each Pareto point, go horizontal then
            # vertical to the next point.
            x_step: List[float] = [float(sorted_pts[0, 0])]
            y_step: List[float] = [float(sorted_pts[0, 1])]
            for k in range(1, len(sorted_pts)):
                # horizontal leg
                x_step.append(float(sorted_pts[k, 0]))
                y_step.append(float(sorted_pts[k - 1, 1]))
                # vertical leg
                x_step.append(float(sorted_pts[k, 0]))
                y_step.append(float(sorted_pts[k, 1]))

            self.ax.plot(
                x_step,
                y_step,
                color="red",
                linewidth=1.5,
                linestyle="-",
                zorder=2,
            )

        # Labels
        xlabel = f"Objective 0 ({self._directions[0]})" if self._directions else "Obj 0"
        ylabel = f"Objective 1 ({self._directions[1]})" if len(self._directions) > 1 else "Obj 1"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title("Pareto Front")

        self.draw()  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_click(self, event: MouseEvent) -> None:
        """Emit ``point_clicked`` when a Pareto-optimal point is clicked."""
        if (
            event.inaxes is not self.ax
            or self._metrics is None
            or self._pareto_mask is None
        ):
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        idx = self._hit_test_pareto_point(x, y)
        if idx is not None:
            self._selected_indices = [idx]
            self.point_clicked.emit(idx)

    def _on_hover(self, event: MouseEvent) -> None:
        """Show tooltip when hovering over a Pareto-optimal point."""
        if (
            event.inaxes is not self.ax
            or self._metrics is None
            or self._pareto_mask is None
        ):
            self._clear_tooltip()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._clear_tooltip()
            return
        idx = self._hit_test_pareto_point(x, y)
        if idx is not None:
            label = self._labels[idx] if idx < len(self._labels) else f"Point {idx}"
            self.ax.set_title(
                f"Pareto Front\nPoint {idx}: {label}\n"
                f"({self._metrics[idx, 0]:.4g}, {self._metrics[idx, 1]:.4g})",
                fontsize=10,
            )
        else:
            self._clear_tooltip()
        self.draw_idle()  # type: ignore[no-untyped-call]

    def _hit_test_pareto_point(
        self, x: float, y: float
    ) -> Optional[int]:
        """Return the index of the Pareto point within ``radius`` pixels, or None."""
        if self._metrics is None or self._pareto_mask is None:
            return None
        pareto_idx = np.where(self._pareto_mask)[0]
        if len(pareto_idx) == 0:
            return None

        pareto_pts = self._metrics[pareto_idx]
        # Transform to display coordinates
        disp = self.ax.transData.transform(pareto_pts)
        click_disp = self.ax.transData.transform([[x, y]])[0]
        dists = np.sqrt(np.sum((disp - click_disp) ** 2, axis=1))
        closest = int(dists.argmin())
        if dists[closest] < 12:  # 12-pixel hit radius
            return int(pareto_idx[closest])
        return None

    def _clear_tooltip(self) -> None:
        """Reset the title to the default (no hover text)."""
        self.ax.set_title("Pareto Front", fontsize=10)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

def main() -> None:
    """Run ParetoCanvas as a standalone window for testing."""
    import sys

    app = QApplication(sys.argv)

    # Synthetic 2-objective data (minimize both)
    rng = np.random.default_rng(42)
    metrics = rng.random((50, 2))
    directions = ["minimize", "minimize"]
    labels = [f"Exp {i}" for i in range(metrics.shape[0])]

    window = QMainWindow()
    window.setWindowTitle("Pareto Front Test")
    central = QWidget()
    layout = QVBoxLayout(central)
    canvas = ParetoCanvas()
    layout.addWidget(canvas)
    window.setCentralWidget(central)
    window.resize(600, 500)

    canvas.set_data(metrics, directions, labels)

    def on_clicked(idx: int) -> None:
        print(f"[point_clicked] Index {idx}: {labels[idx]} = ({metrics[idx, 0]:.4f}, {metrics[idx, 1]:.4f})")

    canvas.point_clicked.connect(on_clicked)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
