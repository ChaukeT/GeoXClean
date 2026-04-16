"""
Reusable results display widgets for estimation and simulation panels.

All estimation panels should embed the same widgets so the user gets a
consistent, professional-grade results display regardless of method.

Widgets:
- GradeStatsWidget  — min/max/mean/median/std table with optional per-domain
- CVTableWidget     — LOO-CV metrics table (slope, R², RMSE, coverage)
- ScatterPlotWidget — Actual vs Estimated scatter with 1:1 line
- SwathPlotWidget   — Grade trends along X, Y, Z axes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QGroupBox, QLabel, QComboBox,
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════
# Grade Statistics Widget
# ═══════════════════════════════════════════════════════════════════

class GradeStatsWidget(QWidget):
    """Table showing min/max/mean/median/std of estimated grades."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._group = QGroupBox("Grade Statistics")
        gl = QVBoxLayout(self._group)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        gl.addWidget(self._table)
        layout.addWidget(self._group)

    def update_stats(
        self,
        grades: np.ndarray,
        variable: str = "Grade",
        unit: str = "",
        domain: str = "",
    ) -> None:
        """Populate the table from an array of grade values."""
        finite = grades[np.isfinite(grades)] if grades is not None else np.array([])
        label = f"{variable} ({unit})" if unit else variable
        if domain:
            label = f"{label} [{domain}]"

        rows = [
            ("Variable", label),
            ("Count (finite)", f"{len(finite)}"),
            ("Mean", f"{np.mean(finite):.4f}" if len(finite) else "N/A"),
            ("Median", f"{np.median(finite):.4f}" if len(finite) else "N/A"),
            ("Std Dev", f"{np.std(finite):.4f}" if len(finite) else "N/A"),
            ("Min", f"{np.min(finite):.4f}" if len(finite) else "N/A"),
            ("Max", f"{np.max(finite):.4f}" if len(finite) else "N/A"),
            ("P5", f"{np.percentile(finite, 5):.4f}" if len(finite) else "N/A"),
            ("P95", f"{np.percentile(finite, 95):.4f}" if len(finite) else "N/A"),
        ]
        self._table.setRowCount(len(rows))
        for i, (name, val) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem(val))


# ═══════════════════════════════════════════════════════════════════
# CV Table Widget
# ═══════════════════════════════════════════════════════════════════

class CVTableWidget(QWidget):
    """LOO-CV metrics table with optional per-domain breakdown."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._group = QGroupBox("Cross-Validation")
        gl = QVBoxLayout(self._group)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        gl.addWidget(self._table)
        layout.addWidget(self._group)

    def update_cv(self, cv_result: Optional[Dict[str, Any]]) -> None:
        """Populate from a cv_result dict (same format as ARBF/OK CV)."""
        if cv_result is None:
            self._table.setRowCount(1)
            self._table.setItem(0, 0, QTableWidgetItem("Status"))
            self._table.setItem(0, 1, QTableWidgetItem("CV not run"))
            return

        def _fmt(key, fmt=".4f"):
            v = cv_result.get(key)
            if v is None:
                return "N/A"
            return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)

        rows = [
            ("R²", _fmt("R2", ".3f")),
            ("RMSE", _fmt("RMSE")),
            ("Slope", _fmt("SLOPE", ".3f")),
            ("Mean Error", _fmt("ME")),
            ("MAE", _fmt("MAE")),
            ("Coverage 90%", _fmt("COVER_90", ".1%")),
            ("Coverage 95%", _fmt("COVER_95", ".1%")),
        ]

        # Per-domain breakdown
        per_domain = cv_result.get("per_domain")
        if per_domain and isinstance(per_domain, dict):
            rows.append(("", ""))
            rows.append(("── Per-Domain ──", ""))
            for dkey, dcv in per_domain.items():
                r2 = dcv.get("R2", float("nan"))
                rmse = dcv.get("RMSE", float("nan"))
                slope = dcv.get("SLOPE", float("nan"))
                n = len(dcv.get("actual", []))
                rows.append((
                    f"  {dkey}",
                    f"R²={r2:.3f}  RMSE={rmse:.4f}  Slope={slope:.3f}  (n={n})",
                ))

        self._table.setRowCount(len(rows))
        for i, (name, val) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem(val))


# ═══════════════════════════════════════════════════════════════════
# Scatter Plot Widget (Actual vs Estimated)
# ═══════════════════════════════════════════════════════════════════

class ScatterPlotWidget(QWidget):
    """Matplotlib scatter plot: actual vs estimated values."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_MPL:
            layout.addWidget(QLabel("Matplotlib not available"))
            self._canvas = None
            return

        self._fig = Figure(figsize=(4, 4), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)

    def update_scatter(
        self,
        actual: np.ndarray,
        estimated: np.ndarray,
        variable: str = "Grade",
        unit: str = "",
    ) -> None:
        """Plot actual vs estimated with 1:1 line and regression."""
        if self._canvas is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        valid = np.isfinite(actual) & np.isfinite(estimated)
        a, e = actual[valid], estimated[valid]
        if len(a) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        ax.scatter(a, e, s=8, alpha=0.5, c="#2196F3", edgecolors="none")

        # 1:1 line
        lo = min(a.min(), e.min())
        hi = max(a.max(), e.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")

        # Regression
        if len(a) > 2:
            slope, intercept = np.polyfit(a, e, 1)
            ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                    "r-", lw=1, label=f"Slope={slope:.3f}")

        label = f"{variable} ({unit})" if unit else variable
        ax.set_xlabel(f"Actual {label}")
        ax.set_ylabel(f"Estimated {label}")
        ax.set_title("Cross-Validation: Actual vs Estimated")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════
# Swath Plot Widget
# ═══════════════════════════════════════════════════════════════════

class SwathPlotWidget(QWidget):
    """Swath plots showing grade trends along X, Y, Z axes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_MPL:
            layout.addWidget(QLabel("Matplotlib not available"))
            self._canvas = None
            return

        # Axis selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["X", "Y", "Z"])
        self._axis_combo.currentTextChanged.connect(self._on_axis_changed)
        top.addWidget(self._axis_combo)
        top.addStretch()
        layout.addLayout(top)

        self._fig = Figure(figsize=(5, 3), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)

        self._swath_data: Optional[Dict] = None

    def update_swath(self, swath_data: Dict[str, Any]) -> None:
        """Store swath data and redraw current axis.

        Expected format: {"x": {"slice_positions": [...], "mean_estimated": [...],
        "mean_actual": [...]}, "y": {...}, "z": {...}}
        """
        self._swath_data = swath_data
        self._on_axis_changed(self._axis_combo.currentText())

    def _on_axis_changed(self, axis: str) -> None:
        if self._canvas is None or self._swath_data is None:
            return
        key = axis.lower()
        data = self._swath_data.get(key)
        if data is None:
            return

        self._fig.clear()
        ax = self._fig.add_subplot(111)

        positions = data.get("slice_positions", [])
        est = data.get("mean_estimated", [])
        act = data.get("mean_actual", [])

        if positions and est:
            ax.plot(positions, est, "b-o", ms=4, label="Estimated (blocks)")
        if positions and act:
            ax.plot(positions, act, "r--s", ms=4, label="Actual (composites)")

        ax.set_xlabel(f"{axis} coordinate")
        ax.set_ylabel("Mean grade")
        ax.set_title(f"Swath Plot — {axis} axis")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw_idle()
