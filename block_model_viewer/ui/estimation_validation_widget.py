"""
Reusable Estimation Validation Widget
======================================

Provides publication-quality CV scatter, swath, and QA diagnostics tabs
for ANY estimation method (OK, SK, UK, ARBF, RBF, etc.).

Reference quality: ARBF estimation panel (arbf_estimation_panel.py:2185-2462).
All estimation panels should embed this widget for consistent QC output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QButtonGroup, QRadioButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from .design_tokens import tokens
    from .modern_styles import ModernColors
    _c = tokens.colors()
    _BG = _c.BG_SURFACE
    _TXT = _c.TEXT_PRIMARY
    _TXT2 = _c.TEXT_SECONDARY
    _BDR = _c.BORDER_SUBTLE
except Exception:
    _BG, _TXT, _TXT2, _BDR = "#252526", "#d4d4d4", "#a0a0a0", "#3c3c3c"

logger = logging.getLogger(__name__)


class EstimationValidationWidget(QWidget):
    """Embeddable validation widget with CV scatter, swath plots, and QA stats.

    Usage from any estimation panel::

        self._validation_widget = EstimationValidationWidget()
        self.results_tabs.addTab(self._validation_widget, "Validation")

        # After estimation completes:
        self._validation_widget.update_results(payload)

    The *payload* dict may contain any subset of:
    - ``cv_result``: object/dict with actual, estimated, r_squared, rmse,
      slope_of_regression, intercept, mean_error, mae
    - ``swath_data`` or ``swath_plots``: per-axis swath data
    - ``qa_summary``: dict with kriging_efficiency_mean, slope_of_regression_mean, etc.
    - ``kriging_variance``: per-block variance array (for standardised errors)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cv_fig: Optional[Figure] = None
        self._swath_fig: Optional[Figure] = None
        self._swath_cache: Optional[dict] = None
        self._current_swath_axis: str = "X"
        self._setup_ui()

    # ──────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()

        # Tab 1: CV Scatter
        self._cv_widget = QWidget()
        self._cv_layout = QVBoxLayout(self._cv_widget)
        self._cv_layout.setContentsMargins(4, 4, 4, 4)
        self._cv_layout.addWidget(QLabel("Run estimation with cross-validation to see results."))
        self._tabs.addTab(self._cv_widget, "CV Scatter")

        # Tab 2: Swath Plots
        self._swath_widget = QWidget()
        swath_outer = QVBoxLayout(self._swath_widget)
        swath_outer.setContentsMargins(4, 4, 4, 4)
        btn_bar = QHBoxLayout()
        self._swath_btns: Dict[str, QRadioButton] = {}
        btn_group = QButtonGroup(self)
        for axis in ["X", "Y", "Z"]:
            btn = QRadioButton(axis)
            btn.setChecked(axis == "X")
            btn.toggled.connect(lambda checked, a=axis: self._on_swath_axis(a) if checked else None)
            btn_bar.addWidget(btn)
            btn_group.addButton(btn)
            self._swath_btns[axis] = btn
        btn_bar.addStretch()
        swath_outer.addLayout(btn_bar)
        self._swath_layout = QVBoxLayout()
        self._swath_layout.addWidget(QLabel("Swath plots will appear after estimation."))
        swath_outer.addLayout(self._swath_layout)
        self._tabs.addTab(self._swath_widget, "CV Swath")

        # Tab 3: Statistics table
        self._stats_widget = QWidget()
        stats_layout = QVBoxLayout(self._stats_widget)
        self._stats_table = QTableWidget()
        self._stats_table.setColumnCount(2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self._stats_table)
        self._tabs.addTab(self._stats_widget, "CV Statistics")

        layout.addWidget(self._tabs)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def update_results(self, payload: Dict[str, Any]):
        """Populate all tabs from an estimation results payload."""
        if payload is None:
            return
        cv = payload.get("cv_result")
        if cv is not None:
            self._populate_cv_scatter(cv)
            self._populate_stats_table(cv, payload.get("qa_summary"))

        swath = (
            payload.get("support_swath_data")
            or payload.get("swath_plots")
            or payload.get("swath_data")
        )
        if swath is not None:
            self._populate_swath(swath)

    def clear(self):
        """Clear all tabs."""
        self._clear_layout(self._cv_layout)
        self._clear_layout(self._swath_layout)
        self._stats_table.setRowCount(0)

    # ──────────────────────────────────────────────────────────
    # CV Scatter
    # ──────────────────────────────────────────────────────────

    def _populate_cv_scatter(self, cv):
        if not MATPLOTLIB_AVAILABLE:
            return
        actual = np.asarray(
            cv.actual if hasattr(cv, 'actual') else cv.get('actual', []),
            dtype=float,
        )
        estimated = np.asarray(
            cv.estimated if hasattr(cv, 'estimated') else cv.get('estimated', []),
            dtype=float,
        )
        mask = np.isfinite(actual) & np.isfinite(estimated)
        actual, estimated = actual[mask], estimated[mask]
        if len(actual) == 0:
            return

        slope = float(cv.slope_of_regression if hasattr(cv, 'slope_of_regression')
                       else cv.get('slope_of_regression', 1.0))
        intercept = float(cv.intercept if hasattr(cv, 'intercept')
                          else cv.get('intercept', 0.0))
        r2 = float(cv.r_squared if hasattr(cv, 'r_squared')
                    else cv.get('r_squared', float('nan')))
        rmse = float(cv.rmse if hasattr(cv, 'rmse')
                     else cv.get('rmse', float('nan')))

        if self._cv_fig is not None:
            plt.close(self._cv_fig)
        self._clear_layout(self._cv_layout)

        fig = Figure(figsize=(7, 6), dpi=110, facecolor=_BG)
        self._cv_fig = fig
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        # Scatter
        ax.scatter(actual, estimated, s=12, alpha=0.5, color="#4FC3F7",
                   edgecolors="none", zorder=3)

        # 1:1 line
        lo = min(actual.min(), estimated.min())
        hi = max(actual.max(), estimated.max())
        ax.plot([lo, hi], [lo, hi], '--', color="#888888", linewidth=1, label="1:1", zorder=2)

        # Regression line
        x_fit = np.array([lo, hi])
        ax.plot(x_fit, slope * x_fit + intercept, '-', color="#FF7043",
                linewidth=1.5, label=f"SoR={slope:.3f}", zorder=4)

        # Annotation
        ann = (f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\n"
               f"Slope = {slope:.3f}\nIntercept = {intercept:.3f}\n"
               f"N = {len(actual):,}")
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=8,
                va='top', ha='left', color=_TXT,
                bbox=dict(boxstyle='round', facecolor=_BG, alpha=0.8, edgecolor=_BDR))

        ax.set_xlabel("Actual", color=_TXT)
        ax.set_ylabel("Estimated", color=_TXT)
        ax.set_title("Cross-Validation: Actual vs Estimated", color=_TXT)
        ax.tick_params(colors=_TXT2)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2, color=_BDR)
        for sp in ax.spines.values():
            sp.set_color(_BDR)
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        self._cv_layout.addWidget(canvas)

    # ──────────────────────────────────────────────────────────
    # Swath Plots
    # ──────────────────────────────────────────────────────────

    def _populate_swath(self, swath_data):
        if hasattr(swath_data, 'axes'):
            self._swath_cache = dict(swath_data.axes)
        elif isinstance(swath_data, dict):
            self._swath_cache = {k.lower(): v for k, v in swath_data.items()}
        elif isinstance(swath_data, list):
            self._swath_cache = {
                (sd.axis if hasattr(sd, 'axis') else sd.get('axis', str(i))).lower(): sd
                for i, sd in enumerate(swath_data)
            }
        else:
            return
        if self._current_swath_axis.lower() not in self._swath_cache:
            self._current_swath_axis = next(iter(self._swath_cache)).upper()
        self._draw_swath(self._current_swath_axis)

    def _on_swath_axis(self, axis: str):
        self._current_swath_axis = axis
        if self._swath_cache:
            self._draw_swath(axis)

    def _draw_swath(self, axis: str):
        if not MATPLOTLIB_AVAILABLE or self._swath_cache is None:
            return
        sd = self._swath_cache.get(axis.lower())
        if sd is None:
            return

        pos = np.asarray(sd.slice_positions if hasattr(sd, 'slice_positions')
                         else sd.get('slice_positions', []), dtype=float)
        est = np.asarray(sd.mean_estimated if hasattr(sd, 'mean_estimated')
                         else sd.get('mean_estimated', []), dtype=float)
        act = np.asarray(sd.mean_actual if hasattr(sd, 'mean_actual')
                         else sd.get('mean_actual', []), dtype=float)

        if len(pos) == 0:
            return

        if self._swath_fig is not None:
            plt.close(self._swath_fig)
        self._clear_layout(self._swath_layout)

        fig = Figure(figsize=(9, 5), dpi=110, facecolor=_BG)
        self._swath_fig = fig
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        ax.plot(pos, est, color="#4FC3F7", linewidth=2, marker="o", markersize=4,
                markerfacecolor="white", markeredgecolor="#4FC3F7",
                label="Estimated (block mean)", zorder=4)
        ax.plot(pos, act, color="#FFB74D", linewidth=2, marker="s", markersize=4,
                markerfacecolor="white", markeredgecolor="#FFB74D",
                label="Actual (composite mean)", zorder=4)

        valid = np.isfinite(est) & np.isfinite(act)
        if np.any(valid):
            ax.fill_between(pos[valid], est[valid], act[valid],
                            alpha=0.08, color="#4FC3F7", zorder=1)

        ax.set_xlabel(f"{axis} coordinate", color=_TXT)
        ax.set_ylabel("Grade", color=_TXT)
        ax.set_title(f"Swath Plot — {axis} direction", color=_TXT)
        ax.tick_params(colors=_TXT2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, color=_BDR)
        for sp in ax.spines.values():
            sp.set_color(_BDR)
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        self._swath_layout.addWidget(canvas)

    # ──────────────────────────────────────────────────────────
    # Statistics Table
    # ──────────────────────────────────────────────────────────

    def _populate_stats_table(self, cv, qa_summary=None):
        rows = []

        def _val(obj, *keys):
            for k in keys:
                v = getattr(obj, k, None) if hasattr(obj, k) else None
                if v is None and isinstance(obj, dict):
                    v = obj.get(k)
                if v is not None:
                    return v
            return None

        r2 = _val(cv, 'r_squared')
        if r2 is not None:
            rows.append(("R²", f"{float(r2):.4f}"))
        rmse = _val(cv, 'rmse')
        if rmse is not None:
            rows.append(("RMSE", f"{float(rmse):.4f}"))
        me = _val(cv, 'mean_error', 'me')
        if me is not None:
            rows.append(("Mean Error (ME)", f"{float(me):.4f}"))
        mae = _val(cv, 'mae')
        if mae is not None:
            rows.append(("MAE", f"{float(mae):.4f}"))
        slope = _val(cv, 'slope_of_regression', 'slope')
        if slope is not None:
            rows.append(("Slope of Regression", f"{float(slope):.4f}"))
        nrmse = _val(cv, 'normalised_rmse', 'nrmse')
        if nrmse is not None:
            rows.append(("Normalised RMSE", f"{float(nrmse):.4f}"))
        n = _val(cv, 'n_samples')
        if n is not None:
            rows.append(("N samples (CV)", f"{int(n):,}"))

        # QA metrics from kriging
        if qa_summary and isinstance(qa_summary, dict):
            for key, label in [
                ('kriging_efficiency_mean', 'Kriging Efficiency (mean)'),
                ('kriging_efficiency_min', 'Kriging Efficiency (min)'),
                ('slope_of_regression_mean', 'Slope of Regression (mean)'),
                ('pct_negative_weights_max', 'Max Negative Weights (%)'),
            ]:
                v = qa_summary.get(key)
                if v is not None:
                    rows.append((label, f"{float(v):.3f}"))

        self._stats_table.setRowCount(len(rows))
        for i, (metric, value) in enumerate(rows):
            self._stats_table.setItem(i, 0, QTableWidgetItem(metric))
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._stats_table.setItem(i, 1, item)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            w = layout.takeAt(0).widget()
            if w:
                w.deleteLater()
