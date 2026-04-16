"""
Variogram Analysis Panel v2 — Interactive Tabbed Plot Design.

Layout:
  ┌─────────────────────────────────────────────────────────────┐
  │ [Variable: Cu ▼] [Data: Declustered ✓] [Compute]           │
  ├─────────────────────────────────────────────────────────────┤
  │  ┌──────┬──────┬──────┬──────┬──────┬──────┐               │
  │  │ Omni │Major │Minor │Vert  │Down  │Model │  ← Plot Tabs  │
  │  ├──────┴──────┴──────┴──────┴──────┴──────┤               │
  │  │                                          │               │
  │  │      LARGE INTERACTIVE MATPLOTLIB PLOT    │               │
  │  │      (one full-size plot per tab)         │               │
  │  │      with NavigationToolbar               │               │
  │  │                                          │               │
  │  ├──────────────────────────────────────────┤               │
  │  │  Parameters  │  Model Summary │  Warnings │  ← Info Tabs │
  │  └──────────────────────────────────────────┘               │
  └─────────────────────────────────────────────────────────────┘

Each variogram direction gets its own FULL-SIZE tab with interactive
matplotlib toolbar (zoom, pan, save).  Parameters and info sit below
the plot in a compact tabbed footer.  No cramped 2×3 grid.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401 – registers the '3d' projection with matplotlib

from .base_analysis_panel import BaseAnalysisPanel
from .mixins.coded_domain_filter_mixin import CodedDomainFilterMixin
from .modern_styles import ModernColors
from .panel_toolkit import (
    action_button,
    form_row,
    hint_label,
    make_combo,
    make_form,
    make_spin,
    section,
    separator,
    PANEL_MARGINS,
    PANEL_SPACING,
)
from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import get_grade_columns
from ..utils.plot_style import PlotDefaults, COLOR_NUGGET, COLOR_SILL, COLOR_RANGE
from ..geostats.variogram_bridge_v2 import run_variogram_pipeline_v2
from ..geostats.variogram_fitting import fit_variogram_model as _fit_variogram_model
from ..geostats.variogram_recommender import recommend_variogram_settings as _recommend_variogram_settings
from ..geostats.variogram_model import MODEL_MAP as _MODEL_MAP
from ..geostats.variogram_gates import compute_data_hash, analyze_nugget_consistency

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Interactive variogram plot widget — one per direction
# ═══════════════════════════════════════════════════════════════════


class _VariogramPlotWidget(QWidget):
    """Full-size interactive variogram plot with toolbar.

    Supports interactive dragging of nugget/sill/range reference lines
    and right-click exclusion of unreliable lag points.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        self.ax.set_xlabel('Lag Distance (m)', fontsize=10)
        self.ax.set_ylabel('\u03b3(h)', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.toolbar = NavigationToolbar(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        # ── Interactive drag state ────────────────────────────────
        self._dragging: Optional[str] = None
        self._nug_line = None
        self._sill_line = None
        self._rng_line = None
        self._nug_ann = None
        self._sill_ann = None
        self._rng_ann = None
        self._dragged: Dict[str, float] = {}
        self._excluded_lags: set = set()

        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    # ── Public API ────────────────────────────────────────────────
    @property
    def dragged_params(self) -> Dict[str, float]:
        return dict(self._dragged)

    def get_excluded_lags(self) -> set:
        return set(self._excluded_lags)

    def reset_excluded_lags(self) -> None:
        self._excluded_lags.clear()

    # ── Mouse handlers ────────────────────────────────────────────
    def _toolbar_active(self) -> bool:
        """Check if NavigationToolbar is in pan/zoom mode."""
        return bool(getattr(self.toolbar, 'mode', ''))

    def _on_press(self, event):
        if event.inaxes is None or self._toolbar_active():
            return

        # Right-click: toggle excluded lag
        if event.button == 3:
            self._toggle_exclude(event)
            return

        if event.button != 1:
            return

        ylim = self.ax.get_ylim()
        xlim = self.ax.get_xlim()
        y_tol = (ylim[1] - ylim[0]) * 0.05
        x_tol = (xlim[1] - xlim[0]) * 0.05

        nug_y = self._dragged.get('nugget')
        if nug_y is not None and self._nug_line and abs(event.ydata - nug_y) < y_tol:
            self._dragging = 'nugget'
            return
        sill_y = self._dragged.get('sill')
        if sill_y is not None and self._sill_line and abs(event.ydata - sill_y) < y_tol:
            self._dragging = 'sill'
            return
        rng_x = self._dragged.get('range')
        if rng_x is not None and self._rng_line and abs(event.xdata - rng_x) < x_tol:
            self._dragging = 'range'
            return

    def _on_motion(self, event):
        if event.inaxes is None:
            if self._dragging is None:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return

        # Cursor feedback when hovering near draggable lines
        if self._dragging is None and not self._toolbar_active():
            ylim = self.ax.get_ylim()
            xlim = self.ax.get_xlim()
            y_tol = (ylim[1] - ylim[0]) * 0.04
            x_tol = (xlim[1] - xlim[0]) * 0.04
            near = False
            for key, val in self._dragged.items():
                if key in ('nugget', 'sill') and val is not None:
                    if abs(event.ydata - val) < y_tol:
                        near = True
                        break
                elif key == 'range' and val is not None:
                    if abs(event.xdata - val) < x_tol:
                        near = True
                        break
            self.canvas.setCursor(
                Qt.CursorShape.OpenHandCursor if near else Qt.CursorShape.ArrowCursor
            )
            return

        if self._dragging is None:
            return

        if self._dragging == 'nugget' and self._nug_line:
            new_y = max(0.0, event.ydata)
            self._nug_line.set_ydata([new_y, new_y])
            self._dragged['nugget'] = new_y
            if self._nug_ann:
                self._nug_ann.set_text(f" Nugget = {new_y:.3f}")
                self._nug_ann.set_position((self._nug_ann.get_position()[0], new_y))
            self.canvas.draw_idle()

        elif self._dragging == 'sill' and self._sill_line:
            new_y = max(0.0, event.ydata)
            self._sill_line.set_ydata([new_y, new_y])
            self._dragged['sill'] = new_y
            if self._sill_ann:
                self._sill_ann.set_text(f" Sill = {new_y:.3f}")
                self._sill_ann.set_position((self._sill_ann.get_position()[0], new_y))
            self.canvas.draw_idle()

        elif self._dragging == 'range' and self._rng_line:
            new_x = max(0.0, event.xdata)
            self._rng_line.set_xdata([new_x, new_x])
            self._dragged['range'] = new_x
            if self._rng_ann:
                self._rng_ann.set_text(f"  Range = {new_x:.1f}m")
                self._rng_ann.set_position((new_x, self._rng_ann.get_position()[1]))
            self.canvas.draw_idle()

    def _on_release(self, event):
        self._dragging = None

    def _toggle_exclude(self, event):
        """Toggle exclusion of the nearest experimental lag point."""
        if not hasattr(self, '_last_lags') or self._last_lags is None:
            return
        lags = self._last_lags
        gamma = self._last_gamma
        if len(lags) == 0:
            return
        # Find nearest point in normalised axis-fraction coordinates
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        dx = (lags - event.xdata) / max(xlim[1] - xlim[0], 1e-9)
        dy = (gamma - event.ydata) / max(ylim[1] - ylim[0], 1e-9)
        dist = np.sqrt(dx**2 + dy**2)
        idx = int(np.argmin(dist))
        if dist[idx] < 0.05:
            if idx in self._excluded_lags:
                self._excluded_lags.discard(idx)
            else:
                self._excluded_lags.add(idx)
            # Replot with exclusions shown
            self._mark_exclusions()

    def _mark_exclusions(self):
        """Redraw excluded points as red X markers."""
        # Remove previous exclusion markers
        for artist in getattr(self, '_excl_artists', []):
            try:
                artist.remove()
            except Exception:
                pass
        self._excl_artists = []
        if not hasattr(self, '_last_lags') or self._last_lags is None:
            return
        for idx in self._excluded_lags:
            if idx < len(self._last_lags):
                a = self.ax.scatter(
                    [self._last_lags[idx]], [self._last_gamma[idx]],
                    s=100, c='#F44336', marker='x', linewidths=2.5, zorder=10,
                )
                self._excl_artists.append(a)
        self.canvas.draw_idle()

    def plot_variogram(self, lags, gamma, model_lags=None, model_gamma=None,
                       counts=None, sample_var=None, fitted_params=None, title=None,
                       all_fitted_models: Optional[Dict] = None,
                       show_confidence: bool = True):
        """Professional variogram plot with smooth curves and annotations.

        Parameters
        ----------
        all_fitted_models : dict, optional
            Maps model_type (str) -> fitted_params dict.  When provided, ALL
            model curves are overlaid with distinct colours and line-styles so
            the user can visually compare Spherical / Exponential / Gaussian
            fits on a single axes.
        show_confidence : bool
            If True and *counts* is available, draw a 95 % confidence envelope
            around the experimental points (Cressie 1993, eq 2.4.12).
        """
        self.ax.clear()
        # Reset drag handles
        self._nug_line = self._sill_line = self._rng_line = None
        self._nug_ann = self._sill_ann = self._rng_ann = None
        self._dragged = {}
        self._last_lags = None
        self._last_gamma = None
        self._excl_artists = []

        if len(lags) == 0:
            self.ax.text(0.5, 0.5, "No data for this direction",
                        ha='center', va='center', fontsize=13, color='grey',
                        transform=self.ax.transAxes)
            if title:
                self.ax.set_title(title, fontsize=12, fontweight='bold')
            self.canvas.draw_idle()
            return

        # ── Experimental points — sized by pair count ──────────────────
        if counts is not None and len(counts) == len(lags):
            max_c = max(counts.max(), 1)
            sizes = np.clip(counts / max_c * 140, 20, 140)
        else:
            sizes = 40
        self.ax.scatter(lags, gamma, s=sizes, c='#2980B9', alpha=0.85,
                       edgecolors='white', linewidths=0.6, zorder=5,
                       label='Experimental')

        # ── Confidence envelopes (Cressie 1993 eq 2.4.12) ─────────────
        if show_confidence and counts is not None and len(counts) == len(lags):
            ci_factor = 1.96 * np.sqrt(2.0 / np.maximum(counts, 1))
            upper = gamma * (1 + ci_factor)
            lower = gamma * (1 - ci_factor)
            lower = np.maximum(lower, 0)  # Can't be negative
            self.ax.fill_between(lags, lower, upper, alpha=0.12,
                                color='#2980B9', zorder=1, label='95% CI')

        # ── Multiple model overlay ─────────────────────────────────────
        _model_styles = {
            'spherical':   ('-',  '#E67E22', 'Spherical'),    # solid orange
            'exponential': ('--', '#2980B9', 'Exponential'),  # dashed blue
            'gaussian':    (':',  '#27AE60', 'Gaussian'),     # dotted green
        }

        if all_fitted_models and len(lags) > 0:
            overlay_lags = np.linspace(0, lags.max() * 1.15, 500)
            for m_type, m_params in all_fitted_models.items():
                if not m_params:
                    continue
                style_info = _model_styles.get(m_type, ('-', '#888888', m_type.title()))
                ls, clr, label_base = style_info
                nug = m_params.get('nugget', 0)
                sill = m_params.get('sill', 0)
                rng = m_params.get('range', 1)
                h_r = overlay_lags / max(rng, 1e-6)
                if m_type == 'spherical':
                    m_gamma = np.where(h_r < 1.0,
                                       nug + sill * (1.5 * h_r - 0.5 * h_r ** 3),
                                       nug + sill)
                elif m_type == 'exponential':
                    m_gamma = nug + sill * (1.0 - np.exp(-3.0 * h_r))
                elif m_type == 'gaussian':
                    m_gamma = nug + sill * (1.0 - np.exp(-3.0 * h_r ** 2))
                else:
                    continue
                # Compute weighted R^2 for label
                r2_text = ''
                try:
                    h_r_exp = lags / max(rng, 1e-6)
                    if m_type == 'spherical':
                        y_pred = np.where(h_r_exp < 1.0,
                                          nug + sill * (1.5 * h_r_exp - 0.5 * h_r_exp ** 3),
                                          nug + sill)
                    elif m_type == 'exponential':
                        y_pred = nug + sill * (1.0 - np.exp(-3.0 * h_r_exp))
                    elif m_type == 'gaussian':
                        y_pred = nug + sill * (1.0 - np.exp(-3.0 * h_r_exp ** 2))
                    else:
                        y_pred = gamma
                    wts = counts if counts is not None and len(counts) == len(lags) else np.ones_like(lags)
                    ss_res = float(np.sum(wts * (gamma - y_pred) ** 2))
                    wmean = float(np.average(gamma, weights=wts))
                    ss_tot = float(np.sum(wts * (gamma - wmean) ** 2))
                    r2w = 1.0 - ss_res / max(ss_tot, 1e-12)
                    r2_text = f' (R\u00b2={r2w:.3f})'
                except Exception:
                    pass
                self.ax.plot(overlay_lags, m_gamma, linestyle=ls, color=clr,
                            linewidth=2.2, zorder=4,
                            label=f'{label_base}{r2_text}',
                            solid_capstyle='round')
        elif model_lags is not None and model_gamma is not None and len(model_lags) > 0:
            # ── Single fitted model curve (legacy path) ────────────────
            self.ax.plot(model_lags, model_gamma, '-', color='#E74C3C',
                        linewidth=2.5, zorder=4, label='Fitted Model',
                        solid_capstyle='round')
            # Fill between nugget and curve
            if fitted_params:
                nug_val = fitted_params.get('nugget', 0)
                self.ax.fill_between(model_lags, nug_val, model_gamma,
                                    alpha=0.06, color='#E74C3C', zorder=1)

        # ── Sample variance reference ──────────────────────────────────
        if sample_var is not None and sample_var > 0:
            self.ax.axhline(y=sample_var, color='#95A5A6', linestyle='--',
                          alpha=0.5, linewidth=0.8, label=f'Sample Var ({sample_var:.1f})')

        # ── Fitted params — draggable reference lines + annotations ────
        # Cache lags/gamma for right-click exclusion
        self._last_lags = np.array(lags, dtype=float)
        self._last_gamma = np.array(gamma, dtype=float)

        if fitted_params:
            sill = fitted_params.get('sill', 0)
            nugget = fitted_params.get('nugget', 0)
            range_ = fitted_params.get('range', 0)
            total_sill = nugget + sill

            # Nugget discontinuity at origin
            if nugget > 1e-9:
                self.ax.plot(0, 0, 'o', color=COLOR_NUGGET, markersize=5,
                            markerfacecolor='white', markeredgewidth=1.2, zorder=6)
                self.ax.plot(0, nugget, 'o', color='#E74C3C', markersize=5,
                            markeredgewidth=1.2, zorder=6)
                self.ax.plot([0, 0], [0, nugget], color=COLOR_NUGGET,
                            linewidth=1.2, alpha=0.6, zorder=3)

            # Draggable reference lines — store handles
            self._nug_line = self.ax.axhline(
                y=nugget, color=COLOR_NUGGET, linestyle=':', linewidth=1.2,
                alpha=0.6, zorder=2, picker=5)
            self._sill_line = self.ax.axhline(
                y=total_sill, color=COLOR_SILL, linestyle='--', linewidth=1.5,
                alpha=0.6, zorder=2, picker=5)
            self._rng_line = self.ax.axvline(
                x=range_, color=COLOR_RANGE, linestyle='--', linewidth=1.5,
                alpha=0.6, zorder=2, picker=5)

            self._dragged = {'nugget': nugget, 'sill': total_sill, 'range': range_}

            # Margin annotations — store for live update during drag
            x_max = float(self.ax.get_xlim()[1]) if self.ax.get_xlim()[1] > 0 else 100
            self._nug_ann = self.ax.text(
                x_max * 0.99, nugget, f" Nugget = {nugget:.3f}",
                va='bottom', ha='right', fontsize=PlotDefaults.ANNOTATION_SIZE, color=COLOR_NUGGET,
                fontstyle='italic', zorder=7)
            self._sill_ann = self.ax.text(
                x_max * 0.99, total_sill, f" Sill = {total_sill:.3f}",
                va='bottom', ha='right', fontsize=PlotDefaults.ANNOTATION_SIZE, color=COLOR_SILL,
                fontstyle='italic', zorder=7)
            self._rng_ann = self.ax.text(
                range_, 0, f"  Range = {range_:.1f}m",
                va='bottom', ha='left', fontsize=PlotDefaults.ANNOTATION_SIZE, color=COLOR_RANGE,
                rotation=90, fontstyle='italic', zorder=7)

            # Model equation box
            nugget_pct = (nugget / total_sill * 100) if total_sill > 0 else 0
            text = (f"Nugget (C\u2080)      {nugget:.3f}  ({nugget_pct:.0f}%)\n"
                    f"Partial Sill (C)  {sill:.3f}\n"
                    f"Range (a)         {range_:.1f} m")
            self.ax.text(0.97, 0.05, text, transform=self.ax.transAxes,
                        fontsize=PlotDefaults.EQUATION_SIZE, verticalalignment='bottom',
                        horizontalalignment='right', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                 edgecolor='#bdbdbd', alpha=0.92, linewidth=0.7))

            # Re-apply any existing exclusion markers
            self._mark_exclusions()

        # ── Axes ───────────────────────────────────────────────────────
        self.ax.set_xlabel('Lag Distance  h  (m)', fontsize=10, labelpad=5)
        self.ax.set_ylabel('\u03b3(h)', fontsize=10, labelpad=5)
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.ax.grid(True, alpha=0.25, linestyle='--')
        self.ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

        if title:
            # Add pair statistics to title
            if counts is not None and len(counts) > 0:
                total_p = int(np.sum(counts))
                title = f"{title}  [{total_p:,} pairs]"
            self.ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

        # Clamp Y-axis so sample variance doesn't compress data
        gamma_vals = np.asarray(gamma) if len(gamma) > 0 else np.array([1.0])
        y_data_max = float(np.nanmax(gamma_vals)) if gamma_vals.size else 1.0
        if model_gamma is not None and len(model_gamma) > 0:
            y_data_max = max(y_data_max, float(np.nanmax(model_gamma)))
        y_top = max(y_data_max * 1.25, 1e-9)
        self.ax.set_ylim(bottom=0.0, top=y_top)

        self.fig.tight_layout()
        self.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════


class VariogramPanel(CodedDomainFilterMixin, BaseAnalysisPanel):
    """Interactive tabbed variogram analysis panel.

    Each direction gets a full-size interactive plot tab with toolbar.
    Parameters and info sit in a compact footer below the plots.
    Variogram results are automatically published to the registry.
    """

    task_name = "variogram"
    panel_title = "3D Variogram Analysis"
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None, main_window=None, **kwargs):
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict] = None
        self._latest_recommendation: Optional[Dict[str, Any]] = None
        self._main_window = main_window
        self._ui_ready = False
        self._using_declustered = False
        self._plot_widgets: Dict[str, _VariogramPlotWidget] = {}

        super().__init__(parent=parent, panel_id=kwargs.get("panel_id", "variogram"))

        self._ui_ready = True
        self._last_experimental: Optional[Dict[str, Any]] = None
        self._init_registry()
        # Auto-load data after a short delay (registry may not be set yet)
        QTimer.singleShot(200, self._auto_load_data)

    # ══════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════

    def setup_ui(self):
        """Build the tabbed plot layout."""
        # ── Top toolbar ──────────────────────────────────────────
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Variable:"))
        self.var_combo = make_combo(tooltip="Grade variable")
        self.var_combo.setMinimumWidth(80)
        toolbar.addWidget(self.var_combo)

        toolbar.addWidget(QLabel("Domain:"))
        self.domain_combo = make_combo(["All Data"], tooltip="Optional coded domain filter")
        self.domain_combo.setMinimumWidth(180)
        toolbar.addWidget(self.domain_combo)

        toolbar.addWidget(QLabel("  "))
        self._data_status = QLabel("No data loaded")
        self._data_status.setStyleSheet("color: #95A5A6; font-size: 9pt;")
        toolbar.addWidget(self._data_status)

        toolbar.addStretch()

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setMinimumHeight(36)
        self.refresh_btn.setMinimumWidth(110)
        self.refresh_btn.setToolTip("Reload latest data from registry")
        self.refresh_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ModernColors.ACCENT_PRIMARY}; color: white;"
            f" font-weight: bold; font-size: 10pt; border-radius: 4px; padding: 4px 12px; }}"
            f"QPushButton:hover {{ background-color: {ModernColors.ELEVATED_BG};"
            f" color: {ModernColors.ACCENT_PRIMARY}; border: 2px solid {ModernColors.ACCENT_PRIMARY}; }}"
        )
        self.refresh_btn.clicked.connect(self._manual_refresh)
        toolbar.addWidget(self.refresh_btn)

        self.recommend_btn = action_button("Recommend & Fit", style="secondary")
        self.recommend_btn.clicked.connect(self._on_recommend_and_fit)
        toolbar.addWidget(self.recommend_btn)

        self.export_rec_btn = QPushButton("Export Recommendation")
        self.export_rec_btn.setMinimumHeight(36)
        self.export_rec_btn.setToolTip("Export the latest recommendation as a JSON file")
        self.export_rec_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ModernColors.ELEVATED_BG}; color: {ModernColors.TEXT_PRIMARY};"
            f" font-weight: bold; font-size: 10pt; border-radius: 4px; padding: 4px 12px;"
            f" border: 1px solid {ModernColors.BORDER_LIGHT}; }}"
            f"QPushButton:hover {{ background-color: {ModernColors.ACCENT_PRIMARY}; color: white; }}"
            f"QPushButton:disabled {{ background-color: {ModernColors.BORDER}; color: {ModernColors.TEXT_DISABLED}; }}"
        )
        self.export_rec_btn.setEnabled(False)
        self.export_rec_btn.clicked.connect(self._export_recommendation)
        toolbar.addWidget(self.export_rec_btn)

        self.compute_btn = action_button("Compute Variogram", style="primary")
        self.compute_btn.clicked.connect(self._on_compute)
        toolbar.addWidget(self.compute_btn)

        self.pdf_export_btn = QPushButton("Export PDF Report")
        self.pdf_export_btn.setMinimumHeight(36)
        self.pdf_export_btn.setToolTip(
            "Export a multi-page PDF report with all variogram plots,\n"
            "anisotropy ellipsoid, variogram map, and model parameters."
        )
        self.pdf_export_btn.setStyleSheet(
            f"QPushButton {{ background-color: {ModernColors.ACCENT_PRIMARY}; color: white;"
            f" font-weight: bold; font-size: 10pt; border-radius: 4px; padding: 4px 12px; }}"
            f"QPushButton:hover {{ background-color: {ModernColors.ELEVATED_BG};"
            f" color: {ModernColors.ACCENT_PRIMARY}; border: 2px solid {ModernColors.ACCENT_PRIMARY}; }}"
        )
        self.pdf_export_btn.clicked.connect(self._export_pdf_report)
        toolbar.addWidget(self.pdf_export_btn)

        self.main_layout.addLayout(toolbar)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        # ── Main area: Plot tabs (top 70%) + Info tabs (bottom 30%) ──
        self._vsplitter = QSplitter(Qt.Orientation.Vertical)

        # PLOT TABS — one full-size interactive plot per direction
        self.plot_tabs = QTabWidget()
        directions = [
            ("Omnidirectional", "omni"),
            ("Major", "major"),
            ("Minor", "minor"),
            ("Vertical", "vertical"),
            ("Downhole", "downhole"),
        ]
        for title, key in directions:
            pw = _VariogramPlotWidget(title)
            self._plot_widgets[key] = pw
            self.plot_tabs.addTab(pw, title)

        # 3D Variogram Map tab — side-by-side: 3D scatter (left) + aniso fit (right)
        self._3d_widget = QWidget()
        _3d_lay = QHBoxLayout(self._3d_widget)
        _3d_lay.setContentsMargins(0, 0, 0, 0)
        _3d_lay.setSpacing(0)

        # Left: 3D scatter map
        _3d_left = QWidget()
        _3d_left_lay = QVBoxLayout(_3d_left)
        _3d_left_lay.setContentsMargins(0, 0, 0, 0)
        self._3d_fig = Figure(figsize=(6, 6), dpi=100, facecolor='white')
        self._3d_canvas = FigureCanvas(self._3d_fig)
        self._3d_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._3d_toolbar = NavigationToolbar(self._3d_canvas, _3d_left)
        _3d_left_lay.addWidget(self._3d_toolbar)
        _3d_left_lay.addWidget(self._3d_canvas, 1)

        # Right: Anisotropic fit plot
        _3d_right = QWidget()
        _3d_right_lay = QVBoxLayout(_3d_right)
        _3d_right_lay.setContentsMargins(0, 0, 0, 0)
        self._anisofit_fig = Figure(figsize=(6, 6), dpi=100, facecolor='white')
        self._anisofit_canvas = FigureCanvas(self._anisofit_fig)
        self._anisofit_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._anisofit_toolbar = NavigationToolbar(self._anisofit_canvas, _3d_right)
        _3d_right_lay.addWidget(self._anisofit_toolbar)
        _3d_right_lay.addWidget(self._anisofit_canvas, 1)

        _splitter = QSplitter(Qt.Orientation.Horizontal)
        _splitter.addWidget(_3d_left)
        _splitter.addWidget(_3d_right)
        _splitter.setSizes([500, 500])
        _3d_lay.addWidget(_splitter)
        self.plot_tabs.addTab(self._3d_widget, "3D Map")

        # Anisotropy Ellipse tab
        self._aniso_widget = QWidget()
        _aniso_lay = QVBoxLayout(self._aniso_widget)
        _aniso_lay.setContentsMargins(0, 0, 0, 0)
        self._aniso_fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self._aniso_canvas = FigureCanvas(self._aniso_fig)
        self._aniso_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._aniso_toolbar = NavigationToolbar(self._aniso_canvas, self._aniso_widget)
        _aniso_lay.addWidget(self._aniso_toolbar)
        _aniso_lay.addWidget(self._aniso_canvas, 1)
        self.plot_tabs.addTab(self._aniso_widget, "Anisotropy")

        # Variogram Map tab — side-by-side: 2D polar (left) + 3D surface (right)
        self._vmap_widget = QWidget()
        _vmap_lay = QHBoxLayout(self._vmap_widget)
        _vmap_lay.setContentsMargins(0, 0, 0, 0)
        _vmap_lay.setSpacing(0)

        # Left: 2D polar heatmap
        _vmap_left = QWidget()
        _vmap_left_lay = QVBoxLayout(_vmap_left)
        _vmap_left_lay.setContentsMargins(0, 0, 0, 0)
        self._vmap_fig = Figure(figsize=(6, 6), dpi=100, facecolor='white')
        self._vmap_canvas = FigureCanvas(self._vmap_fig)
        self._vmap_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._vmap_toolbar = NavigationToolbar(self._vmap_canvas, _vmap_left)
        _vmap_left_lay.addWidget(self._vmap_toolbar)
        _vmap_left_lay.addWidget(self._vmap_canvas, 1)

        # Right: 3D surface
        _vmap_right = QWidget()
        _vmap_right_lay = QVBoxLayout(_vmap_right)
        _vmap_right_lay.setContentsMargins(0, 0, 0, 0)
        self._vmap3d_fig = Figure(figsize=(6, 6), dpi=100, facecolor='white')
        self._vmap3d_canvas = FigureCanvas(self._vmap3d_fig)
        self._vmap3d_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._vmap3d_toolbar = NavigationToolbar(self._vmap3d_canvas, _vmap_right)
        _vmap_right_lay.addWidget(self._vmap3d_toolbar)
        _vmap_right_lay.addWidget(self._vmap3d_canvas, 1)

        _vmap_splitter = QSplitter(Qt.Orientation.Horizontal)
        _vmap_splitter.addWidget(_vmap_left)
        _vmap_splitter.addWidget(_vmap_right)
        _vmap_splitter.setSizes([500, 500])
        _vmap_lay.addWidget(_vmap_splitter)
        self.plot_tabs.addTab(self._vmap_widget, "Variogram Map")

        # H-Scatterplot tab (Z(x) vs Z(x+h) per lag)
        self._hscatter_widget = QWidget()
        _hscatter_lay = QVBoxLayout(self._hscatter_widget)
        _hscatter_lay.setContentsMargins(0, 0, 0, 0)
        self._hscatter_fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self._hscatter_canvas = FigureCanvas(self._hscatter_fig)
        self._hscatter_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._hscatter_toolbar = NavigationToolbar(self._hscatter_canvas, self._hscatter_widget)
        _hscatter_lay.addWidget(self._hscatter_toolbar)
        _hscatter_lay.addWidget(self._hscatter_canvas, 1)
        self.plot_tabs.addTab(self._hscatter_widget, "H-Scatter")

        self._vsplitter.addWidget(self.plot_tabs)

        # INFO TABS — parameters, model summary, warnings
        self._info_tabs = QTabWidget()
        self._info_tabs.setMaximumHeight(260)

        # Parameters tab
        self._build_params_tab()

        # Model Summary tab
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setFont(QFont("Consolas", 9))
        self._summary_text.setPlaceholderText("Run variogram to see model summary...")
        self._info_tabs.addTab(self._summary_text, "Model Summary")

        self._recommend_text = QTextEdit()
        self._recommend_text.setReadOnly(True)
        self._recommend_text.setFont(QFont("Consolas", 9))
        self._recommend_text.setPlaceholderText("Use Recommend & Fit to generate data-driven settings.")
        self._info_tabs.addTab(self._recommend_text, "Recommendations")

        # Warnings tab
        self._warnings_text = QTextEdit()
        self._warnings_text.setReadOnly(True)
        self._warnings_text.setFont(QFont("Consolas", 9))
        self._info_tabs.addTab(self._warnings_text, "Warnings")

        self._vsplitter.addWidget(self._info_tabs)
        self._vsplitter.setSizes([500, 200])

        self.main_layout.addWidget(self._vsplitter, 1)

    def _build_params_tab(self):
        """Build the parameters tab inside the info area."""
        params_widget = QWidget()
        h = QHBoxLayout(params_widget)
        h.setContentsMargins(6, 4, 6, 4)

        # Column 1: Lags
        f1 = QFormLayout()
        f1.setContentsMargins(0, 0, 8, 0)
        self.cb_auto_lags = QCheckBox("Auto lags")
        self.cb_auto_lags.setChecked(True)
        self.cb_auto_lags.toggled.connect(self._on_auto_lags_toggled)
        f1.addRow(self.cb_auto_lags)
        self.nlag_spin = QSpinBox(); self.nlag_spin.setRange(5, 50); self.nlag_spin.setValue(15)
        f1.addRow("N Lags:", self.nlag_spin)
        self.lag_dist_spin = make_spin(0.1, 10000.0, 25.0, 1)
        f1.addRow("Lag Dist:", self.lag_dist_spin)
        self.lag_tol_spin = make_spin(0.1, 5000.0, 12.5, 1)
        f1.addRow("Tolerance:", self.lag_tol_spin)
        h.addLayout(f1)
        self._on_auto_lags_toggled(True)

        # Column 2: Directions
        f2 = QFormLayout()
        f2.setContentsMargins(8, 0, 8, 0)
        self.cb_manual_azimuth = QCheckBox("Manual az/dip")
        self.cb_manual_azimuth.toggled.connect(self._on_manual_az_toggled)
        f2.addRow(self.cb_manual_azimuth)
        self.major_az_spin = make_spin(0.0, 360.0, 0.0, 1)
        f2.addRow("Azimuth:", self.major_az_spin)
        self.major_dip_spin = make_spin(-90.0, 90.0, 0.0, 1)
        f2.addRow("Dip:", self.major_dip_spin)
        self.cone_tol_spin = make_spin(1.0, 90.0, 22.5, 1)
        f2.addRow("Cone Tol:", self.cone_tol_spin)
        h.addLayout(f2)
        self._on_manual_az_toggled(False)

        # Column 3: Fitting
        f3 = QFormLayout()
        f3.setContentsMargins(8, 0, 8, 0)
        self.model_combo = make_combo(["Spherical", "Exponential", "Gaussian"])
        f3.addRow("Model:", self.model_combo)
        self.struct_spin = QSpinBox(); self.struct_spin.setRange(1, 3); self.struct_spin.setValue(1)
        f3.addRow("Structures:", self.struct_spin)
        self.cb_global_nugget = QCheckBox("Lock nugget")
        self.cb_global_nugget.toggled.connect(lambda c: self.global_nugget_spin.setEnabled(c))
        f3.addRow(self.cb_global_nugget)
        self.global_nugget_spin = make_spin(0.0, 1e15, 0.0, 4)
        self.global_nugget_spin.setEnabled(False)
        f3.addRow("Nugget:", self.global_nugget_spin)
        # Auto-refit when model type changes (no recompute of experimental)
        self.model_combo.currentTextChanged.connect(self._on_model_type_changed)
        h.addLayout(f3)

        # Column 4: Options
        f4 = QFormLayout()
        f4.setContentsMargins(8, 0, 0, 0)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999); self.seed_spin.setValue(42)
        f4.addRow("Seed:", self.seed_spin)
        self.cb_bandwidth = QCheckBox("Bandwidth")
        f4.addRow(self.cb_bandwidth)
        self.bandwidth_spin = make_spin(0.0, 10000.0, 0.0, 1)
        f4.addRow("BW (m):", self.bandwidth_spin)
        h.addLayout(f4)

        self._info_tabs.addTab(params_widget, "Parameters")

    # ══════════════════════════════════════════════════════════════
    # TOGGLE HANDLERS
    # ══════════════════════════════════════════════════════════════

    def _on_auto_lags_toggled(self, checked):
        self.nlag_spin.setEnabled(not checked)
        self.lag_dist_spin.setEnabled(not checked)
        self.lag_tol_spin.setEnabled(not checked)

    def _on_manual_az_toggled(self, checked):
        self.major_az_spin.setEnabled(checked)
        self.major_dip_spin.setEnabled(checked)

    def _on_model_type_changed(self, model_type: str):
        """Re-fit the model to existing experimental data when model type changes.

        This avoids the expensive experimental variogram recomputation.
        Only the fitted curves and parameters are updated.
        """
        if not self._last_experimental or not self.variogram_results:
            return  # No previous results to refit

        results = self._last_experimental.get('results')
        if results is None:
            return

        var = self._last_experimental.get('variable', '')
        m_type = model_type.lower()

        try:
            fitted_models = results.get('fitted_models', {})
            sample_var = results.get('metadata', {}).get(
                'sample_variance', results.get('sample_variance', 1.0)
            )

            direction_data_keys = {
                'omni': 'omni_variogram',
                'major': 'major_variogram',
                'minor': 'minor_variogram',
                'vertical': 'vertical_variogram',
                'downhole': 'downhole_variogram',
            }

            for direction, data_key in direction_data_keys.items():
                vg_df = results.get(data_key)
                if vg_df is None or not isinstance(vg_df, pd.DataFrame) or vg_df.empty:
                    continue
                if 'distance' not in vg_df.columns or 'gamma' not in vg_df.columns:
                    continue

                distances = vg_df['distance'].values
                gamma = vg_df['gamma'].values
                counts = vg_df['npairs'].values if 'npairs' in vg_df.columns else None

                valid_mask = ~(np.isnan(distances) | np.isnan(gamma))
                if int(np.sum(valid_mask)) < 3:
                    continue

                try:
                    # Cressie (1985) weighting: w = N(h) / gamma(h)^2 with
                    # a 10%-of-median floor to stop the first lag from
                    # dominating. Previously the legacy Variogram3D.fit_model
                    # wrapper did this; now inlined here so we can call the
                    # shared fit_variogram_model directly.
                    fit_weights = None
                    if counts is not None:
                        pc = np.asarray(counts, dtype=float)[valid_mask]
                        g_valid = gamma[valid_mask]
                        g_sq = g_valid ** 2
                        g_sq_floor = max(float(np.median(g_sq)) * 0.1, 1e-6)
                        g_sq = np.maximum(g_sq, g_sq_floor)
                        fit_weights = pc / g_sq
                    max_lag = float(np.nanmax(distances[valid_mask]))
                    nugget, total_sill, rng = _fit_variogram_model(
                        distances[valid_mask],
                        gamma[valid_mask],
                        model_type=m_type,
                        weights=fit_weights,
                        max_lag=max_lag,
                        sill_cap=sample_var,
                    )
                    psill = max(total_sill - nugget, 0.0)
                    fitted_models.setdefault(direction, {})[m_type] = {
                        'nugget': nugget,
                        'sill': psill,
                        'range': rng,
                        'total_sill': total_sill,
                        'model_type': m_type,
                    }
                except Exception as exc:
                    logger.debug("Refit failed for %s/%s: %s", direction, m_type, exc)

            results['fitted_models'] = fitted_models

            # Rebuild combined model and update all plots
            combined = self._build_combined_model(results, var)
            results['combined_3d_model'] = combined
            self.variogram_results = results

            self._update_all_plots(results)
            self._update_summary(results, combined)
            self._update_warnings(results)

            logger.info("Model type changed to %s — refitted all directions", m_type)
        except Exception as exc:
            logger.warning("Model type refit failed: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # DATA LOADING — automatic
    # ══════════════════════════════════════════════════════════════

    def _init_registry(self):
        try:
            registry = self.get_registry()
            if registry is None:
                return
            self.registry = registry
            registry.drillholeDataLoaded.connect(self._on_data_loaded)
            # Also listen for composites and declustering results so the
            # panel automatically picks up data loaded by other panels.
            if hasattr(registry, 'compositesLoaded'):
                registry.compositesLoaded.connect(self._on_data_loaded)
            if hasattr(registry, 'declusteringResultsLoaded'):
                registry.declusteringResultsLoaded.connect(self._on_data_loaded)
            if hasattr(registry, 'indicatorRBFDomainLoaded'):
                registry.indicatorRBFDomainLoaded.connect(self._on_indicator_rbf_domain_loaded)
        except Exception:
            pass

    def showEvent(self, event):
        """Auto-load data when the panel becomes visible."""
        super().showEvent(event)
        if self._ui_ready and self.drillhole_data is None:
            QTimer.singleShot(100, self._auto_load_data)

    def _auto_load_data(self):
        """Auto-load data on panel open — uses declustered if available."""
        self._on_data_loaded()

    def _on_data_loaded(self, data=None):
        if not self._ui_ready:
            return
        try:
            self._latest_recommendation = None
            registry = getattr(self, 'registry', None) or self.get_registry()
            if registry is None:
                return

            # Get the primary drillhole_data (contains transformed columns)
            primary_data = registry.get_drillhole_data(copy_data=False)
            primary_df = None
            if isinstance(primary_data, dict):
                for key in ("composites", "composites_df", "assays", "assays_df"):
                    candidate = primary_data.get(key)
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        primary_df = candidate
                        break

            # Prefer declustered data
            declust = registry.get_declustering_results() if hasattr(registry, 'get_declustering_results') else None
            if declust and 'weighted_dataframe' in declust:
                df = declust['weighted_dataframe']
                if not df.empty:
                    df = ensure_xyz_columns(df)
                    # Merge transformed columns from primary data
                    if primary_df is not None:
                        for col in primary_df.columns:
                            if col not in df.columns:
                                df = df.copy()
                                df[col] = primary_df[col].reindex(df.index)
                    self.drillhole_data = self._prepare_domain_filter_dataframe(
                        df,
                        registry_payload=primary_data,
                        populate_combo=True,
                        combo=self.domain_combo,
                        all_label="All Data",
                    )
                    self._using_declustered = True
                    self._data_status.setText(f"✓ Declustered ({len(self.drillhole_data)} samples)")
                    self._data_status.setStyleSheet("color: #27AE60; font-size: 9pt; font-weight: bold;")
                    self._refresh_variables(self.drillhole_data)
                    return

            # Fallback to composites/assays
            data = data or registry.get_estimation_ready_data() or registry.get_drillhole_data()
            df = None
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                for key in ("composites", "composites_df", "assays", "assays_df"):
                    candidate = data.get(key)
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        df = candidate
                        break
            # Merge transformed columns from primary data if using a different source
            if df is not None and primary_df is not None and df is not primary_df:
                for col in primary_df.columns:
                    if col not in df.columns:
                        df = df.copy()
                        df[col] = primary_df[col].reindex(df.index)
            if df is not None:
                df = ensure_xyz_columns(df)
                self.drillhole_data = self._prepare_domain_filter_dataframe(
                    df,
                    registry_payload=data,
                    populate_combo=True,
                    combo=self.domain_combo,
                    all_label="All Data",
                )
                self._using_declustered = False
                self._data_status.setText(f"Composites ({len(self.drillhole_data)} samples)")
                self._data_status.setStyleSheet("color: #F39C12; font-size: 9pt;")
                self._refresh_variables(self.drillhole_data)
        except Exception as exc:
            logger.debug("Variogram data load failed: %s", exc)

    def _manual_refresh(self):
        """Manual refresh — reload latest data from registry."""
        self._on_data_loaded()
        logger.info("VariogramAnalysisPanel: Manual refresh completed")

    def _refresh_variables(self, df):
        self.var_combo.clear()
        for col in get_grade_columns(df):
            self.var_combo.addItem(col)
        self._populate_domain_filter_combo(df, combo=self.domain_combo, all_label="All Data")

    def set_drillhole_data(self, data):
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = None
            for key in ("composites", "composites_df", "assays"):
                if key in data and isinstance(data[key], pd.DataFrame):
                    df = data[key]
                    break
        else:
            df = None
        if isinstance(df, pd.DataFrame):
            df = ensure_xyz_columns(df)
            self.drillhole_data = self._prepare_domain_filter_dataframe(
                df,
                registry_payload=data if isinstance(data, dict) else None,
                populate_combo=True,
                combo=self.domain_combo,
                all_label="All Data",
            )
        if self.drillhole_data is not None and self._ui_ready:
            self._refresh_variables(self.drillhole_data)
            self._data_status.setText(f"{len(self.drillhole_data)} samples loaded")

    def _get_filtered_variogram_inputs(self, variable: str):
        """Return the active data slice, domain metadata, and optional declustering weights."""
        df = self.drillhole_data.dropna(subset=['X', 'Y', 'Z', variable])
        df, domain_filter_metadata = self._apply_domain_filter(
            df,
            combo=self.domain_combo,
            all_label="All Data",
        )
        if df is None or df.empty:
            raise ValueError("No samples remain after applying the selected domain filter.")

        ext_weights = None
        if self._using_declustered and 'declust_weight' in df.columns:
            ext_weights = df['declust_weight'].to_numpy(float)
            logger.info("Using declustering weights: %d samples", len(ext_weights))

        return df, domain_filter_metadata, ext_weights

    def _apply_recommendation_to_ui(self, recommendation: Dict[str, Any]) -> None:
        """Apply recommended settings to the live widgets."""
        settings = recommendation.get("settings", {})

        self.cb_auto_lags.setChecked(bool(settings.get("auto_lags", True)))
        self.nlag_spin.setValue(int(settings.get("nlag", self.nlag_spin.value())))
        self.lag_dist_spin.setValue(float(settings.get("lag_distance", self.lag_dist_spin.value())))
        self.lag_tol_spin.setValue(float(settings.get("lag_tolerance", self.lag_tol_spin.value())))

        self.cb_manual_azimuth.setChecked(bool(settings.get("manual_azimuth", False)))
        self.major_az_spin.setValue(float(settings.get("default_azimuth", self.major_az_spin.value())) % 360.0)
        self.major_dip_spin.setValue(float(settings.get("default_dip", self.major_dip_spin.value())))
        self.cone_tol_spin.setValue(float(settings.get("cone_tolerance", self.cone_tol_spin.value())))

        model_label = str(settings.get("model_type", self.model_combo.currentText())).title()
        idx = self.model_combo.findText(model_label, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.struct_spin.setValue(int(settings.get("n_structures", self.struct_spin.value())))

        global_nugget = settings.get("global_nugget")
        self.cb_global_nugget.setChecked(global_nugget is not None)
        if global_nugget is not None:
            self.global_nugget_spin.setValue(float(global_nugget))

        bandwidth = settings.get("bandwidth")
        self.cb_bandwidth.setChecked(bandwidth is not None)
        if bandwidth is not None:
            self.bandwidth_spin.setValue(float(bandwidth))

    def _update_recommendations(self, recommendation: Dict[str, Any]) -> None:
        """Render the recommendation report in the Recommendations tab."""
        settings = recommendation.get("settings", {})
        analysis = recommendation.get("analysis", {})
        rationale = recommendation.get("rationale", [])
        bandwidth = settings.get("bandwidth")
        bandwidth_text = f"{float(bandwidth):.1f} m" if bandwidth is not None else "disabled"
        spacing_value = analysis.get("median_spacing", float("nan"))
        spacing_text = f"{float(spacing_value):.1f} m" if np.isfinite(spacing_value) else "unavailable"
        lines = [
            "DEEP VARIOGRAM ANALYSIS",
            "=======================",
            "",
            "Applied Settings",
            "----------------",
            f"Model family:     {settings.get('model_type', 'spherical')}",
            f"Auto lags:        {settings.get('auto_lags', True)}",
            f"N lags:           {settings.get('nlag', '-')}",
            f"Lag distance:     {float(settings.get('lag_distance', 0.0)):.1f} m",
            f"Lag tolerance:    {float(settings.get('lag_tolerance', 0.0)):.1f} m",
            f"Manual azimuth:   {settings.get('manual_azimuth', False)}",
            f"Azimuth / Dip:    {settings.get('default_azimuth', 0.0):.1f}° / {settings.get('default_dip', 0.0):.1f}°",
            f"Cone tolerance:   {settings.get('cone_tolerance', 0.0):.1f}°",
            f"Structures:       {settings.get('n_structures', 1)}",
            f"Global nugget:    {settings.get('global_nugget') if settings.get('global_nugget') is not None else 'not locked'}",
            f"Bandwidth:        {bandwidth_text}",
            "",
            "Data Analysis",
            "-------------",
            f"Samples:          {analysis.get('n_samples', 0)}",
            f"Sample variance:  {float(analysis.get('sample_variance', 0.0)):.4f}",
            f"Support points:   {analysis.get('support_points', 0)}",
            f"Median spacing:   {spacing_text}",
            f"Orientation src:  {analysis.get('orientation_source', 'unknown')}",
            f"Orientation ratio:{float(analysis.get('orientation_ratio', 0.0)):.2f}",
            f"Probe cone:       {analysis.get('probe_cone_tolerance', 0.0):.1f}°",
            f"Model scores:     {analysis.get('probe_model_scores', {})}",
            f"Weak directions:  {analysis.get('critical_weak_directions', []) or 'none critical'}",
            "",
            "Rationale",
            "---------",
        ]
        lines.extend(f"- {reason}" for reason in rationale)
        self._recommend_text.setPlainText("\n".join(lines))
        self._info_tabs.setCurrentWidget(self._recommend_text)

    def _on_recommend_and_fit(self):
        """Deep-analyze the current data, apply recommended settings, and compute."""
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "No drillhole data loaded.\nData loads automatically from the registry.")
            return

        var = self.var_combo.currentText()
        if not var:
            QMessageBox.warning(self, "No Variable", "Select a grade variable.")
            return

        self.recommend_btn.setEnabled(False)
        self.compute_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        try:
            df, domain_filter_metadata, ext_weights = self._get_filtered_variogram_inputs(var)

            def _progress(pct, msg):
                self.progress_bar.setValue(max(0, min(int(pct), 100)))
                self.progress_bar.setFormat(f"{int(pct)}% — {msg}")
                QApplication.processEvents()

            recommendation = _recommend_variogram_settings(
                df,
                vcol=var,
                z_positive_up=True,
                random_state=self.seed_spin.value(),
                sample_weights=ext_weights,
                progress_callback=_progress,
            )
            recommendation["variable"] = var
            recommendation["domain_filter_metadata"] = domain_filter_metadata
            self._latest_recommendation = recommendation
            if hasattr(self, 'export_rec_btn'):
                self.export_rec_btn.setEnabled(True)
            self._apply_recommendation_to_ui(recommendation)
            self._update_recommendations(recommendation)
        except Exception as exc:
            logger.error("Variogram recommendation failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Recommendation Error", f"Recommendation failed:\n{exc}")
            self.recommend_btn.setEnabled(True)
            self.compute_btn.setEnabled(True)
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            return

        self.recommend_btn.setEnabled(True)
        self.compute_btn.setEnabled(True)
        self._on_compute()

    # ══════════════════════════════════════════════════════════════
    # COMPUTE VARIOGRAM
    # ══════════════════════════════════════════════════════════════

    def _on_compute(self):
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "No drillhole data loaded.\nData loads automatically from the registry.")
            return

        var = self.var_combo.currentText()
        if not var:
            QMessageBox.warning(self, "No Variable", "Select a grade variable.")
            return

        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        try:
            try:
                df, domain_filter_metadata, _ext_weights = self._get_filtered_variogram_inputs(var)
            except ValueError as exc:
                QMessageBox.warning(self, "No Domain Data", str(exc))
                return

            def _progress(pct, msg):
                self.progress_bar.setValue(pct)
                self.progress_bar.setFormat(f"{pct}% — {msg}")
                QApplication.processEvents()

            default_az = self.major_az_spin.value() if self.cb_manual_azimuth.isChecked() else None
            default_dip = self.major_dip_spin.value() if self.cb_manual_azimuth.isChecked() else None
            global_nugget = self.global_nugget_spin.value() if self.cb_global_nugget.isChecked() else None
            bw = self.bandwidth_spin.value() if self.cb_bandwidth.isChecked() else None

            # ── Skewness / CV check: warn when raw data needs transform ──
            if var in df.columns:
                _vals = df[var].dropna()
                if len(_vals) > 30:
                    _skew = float(_vals.skew())
                    _cv = float(_vals.std() / max(abs(_vals.mean()), 1e-12))
                    if _skew > 2.0 or _cv > 1.5:
                        _ns_col = f"{var}_NS"
                        _ns_hint = (
                            f"\n\nTip: A pre-computed '{_ns_col}' column exists in "
                            f"your data — select it as the variable for better results."
                            if _ns_col in df.columns else ""
                        )
                        _reply = QMessageBox.warning(
                            self,
                            "Highly Skewed Data",
                            f"'{var}' is highly skewed (skewness={_skew:.1f}, "
                            f"CV={_cv:.1f}).\n\n"
                            f"Raw variograms on skewed data produce unreliable "
                            f"fits (sills will be wrong, ranges may hit "
                            f"boundaries).{_ns_hint}\n\n"
                            f"Continue with raw data anyway?",
                            QMessageBox.StandardButton.Yes
                            | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )
                        if _reply == QMessageBox.StandardButton.No:
                            return

            results = run_variogram_pipeline_v2(
                df, vcol=var,
                nlag=self.nlag_spin.value(),
                lag_distance=self.lag_dist_spin.value(),
                lag_tolerance=self.lag_tol_spin.value(),
                model_types=[self.model_combo.currentText().lower()],
                z_positive_up=True,
                default_azimuth=default_az,
                default_dip=default_dip,
                azimuth_tolerance=self.cone_tol_spin.value(),
                dip_tolerance=self.cone_tol_spin.value(),
                auto_lags=self.cb_auto_lags.isChecked(),
                n_structures=self.struct_spin.value(),
                global_nugget=global_nugget,
                progress_callback=_progress,
                random_state=self.seed_spin.value(),
                sample_weights=_ext_weights,
                bandwidth=bw,
            )

            self.variogram_results = results

            # Cache experimental variograms for fast model-type refit
            self._last_experimental = {
                'results': results,
                'variable': var,
                'domain_filter_metadata': domain_filter_metadata,
            }

            # Build combined model
            combined = self._build_combined_model(results, var)
            results['combined_3d_model'] = combined
            results['variable'] = var

            # Data lineage
            try:
                results['source_data_hash'] = compute_data_hash(df, var)
                results['data_source_type'] = 'declustered' if self._using_declustered else 'raw'
                results.setdefault('metadata', {})['source_data_n_samples'] = len(df)
                if domain_filter_metadata:
                    results.setdefault('metadata', {}).update(domain_filter_metadata)
                rec_meta = (self._latest_recommendation or {}).get("domain_filter_metadata", {})
                same_domain = (
                    rec_meta.get("domain_filter_selection")
                    == (domain_filter_metadata or {}).get("domain_filter_selection")
                )
                if (
                    self._latest_recommendation is not None
                    and self._latest_recommendation.get("variable") == var
                    and same_domain
                ):
                    results['recommendation'] = self._latest_recommendation
                    results.setdefault('metadata', {}).update({
                        'recommendation_applied': True,
                        'recommendation_version': self._latest_recommendation.get('recommendation_version'),
                        'recommended_model_type': self._latest_recommendation.get('settings', {}).get('model_type'),
                    })
            except Exception:
                pass

            # Update plots
            self._update_all_plots(results)
            self._update_summary(results, combined)
            self._update_warnings(results)

            # AUTO-PUBLISH to registry — all estimation methods pick it up automatically
            self._publish_to_registry(results, var)

            self.progress_bar.setValue(100)

        except Exception as exc:
            logger.error("Variogram computation failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Error", f"Variogram failed:\n{exc}")
        finally:
            self.compute_btn.setEnabled(True)
            self.compute_btn.setText("Compute Variogram")
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    # ══════════════════════════════════════════════════════════════
    # UPDATE PLOTS — one full-size plot per tab
    # ══════════════════════════════════════════════════════════════

    def _update_all_plots(self, results: Dict):
        """Update plots using the actual variogram3d result structure.

        Results keys: 'omni_variogram', 'major_variogram', etc. (DataFrames with
        columns: distance, gamma, npairs).  Fitted models at
        results['fitted_models'][direction][model_type] = {nugget, sill, range, total_sill}.
        """
        sample_var = results.get('metadata', {}).get('sample_variance', None)
        if sample_var is None:
            sample_var = results.get('sample_variance', 1.0)

        fitted_models = results.get('fitted_models', {})
        model_type = self.model_combo.currentText().lower()

        direction_map = {
            'omni': ('Omnidirectional', 'omni_variogram'),
            'major': (f"Major (az={results.get('major_azimuth', 0):.0f}°)", 'major_variogram'),
            'minor': (f"Minor (az={results.get('minor_azimuth', 90):.0f}°)", 'minor_variogram'),
            'vertical': ('Vertical', 'vertical_variogram'),
            'downhole': ('Downhole', 'downhole_variogram'),
        }

        for key, (title, data_key) in direction_map.items():
            pw = self._plot_widgets.get(key)
            if pw is None:
                continue

            vg_df = results.get(data_key)
            if vg_df is None or (isinstance(vg_df, pd.DataFrame) and vg_df.empty):
                pw.plot_variogram(lags=np.array([]), gamma=np.array([]), title=title)
                continue

            # Extract from DataFrame
            if isinstance(vg_df, pd.DataFrame):
                lags = vg_df['distance'].values if 'distance' in vg_df.columns else np.array([])
                gamma = vg_df['gamma'].values if 'gamma' in vg_df.columns else np.array([])
                counts = vg_df['npairs'].values if 'npairs' in vg_df.columns else None
            else:
                lags = np.array([])
                gamma = np.array([])
                counts = None

            # Get fitted model parameters — primary (user-selected) model
            fitted = fitted_models.get(key, {}).get(model_type, {})

            # Build dict of ALL fitted model types for this direction
            all_dir_models: Dict[str, Dict] = {}
            dir_models = fitted_models.get(key, {})
            for mt in ('spherical', 'exponential', 'gaussian'):
                m_params = dir_models.get(mt, {})
                if m_params and m_params.get('range', 0) > 0:
                    all_dir_models[mt] = m_params
            # Only use overlay when we have more than one model type
            overlay = all_dir_models if len(all_dir_models) > 1 else None

            # Generate single model curve as fallback (when only one type fitted)
            model_lags = None
            model_gamma = None
            if overlay is None and fitted and len(lags) > 0:
                model_lags = np.linspace(0, lags.max() * 1.15, 500)
                nug = fitted.get('nugget', 0)
                sill = fitted.get('sill', 0)
                rng = fitted.get('range', 1)
                h_r = model_lags / max(rng, 1e-6)
                if model_type == 'spherical':
                    model_gamma = np.where(
                        h_r < 1.0,
                        nug + sill * (1.5 * h_r - 0.5 * h_r**3),
                        nug + sill,
                    )
                elif model_type == 'exponential':
                    model_gamma = nug + sill * (1.0 - np.exp(-3.0 * h_r))
                elif model_type == 'gaussian':
                    model_gamma = nug + sill * (1.0 - np.exp(-3.0 * h_r**2))

            pw.plot_variogram(
                lags=lags, gamma=gamma,
                model_lags=model_lags, model_gamma=model_gamma,
                counts=counts, sample_var=sample_var,
                fitted_params=fitted, title=title,
                all_fitted_models=overlay,
            )

        # Update 3D map, anisotropy plots, variogram map, and h-scatterplot
        self._update_3d_map(results)
        self._update_anisotropy_plot(results)
        self._update_variogram_map(results)
        self._update_h_scatter(results)

    def _update_3d_map(self, results: Dict):
        """Plot 3D variogram map — shows semivariance as colour in 3D space."""
        self._3d_fig.clear()
        try:
            vgm = results.get('variogram_object')
            has_model = (vgm is not None) or bool(results.get('combined_3d_model'))
            if not has_model:
                ax = self._3d_fig.add_subplot(111)
                ax.text(0.5, 0.5, "No variogram object available",
                       ha='center', va='center', fontsize=14, color='grey')
                self._3d_canvas.draw_idle()
                return

            ax = self._3d_fig.add_subplot(111, projection='3d')

            az_major = results.get('major_azimuth', 0.0)
            az_minor = results.get('minor_azimuth', 90.0)

            direction_configs = [
                ('omni', 'omni_variogram', '#2980B9', 'Omni'),
                ('major', 'major_variogram', '#E74C3C', f'Major ({az_major:.0f}°)'),
                ('minor', 'minor_variogram', '#27AE60', f'Minor ({az_minor:.0f}°)'),
                ('vertical', 'vertical_variogram', '#8E44AD', 'Vertical'),
            ]

            for key, data_key, color, label in direction_configs:
                vg_df = results.get(data_key)
                if vg_df is None or not isinstance(vg_df, pd.DataFrame) or vg_df.empty:
                    continue
                lags = vg_df['distance'].values if 'distance' in vg_df.columns else np.array([])
                if len(lags) == 0:
                    continue

                az_rad = np.radians(az_major if key == 'major' else (az_minor if key == 'minor' else 0))
                if key == 'vertical':
                    xs = np.zeros_like(lags)
                    ys = np.zeros_like(lags)
                    zs = lags
                elif key == 'omni':
                    xs = lags * np.cos(az_rad)
                    ys = lags * np.sin(az_rad)
                    zs = np.zeros_like(lags)
                else:
                    xs = lags * np.sin(az_rad)
                    ys = lags * np.cos(az_rad)
                    zs = np.zeros_like(lags)

                ax.scatter(xs, ys, zs, c=color, s=30, alpha=0.7, label=label)

            combined = results.get('combined_3d_model', {})
            if combined:
                r_max = combined.get('major_range', 100)
                r_mid = combined.get('minor_range', 100)
                r_min = combined.get('vertical_range', 100)

                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                x_e = r_max * np.outer(np.cos(u), np.sin(v))
                y_e = r_mid * np.outer(np.sin(u), np.sin(v))
                z_e = r_min * np.outer(np.ones_like(u), np.cos(v))
                ax.plot_wireframe(x_e, y_e, z_e, alpha=0.15, color='orange', linewidth=0.5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Variogram Map', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
        except Exception as exc:
            logger.warning("3D variogram map failed: %s", exc)
            ax = self._3d_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"3D plot error: {exc}", ha='center', va='center', color='red')

        self._3d_fig.tight_layout()
        self._3d_canvas.draw_idle()

        # Update the companion anisotropic fit plot
        self._update_aniso_fit(results)

    def _update_aniso_fit(self, results: Dict):
        """Plot all-direction experimental variograms + fitted curves side-by-side with the 3D map."""
        self._anisofit_fig.clear()
        try:
            fitted = results.get('fitted_models', {})
            model_type = self.model_combo.currentText().lower()
            combined = results.get('combined_3d_model', {})
            total_sill = combined.get('total_sill', 1.0) if combined else 1.0
            cm_type = combined.get('model_type', model_type) if combined else model_type

            az_major = results.get('major_azimuth', 0.0)
            az_minor = results.get('minor_azimuth', 90.0)

            ax = self._anisofit_fig.add_subplot(111)

            def _eval_model(h, m_type, nug, psill, rng):
                """Evaluate a variogram model at the given lag distances.

                Replaces the legacy ``Variogram3D.evaluate_model`` call —
                the shared MODEL_MAP kernels take total sill (C0 + C).
                """
                func = _MODEL_MAP.get(m_type, _MODEL_MAP["spherical"])
                return func(h, rng, psill + nug, nug)

            direction_configs = [
                ('omni', 'omni_variogram', '#2980B9', 'Omni'),
                ('major', 'major_variogram', '#E74C3C', f'Major ({az_major:.0f}°)'),
                ('minor', 'minor_variogram', '#27AE60', f'Minor ({az_minor:.0f}°)'),
                ('vertical', 'vertical_variogram', '#8E44AD', 'Vertical'),
            ]

            for key, data_key, color, label in direction_configs:
                vg_df = results.get(data_key)
                if vg_df is not None and isinstance(vg_df, pd.DataFrame) and not vg_df.empty:
                    lags = vg_df['distance'].values
                    gamma = vg_df['gamma'].values
                    if len(lags) > 0:
                        ax.scatter(lags, gamma, color=color, s=30, alpha=0.7, zorder=3,
                                   label=label)

                params = fitted.get(key, {}).get(model_type)
                if params:
                    p_nug = params.get('nugget', 0)
                    p_sill = params.get('partial_sill', params.get('sill', 1))
                    p_range = params.get('range', params.get('effective_range', 100))
                    p_type = params.get('model_type', cm_type)
                    max_h = p_range * 1.5
                    if vg_df is not None and isinstance(vg_df, pd.DataFrame) and not vg_df.empty:
                        max_h = max(max_h, vg_df['distance'].max() * 1.1)
                    h = np.linspace(0, max_h, 300)
                    gamma_fit = _eval_model(h, p_type, p_nug, p_sill, p_range)
                    ax.plot(h, gamma_fit, color=color, lw=2.0, solid_capstyle='round',
                            label=f'fit on {label.split("(")[0].strip().lower()}')

            ax.axhline(y=total_sill, color='grey', linestyle='--', alpha=0.3, linewidth=1.0)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Lag Distance')
            ax.set_ylabel('Semivariance γ(h)')
            ax.set_title('Fitting an anisotropic model', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='lower right')
        except Exception as exc:
            logger.warning("Anisotropic fit plot failed: %s", exc)
            ax = self._anisofit_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Fit plot error: {exc}", ha='center', va='center', color='red')

        self._anisofit_fig.tight_layout()
        self._anisofit_canvas.draw_idle()

    def _update_anisotropy_plot(self, results: Dict):
        """Professional anisotropy visualisation with compass rose, directional
        variogram curves overlaid on ellipse axes, range tick marks, drill collar
        positions, and geological notation."""
        self._aniso_fig.clear()
        combined = results.get('combined_3d_model', {})

        if not combined:
            ax = self._aniso_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No model fitted", ha='center', va='center',
                    fontsize=14, color='grey')
            self._aniso_canvas.draw_idle()
            return

        r_max = combined.get('major_range', 100)
        r_mid = combined.get('minor_range', 100)
        r_min = combined.get('vertical_range', 100)
        azimuth = combined.get('azimuth', 0.0)
        dip = combined.get('dip', 0.0)
        nugget = combined.get('nugget', 0.0)
        psill = combined.get('partial_sill', combined.get('sill', 0.0))
        total_sill = combined.get('total_sill', nugget + psill)
        nugget_pct = nugget / total_sill * 100 if total_sill > 0 else 0
        model_type = combined.get('model_type', 'spherical')
        aniso_h = r_max / max(r_mid, 1e-3)
        aniso_v = r_max / max(r_min, 1e-3)

        theta = np.linspace(0, 2 * np.pi, 200)
        az_rad = np.radians(azimuth)
        lim = max(r_max, r_mid, r_min) * 1.35

        # ── Colours ────────────────────────────────────────────────
        C_MAJOR = '#E53935'    # red
        C_MINOR = '#43A047'    # green
        C_VERT  = '#7E57C2'    # purple
        C_GRID  = '#BDBDBD'
        C_BG    = '#FAFAFA'

        # ══════════════════════════════════════════════════════════
        # LEFT: Plan View (XY) with compass rose
        # ══════════════════════════════════════════════════════════
        ax1 = self._aniso_fig.add_subplot(121)
        ax1.set_facecolor(C_BG)

        # Compass rose — concentric range circles
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = lim * frac * 0.75
            circle = plt.Circle((0, 0), r, fill=False, color=C_GRID,
                                linewidth=0.4, linestyle='--')
            ax1.add_patch(circle)
            if frac < 1.0:
                ax1.text(r * 0.05, r + lim * 0.02, f'{r:.0f}m',
                         fontsize=6, color='#9E9E9E', ha='left')

        # Compass labels
        compass_r = lim * 0.82
        for label, angle in [('N', 90), ('E', 0), ('S', 270), ('W', 180)]:
            x = compass_r * np.cos(np.radians(angle))
            y = compass_r * np.sin(np.radians(angle))
            ax1.text(x, y, label, ha='center', va='center', fontsize=11,
                     fontweight='bold', color='#616161')
        # Inter-cardinal
        for label, angle in [('NE', 45), ('SE', 315), ('SW', 225), ('NW', 135)]:
            x = compass_r * np.cos(np.radians(angle))
            y = compass_r * np.sin(np.radians(angle))
            ax1.text(x, y, label, ha='center', va='center', fontsize=7,
                     color='#9E9E9E')

        # Thin compass lines (N-S, E-W)
        ax1.plot([-lim, lim], [0, 0], color=C_GRID, linewidth=0.5)
        ax1.plot([0, 0], [-lim, lim], color=C_GRID, linewidth=0.5)

        # Range ellipse — rotated (GSLIB convention: az CW from North/Y+)
        ex = r_max * np.cos(theta)
        ey = r_mid * np.sin(theta)
        # Geographic rotation: major along azimuth from North
        rx = ex * np.sin(az_rad) + ey * np.cos(az_rad)
        ry = ex * np.cos(az_rad) - ey * np.sin(az_rad)
        ax1.fill(rx, ry, alpha=0.08, color=C_MAJOR)
        ax1.plot(rx, ry, color=C_MAJOR, linewidth=2.0, solid_capstyle='round')

        # Major direction arrow + range ticks
        maj_dx = r_max * np.sin(az_rad)
        maj_dy = r_max * np.cos(az_rad)
        ax1.annotate('', xy=(maj_dx, maj_dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=C_MAJOR, lw=2.2,
                                     shrinkA=0, shrinkB=0))
        ax1.annotate('', xy=(-maj_dx, -maj_dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=C_MAJOR, lw=1.2,
                                     alpha=0.4, shrinkA=0, shrinkB=0))
        # Range tick marks along major
        for frac in (0.25, 0.5, 0.75):
            tx, ty = maj_dx * frac, maj_dy * frac
            perp_x, perp_y = -maj_dy / r_max * 4, maj_dx / r_max * 4
            ax1.plot([tx - perp_x, tx + perp_x], [ty - perp_y, ty + perp_y],
                     color=C_MAJOR, linewidth=1.0, alpha=0.5)

        # Major label at tip
        ax1.text(maj_dx * 1.08, maj_dy * 1.08, f'{r_max:.0f}m',
                 fontsize=8, color=C_MAJOR, fontweight='bold', ha='center', va='center')

        # Minor direction arrow + range ticks
        min_dx = r_mid * np.cos(az_rad)
        min_dy = -r_mid * np.sin(az_rad)
        ax1.annotate('', xy=(min_dx, min_dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=C_MINOR, lw=2.0,
                                     shrinkA=0, shrinkB=0))
        ax1.text(min_dx * 1.12, min_dy * 1.12, f'{r_mid:.0f}m',
                 fontsize=8, color=C_MINOR, fontweight='bold', ha='center', va='center')

        # Drill collar positions (if source data available)
        src = results.get('_source_data')
        xcol = results.get('_xcol', 'X')
        ycol = results.get('_ycol', 'Y')
        if src is not None and xcol in src.columns and ycol in src.columns:
            sx = src[xcol].values - src[xcol].mean()
            sy = src[ycol].values - src[ycol].mean()
            # Scale to fit within ellipse display
            data_span = max(np.max(sx) - np.min(sx), np.max(sy) - np.min(sy), 1)
            scale = lim * 0.6 / (data_span * 0.5)
            ax1.scatter(sx * scale, sy * scale, s=3, color='#90A4AE',
                        alpha=0.3, zorder=1, label='Drill collars')

        # Azimuth arc annotation
        arc_r = lim * 0.45
        arc_theta = np.linspace(np.pi / 2, np.pi / 2 - az_rad, 50)
        ax1.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta),
                 color=C_MAJOR, linewidth=1.0, alpha=0.5, linestyle='-')
        mid_arc = np.pi / 2 - az_rad / 2
        ax1.text(arc_r * 0.85 * np.cos(mid_arc), arc_r * 0.85 * np.sin(mid_arc),
                 f'{azimuth:.0f}\u00b0', fontsize=8, color=C_MAJOR, ha='center',
                 va='center', fontstyle='italic')

        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_aspect('equal')
        ax1.set_xlabel('Easting (m)', fontsize=9)
        ax1.set_ylabel('Northing (m)', fontsize=9)
        ax1.set_title('Plan View', fontsize=12, fontweight='bold', pad=10)
        ax1.tick_params(labelsize=7)

        # ══════════════════════════════════════════════════════════
        # RIGHT: Section View (along strike) with variogram curve overlay
        # ══════════════════════════════════════════════════════════
        ax2 = self._aniso_fig.add_subplot(122)
        ax2.set_facecolor(C_BG)

        # Section ellipse (major × vertical)
        sx_e = r_max * np.cos(theta)
        sz_e = r_min * np.sin(theta)
        ax2.fill(sx_e, sz_e, alpha=0.06, color=C_VERT)
        ax2.plot(sx_e, sz_e, color=C_VERT, linewidth=2.0, solid_capstyle='round')

        # Horizontal axis arrow (major range)
        ax2.annotate('', xy=(r_max, 0), xytext=(-r_max, 0),
                     arrowprops=dict(arrowstyle='<->', color=C_MAJOR, lw=1.5))
        ax2.text(0, -r_min * 0.15, f'Major: {r_max:.0f}m',
                 fontsize=8, color=C_MAJOR, ha='center', va='top', fontweight='bold')

        # Vertical axis arrow (vertical range)
        ax2.annotate('', xy=(0, r_min), xytext=(0, -r_min),
                     arrowprops=dict(arrowstyle='<->', color=C_VERT, lw=1.5))
        ax2.text(r_max * 0.08, 0, f'Vert: {r_min:.0f}m',
                 fontsize=8, color=C_VERT, ha='left', va='center', fontweight='bold',
                 rotation=90)

        # Range tick marks
        for frac in (0.25, 0.5, 0.75):
            tx = r_max * frac
            ax2.plot([tx, tx], [-3, 3], color=C_MAJOR, linewidth=0.8, alpha=0.4)
            ax2.plot([-tx, -tx], [-3, 3], color=C_MAJOR, linewidth=0.8, alpha=0.4)
            tz = r_min * frac
            ax2.plot([-3, 3], [tz, tz], color=C_VERT, linewidth=0.8, alpha=0.4)
            ax2.plot([-3, 3], [-tz, -tz], color=C_VERT, linewidth=0.8, alpha=0.4)

        # Dip indicator (if non-zero)
        if abs(dip) > 0.5:
            dip_rad = np.radians(dip)
            dip_len = r_max * 0.4
            ax2.annotate('', xy=(dip_len * np.cos(dip_rad), -dip_len * np.sin(dip_rad)),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
            ax2.text(dip_len * 0.5 * np.cos(dip_rad),
                     -dip_len * 0.5 * np.sin(dip_rad) - 5,
                     f'Dip: {dip:.0f}\u00b0', fontsize=8, color='#FF9800',
                     fontstyle='italic')

        # Grid
        ax2.axhline(0, color=C_GRID, linewidth=0.5)
        ax2.axvline(0, color=C_GRID, linewidth=0.5)

        v_lim = max(r_max, r_min) * 1.3
        ax2.set_xlim(-v_lim, v_lim)
        ax2.set_ylim(-v_lim, v_lim)
        ax2.set_aspect('equal')
        ax2.set_xlabel('Along Strike (m)', fontsize=9)
        ax2.set_ylabel('Elevation (m)', fontsize=9)
        ax2.set_title('Section View', fontsize=12, fontweight='bold', pad=10)
        ax2.tick_params(labelsize=7)

        # ── Model parameters table (centre-bottom) ────────────────
        param_text = (
            f"{model_type.title()} Model\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"C\u2080 (nugget)    {nugget:>8.2f}  ({nugget_pct:.0f}%)\n"
            f"C  (partial sill) {psill:>8.2f}\n"
            f"C\u2080+C (total)   {total_sill:>8.2f}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Major range    {r_max:>8.1f} m\n"
            f"Minor range    {r_mid:>8.1f} m  (ratio {aniso_h:.1f}:1)\n"
            f"Vertical range {r_min:>8.1f} m  (ratio {aniso_v:.1f}:1)\n"
            f"Azimuth        {azimuth:>8.0f}\u00b0\n"
            f"Dip            {dip:>8.0f}\u00b0"
        )
        self._aniso_fig.text(
            0.5, 0.01, param_text, fontsize=8, fontfamily='monospace',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#BDBDBD', alpha=0.95, linewidth=0.8),
        )

        self._aniso_fig.suptitle('Search Ellipsoid & Anisotropy',
                                 fontsize=13, fontweight='bold', y=0.98)
        try:
            self._aniso_fig.tight_layout(rect=[0.02, 0.18, 0.98, 0.94])
        except Exception:
            pass
        self._aniso_canvas.draw_idle()

    def _update_variogram_map(self, results: Dict):
        """Render the 2D variogram map as a polar heatmap.

        The variogram map shows semivariance as a function of both direction
        (azimuth) and lag distance.  High-semivariance directions indicate low
        continuity; low-semivariance directions indicate high continuity —
        this is the standard tool for identifying anisotropy axes before
        fitting a directional variogram model (Deutsch & Journel, 1998).
        """
        self._vmap_fig.clear()

        # Try to compute the variogram map from the data stored in results
        vmap = results.get("variogram_map")

        # If the pipeline didn't return a variogram map, compute one on the fly
        if vmap is None:
            try:
                from ..geostats.variogram_assistant import _compute_variogram_map
                data = results.get("_source_data")
                if data is None:
                    # Try to reconstruct from the variogram object
                    vgm = results.get("variogram_object")
                    if vgm is not None and hasattr(vgm, "data"):
                        data = vgm.data
                if data is None:
                    ax = self._vmap_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "No source data available for variogram map",
                            ha='center', va='center', fontsize=12, color='grey',
                            transform=ax.transAxes)
                    self._vmap_canvas.draw_idle()
                    self._update_variogram_map_3d(results, None)
                    return

                xcol = results.get("_xcol", "X")
                ycol = results.get("_ycol", "Y")
                zcol = results.get("_zcol", "Z")
                vcol = results.get("_vcol", "GRADE")
                cols_present = [c for c in [xcol, ycol, zcol, vcol] if c in data.columns]
                if len(cols_present) < 4:
                    ax = self._vmap_fig.add_subplot(111)
                    ax.text(0.5, 0.5, "Incomplete columns for variogram map",
                            ha='center', va='center', fontsize=12, color='grey',
                            transform=ax.transAxes)
                    self._vmap_canvas.draw_idle()
                    self._update_variogram_map_3d(results, None)
                    return

                coords = data[[xcol, ycol, zcol]].values
                values = data[vcol].values
                mask = np.isfinite(values)
                coords, values = coords[mask], values[mask]

                vmap = _compute_variogram_map(coords, values)
            except Exception as exc:
                logger.warning("Variogram map computation failed: %s", exc)
                ax = self._vmap_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Variogram map error:\n{exc}",
                        ha='center', va='center', fontsize=10, color='red',
                        transform=ax.transAxes)
                self._vmap_canvas.draw_idle()
                self._update_variogram_map_3d(results, None)
                return

        if vmap is None:
            ax = self._vmap_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No variogram map available",
                    ha='center', va='center', fontsize=12, color='grey',
                    transform=ax.transAxes)
            self._vmap_canvas.draw_idle()
            self._update_variogram_map_3d(results, None)
            return

        try:
            gamma_matrix = np.array(vmap["gamma_matrix"], dtype=float)
            azimuths = np.array(vmap["azimuths"], dtype=float)
            distances = np.array(vmap["distances"], dtype=float)
            pair_counts = np.array(vmap.get("pair_counts", np.ones_like(gamma_matrix)))

            n_az, n_lag = gamma_matrix.shape

            # ── Polar heatmap ──────────────────────────────────────────
            ax = self._vmap_fig.add_subplot(111, projection='polar')

            # Mirror the map to full 360° (input is 0-165°)
            full_az = np.concatenate([azimuths, azimuths + 180.0])
            full_gamma = np.concatenate([gamma_matrix, gamma_matrix], axis=0)
            full_pairs = np.concatenate([pair_counts, pair_counts], axis=0)

            # Convert azimuth to polar theta (geographic: 0=N, CW)
            theta_edges = np.deg2rad(90 - np.concatenate([
                full_az - (azimuths[1] - azimuths[0]) / 2 if len(azimuths) > 1 else full_az,
                [full_az[-1] + (azimuths[1] - azimuths[0]) / 2 if len(azimuths) > 1 else full_az[-1] + 15],
            ]))
            r_edges = np.concatenate([[0], distances])

            # Create meshgrid for pcolormesh
            T, R = np.meshgrid(theta_edges, r_edges, indexing='ij')

            # Mask cells with too few pairs
            masked_gamma = np.ma.masked_where(
                (full_pairs < 5) | np.isnan(full_gamma), full_gamma,
            )

            pc = ax.pcolormesh(
                T, R, masked_gamma,
                cmap='RdYlBu_r', shading='flat',
            )

            # Compass formatting
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            tick_pos = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                               fontsize=9)

            ax.set_title('Variogram Map\n(semivariance by direction & distance)',
                         fontsize=11, fontweight='bold', pad=20)

            # Colorbar
            cbar = self._vmap_fig.colorbar(pc, ax=ax, pad=0.12, shrink=0.8)
            cbar.set_label('\u03b3(h)', fontsize=10)

            # Annotate: directions of min/max continuity
            valid_mask = ~np.ma.getmaskarray(masked_gamma)
            if np.any(valid_mask):
                # Average gamma per azimuth (mean over distance lags)
                mean_gamma_per_az = np.nanmean(
                    np.where(valid_mask, full_gamma, np.nan), axis=1,
                )
                finite = np.isfinite(mean_gamma_per_az)
                if np.sum(finite) >= 2:
                    i_min = np.nanargmin(mean_gamma_per_az)
                    i_max = np.nanargmax(mean_gamma_per_az)
                    ax.annotate(
                        f"Max continuity\n({full_az[i_min]:.0f}\u00b0)",
                        xy=(np.deg2rad(90 - full_az[i_min]), distances[-1] * 0.6),
                        fontsize=8, color='#2196F3', fontweight='bold',
                        ha='center', va='center',
                    )
                    ax.annotate(
                        f"Min continuity\n({full_az[i_max]:.0f}\u00b0)",
                        xy=(np.deg2rad(90 - full_az[i_max]), distances[-1] * 0.6),
                        fontsize=8, color='#F44336', fontweight='bold',
                        ha='center', va='center',
                    )

        except Exception as exc:
            logger.warning("Variogram map rendering failed: %s", exc)
            ax = self._vmap_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Rendering error:\n{exc}",
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax.transAxes)

        try:
            self._vmap_fig.tight_layout()
        except Exception:
            pass
        self._vmap_canvas.draw_idle()

        # Update the companion 3D surface plot
        self._update_variogram_map_3d(results, vmap)

    def _update_variogram_map_3d(self, results: Dict, vmap: Optional[Dict] = None):
        """Render the variogram map as a 3D surface (azimuth × distance → γ)."""
        self._vmap3d_fig.clear()

        if vmap is None:
            ax = self._vmap3d_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No variogram map data for 3D view",
                    ha='center', va='center', fontsize=12, color='grey',
                    transform=ax.transAxes)
            self._vmap3d_canvas.draw_idle()
            return

        try:
            gamma_matrix = np.array(vmap["gamma_matrix"], dtype=float)
            azimuths = np.array(vmap["azimuths"], dtype=float)
            distances = np.array(vmap["distances"], dtype=float)
            pair_counts = np.array(vmap.get("pair_counts", np.ones_like(gamma_matrix)))

            # Mirror to full 360°
            full_az = np.concatenate([azimuths, azimuths + 180.0])
            full_gamma = np.concatenate([gamma_matrix, gamma_matrix], axis=0)
            full_pairs = np.concatenate([pair_counts, pair_counts], axis=0)

            # Mask low-pair cells
            full_gamma = np.where((full_pairs < 5) | np.isnan(full_gamma), np.nan, full_gamma)

            # Build meshgrid: azimuth (degrees) × distance → Cartesian X, Y
            Az, Dist = np.meshgrid(full_az, distances, indexing='ij')
            az_rad = np.deg2rad(Az)

            # Convert polar to Cartesian for 3D surface
            X = Dist * np.sin(az_rad)   # East
            Y = Dist * np.cos(az_rad)   # North
            Z = full_gamma

            ax = self._vmap3d_fig.add_subplot(111, projection='3d')

            # Mask NaNs for surface
            Z_masked = np.ma.masked_invalid(Z)

            # Colour by semivariance
            from matplotlib import cm
            norm = plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
            facecolors = cm.RdYlBu_r(norm(Z_masked))

            ax.plot_surface(X, Y, Z_masked, facecolors=facecolors,
                            rstride=1, cstride=1, alpha=0.85, shade=True,
                            linewidth=0.2, edgecolor='grey')

            # Add directional annotations
            az_major = results.get('major_azimuth', 0.0)
            az_minor = results.get('minor_azimuth', 90.0)
            d_max = distances[-1] if len(distances) > 0 else 100
            for az_deg, label, color in [
                (az_major, 'Major', '#E74C3C'),
                (az_minor, 'Minor', '#27AE60'),
            ]:
                rad = np.deg2rad(az_deg)
                ax.plot([0, d_max * np.sin(rad)],
                        [0, d_max * np.cos(rad)],
                        [0, 0], color=color, lw=2.0, alpha=0.8, label=label)

            ax.set_xlabel('E (m)')
            ax.set_ylabel('N (m)')
            ax.set_zlabel('γ(h)')
            ax.set_title('Variogram Map — 3D Surface', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')

            # Colorbar
            mappable = cm.ScalarMappable(norm=norm, cmap='RdYlBu_r')
            mappable.set_array(Z_masked)
            self._vmap3d_fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1,
                                       label='γ(h)')

        except Exception as exc:
            logger.warning("3D variogram map surface failed: %s", exc)
            ax = self._vmap3d_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"3D surface error:\n{exc}",
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax.transAxes)

        try:
            self._vmap3d_fig.tight_layout()
        except Exception:
            pass
        self._vmap3d_canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════
    # H-SCATTERPLOT — Z(x) vs Z(x+h) per lag
    # ══════════════════════════════════════════════════════════════

    def _update_h_scatter(self, results: Dict):
        """Render h-scatterplots (Z(x) vs Z(x+h)) for the first 6 lags.

        These are critical for checking stationarity: if the cloud shifts
        position across lags, the data is non-stationary.
        """
        self._hscatter_fig.clear()
        try:
            # We need the raw data and lag parameters
            var = results.get('variable', '')
            if self.drillhole_data is None or var not in self.drillhole_data.columns:
                ax = self._hscatter_fig.add_subplot(111)
                ax.text(0.5, 0.5, "No source data for h-scatterplot",
                        ha='center', va='center', fontsize=12, color='grey',
                        transform=ax.transAxes)
                self._hscatter_canvas.draw_idle()
                return

            df = self.drillhole_data.dropna(subset=['X', 'Y', 'Z', var])
            coords = df[['X', 'Y', 'Z']].values.astype(float)
            values = df[var].values.astype(float)
            n = len(values)

            if n < 10:
                ax = self._hscatter_fig.add_subplot(111)
                ax.text(0.5, 0.5, "Too few samples for h-scatterplot",
                        ha='center', va='center', fontsize=12, color='grey',
                        transform=ax.transAxes)
                self._hscatter_canvas.draw_idle()
                return

            # Get lag parameters from the omni variogram
            omni_df = results.get('omni_variogram')
            if omni_df is None or not isinstance(omni_df, pd.DataFrame) or omni_df.empty:
                ax = self._hscatter_fig.add_subplot(111)
                ax.text(0.5, 0.5, "No omnidirectional variogram for h-scatterplot",
                        ha='center', va='center', fontsize=12, color='grey',
                        transform=ax.transAxes)
                self._hscatter_canvas.draw_idle()
                return

            lag_dists = omni_df['distance'].values
            n_show = min(6, len(lag_dists))
            if n_show == 0:
                ax = self._hscatter_fig.add_subplot(111)
                ax.text(0.5, 0.5, "No lags computed",
                        ha='center', va='center', fontsize=12, color='grey',
                        transform=ax.transAxes)
                self._hscatter_canvas.draw_idle()
                return

            # Determine lag tolerance (half the spacing between first two lags)
            if len(lag_dists) > 1:
                lag_tol = (lag_dists[1] - lag_dists[0]) / 2.0
            else:
                lag_tol = lag_dists[0] / 2.0

            # Build KDTree for efficient pair finding
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            max_lag = lag_dists[n_show - 1] + lag_tol

            # Find all pairs within max_lag distance (cap to avoid memory issues)
            if n > 5000:
                # Subsample for performance
                rng_state = np.random.RandomState(42)
                idx = rng_state.choice(n, size=min(n, 3000), replace=False)
                coords_sub = coords[idx]
                values_sub = values[idx]
                tree_sub = cKDTree(coords_sub)
                pairs = tree_sub.query_pairs(max_lag, output_type='ndarray')
                pair_dists = np.linalg.norm(
                    coords_sub[pairs[:, 0]] - coords_sub[pairs[:, 1]], axis=1
                )
                z_tail = values_sub[pairs[:, 0]]
                z_head = values_sub[pairs[:, 1]]
            else:
                pairs = tree.query_pairs(max_lag, output_type='ndarray')
                pair_dists = np.linalg.norm(
                    coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1
                )
                z_tail = values[pairs[:, 0]]
                z_head = values[pairs[:, 1]]

            # Determine subplot layout
            if n_show <= 4:
                nrows, ncols = 2, 2
            else:
                nrows, ncols = 2, 3

            axes = self._hscatter_fig.subplots(nrows, ncols)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes_flat = axes.flatten()

            val_min = float(np.nanmin(values))
            val_max = float(np.nanmax(values))

            for i in range(n_show):
                ax = axes_flat[i]
                lag_center = lag_dists[i]
                # Find pairs in this lag bin
                mask = np.abs(pair_dists - lag_center) <= lag_tol
                zt = z_tail[mask]
                zh = z_head[mask]

                if len(zt) < 2:
                    ax.text(0.5, 0.5, f"Lag {i+1}: no pairs",
                            ha='center', va='center', fontsize=9, color='grey',
                            transform=ax.transAxes)
                    ax.set_title(f"h = {lag_center:.1f}m", fontsize=9)
                    continue

                # Compute correlation
                corr = float(np.corrcoef(zt, zh)[0, 1]) if len(zt) > 2 else 0.0

                # Scatter with density coloring
                ax.scatter(zt, zh, s=4, alpha=0.3, c='#2980B9', edgecolors='none')

                # 45-degree reference line (perfect correlation)
                ref_line = [val_min, val_max]
                ax.plot(ref_line, ref_line, '--', color='#E74C3C',
                        linewidth=1.0, alpha=0.6)

                ax.set_xlim(val_min, val_max)
                ax.set_ylim(val_min, val_max)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"h={lag_center:.0f}m  r={corr:.2f}  n={len(zt)}",
                             fontsize=8, fontweight='bold')
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)

                if i >= ncols * (nrows - 1):  # bottom row
                    ax.set_xlabel('Z(x)', fontsize=8)
                if i % ncols == 0:  # left column
                    ax.set_ylabel('Z(x+h)', fontsize=8)

            # Hide unused axes
            for j in range(n_show, len(axes_flat)):
                axes_flat[j].set_visible(False)

            self._hscatter_fig.suptitle(
                f'H-Scatterplots: {var} -- Z(x) vs Z(x+h)',
                fontsize=11, fontweight='bold',
            )

        except Exception as exc:
            logger.warning("H-scatterplot failed: %s", exc, exc_info=True)
            ax = self._hscatter_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"H-scatterplot error:\n{exc}",
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax.transAxes)

        try:
            self._hscatter_fig.tight_layout(rect=[0, 0, 1, 0.94])
        except Exception:
            pass
        self._hscatter_canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════
    # UPDATE INFO TABS
    # ══════════════════════════════════════════════════════════════

    def _update_summary(self, results: Dict, combined: Dict):
        meta = results.get('metadata', {})
        lines = []
        lines.append(f"Samples: {meta.get('source_data_n_samples', '?')}")
        lines.append(f"Sample Variance: {meta.get('sample_variance', 0):.4f}")
        lines.append("")
        if combined:
            lines.append("─── 3D Combined Model ───")
            lines.append(f"Total Sill:     {combined.get('total_sill', 0):.4f}")
            lines.append(f"Partial Sill:   {combined.get('sill', 0):.4f}")
            lines.append(f"Nugget:         {combined.get('nugget', 0):.4f}")
            lines.append(f"Nugget Ratio:   {combined.get('nugget_ratio', 0)*100:.1f}%")
            lines.append("")
            lines.append(f"Range Max:      {combined.get('major_range', 0):.1f} m")
            lines.append(f"Range Mid:      {combined.get('minor_range', 0):.1f} m")
            lines.append(f"Range Min:      {combined.get('vertical_range', 0):.1f} m")
            lines.append(f"Azimuth:        {combined.get('azimuth', 0):.1f}°")
            lines.append(f"Dip:            {combined.get('dip', 0):.1f}°")
        self._summary_text.setPlainText("\n".join(lines))

    def _update_warnings(self, results: Dict):
        warnings = []

        # 1. Nugget consistency check
        try:
            report = analyze_nugget_consistency(results)
            if not report.is_consistent:
                warnings.append(f"NUGGET INCONSISTENCY: {report.message}")
        except Exception as exc:
            logger.debug("Nugget consistency check failed: %s", exc)

        # 2. Weak direction warnings from metadata
        meta = results.get('metadata', {})
        weak_dirs = meta.get('weak_directions', [])
        for wd in weak_dirs:
            if isinstance(wd, dict):
                name = wd.get('direction', '?')
                pairs = wd.get('total_pairs', 0)
                avg = wd.get('avg_pairs_per_lag', 0)
                is_crit = wd.get('is_critical', False)
                sev = "CRITICAL" if is_crit else "WEAK"
                warnings.append(
                    f"{sev} DIRECTION: {name} has {avg:.0f} pairs/lag "
                    f"(total {pairs} pairs, need ≥30/lag for reliable fit)"
                )
            elif isinstance(wd, str):
                warnings.append(wd)

        # 3. Sill vs sample variance check
        sample_var = meta.get('sample_variance', 0)
        combined = results.get('combined_3d_model', {})
        if combined and sample_var > 0:
            total_sill = combined.get('total_sill', 0)
            if total_sill > sample_var * 1.2:
                warnings.append(
                    f"SILL EXCESS: Total sill ({total_sill:.1f}) exceeds "
                    f"sample variance ({sample_var:.1f}) by "
                    f"{total_sill/sample_var:.1f}×. Model may be over-fitted."
                )
            nugget_ratio = combined.get('nugget_ratio', 0)
            if nugget_ratio > 0.40:
                warnings.append(
                    f"HIGH NUGGET: Nugget ratio is {nugget_ratio*100:.0f}% — "
                    f"the estimator treats {nugget_ratio*100:.0f}% of grade "
                    f"variation as random noise. Check variogram fit."
                )

        # 4. Range checks — extreme anisotropy
        if combined:
            r_max = combined.get('major_range', 0)
            r_mid = combined.get('minor_range', 0)
            r_min = combined.get('vertical_range', 0)
            if r_max > 0 and r_min > 0 and r_max / r_min > 5:
                warnings.append(
                    f"EXTREME ANISOTROPY: Range ratio {r_max/r_min:.1f}:1 "
                    f"({r_max:.0f}m / {r_min:.0f}m). Verify directional "
                    f"variogram fits -- this may indicate fitting artifacts."
                )
            if r_max > 0 and r_mid > 0 and r_max / r_mid > 5:
                warnings.append(
                    f"EXTREME HORIZONTAL ANISOTROPY: Major/Minor range ratio "
                    f"{r_max/r_mid:.1f}:1 ({r_max:.0f}m / {r_mid:.0f}m). "
                    f"This may indicate the search azimuth is not aligned "
                    f"with the true major continuity direction, or there is "
                    f"genuine extreme horizontal anisotropy in the deposit."
                )

        # 5. Zonal anisotropy detection — variogram does not reach the sill
        fitted_models = results.get('fitted_models', {})
        model_type = self.model_combo.currentText().lower()
        combined_total_sill = (combined or {}).get('total_sill', 0)
        if combined_total_sill > 0:
            direction_labels = {
                'omni': ('Omnidirectional', 'omni_variogram'),
                'major': ('Major', 'major_variogram'),
                'minor': ('Minor', 'minor_variogram'),
                'vertical': ('Vertical', 'vertical_variogram'),
                'downhole': ('Downhole', 'downhole_variogram'),
            }
            zonal_dirs = []
            for dir_key, (dir_label, data_key) in direction_labels.items():
                vg_df = results.get(data_key)
                if vg_df is None or not isinstance(vg_df, pd.DataFrame) or vg_df.empty:
                    continue
                gamma_vals = vg_df['gamma'].values
                if len(gamma_vals) == 0:
                    continue
                # Get the fitted total_sill for this direction
                dir_fitted = fitted_models.get(dir_key, {}).get(model_type, {})
                dir_total_sill = dir_fitted.get('total_sill', combined_total_sill)
                if dir_total_sill <= 0:
                    dir_total_sill = combined_total_sill
                # Check if last-lag gamma is < 70% of the fitted total sill
                last_gamma = float(gamma_vals[-1])
                if last_gamma < 0.70 * dir_total_sill:
                    zonal_dirs.append(
                        f"{dir_label} (last-lag gamma={last_gamma:.2f}, "
                        f"sill={dir_total_sill:.2f}, ratio={last_gamma/dir_total_sill:.0%})"
                    )
            if zonal_dirs:
                warnings.append(
                    f"ZONAL ANISOTROPY DETECTED: The following direction(s) do "
                    f"not reach the fitted sill within the computed lag range:\n"
                    + "\n".join(f"  - {d}" for d in zonal_dirs)
                    + "\n\nZonal anisotropy means the variogram sill differs by "
                    "direction. Unlike geometric anisotropy (where only the "
                    "range changes), zonal anisotropy indicates that the total "
                    "variance of the regionalized variable depends on direction. "
                    "This can occur when a long-range trend or drift exists in "
                    "some directions but not others. Consider: (1) increasing "
                    "the lag range, (2) using a nested model with a long-range "
                    "structure, or (3) applying a trend removal before variogram "
                    "modelling."
                )

        # Display
        if warnings:
            self._warnings_text.setPlainText(
                "\n\n".join(f"WARNING: {w}" for w in warnings)
            )
            self._info_tabs.setTabText(3, f"Warnings ({len(warnings)})")
        else:
            self._warnings_text.setPlainText("No warnings -- all checks passed.")
            self._info_tabs.setTabText(3, "Warnings")

    # ══════════════════════════════════════════════════════════════
    # BUILD COMBINED MODEL
    # ══════════════════════════════════════════════════════════════

    def _build_combined_model(self, results: Dict, variable: str) -> Dict:
        """Build combined 3D model from fitted_models dict.

        Results structure: results['fitted_models'][direction][model_type]
        = {'nugget': ..., 'sill': ..., 'range': ..., 'total_sill': ...}

        Applies the same reliability guards as the v2 bridge so a noisy
        directional fit can't be silently re-labelled as "major" by the
        sort-by-value step. A directional fit is rejected when its total
        sill exceeds 1.3x the sample variance OR its range exceeds the
        major range by more than 15% (geometrically impossible for the
        minor axis of an anisotropy ellipsoid).
        """
        fitted = results.get('fitted_models', {})
        model_type = self.model_combo.currentText().lower()

        def _get_params(direction):
            return fitted.get(direction, {}).get(model_type, {})

        omni_p = _get_params('omni')
        major_p = _get_params('major')
        minor_p = _get_params('minor')
        vert_p = _get_params('vertical')
        down_p = _get_params('downhole')

        sample_var = results.get('metadata', {}).get('sample_variance',
                     results.get('sample_variance', 1.0))

        base_sill = omni_p.get('sill', sample_var * 0.9)
        base_nugget = omni_p.get('nugget', 0.0)
        base_range = omni_p.get('range', 100.0)

        # Nugget: prefer omni first-lag gamma as proxy
        nugget = base_nugget
        omni_df = results.get('omni_variogram')
        if isinstance(omni_df, pd.DataFrame) and 'gamma' in omni_df.columns and len(omni_df) > 0:
            proxy = float(omni_df['gamma'].iloc[0])
            if proxy > 0:
                nugget = proxy

        # Sill cap
        total_sill = base_sill + nugget
        if total_sill > sample_var * 1.3:
            total_sill = sample_var

        # Ranges from each direction (raw, before reliability guard)
        major_range = major_p.get('range', base_range)
        minor_range = minor_p.get('range', base_range)
        vert_range = vert_p.get('range', down_p.get('range', base_range))

        # ── Reliability guard ────────────────────────────────────────
        # Mirror the v2 bridge's `_minor_fit_unreliable` check so a
        # noisy minor fit can't slip past sort-by-value relabelling.
        def _is_reliable(params: Dict, label: str) -> bool:
            if not params:
                return False
            tsill = params.get('total_sill')
            if tsill is None:
                tsill = params.get('sill', 0.0) + params.get('nugget', 0.0)
            rng = params.get('range')
            if rng is None or not (rng > 0):
                return False
            if sample_var > 0 and tsill > 1.3 * sample_var:
                logger.warning(
                    "Rejecting %s directional fit: total_sill %.3f > "
                    "1.3x sample variance %.3f", label, tsill, sample_var,
                )
                return False
            return True

        major_ok = _is_reliable(major_p, 'major')
        minor_ok = _is_reliable(minor_p, 'minor')
        vert_ok = _is_reliable(vert_p, 'vertical')

        # When the major direction is rejected, fall back to omni so
        # we still have a defensible reference range.
        if not major_ok:
            omni_range_fallback = omni_p.get('range') or base_range
            if omni_range_fallback and omni_range_fallback > 0:
                major_range = float(omni_range_fallback)

        # When the minor direction is rejected, fall back to
        # min(major, omni) and KEEP the azimuth anchored to major.
        # When the minor is accepted but its range exceeds major by >15%,
        # clamp it (geometric constraint: minor cannot exceed major).
        if not minor_ok:
            omni_r = omni_p.get('range') or major_range
            minor_range = min(float(major_range), float(omni_r))
            logger.warning(
                "Minor directional fit rejected; falling back to "
                "min(major, omni) = %.1fm", minor_range,
            )
        elif minor_range > major_range * 1.15:
            logger.warning(
                "Minor range %.1fm exceeds major %.1fm by more than 15 "
                "percent - clamping to major (geometric constraint).",
                minor_range, major_range,
            )
            minor_range = major_range
        # Final defensive clamp
        if minor_range > major_range:
            minor_range = major_range

        # Vertical: when rejected, fall back to half-major.
        if not vert_ok:
            vert_range = max(major_range * 0.5, 1e-6)

        # Sanity ordering: enforce vertical ≤ minor ≤ major. We do NOT
        # sort by value here — the directional labels carry geological
        # meaning that sort-by-value would silently destroy.
        if vert_range > minor_range:
            vert_range = minor_range

        azimuth = results.get('major_azimuth', 0.0)
        dip = results.get('major_dip', 0.0)

        combined = {
            'model_type': model_type,
            'total_sill': total_sill,
            'nugget': nugget,
            'sill': total_sill - nugget,
            'major_range': float(major_range),
            'minor_range': float(minor_range),
            'vertical_range': float(vert_range),
            'azimuth': azimuth,
            'dip': dip,
            'nugget_ratio': nugget / max(total_sill, 1e-12),
            'variable': variable,
        }
        logger.info(
            "Combined model: sill=%.2f, nugget=%.2f, ranges=(%.1f, %.1f, %.1f), az=%.1f°",
            combined['sill'], nugget, major_range, minor_range, vert_range, azimuth,
        )
        return combined

    # ══════════════════════════════════════════════════════════════
    # AUTO-PUBLISH TO REGISTRY
    # ══════════════════════════════════════════════════════════════

    def _publish_to_registry(self, results: Dict, variable: str):
        """Automatically publish to registry — all estimation methods pick this up."""
        registry = getattr(self, 'registry', None) or self.get_registry()
        if registry is None:
            return
        try:
            payload = results.copy()
            payload['variable'] = variable

            # Use the dedicated variogram registration method which:
            #  - stores per-variable (variogram_results_{var})
            #  - tracks _latest_variogram_variable
            #  - maintains _variogram_variables index
            #  - emits variogramResultsLoaded signal
            if hasattr(registry, 'register_variogram_results'):
                registry.register_variogram_results(
                    payload, source_panel="Variogram",
                )
            else:
                # Fallback for older registry versions
                registry.register_results(
                    "variogram_results", payload, source_panel="Variogram",
                )

            self._data_status.setText(f"✓ Published ({variable})")
            logger.info("Variogram auto-published: variogram_results_%s", variable)
        except Exception as exc:
            logger.warning("Auto-publish failed: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # PDF EXPORT
    # ══════════════════════════════════════════════════════════════

    def _export_pdf_report(self):
        """Export a multi-page PDF report with all variogram plots and parameters."""
        if not self.variogram_results:
            QMessageBox.warning(
                self, "No Results",
                "No variogram results available.\nRun the analysis first."
            )
            return

        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "variogram_report.pdf",
            "PDF Files (*.pdf)"
        )
        if not path:
            return

        try:
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.figure import Figure as MplFigure
            import matplotlib.pyplot as plt

            results = self.variogram_results
            combined = results.get('combined_3d_model', {})
            meta = results.get('metadata', {})
            fits = results.get('fitted_models', {})
            m_type = self.model_combo.currentText().lower()

            with PdfPages(path) as pdf:
                # ── Page 1: Directional variograms (2x2 grid) ──────────
                fig1 = MplFigure(figsize=(11, 8.5), dpi=150)
                fig1.suptitle('Directional Variograms', fontsize=14, fontweight='bold')

                plot_keys = [
                    ('omni', 'Omnidirectional'),
                    ('major', 'Major Direction'),
                    ('minor', 'Minor Direction'),
                    ('vertical', 'Vertical'),
                ]
                for idx, (key, title) in enumerate(plot_keys):
                    ax = fig1.add_subplot(2, 2, idx + 1)
                    pw = self._plot_widgets.get(key)
                    if pw and pw.fig.axes:
                        src_ax = pw.fig.axes[0]
                        for line in src_ax.get_lines():
                            ax.plot(
                                line.get_xdata(), line.get_ydata(),
                                color=line.get_color(),
                                linewidth=line.get_linewidth(),
                                linestyle=line.get_linestyle(),
                                label=line.get_label() if not line.get_label().startswith('_') else None,
                            )
                        for coll in src_ax.collections:
                            try:
                                offsets = coll.get_offsets()
                                if len(offsets) > 0:
                                    ax.scatter(
                                        offsets[:, 0], offsets[:, 1],
                                        s=15, alpha=0.7, label='Experimental',
                                    )
                            except Exception:
                                pass
                        ax.set_xlim(src_ax.get_xlim())
                        ax.set_ylim(src_ax.get_ylim())
                        ax.set_xlabel(src_ax.get_xlabel())
                        ax.set_ylabel(src_ax.get_ylabel())
                        if src_ax.get_legend():
                            ax.legend(fontsize=7)
                    ax.set_title(title, fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                fig1.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig1)
                plt.close(fig1)

                # ── Page 2: 3D anisotropy ellipsoid ────────────────────
                if self._aniso_fig.axes:
                    pdf.savefig(self._aniso_fig)

                # ── Page 3: Variogram map heatmap ──────────────────────
                if self._vmap_fig.axes:
                    pdf.savefig(self._vmap_fig)

                # ── Page 4: Text summary with all parameters ──────────
                fig_text = MplFigure(figsize=(11, 8.5), dpi=150)
                ax_text = fig_text.add_subplot(111)
                ax_text.axis('off')

                lines = []
                lines.append("VARIOGRAM MODEL SUMMARY")
                lines.append("=" * 50)
                lines.append("")
                var_name = results.get('variable', self.var_combo.currentText())
                lines.append(f"Variable:           {var_name}")
                lines.append(f"Model Type:         {m_type.title()}")
                lines.append(f"Samples:            {meta.get('source_data_n_samples', '?')}")
                lines.append(f"Sample Variance:    {meta.get('sample_variance', 0):.4f}")
                lines.append("")

                if combined:
                    nugget = combined.get('nugget', 0)
                    total_sill = combined.get('total_sill', 0)
                    psill = combined.get('sill', 0)
                    nugget_ratio = combined.get('nugget_ratio', 0)
                    lines.append("COMBINED 3D MODEL")
                    lines.append("-" * 50)
                    lines.append(f"Nugget (C0):        {nugget:.4f}")
                    lines.append(f"Partial Sill (C):   {psill:.4f}")
                    lines.append(f"Total Sill (C0+C):  {total_sill:.4f}")
                    lines.append(f"Nugget Ratio:       {nugget_ratio*100:.1f}%")
                    lines.append("")
                    lines.append(f"Major Range:        {combined.get('major_range', 0):.1f} m")
                    lines.append(f"Minor Range:        {combined.get('minor_range', 0):.1f} m")
                    lines.append(f"Vertical Range:     {combined.get('vertical_range', 0):.1f} m")
                    lines.append(f"Azimuth:            {combined.get('azimuth', 0):.1f} deg")
                    lines.append(f"Dip:                {combined.get('dip', 0):.1f} deg")
                    lines.append("")
                    r_max = combined.get('major_range', 1)
                    r_min_h = combined.get('minor_range', 1)
                    r_vert = combined.get('vertical_range', 1)
                    lines.append(f"Aniso Ratio (H):    {r_max / max(r_min_h, 1e-6):.2f}")
                    lines.append(f"Aniso Ratio (V):    {r_max / max(r_vert, 1e-6):.2f}")

                lines.append("")
                lines.append("DIRECTIONAL FIT PARAMETERS")
                lines.append("-" * 50)
                for direction in ['omni', 'major', 'minor', 'vertical', 'downhole']:
                    params = fits.get(direction, {}).get(m_type, {})
                    if params:
                        lines.append(
                            f"  {direction.capitalize():12s}  "
                            f"C0={params.get('nugget', 0):.3f}  "
                            f"C={params.get('sill', 0):.3f}  "
                            f"a={params.get('range', 0):.1f}m"
                        )

                # Warnings
                warnings_list = []
                try:
                    report = analyze_nugget_consistency(results)
                    if not report.is_consistent:
                        warnings_list.append(f"NUGGET INCONSISTENCY: {report.message}")
                except Exception:
                    pass
                if combined:
                    nugget_ratio_val = combined.get('nugget_ratio', 0)
                    if nugget_ratio_val > 0.40:
                        warnings_list.append(
                            f"HIGH NUGGET: {nugget_ratio_val*100:.0f}% of variation is noise"
                        )
                if warnings_list:
                    lines.append("")
                    lines.append("WARNINGS")
                    lines.append("-" * 50)
                    for w in warnings_list:
                        lines.append(f"  ! {w}")

                text = "\n".join(lines)
                ax_text.text(
                    0.05, 0.95, text,
                    transform=ax_text.transAxes,
                    fontsize=9, fontfamily='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='#ccc'),
                )
                fig_text.tight_layout()
                pdf.savefig(fig_text)
                plt.close(fig_text)

            QMessageBox.information(
                self, "PDF Exported",
                f"Variogram report saved to:\n{path}"
            )
            logger.info("PDF variogram report exported to %s", path)

        except Exception as exc:
            logger.error("PDF export failed: %s", exc, exc_info=True)
            QMessageBox.critical(
                self, "Export Error",
                f"PDF export failed:\n{exc}"
            )

    # ══════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ══════════════════════════════════════════════════════════════

    def get_registry(self):
        if hasattr(self, 'registry') and self.registry is not None:
            return self.registry
        parent = self.parent()
        while parent is not None:
            for attr in ('registry', 'data_registry'):
                if hasattr(parent, attr):
                    self.registry = getattr(parent, attr)
                    return self.registry
            parent = parent.parent()
        try:
            from ..core.data_registry import DataRegistry
            self.registry = DataRegistry.instance()
            return self.registry
        except Exception:
            return None

    def show_progress(self, message: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(message)

    def hide_progress(self) -> None:
        self.progress_bar.setVisible(False)

    def refresh_theme(self) -> None:
        pass

    def clear_results(self):
        self.variogram_results = None
        for pw in self._plot_widgets.values():
            pw.ax.clear()
            pw.canvas.draw_idle()
        self._summary_text.clear()
        self._recommend_text.clear()
        self._warnings_text.clear()

    # ══════════════════════════════════════════════════════════════
    # EXPORT RECOMMENDATION
    # ══════════════════════════════════════════════════════════════

    def _export_recommendation(self):
        """Export the latest recommendation dict as a JSON file."""
        if not self._latest_recommendation:
            QMessageBox.information(self, "No Recommendation",
                                    "No recommendation available. Run 'Recommend & Fit' first.")
            return

        from PyQt6.QtWidgets import QFileDialog
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Recommendation", "variogram_recommendation.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            # Build a JSON-serialisable copy (numpy types are not serialisable)
            def _sanitise(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _sanitise(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitise(v) for v in obj]
                return obj

            data = _sanitise(self._latest_recommendation)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info("Recommendation exported to %s", path)
            QMessageBox.information(self, "Export Complete",
                                    f"Recommendation saved to:\n{path}")
        except Exception as exc:
            logger.error("Recommendation export failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export recommendation:\n{exc}")

    # ══════════════════════════════════════════════════════════════
    # PROJECT SAVE / RESTORE
    # ══════════════════════════════════════════════════════════════

    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Return settings dict for project save."""
        settings: Dict[str, Any] = {}
        try:
            settings['variable'] = self.var_combo.currentText()
            settings['domain'] = self.domain_combo.currentText()
            settings['nlag'] = self.nlag_spin.value()
            settings['lag_dist'] = self.lag_dist_spin.value()
            settings['lag_tol'] = self.lag_tol_spin.value()
            settings['auto_lags'] = self.cb_auto_lags.isChecked()
            settings['model_type'] = self.model_combo.currentText()
            settings['structures'] = self.struct_spin.value()
            settings['manual_azimuth'] = self.cb_manual_azimuth.isChecked()
            settings['major_azimuth'] = self.major_az_spin.value()
            settings['major_dip'] = self.major_dip_spin.value()
            settings['cone_tolerance'] = self.cone_tol_spin.value()
            settings['global_nugget_enabled'] = self.cb_global_nugget.isChecked()
            settings['global_nugget_value'] = self.global_nugget_spin.value()
            settings['bandwidth_enabled'] = self.cb_bandwidth.isChecked()
            settings['bandwidth'] = self.bandwidth_spin.value()
            settings['seed'] = self.seed_spin.value()

            if self._latest_recommendation:
                settings['recommendation'] = self._latest_recommendation

            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            return settings if settings else None
        except Exception as exc:
            logger.warning("Could not save variogram panel settings: %s", exc)
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Restore settings from project load."""
        if not settings:
            return
        try:
            if 'variable' in settings:
                idx = self.var_combo.findText(settings['variable'])
                if idx >= 0:
                    self.var_combo.setCurrentIndex(idx)
            if 'domain' in settings:
                idx = self.domain_combo.findText(settings['domain'])
                if idx >= 0:
                    self.domain_combo.setCurrentIndex(idx)
            if 'nlag' in settings:
                self.nlag_spin.setValue(int(settings['nlag']))
            if 'lag_dist' in settings:
                self.lag_dist_spin.setValue(float(settings['lag_dist']))
            if 'lag_tol' in settings:
                self.lag_tol_spin.setValue(float(settings['lag_tol']))
            if 'auto_lags' in settings:
                self.cb_auto_lags.setChecked(bool(settings['auto_lags']))
            if 'model_type' in settings:
                idx = self.model_combo.findText(str(settings['model_type']).title())
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            if 'structures' in settings:
                self.struct_spin.setValue(int(settings['structures']))
            if 'manual_azimuth' in settings:
                self.cb_manual_azimuth.setChecked(bool(settings['manual_azimuth']))
            if 'major_azimuth' in settings:
                self.major_az_spin.setValue(float(settings['major_azimuth']))
            if 'major_dip' in settings:
                self.major_dip_spin.setValue(float(settings['major_dip']))
            if 'cone_tolerance' in settings:
                self.cone_tol_spin.setValue(float(settings['cone_tolerance']))
            if 'global_nugget_enabled' in settings:
                self.cb_global_nugget.setChecked(bool(settings['global_nugget_enabled']))
            if 'global_nugget_value' in settings:
                self.global_nugget_spin.setValue(float(settings['global_nugget_value']))
            if 'bandwidth_enabled' in settings:
                self.cb_bandwidth.setChecked(bool(settings['bandwidth_enabled']))
            if 'bandwidth' in settings:
                self.bandwidth_spin.setValue(float(settings['bandwidth']))
            if 'seed' in settings:
                self.seed_spin.setValue(int(settings['seed']))

            rec = settings.get('recommendation')
            if rec:
                self._latest_recommendation = rec
                self._apply_recommendation_to_ui(rec)
                self._update_recommendations(rec)
                if hasattr(self, 'export_rec_btn'):
                    self.export_rec_btn.setEnabled(True)

            logger.info("Restored variogram analysis panel settings from project")
        except Exception as exc:
            logger.warning("Could not restore variogram panel settings: %s", exc)
