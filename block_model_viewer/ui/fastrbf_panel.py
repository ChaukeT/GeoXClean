"""
FastRBF Interpolation Estimator Panel  (v2)
============================================

Dedicated panel for JORC-compliant geostatistical RBF interpolation using
the standalone geostats.estimation FastRBF engine.

v2 consolidates 7 tabs → 4 with a left-to-right workflow:

  Tab 0 — Data & Model     : data source (radio), variable, kernel config,
                              variogram import + live curve canvas
  Tab 1 — Search & Grid    : drift, anisotropy (toggle), search neighbourhood,
                              grid mode (Auto/BM/Manual), block sizes
  Tab 2 — Run & Results    : run/stop, progress, output config, JORC table
  Tab 3 — Validation       : pre-flight checks, LOO-CV, metric cards, bias

Key changes from v1:
  - Merged Interpolant + Variogram into Data & Model (variogram tab was redundant)
  - Merged Trend + Query + Outputs grid → Search & Grid
  - Moved Run button into a Run & Results tab (no pinned run-bar)
  - Anisotropy controls hidden until toggled on
  - LOO-CV can run independently via dedicated button
  - JORC classification in Run & Results tab
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QTextEdit, QCheckBox,
    QWidget, QFrame, QTabWidget, QFormLayout, QLineEdit,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QSizePolicy, QScrollArea,
    QApplication, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSignalBlocker
from PyQt6.QtGui import QFont

# Matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    FigureCanvas = None
    Figure = None
    MATPLOTLIB_AVAILABLE = False

from .base_analysis_panel import BaseAnalysisPanel
from .design_tokens import tokens
from .collapsible_group import CollapsibleGroup
from .panel_toolkit import (
    section, make_form, form_row, make_combo, make_spin, make_int_spin,
    action_button, hint_label, info_display, separator,
    PANEL_MARGINS, PANEL_SPACING,
)
from .modern_styles import ModernColors, get_theme_colors
from ..utils.variable_utils import get_grade_columns, populate_variable_combo
from .panel_utils import resolve_variogram_for_variable
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)

try:
    from ..geostats.fastrbf_bridge import FASTRBF_AVAILABLE
except ImportError:
    FASTRBF_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# Reusable Matplotlib Canvas (same pattern as VariogramCanvas)
# ═══════════════════════════════════════════════════════════════════

class _FastRBFCanvas(FigureCanvas):
    """Dark-themed matplotlib canvas with safe draw and theme helpers."""

    def __init__(self, parent=None, width=6, height=4, dpi=100):
        colors = get_theme_colors()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(colors.CARD_BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 220)
        self.updateGeometry()

    # ── theme ──

    def _apply_theme(self, ax, is_3d=False):
        colors = get_theme_colors()
        ax.set_facecolor(colors.CARD_BG)
        ax.tick_params(colors=colors.TEXT_PRIMARY, labelsize=8)
        ax.xaxis.label.set_color(colors.TEXT_PRIMARY)
        ax.yaxis.label.set_color(colors.TEXT_PRIMARY)
        if is_3d and hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_color(colors.TEXT_PRIMARY)
            ax.zaxis.set_tick_params(colors=colors.TEXT_PRIMARY, labelsize=7)
        ax.title.set_color(colors.TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_color(colors.BORDER)
        ax.grid(True, color=colors.BORDER, linestyle="--", alpha=0.4)

    # ── safe draw ──

    def _safe_draw(self):
        try:
            self.draw_idle()
            self.flush_events()
        except RuntimeError as e:
            if "QAction" in str(e) or "deleted" in str(e):
                try:
                    self.update()
                    self.repaint()
                except Exception:
                    pass
            else:
                raise
        except Exception:
            try:
                self.update()
                self.repaint()
            except Exception:
                pass

    # ── variogram model curve ──

    def plot_variogram_model(self, params: Dict[str, Any]):
        """Plot fitted variogram model curve with optional directional curves."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._apply_theme(ax)
        colors = get_theme_colors()

        if not params:
            ax.text(0.5, 0.5, "No variogram model loaded\n\nImport from registry",
                    ha="center", va="center", color=colors.TEXT_HINT,
                    fontsize=10, transform=ax.transAxes)
            self._safe_draw()
            return

        nugget = params.get("nugget", 0.0)
        psill = params.get("sill", params.get("partial_sill", 1.0))
        total_sill = params.get("total_sill", nugget + psill)
        rng = params.get("major_range", params.get("range", 100.0))
        model_type = params.get("model_type", "spherical")

        max_h = rng * 1.5
        h = np.linspace(0, max_h, 500)

        gamma = self._eval_variogram(h, model_type, nugget, psill, rng)

        # Main model curve
        ax.plot(h, gamma, color="#ffb74d", linewidth=2.5,
                label=f"{model_type.title()} (Major: {rng:.0f})", zorder=3)

        # Directional curves (if anisotropic)
        semi_range = params.get("semi_range")
        minor_range = params.get("minor_range")
        if semi_range and semi_range != rng:
            g_semi = self._eval_variogram(h, model_type, nugget, psill, semi_range)
            ax.plot(h, g_semi, color="#2979FF", linewidth=1.8, linestyle="--",
                    label=f"Semi ({semi_range:.0f})", zorder=2)
        if minor_range and minor_range != rng:
            g_minor = self._eval_variogram(h, model_type, nugget, psill, minor_range)
            ax.plot(h, g_minor, color="#FF6D00", linewidth=1.8, linestyle=":",
                    label=f"Minor ({minor_range:.0f})", zorder=2)

        # Reference lines
        ax.axhline(y=nugget, color="#9e9e9e", linestyle=":", alpha=0.7,
                    label=f"Nugget = {nugget:.3f}")
        ax.axhline(y=total_sill, color="#ef5350", linestyle="--", alpha=0.7,
                    label=f"Sill = {total_sill:.3f}")
        ax.axvline(x=rng, color="#66bb6a", linestyle="--", alpha=0.7,
                    label=f"Range = {rng:.1f}")

        ax.fill_between(h, nugget, gamma, alpha=0.08, color="#ffb74d")

        ax.set_xlabel("Distance (h)", fontsize=9)
        ax.set_ylabel("Semivariance \u03b3(h)", fontsize=9)
        ax.set_title("Fitted Variogram Model", fontsize=11, fontweight="bold", pad=8)
        ax.legend(loc="lower right", facecolor=colors.CARD_BG,
                  edgecolor=colors.BORDER, labelcolor=colors.TEXT_PRIMARY, fontsize=7)
        ax.set_xlim(0, max_h)
        ax.set_ylim(0, total_sill * 1.25)

        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self._safe_draw()

    @staticmethod
    def _eval_variogram(h, model_type, nugget, psill, rng):
        """Evaluate standard variogram models."""
        h = np.asarray(h, dtype=float)
        gamma = np.full_like(h, nugget + psill)
        mt = model_type.lower()

        if mt in ("spherical", "spheroidal"):
            mask = h < rng
            hr = h[mask] / rng
            gamma[mask] = nugget + psill * (1.5 * hr - 0.5 * hr ** 3)
            gamma[h == 0] = 0.0
        elif mt == "exponential":
            gamma = nugget + psill * (1.0 - np.exp(-3.0 * h / rng))
            gamma[h == 0] = 0.0
        elif mt == "gaussian":
            gamma = nugget + psill * (1.0 - np.exp(-3.0 * (h / rng) ** 2))
            gamma[h == 0] = 0.0
        elif mt == "linear":
            mask = h < rng
            gamma[mask] = nugget + psill * (h[mask] / rng)
            gamma[h == 0] = 0.0
        elif mt == "cubic":
            mask = h < rng
            hr = h[mask] / rng
            gamma[mask] = nugget + psill * (7 * hr ** 2 - 8.75 * hr ** 3
                                            + 3.5 * hr ** 5 - 0.75 * hr ** 7)
            gamma[h == 0] = 0.0
        else:
            mask = h < rng
            hr = h[mask] / rng
            gamma[mask] = nugget + psill * (1.5 * hr - 0.5 * hr ** 3)
            gamma[h == 0] = 0.0
        return gamma


# ═══════════════════════════════════════════════════════════════════
# FastRBF Panel  (v2 — 4 tabs)
# ═══════════════════════════════════════════════════════════════════

from .mixins.domain_mask_mixin import DomainMaskMixin
from .mixins.coded_domain_filter_mixin import CodedDomainFilterMixin


class FastRBFPanel(CodedDomainFilterMixin, DomainMaskMixin, BaseAnalysisPanel):
    """
    Professional FastRBF Interpolation Estimator panel.

    Tabs: Data & Model | Search & Grid | Run & Results | Validation
    """
    PANEL_ID = "FastRBFPanel"
    PANEL_NAME = "FastRBF Interpolation"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    task_name = "fastrbf"
    request_visualization = pyqtSignal(dict)
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None):
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.fastrbf_results: Optional[Dict[str, Any]] = None
        self.registry = None
        self._pending_drillhole_data = None
        self._pending_variogram_results = None
        self._ui_ready = False
        self._registry_data: Optional[Dict] = None
        self.block_grid_spec: Optional[Dict[str, Any]] = None
        self.main_window = None
        super().__init__(parent=parent, panel_id="fastrbf")

    def refresh_theme(self):
        pass

    # ──────────────────────────────────────────────────────────────
    # OVERRIDE _setup_base_ui: Header + 4 Tabs (no pinned run bar)
    # ──────────────────────────────────────────────────────────────

    def _setup_base_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        root.addWidget(self.tab_widget, stretch=1)

        self._build_data_model_tab()      # Tab 0
        self._build_search_grid_tab()     # Tab 1
        self._build_run_results_tab()     # Tab 2
        self._build_validation_tab()      # Tab 3

        # Finalize
        self._ui_ready = True
        self._init_registry_connections()
        self.progress_updated.connect(self._on_progress)
        self._process_pending_data()
        QTimer.singleShot(0, self._connect_registry_notice)
        self._is_initialized = True

    # ──────────────────────────────────────────────────────────────
    # HEADER (compact)
    # ──────────────────────────────────────────────────────────────

    def _build_header(self) -> QFrame:
        f = QFrame()
        f.setObjectName("Card")
        f.setStyleSheet(f"""
            QFrame#Card {{
                background-color: {ModernColors.ELEVATED_BG};
                border-bottom: 1px solid {ModernColors.DIVIDER};
            }}
        """)
        lay = QHBoxLayout(f)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)

        title = QLabel("FastRBF Interpolation Estimator")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        lay.addWidget(title)

        lay.addStretch()

        sub = QLabel("Leapfrog Geo-Style RBF  |  3D Implicit  |  JORC 2012")
        sub.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        lay.addWidget(sub)
        return f

    # ══════════════════════════════════════════════════════════════
    # TAB 0 — DATA & MODEL
    # ══════════════════════════════════════════════════════════════

    def _build_data_model_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── Data Source ──
        src_grp = section("Data Source")
        src_lay = QVBoxLayout()

        src_header = QHBoxLayout()
        src_title = QLabel("DATA SOURCE")
        src_title.setStyleSheet(
            f"color: {ModernColors.TEXT_HINT}; font-weight: bold; font-size: 9pt;"
        )
        self.data_refresh_btn = QPushButton("Refresh")
        self.data_refresh_btn.setMinimumHeight(30)
        self.data_refresh_btn.setToolTip("Reload data from registry")
        self.data_refresh_btn.clicked.connect(self._manual_refresh)
        src_header.addWidget(src_title)
        src_header.addStretch()
        src_header.addWidget(self.data_refresh_btn)
        src_lay.addLayout(src_header)

        self.data_source_btn_group = QButtonGroup()
        self.data_source_composited = QRadioButton("Composited Data (recommended)")
        self.data_source_composited.setToolTip("Use composited drillhole data")
        self.data_source_declustered = QRadioButton("Declustered Data")
        self.data_source_declustered.setToolTip("Use declustered data with weights")
        self.data_source_raw = QRadioButton("Raw Assay Data")
        self.data_source_raw.setToolTip("Use raw drillhole assay data")
        self.data_source_btn_group.addButton(self.data_source_composited, 0)
        self.data_source_btn_group.addButton(self.data_source_declustered, 1)
        self.data_source_btn_group.addButton(self.data_source_raw, 2)
        self.data_source_composited.setChecked(True)
        self.data_source_btn_group.buttonClicked.connect(
            self._on_data_source_radio_changed
        )
        src_lay.addWidget(self.data_source_composited)
        src_lay.addWidget(self.data_source_declustered)
        src_lay.addWidget(self.data_source_raw)

        self.data_source_status = QLabel("No data loaded")
        self.data_source_status.setWordWrap(True)
        self.data_source_status.setStyleSheet(
            f"font-size: 9pt; color: {ModernColors.TEXT_HINT}; margin-top: 6px;"
        )
        src_lay.addWidget(self.data_source_status)
        src_grp.add_layout(src_lay)
        lay.addWidget(src_grp)

        # ── Variable Selection ──
        var_grp = section("Variable Selection")
        form = make_form()
        self.variable_combo = make_combo()
        self.variable_combo.currentTextChanged.connect(self._on_variable_changed)
        form_row(form, "Variable:", self.variable_combo)
        self.domain_combo = make_combo()
        self.domain_combo.addItem("All Data")
        self.domain_combo.setToolTip("Filter composites by domain")
        self.domain_combo.currentTextChanged.connect(self._on_domain_filter_selection_changed)
        self.domain_combo.currentTextChanged.connect(
            lambda _: self._reload_variogram_for_domain()
        )
        form_row(form, "Domain:", self.domain_combo)
        var_grp.add_layout(form)
        lay.addWidget(var_grp)

        # ── Data Preview ──
        prev = section("Data Preview")
        self.data_preview_label = info_display("No data loaded")
        pl = QVBoxLayout()
        pl.addWidget(self.data_preview_label)
        prev.add_layout(pl)
        lay.addWidget(prev)

        # ── Interpolant Configuration ──
        cfg = section("Interpolant Configuration")
        cfg_form = make_form()
        self.kernel_combo = make_combo([
            "Spheroidal (Leapfrog default)", "Spherical", "Gaussian",
            "Exponential", "Linear", "Cubic",
        ])
        form_row(cfg_form, "Kernel:", self.kernel_combo, "RBF kernel function")

        self.alpha_combo = make_combo(["3", "5", "7", "9"])
        self.alpha_combo.setCurrentIndex(1)
        form_row(cfg_form, "Alpha:", self.alpha_combo, "Shape exponent")

        self.sill_spin = make_spin(0.001, 1e6, 1.0, 4, tooltip="Total sill")
        form_row(cfg_form, "Total Sill:", self.sill_spin)

        self.nugget_spin = make_spin(0.0, 1e6, 0.0, 4, tooltip="Nugget")
        form_row(cfg_form, "Nugget:", self.nugget_spin)

        self.range_spin = make_spin(0.1, 1e6, 100.0, 1, tooltip="Major range (m)")
        form_row(cfg_form, "Range (m):", self.range_spin)
        cfg.add_layout(cfg_form)
        lay.addWidget(cfg)

        # ── Variogram Integration ──
        vi = section("Variogram Integration")
        vl = QVBoxLayout()
        self.import_vario_btn = action_button("Import from Variogram", "primary")
        self.import_vario_btn.clicked.connect(self._load_variogram_from_registry)
        vl.addWidget(self.import_vario_btn)
        self.vario_status_label = hint_label(
            "No variogram loaded \u2014 configure manually or import"
        )
        vl.addWidget(self.vario_status_label)
        vi.add_layout(vl)
        lay.addWidget(vi)

        # ── Live Variogram Model Curve ──
        if MATPLOTLIB_AVAILABLE:
            self.vario_curve_canvas = _FastRBFCanvas(parent=container, width=5, height=3)
            lay.addWidget(self.vario_curve_canvas)
        else:
            self.vario_curve_canvas = None
            lay.addWidget(QLabel("matplotlib not available"))

        # Wire spin changes to update model curve
        for w in (self.sill_spin, self.nugget_spin, self.range_spin):
            w.valueChanged.connect(self._update_interpolant_curve)
        self.kernel_combo.currentTextChanged.connect(self._update_interpolant_curve)

        QTimer.singleShot(200, self._update_interpolant_curve)

        lay.addStretch()
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Data && Model")

    def _update_interpolant_curve(self):
        if not self.vario_curve_canvas:
            return
        kernel_text = self.kernel_combo.currentText().split("(")[0].strip().lower()
        params = {
            "model_type": kernel_text,
            "nugget": self.nugget_spin.value(),
            "sill": max(0, self.sill_spin.value() - self.nugget_spin.value()),
            "total_sill": self.sill_spin.value(),
            "major_range": self.range_spin.value(),
        }
        # Add directional curves if anisotropy is enabled
        if hasattr(self, "anisotropy_enabled") and self.anisotropy_enabled.isChecked():
            params["semi_range"] = self.range_semi_spin.value()
            params["minor_range"] = self.range_minor_spin.value()
        self.vario_curve_canvas.plot_variogram_model(params)

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — SEARCH & GRID
    # ══════════════════════════════════════════════════════════════

    def _build_search_grid_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── Drift ──
        dg = section("Drift (Trend Removal)")
        df = make_form()
        self.drift_combo = make_combo(["Constant (default)", "Linear", "None"])
        form_row(df, "Drift:", self.drift_combo)
        dg.add_layout(df)
        lay.addWidget(dg)

        # ── Search Neighbourhood ──
        sg = section("Search Neighbourhood")
        sf = make_form()
        self.max_samples_spin = make_int_spin(1, 200, 24, tooltip="Max samples")
        form_row(sf, "Max Samples:", self.max_samples_spin)
        self.min_samples_spin = make_int_spin(1, 100, 8, tooltip="Min samples")
        form_row(sf, "Min Samples:", self.min_samples_spin)
        self.max_per_octant_spin = make_int_spin(0, 50, 4, tooltip="Max per octant")
        form_row(sf, "Max per Octant:", self.max_per_octant_spin)
        self.min_octants_spin = make_int_spin(0, 8, 2, tooltip="Min octants")
        form_row(sf, "Min Octants:", self.min_octants_spin)
        sg.add_layout(sf)
        lay.addWidget(sg)

        # ── Anisotropy ──
        ag = section("Anisotropy")
        al = QVBoxLayout()
        self.anisotropy_enabled = QCheckBox("Enable anisotropic search")
        self.anisotropy_enabled.setChecked(False)
        self.anisotropy_enabled.toggled.connect(self._on_anisotropy_toggled)
        al.addWidget(self.anisotropy_enabled)
        al.addWidget(hint_label(
            "When enabled, the search neighbourhood and kernel are stretched "
            "along the major/semi/minor axes."
        ))

        # Anisotropy controls — hidden until toggled on
        self.aniso_widget = QWidget()
        aniso_lay = QVBoxLayout(self.aniso_widget)
        aniso_lay.setContentsMargins(0, 8, 0, 0)
        aniso_lay.setSpacing(6)

        af = make_form()
        self.range_major_spin = make_spin(0.01, 1e6, 100.0, 1,
                                          tooltip="Major range (m)")
        form_row(af, "Major Range (m):", self.range_major_spin)
        self.range_semi_spin = make_spin(0.01, 1e6, 100.0, 1,
                                         tooltip="Semi-major range (m)")
        form_row(af, "Semi-Major Range (m):", self.range_semi_spin)
        self.range_minor_spin = make_spin(0.01, 1e6, 50.0, 1,
                                          tooltip="Minor range (m)")
        form_row(af, "Minor Range (m):", self.range_minor_spin)
        aniso_lay.addLayout(af)

        orient_label = QLabel("Orientation")
        orient_label.setStyleSheet(
            f"color: {ModernColors.ACCENT_PRIMARY}; font-weight: bold; font-size: 10pt;"
        )
        aniso_lay.addWidget(orient_label)

        of = make_form()
        self.azimuth_spin = make_spin(0.0, 360.0, 0.0, 0, suffix="\u00b0")
        form_row(of, "Azimuth (\u00b0):", self.azimuth_spin)
        self.dip_spin = make_spin(-90.0, 90.0, 0.0, 0, suffix="\u00b0")
        form_row(of, "Dip (\u00b0):", self.dip_spin)
        self.pitch_spin = make_spin(-90.0, 90.0, 0.0, 0, suffix="\u00b0")
        form_row(of, "Pitch (\u00b0):", self.pitch_spin)
        aniso_lay.addLayout(of)

        self.aniso_widget.setVisible(False)
        al.addWidget(self.aniso_widget)
        ag.add_layout(al)
        lay.addWidget(ag)

        # Wire anisotropy spins to refresh variogram curve
        for w in (self.range_major_spin, self.range_semi_spin, self.range_minor_spin):
            w.valueChanged.connect(self._update_interpolant_curve)

        # ── Grid Specification ──
        gg = section("Grid Specification")
        gl = QVBoxLayout()

        gf = make_form()
        self.grid_mode_combo = make_combo([
            "Auto (from data extent)",
            "From Block Model",
            "Manual",
        ])
        self.grid_mode_combo.currentTextChanged.connect(self._on_grid_mode_changed)
        form_row(gf, "Grid Mode:", self.grid_mode_combo,
                 "Auto: detects from drillhole/renderer bounds. "
                 "Block Model: aligns to existing BM grid.")
        gl.addLayout(gf)

        # Auto-detect + auto-fit
        auto_row = QHBoxLayout()
        self.auto_detect_btn = action_button("Auto-Detect Grid", "primary")
        self.auto_detect_btn.clicked.connect(self._auto_detect_grid)
        auto_row.addWidget(self.auto_detect_btn)
        self.auto_fit_check = QCheckBox("Auto-fit on data load")
        self.auto_fit_check.setChecked(True)
        self.auto_fit_check.setToolTip(
            "Automatically compute grid from data extent when new drillhole data loads"
        )
        auto_row.addWidget(self.auto_fit_check)
        auto_row.addStretch()
        gl.addLayout(auto_row)

        self.grid_info_label = info_display(
            "Grid will be computed from drillhole/block model extent"
        )
        gl.addWidget(self.grid_info_label)

        # Block size controls
        bsf = make_form()
        self.dx_spin = make_spin(0.1, 10000.0, 10.0, 1, tooltip="Block size X (m)")
        self.dy_spin = make_spin(0.1, 10000.0, 10.0, 1, tooltip="Block size Y (m)")
        self.dz_spin = make_spin(0.1, 10000.0, 5.0, 1, tooltip="Block size Z (m)")
        self.dx_spin.valueChanged.connect(self._on_block_size_changed)
        self.dy_spin.valueChanged.connect(self._on_block_size_changed)
        self.dz_spin.valueChanged.connect(self._on_block_size_changed)
        form_row(bsf, "Block Size X (m):", self.dx_spin)
        form_row(bsf, "Block Size Y (m):", self.dy_spin)
        form_row(bsf, "Block Size Z (m):", self.dz_spin)
        gl.addLayout(bsf)

        # Full grid controls (editable in Manual, read-only in Auto)
        self.grid_params_widget = QWidget()
        gpf = QFormLayout(self.grid_params_widget)
        gpf.setContentsMargins(0, 8, 0, 0)

        origin_label = QLabel("Grid Origin (corner of first block):")
        origin_label.setStyleSheet(
            f"color: {ModernColors.TEXT_SECONDARY}; font-size: 9pt;"
        )
        gpf.addRow(origin_label)
        self.xmin_spin = make_spin(-1e9, 1e9, 0.0, 1, tooltip="X origin")
        self.ymin_spin = make_spin(-1e9, 1e9, 0.0, 1, tooltip="Y origin")
        self.zmin_spin = make_spin(-1e9, 1e9, 0.0, 1, tooltip="Z origin")
        gpf.addRow("X Origin:", self.xmin_spin)
        gpf.addRow("Y Origin:", self.ymin_spin)
        gpf.addRow("Z Origin:", self.zmin_spin)

        blocks_label = QLabel("Number of Blocks:")
        blocks_label.setStyleSheet(
            f"color: {ModernColors.TEXT_SECONDARY}; font-size: 9pt;"
        )
        gpf.addRow(blocks_label)
        self.nx_spin = make_int_spin(1, 1000, 50)
        self.ny_spin = make_int_spin(1, 1000, 50)
        self.nz_spin = make_int_spin(1, 500, 20)
        gpf.addRow("NX:", self.nx_spin)
        gpf.addRow("NY:", self.ny_spin)
        gpf.addRow("NZ:", self.nz_spin)

        self._set_grid_controls_editable(False)
        gl.addWidget(self.grid_params_widget)
        gg.add_layout(gl)
        lay.addWidget(gg)

        lay.addWidget(self._build_domain_mask_group(default_enabled=False))
        self.mask_checkbox.setToolTip(
            "FastRBF produces smooth interpolated fields where extrapolation may be intentional "
            "(e.g. geophysics, DTM). Enable this to mask blocks outside the data footprint."
        )

        lay.addStretch()
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Search && Grid")

    def _on_anisotropy_toggled(self, checked: bool):
        self.aniso_widget.setVisible(checked)
        self._update_interpolant_curve()

    def _set_grid_controls_editable(self, editable: bool):
        for w in (self.xmin_spin, self.ymin_spin, self.zmin_spin,
                  self.nx_spin, self.ny_spin, self.nz_spin):
            w.setReadOnly(not editable)
            w.setButtonSymbols(
                QSpinBox.ButtonSymbols.UpDownArrows if editable
                else QSpinBox.ButtonSymbols.NoButtons
            )

    def _on_grid_mode_changed(self, text):
        is_manual = (text == "Manual")
        self._set_grid_controls_editable(is_manual)
        if text == "From Block Model":
            self._apply_block_model_grid()
        elif not is_manual:
            self._auto_detect_grid()

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — RUN & RESULTS
    # ══════════════════════════════════════════════════════════════

    def _build_run_results_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── Run Controls ──
        run_grp = section("Run Interpolation")
        rl = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run FastRBF Interpolation")
        self.run_btn.setObjectName("btn_primary")
        self.run_btn.setMinimumHeight(42)
        self.run_btn.setMinimumWidth(260)
        self.run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("btn_danger")
        self.stop_btn.setMinimumHeight(42)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        rl.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setVisible(False)
        rl.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(
            f"color: {ModernColors.TEXT_HINT}; font-size: 11px;"
        )
        rl.addWidget(self.progress_label)
        run_grp.add_layout(rl)
        lay.addWidget(run_grp)

        # ── Output Configuration ──
        og = section("Output Configuration")
        of = make_form()
        self.output_name_edit = QLineEdit("FastRBF_")
        form_row(of, "Property Name:", self.output_name_edit)
        self.discretisation_spin = make_int_spin(1, 16, 4, tooltip="Points per dim")
        form_row(of, "Discretisation:", self.discretisation_spin)
        self.clip_enabled = QCheckBox("Enable value clipping")
        of.addRow("Clipping:", self.clip_enabled)
        self.clip_min_spin = make_spin(-1e6, 1e6, 0.0, 4)
        self.clip_max_spin = make_spin(-1e6, 1e6, 100.0, 4)
        self.clip_min_spin.setEnabled(False)
        self.clip_max_spin.setEnabled(False)
        form_row(of, "Clip Min:", self.clip_min_spin)
        form_row(of, "Clip Max:", self.clip_max_spin)
        self.clip_enabled.toggled.connect(self.clip_min_spin.setEnabled)
        self.clip_enabled.toggled.connect(self.clip_max_spin.setEnabled)
        og.add_layout(of)
        lay.addWidget(og)

        # ── JORC Classification ──
        jg = section("JORC Classification")
        self.class_table = QTableWidget(4, 3)
        self.class_table.setHorizontalHeaderLabels(["Category", "Blocks", "%"])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setMaximumHeight(160)
        for i, cat in enumerate(["Measured", "Indicated", "Inferred", "Unclassified"]):
            self.class_table.setItem(i, 0, QTableWidgetItem(cat))
            self.class_table.setItem(i, 1, QTableWidgetItem("--"))
            self.class_table.setItem(i, 2, QTableWidgetItem("--"))
        jl = QVBoxLayout()
        jl.addWidget(self.class_table)
        jg.add_layout(jl)
        lay.addWidget(jg)

        lay.addStretch()
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Run && Results")

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — VALIDATION
    # ══════════════════════════════════════════════════════════════

    def _build_validation_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # CV / check buttons
        bar = QHBoxLayout()
        self.cv_run_btn = action_button(
            "Leave-One-Out Cross-Validation", "primary"
        )
        self.cv_run_btn.clicked.connect(self._run_cv)
        bar.addWidget(self.cv_run_btn)
        self.recheck_btn = action_button("Re-run Checks", "secondary")
        self.recheck_btn.clicked.connect(self._run_data_checks)
        bar.addWidget(self.recheck_btn)
        bar.addStretch()
        lay.addLayout(bar)

        # Pre-flight checks
        cg = section("Data & Parameter Checks")
        self.checks_layout = QVBoxLayout()
        self._check_labels: Dict[str, QLabel] = {}
        for cid, txt in [
            ("data_count", "Data count"),
            ("duplicates", "No duplicate locations"),
            ("value_range", "Value range"),
            ("cv_of_var", "Coefficient of variation"),
            ("nugget_sill", "Nugget/Sill ratio"),
            ("range_extent", "Range proportional to data extent"),
        ]:
            lbl = QLabel(f"\u2022 {txt}: pending")
            lbl.setStyleSheet(f"""
                QLabel {{
                    background-color: {ModernColors.ELEVATED_BG};
                    padding: 8px 12px;
                    border-radius: 4px;
                    color: {ModernColors.TEXT_SECONDARY};
                    font-family: Consolas, monospace;
                    font-size: 11px;
                }}
            """)
            self._check_labels[cid] = lbl
            self.checks_layout.addWidget(lbl)
        cg.add_layout(self.checks_layout)
        lay.addWidget(cg)

        # CV metrics (grid)
        cv = section("Cross-Validation Metrics", collapsed=True)
        cvg = QGridLayout()
        cvg.setSpacing(8)
        self.cv_rmse_label = self._metric_card("RMSE", cvg, 0, 0)
        self.cv_mae_label = self._metric_card("MAE", cvg, 0, 1)
        self.cv_r2_label = self._metric_card("R\u00b2", cvg, 0, 2)
        self.cv_me_label = self._metric_card("Mean Error", cvg, 1, 0)
        self.cv_corr_label = self._metric_card("Correlation", cvg, 1, 1)
        self.cv_slope_label = self._metric_card("Slope of Regression", cvg, 1, 2)
        cv.add_layout(cvg)
        lay.addWidget(cv)

        # Bias
        bg = section("Bias Assessment", collapsed=True)
        bf = make_form()
        self.bias_pct_label = QLabel("--")
        self.bias_pct_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        form_row(bf, "Global Bias:", self.bias_pct_label)
        self.bias_flag_label = QLabel("--")
        self.bias_flag_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        form_row(bf, "Conditional Bias:", self.bias_flag_label)
        bg.add_layout(bf)
        lay.addWidget(bg)

        lay.addStretch()
        self.tab_widget.addTab(tab, "Validation")

    def _metric_card(self, title: str, grid: QGridLayout, row: int, col: int) -> QLabel:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {ModernColors.ELEVATED_BG};
                border: 1px solid {ModernColors.DIVIDER};
                border-radius: 6px;
            }}
        """)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(
            f"color: {ModernColors.TEXT_HINT}; font-size: 10px; border: none;"
        )
        cl.addWidget(t)
        v = QLabel("--")
        v.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        v.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; border: none;")
        cl.addWidget(v)
        grid.addWidget(card, row, col)
        return v

    # ══════════════════════════════════════════════════════════════
    # REGISTRY CONNECTIONS
    # ══════════════════════════════════════════════════════════════

    def _init_registry_connections(self):
        try:
            if hasattr(self, "controller") and self.controller:
                self.registry = getattr(self.controller, "registry", None)
            if not self.registry:
                self.registry = self.get_registry()
        except Exception:
            pass
        if not self.registry:
            return
        try:
            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
        except Exception:
            pass
        try:
            if hasattr(self.registry, "compositesLoaded"):
                self.registry.compositesLoaded.connect(self._on_composites_refreshed)
        except Exception:
            pass
        try:
            if hasattr(self.registry, "indicatorRBFDomainLoaded"):
                self.registry.indicatorRBFDomainLoaded.connect(self._on_indicator_rbf_domain_loaded)
        except Exception:
            pass
        try:
            self.registry.variogramResultsLoaded.connect(self._on_variogram_loaded)
        except Exception:
            pass
        if hasattr(self.registry, "blockModelGenerated"):
            try:
                self.registry.blockModelGenerated.connect(
                    self._on_block_model_loaded
                )
            except Exception:
                pass
        # Load existing data
        try:
            d = self.registry.get_drillhole_data()
            if d is not None:
                self._on_drillhole_data_loaded(d)
        except Exception:
            pass
        try:
            v = self.registry.get_variogram_results()
            if v:
                self._on_variogram_loaded(v)
        except Exception:
            pass
        try:
            bm = self.registry.get_block_model()
            if bm is not None:
                self._on_block_model_loaded(bm)
        except Exception:
            pass
        # Subscribe to shared BlockModelDefinition updates and seed now
        if hasattr(self.registry, "blockModelDefinitionChanged"):
            try:
                self.registry.blockModelDefinitionChanged.connect(
                    lambda _defn: self._apply_shared_block_model_definition()
                )
            except Exception:
                pass
        try:
            self._apply_shared_block_model_definition()
        except Exception:
            pass

    def _apply_shared_block_model_definition(self):
        """Seed grid spinboxes from the shared BlockModelDefinition (if any)."""
        try:
            reg = getattr(self, "registry", None)
            if reg is None or not hasattr(reg, "get_block_model_definition"):
                return
            defn = reg.get_block_model_definition()
            if defn is None:
                return
            from PyQt6.QtCore import QSignalBlocker
            ox, oy, oz = defn.origin
            nx, ny, nz = defn.dims
            dx, dy, dz = defn.block_size
            self._suspend_block_size_refit = True
            try:
                with QSignalBlocker(self.dx_spin), QSignalBlocker(self.dy_spin), QSignalBlocker(self.dz_spin):
                    self.xmin_spin.setValue(float(ox))
                    self.ymin_spin.setValue(float(oy))
                    self.zmin_spin.setValue(float(oz))
                    self.nx_spin.setValue(int(nx))
                    self.ny_spin.setValue(int(ny))
                    self.nz_spin.setValue(int(nz))
                    self.dx_spin.setValue(float(dx))
                    self.dy_spin.setValue(float(dy))
                    self.dz_spin.setValue(float(dz))
            finally:
                self._suspend_block_size_refit = False
            logger.info(
                "FastRBFPanel: applied shared BlockModelDefinition '%s' "
                "(%dx%dx%d, %gx%gx%g)",
                defn.name, nx, ny, nz, dx, dy, dz,
            )
        except Exception as exc:
            logger.debug("FastRBFPanel: failed to apply shared BMD: %s", exc)

    def _process_pending_data(self):
        if self._pending_drillhole_data:
            self._on_drillhole_data_loaded(self._pending_drillhole_data)
            self._pending_drillhole_data = None
        if self._pending_variogram_results:
            self._on_variogram_loaded(self._pending_variogram_results)
            self._pending_variogram_results = None

    # ══════════════════════════════════════════════════════════════
    # DATA HANDLING
    # ══════════════════════════════════════════════════════════════

    def _on_drillhole_data_loaded(self, data):
        if not self._ui_ready:
            self._pending_drillhole_data = data
            return
        self._registry_data = data if isinstance(data, dict) else None

        composites = None
        declustered = None
        assays = None

        if isinstance(data, dict):
            composites = data.get("composites")
            if composites is None:
                composites = data.get("composites_df")
            declustered = data.get("declustered")
            assays = data.get("assays")
            if assays is None:
                assays = data.get("assays_df")
        elif isinstance(data, pd.DataFrame):
            composites = data

        comp_ok = isinstance(composites, pd.DataFrame) and not composites.empty
        decl_ok = isinstance(declustered, pd.DataFrame) and not declustered.empty
        raw_ok = isinstance(assays, pd.DataFrame) and not assays.empty

        self.data_source_composited.setEnabled(comp_ok)
        self.data_source_declustered.setEnabled(decl_ok)
        self.data_source_raw.setEnabled(raw_ok)

        df = None
        if self.data_source_composited.isChecked() and comp_ok:
            df = composites
        elif self.data_source_declustered.isChecked() and decl_ok:
            df = declustered
        elif self.data_source_raw.isChecked() and raw_ok:
            df = assays
        elif comp_ok:
            df = composites
            self.data_source_composited.setChecked(True)
        elif decl_ok:
            df = declustered
            self.data_source_declustered.setChecked(True)
        elif raw_ok:
            df = assays
            self.data_source_raw.setChecked(True)

        parts = []
        if comp_ok:
            parts.append(f"Composites: {len(composites):,}")
        if decl_ok:
            parts.append(f"Declustered: {len(declustered):,}")
        if raw_ok:
            parts.append(f"Raw Assays: {len(assays):,}")
        if parts and df is not None:
            source = "Composited" if df is composites else (
                "Declustered" if df is declustered else "Raw Assays"
            )
            self.data_source_status.setText(
                f"Active: {source} ({len(df):,} samples)\n"
                + " | ".join(parts)
            )
            self.data_source_status.setStyleSheet(
                f"font-size: 9pt; color: {ModernColors.SUCCESS}; margin-top: 6px;"
            )
        else:
            self.data_source_status.setText("No data loaded")
            self.data_source_status.setStyleSheet(
                f"font-size: 9pt; color: {ModernColors.TEXT_HINT}; margin-top: 6px;"
            )

        if df is not None:
            self.drillhole_data = df
            self._update_variable_combo()
            self._update_data_preview()
            if self.auto_fit_check.isChecked():
                self._auto_detect_grid()

    def _on_variogram_loaded(self, results):
        if not self._ui_ready:
            self._pending_variogram_results = results
            return
        self.variogram_results = results
        self._apply_variogram_to_ui(results)

    def _on_data_source_radio_changed(self, button):
        if self._registry_data:
            self._on_drillhole_data_loaded(self._registry_data)
        elif self.drillhole_data is not None:
            self._update_variable_combo()
            self._update_data_preview()

    def _manual_refresh(self):
        try:
            reg = self.get_registry()
            if not reg:
                return
            d = reg.get_drillhole_data()
            if d:
                self._on_drillhole_data_loaded(d)
            v = reg.get_variogram_results()
            if v:
                self._on_variogram_loaded(v)
            bm = reg.get_block_model()
            if bm is not None:
                self._on_block_model_loaded(bm)
        except Exception as e:
            logger.warning("Refresh failed: %s", e)

    def _on_block_model_loaded(self, block_model):
        try:
            if hasattr(block_model, "to_dataframe"):
                df = block_model.to_dataframe()
            elif isinstance(block_model, pd.DataFrame):
                df = block_model.copy()
            else:
                return
            if df is None or df.empty:
                return

            from ..utils.coordinate_utils import ensure_xyz_columns
            df = ensure_xyz_columns(df)
            if not all(c in df.columns for c in ("X", "Y", "Z")):
                return

            x_coords = np.unique(df["X"].values.astype(float))
            y_coords = np.unique(df["Y"].values.astype(float))
            z_coords = np.unique(df["Z"].values.astype(float))
            if len(x_coords) == 0 or len(y_coords) == 0 or len(z_coords) == 0:
                return

            nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
            xinc = float(np.mean(np.diff(np.sort(x_coords)))) if nx > 1 else 1.0
            yinc = float(np.mean(np.diff(np.sort(y_coords)))) if ny > 1 else 1.0
            zinc = float(np.mean(np.diff(np.sort(z_coords)))) if nz > 1 else 1.0

            xmin = float(np.min(x_coords) - xinc / 2.0)
            ymin = float(np.min(y_coords) - yinc / 2.0)
            zmin = float(np.min(z_coords) - zinc / 2.0)

            self.block_grid_spec = {
                "nx": nx, "ny": ny, "nz": nz,
                "xmin": xmin, "ymin": ymin, "zmin": zmin,
                "xinc": xinc, "yinc": yinc, "zinc": zinc,
            }
            logger.info(
                "FastRBF: Captured block model grid "
                "(nx=%d, ny=%d, nz=%d, inc=(%.1f, %.1f, %.1f))",
                nx, ny, nz, xinc, yinc, zinc,
            )
        except Exception as e:
            logger.warning("FastRBF: Failed to infer grid from block model: %s", e)

    def _on_variable_changed(self, text):
        if text and hasattr(self, "output_name_edit"):
            self.output_name_edit.setText(f"FastRBF_{text}")

    def _update_variable_combo(self):
        if self.drillhole_data is None:
            return
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        skip = {"X", "Y", "Z", "HOLEID", "HOLE_ID", "FROM", "TO",
                "LENGTH", "DEPTH", "SAMPLE_ID"}
        cols = [c for c in self.drillhole_data.select_dtypes(
            include=[np.number]).columns if c.upper() not in skip]
        self.variable_combo.addItems(cols)
        self.variable_combo.blockSignals(False)
        if cols:
            self.variable_combo.setCurrentIndex(0)
            self._on_variable_changed(cols[0])
        if hasattr(self, "domain_combo") and self.drillhole_data is not None:
            try:
                self._populate_domain_filter_combo(
                    self.drillhole_data, combo=self.domain_combo, all_label="All Data",
                )
            except Exception:
                pass

    def _update_data_preview(self):
        if self.drillhole_data is None:
            self.data_preview_label.setText("No data loaded")
            return
        df = self.drillhole_data
        n = len(df)
        var = self.variable_combo.currentText()
        lines = [f"Samples: {n:,}"]
        if all(c in df.columns for c in ["X", "Y", "Z"]):
            lines.append(
                f"X: [{df['X'].min():.1f}, {df['X'].max():.1f}]  "
                f"Y: [{df['Y'].min():.1f}, {df['Y'].max():.1f}]  "
                f"Z: [{df['Z'].min():.1f}, {df['Z'].max():.1f}]"
            )
        if var and var in df.columns:
            c = df[var].dropna()
            lines.append(
                f"{var}: mean={c.mean():.4f}, std={c.std():.4f}, "
                f"min={c.min():.4f}, max={c.max():.4f}"
            )
        self.data_preview_label.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════
    # GRID AUTO-DETECTION (renderer → block model → DataFrame)
    # ══════════════════════════════════════════════════════════════

    def _find_renderer(self):
        if hasattr(self, "main_window") and self.main_window is not None:
            if hasattr(self.main_window, "viewer_widget"):
                r = getattr(self.main_window.viewer_widget, "renderer", None)
                if r:
                    return r
        if self.controller:
            r = getattr(self.controller, "r", None)
            if r:
                return r
        try:
            w = self
            for _ in range(10):
                w = w.parent() if hasattr(w, "parent") else None
                if w is None:
                    break
                if hasattr(w, "viewer_widget"):
                    r = getattr(w.viewer_widget, "renderer", None)
                    if r:
                        return r
        except Exception:
            pass
        try:
            app = QApplication.instance()
            if app:
                for win in app.topLevelWidgets():
                    if hasattr(win, "viewer_widget"):
                        r = getattr(win.viewer_widget, "renderer", None)
                        if r:
                            return r
        except Exception:
            pass
        return None

    def _get_rendered_bounds(self) -> Optional[Dict[str, float]]:
        renderer = self._find_renderer()
        if not renderer:
            return None
        cache = getattr(renderer, "_drillhole_polylines_cache", None)
        if not cache or "hole_polys" not in cache:
            return None
        try:
            all_pts = []
            for _, poly in cache["hole_polys"].items():
                if hasattr(poly, "points") and poly.n_points > 0:
                    all_pts.append(poly.points)
            if not all_pts:
                return None
            pts = np.vstack(all_pts)
            return {
                "x_min": float(pts[:, 0].min()),
                "x_max": float(pts[:, 0].max()),
                "y_min": float(pts[:, 1].min()),
                "y_max": float(pts[:, 1].max()),
                "z_min": float(pts[:, 2].min()),
                "z_max": float(pts[:, 2].max()),
            }
        except Exception as e:
            logger.debug("Rendered bounds failed: %s", e)
            return None

    def _on_block_size_changed(self, *_):
        """Re-fit grid origin and NX/NY/NZ when DX/DY/DZ change."""
        if getattr(self, "_suspend_block_size_refit", False):
            return
        if getattr(self, "drillhole_data", None) is None:
            return
        df = self.drillhole_data
        try:
            if df.empty:
                return
        except Exception:
            return
        self._suspend_block_size_refit = True
        try:
            self._auto_detect_grid()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("refit after block-size change failed: %s", exc)
        finally:
            self._suspend_block_size_refit = False

    def _auto_fix_grid_origin_if_needed(self):
        """Auto-run Auto-Detect if origin is (0,0,0) but drillhole centroid is far away.

        Catches the UTM vs local coordinate mismatch case where the user loaded
        UTM drillholes but never clicked Auto-Detect, leaving the grid at its
        default origin ~500 km from the data.
        """
        try:
            x0 = self.xmin_spin.value()
            y0 = self.ymin_spin.value()
            z0 = self.zmin_spin.value()
        except AttributeError:
            return
        if x0 != 0.0 or y0 != 0.0 or z0 != 0.0:
            return
        df = getattr(self, "drillhole_data", None)
        if df is None or getattr(df, "empty", True):
            return
        coord_cols = None
        for cx, cy, cz in [("X", "Y", "Z"), ("x", "y", "z"), ("EAST", "NORTH", "RL")]:
            if cx in df.columns and cy in df.columns and cz in df.columns:
                coord_cols = (cx, cy, cz)
                break
        if coord_cols is None:
            return
        import numpy as _np
        x = df[coord_cols[0]].dropna().to_numpy()
        y = df[coord_cols[1]].dropna().to_numpy()
        z = df[coord_cols[2]].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            return
        cx = float(_np.mean(x))
        cy = float(_np.mean(y))
        cz = float(_np.mean(z)) if len(z) else 0.0
        if abs(cx) < 1000 and abs(cy) < 1000:
            return
        import logging as _logging
        _logging.getLogger(__name__).info(
            "%s: grid origin is (0,0,0) but drillhole centroid is (%.0f, %.0f, %.0f). "
            "Auto-detecting grid from drillhole extents.",
            type(self).__name__, cx, cy, cz,
        )
        try:
            self._auto_detect_grid()
        except Exception as _exc:
            _logging.getLogger(__name__).debug("auto-detect from UTM fix failed: %s", _exc)

    def _auto_detect_grid(self):
        # IMPORTANT: Always prefer DataFrame bounds over rendered bounds.
        # The estimation engine receives data from the DataFrame (original
        # coords), NOT from the renderer (which may have shifted to local).
        # Using rendered bounds here would create a coordinate mismatch.
        df = self.drillhole_data
        if df is not None and not df.empty and all(c in df.columns for c in ["X", "Y", "Z"]):
            x_min = float(df["X"].min())
            x_max = float(df["X"].max())
            y_min = float(df["Y"].min())
            y_max = float(df["Y"].max())
            z_min = float(df["Z"].min())
            z_max = float(df["Z"].max())
            logger.info("FastRBF grid: using DataFrame bounds (same coords as estimation)")
        elif df is not None and not df.empty:
            try:
                from ..utils.coordinate_utils import get_spatial_extent
                extent = get_spatial_extent(df, registry=self.get_registry())
                if extent:
                    x_min, x_max = extent["x_min"], extent["x_max"]
                    y_min, y_max = extent["y_min"], extent["y_max"]
                    z_min, z_max = extent["z_min"], extent["z_max"]
                else:
                    self.grid_info_label.setText("Missing XYZ coordinates")
                    return
            except Exception:
                self.grid_info_label.setText("Missing XYZ coordinates")
                return
        else:
            # Last resort: renderer bounds (may not match data coords)
            bounds = self._get_rendered_bounds()
            if bounds:
                x_min, x_max = bounds["x_min"], bounds["x_max"]
                y_min, y_max = bounds["y_min"], bounds["y_max"]
                z_min, z_max = bounds["z_min"], bounds["z_max"]
                logger.warning("FastRBF grid: using rendered bounds (may not match data coords)")
            else:
                self.grid_info_label.setText("No data loaded for grid detection")
                return

        dx = self.dx_spin.value()
        dy = self.dy_spin.value()
        dz = self.dz_spin.value()

        x_range = max(x_max - x_min, dx)
        y_range = max(y_max - y_min, dy)
        z_range = max(z_max - z_min, dz)

        x_pad = max(dx, x_range * 0.05)
        y_pad = max(dy, y_range * 0.05)
        z_pad = max(dz, z_range * 0.05)

        xmin = np.floor((x_min - x_pad) / dx) * dx
        ymin = np.floor((y_min - y_pad) / dy) * dy
        zmin = np.floor((z_min - z_pad) / dz) * dz

        xmax = np.ceil((x_max + x_pad) / dx) * dx
        ymax = np.ceil((y_max + y_pad) / dy) * dy
        zmax = np.ceil((z_max + z_pad) / dz) * dz

        nx = max(1, int(np.round((xmax - xmin) / dx)))
        ny = max(1, int(np.round((ymax - ymin) / dy)))
        nz = max(1, int(np.round((zmax - zmin) / dz)))

        if zmin + nz * dz < z_max:
            nz = int(np.ceil((z_max + z_pad - zmin) / dz))

        self.xmin_spin.setValue(xmin)
        self.ymin_spin.setValue(ymin)
        self.zmin_spin.setValue(zmin)
        self.nx_spin.setValue(nx)
        self.ny_spin.setValue(ny)
        self.nz_spin.setValue(nz)

        total = nx * ny * nz
        txt = (
            f"Grid: {nx} x {ny} x {nz} = {total:,} blocks\n"
            f"Origin: ({xmin:.1f}, {ymin:.1f}, {zmin:.1f})\n"
            f"Block size: ({dx:.1f}, {dy:.1f}, {dz:.1f}) m\n"
            f"Data extent: X=[{x_min:.1f}, {x_max:.1f}], "
            f"Y=[{y_min:.1f}, {y_max:.1f}], Z=[{z_min:.1f}, {z_max:.1f}]"
        )
        self.grid_info_label.setText(txt)

        logger.info(
            "FastRBF grid: %dx%dx%d = %s blocks, "
            "origin=(%.1f, %.1f, %.1f), inc=(%.1f, %.1f, %.1f)",
            nx, ny, nz, f"{total:,}", xmin, ymin, zmin, dx, dy, dz,
        )

    def _apply_block_model_grid(self):
        if not self.block_grid_spec:
            self.grid_info_label.setText(
                "No block model grid available. Load a block model first."
            )
            return
        g = self.block_grid_spec
        self.dx_spin.setValue(g["xinc"])
        self.dy_spin.setValue(g["yinc"])
        self.dz_spin.setValue(g["zinc"])
        self.xmin_spin.setValue(g["xmin"])
        self.ymin_spin.setValue(g["ymin"])
        self.zmin_spin.setValue(g["zmin"])
        self.nx_spin.setValue(g["nx"])
        self.ny_spin.setValue(g["ny"])
        self.nz_spin.setValue(g["nz"])

        total = g["nx"] * g["ny"] * g["nz"]
        txt = (
            f"Aligned to Block Model grid\n"
            f"Grid: {g['nx']} x {g['ny']} x {g['nz']} = {total:,} blocks\n"
            f"Origin: ({g['xmin']:.1f}, {g['ymin']:.1f}, {g['zmin']:.1f})\n"
            f"Block size: ({g['xinc']:.1f}, {g['yinc']:.1f}, {g['zinc']:.1f}) m"
        )
        self.grid_info_label.setText(txt)

    def _compute_grid_spec_from_ui(self) -> Dict[str, Any]:
        return {
            "nx": self.nx_spin.value(),
            "ny": self.ny_spin.value(),
            "nz": self.nz_spin.value(),
            "xmin": self.xmin_spin.value(),
            "ymin": self.ymin_spin.value(),
            "zmin": self.zmin_spin.value(),
            "xinc": self.dx_spin.value(),
            "yinc": self.dy_spin.value(),
            "zinc": self.dz_spin.value(),
        }

    # ══════════════════════════════════════════════════════════════
    # VARIOGRAM INTEGRATION
    # ══════════════════════════════════════════════════════════════

    def _load_variogram_from_registry(self):
        try:
            selected_var = self.variable_combo.currentText() if hasattr(self, 'variable_combo') else None
            vario = resolve_variogram_for_variable(self.registry, selected_var, self)
            if not vario:
                return
            self.variogram_results = vario
            self._apply_variogram_to_ui(vario)
            self.vario_status_label.setText("Variogram loaded successfully")
            self.vario_status_label.setStyleSheet(f"color: {ModernColors.SUCCESS};")
        except Exception as e:
            logger.error("Failed to load variogram: %s", e)
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")

    def _apply_variogram_to_ui(self, vario: Dict):
        combined = vario.get("combined_3d_model", {})
        omni = vario.get("omni_variogram", {})
        source = combined if combined else omni
        if not source:
            return

        # NOTE: "sill" key in variogram results is the TOTAL sill (C0+C1), not partial
        nug = float(source.get("nugget", 0.0))
        raw_sill = float(source.get("sill", 1.0))
        tsill = float(source.get("total_sill", raw_sill))
        if tsill <= nug:
            tsill = nug + max(raw_sill, 0.01)

        self.sill_spin.setValue(tsill)
        self.nugget_spin.setValue(nug)

        mr = float(source.get("major_range", source.get("range", 100.0)))
        self.range_spin.setValue(mr)

        mt = source.get("model_type", "spheroidal").lower()
        idx = {"spheroidal": 0, "spherical": 1, "gaussian": 2,
               "exponential": 3, "linear": 4, "cubic": 5}.get(mt, 0)
        self.kernel_combo.setCurrentIndex(idx)

        alpha = source.get("alpha", 5)
        ai = {"3": 0, "5": 1, "7": 2, "9": 3}.get(str(alpha), 1)
        self.alpha_combo.setCurrentIndex(ai)

        mnr = float(source.get("minor_range", mr))
        vr = float(source.get("vertical_range", mr))
        self.range_major_spin.setValue(mr)
        self.range_semi_spin.setValue(mnr)
        self.range_minor_spin.setValue(vr)

        has_aniso = abs(mr - mnr) > 1e-6 or abs(mr - vr) > 1e-6
        self.anisotropy_enabled.setChecked(has_aniso)

        self.azimuth_spin.setValue(float(source.get("azimuth", 0.0)))
        self.dip_spin.setValue(float(source.get("dip", 0.0)))
        self.pitch_spin.setValue(
            float(source.get("plunge", source.get("pitch", 0.0)))
        )

        self._update_interpolant_curve()

    # ══════════════════════════════════════════════════════════════
    # DATA & PARAMETER CHECKS
    # ══════════════════════════════════════════════════════════════

    def _run_data_checks(self):
        df = self.drillhole_data
        var = self.variable_combo.currentText()

        def _set(cid, ok, txt):
            lbl = self._check_labels.get(cid)
            if not lbl:
                return
            tick = "\u2713" if ok else "\u2717"
            clr = ModernColors.SUCCESS if ok else ModernColors.ERROR
            lbl.setText(f"{tick} {txt}")
            lbl.setStyleSheet(f"""
                QLabel {{
                    background-color: {ModernColors.ELEVATED_BG};
                    padding: 8px 12px;
                    border-radius: 4px;
                    border-left: 3px solid {clr};
                    color: {clr};
                    font-family: Consolas, monospace;
                    font-size: 11px;
                }}
            """)

        if df is None or df.empty:
            for cid in self._check_labels:
                _set(cid, False, "No data loaded")
            return

        n = len(df)
        _set("data_count", n >= 4, f"Data count: {n} points")

        if all(c in df.columns for c in ["X", "Y", "Z"]):
            dupes = df.duplicated(subset=["X", "Y", "Z"]).sum()
            _set("duplicates", dupes == 0,
                 "No duplicate locations" if dupes == 0
                 else f"{dupes} duplicate locations found")
        else:
            _set("duplicates", False, "Missing XYZ columns")

        if var and var in df.columns:
            col = df[var].dropna()
            vmin, vmax = col.min(), col.max()
            _set("value_range", vmax > vmin,
                 f"Value range: {vmin:.2f} to {vmax:.2f}")
            cv = col.std() / col.mean() if col.mean() != 0 else float("inf")
            _set("cv_of_var", cv < 5.0,
                 f"Coefficient of variation: {cv:.2f}")
        else:
            _set("value_range", False, "No variable selected")
            _set("cv_of_var", False, "No variable selected")

        sill = self.sill_spin.value()
        nug = self.nugget_spin.value()
        if sill > 0:
            ratio = nug / sill * 100
            _set("nugget_sill", nug < sill,
                 f"Nugget/Sill ratio: {ratio:.0f}%")
        else:
            _set("nugget_sill", False, "Sill must be > 0")

        if all(c in df.columns for c in ["X", "Y", "Z"]):
            ext = max(
                df["X"].max() - df["X"].min(),
                df["Y"].max() - df["Y"].min(),
                max(df["Z"].max() - df["Z"].min(), 1.0),
            )
            rng = self.range_spin.value()
            ok = 0.01 * ext < rng < 3.0 * ext
            _set("range_extent", ok, "Range proportional to data extent")
        else:
            _set("range_extent", False, "Missing XYZ columns")

    def _run_cv(self):
        if not self.validate_inputs():
            return
        try:
            params = self.gather_parameters()
            params["run_cv_only"] = True
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return

        if self.drillhole_data is not None and not self.drillhole_data.empty:
            params["data"] = self.drillhole_data

        self.cv_run_btn.setEnabled(False)
        self.cv_run_btn.setText("Running LOO-CV...")

        def _cv_done(result):
            self.cv_run_btn.setEnabled(True)
            self.cv_run_btn.setText("Leave-One-Out Cross-Validation")
            if result is None:
                return
            if isinstance(result, dict) and "error" in result:
                QMessageBox.warning(self, "CV Error", result["error"])
                return
            diag = result.get("diagnostics", {})
            self._update_cv_results(diag)

        try:
            self.controller.run_fastrbf_interpolation(
                params=params,
                callback=_cv_done,
                progress_callback=self._on_progress,
            )
        except Exception as e:
            self.cv_run_btn.setEnabled(True)
            self.cv_run_btn.setText("Leave-One-Out Cross-Validation")
            QMessageBox.critical(self, "Error", f"CV failed:\n{e}")

    # ══════════════════════════════════════════════════════════════
    # GATHER / VALIDATE / RUN
    # ══════════════════════════════════════════════════════════════

    def gather_parameters(self) -> Dict[str, Any]:
        self._auto_fix_grid_origin_if_needed()
        var = self.variable_combo.currentText()
        if not var:
            raise ValueError("No variable selected")

        dt = self.drift_combo.currentText().lower()
        if dt.startswith("constant"):
            drift = "constant"
        elif dt.startswith("linear"):
            drift = "linear"
        else:
            drift = "none"

        params: Dict[str, Any] = {
            "variable": var,
            "engine": "fastrbf",
            "fastrbf_kernel": self.kernel_combo.currentText().split("(")[0].strip().lower(),
            "fastrbf_alpha": int(self.alpha_combo.currentText()),
            "fastrbf_sill": self.sill_spin.value(),
            "fastrbf_nugget": self.nugget_spin.value(),
            "fastrbf_range": self.range_spin.value(),
            "fastrbf_drift": drift,
            "fastrbf_max_samples": self.max_samples_spin.value(),
            "fastrbf_min_samples": self.min_samples_spin.value(),
            "fastrbf_max_per_octant": self.max_per_octant_spin.value(),
            "fastrbf_min_octants": self.min_octants_spin.value(),
            "fastrbf_discretisation": self.discretisation_spin.value(),
            "run_diagnostics": True,
            "anisotropy_enabled": self.anisotropy_enabled.isChecked(),
            "grid_spec": self._get_grid_spec(),
            "output_property_name": self.output_name_edit.text(),
        }
        if self.clip_enabled.isChecked():
            params["fastrbf_clip_min"] = self.clip_min_spin.value()
            params["fastrbf_clip_max"] = self.clip_max_spin.value()
        if params["anisotropy_enabled"]:
            params.update({
                "range_x": self.range_major_spin.value(),
                "range_y": self.range_semi_spin.value(),
                "range_z": self.range_minor_spin.value(),
                "azimuth": self.azimuth_spin.value(),
                "dip": self.dip_spin.value(),
                "plunge": self.pitch_spin.value(),
            })
        if self.variogram_results:
            params["variogram_results"] = self.variogram_results
        _dm = getattr(self, "_active_domain_filter_metadata", {}) or {}
        params["domain_column"] = _dm.get("domain_filter_column")
        params["domain_value"] = _dm.get("domain_filter_value")
        return params

    def _get_grid_spec(self) -> Optional[Dict[str, Any]]:
        mode = self.grid_mode_combo.currentText()
        if mode == "Manual" or mode == "From Block Model":
            return self._compute_grid_spec_from_ui()
        if self.nx_spin.value() > 0 and self.ny_spin.value() > 0:
            return self._compute_grid_spec_from_ui()
        self._auto_detect_grid()
        return self._compute_grid_spec_from_ui()

    def validate_inputs(self) -> bool:
        if not FASTRBF_AVAILABLE:
            QMessageBox.warning(self, "Engine Not Available",
                                "FastRBF engine is not installed.")
            return False
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "Load drillhole data first.")
            return False
        var = self.variable_combo.currentText()
        if not var or var not in self.drillhole_data.columns:
            QMessageBox.warning(self, "Invalid Variable",
                                "Select a valid grade variable.")
            return False
        missing = [c for c in ["X", "Y", "Z"]
                   if c not in self.drillhole_data.columns]
        if missing:
            QMessageBox.warning(self, "Missing Coordinates",
                                f"Missing: {missing}")
            return False
        valid = self.drillhole_data.dropna(subset=["X", "Y", "Z", var])
        if len(valid) < 4:
            QMessageBox.warning(self, "Insufficient Data",
                                f"Need >= 4 valid samples, found {len(valid)}.")
            return False
        if self.nugget_spin.value() > self.sill_spin.value():
            QMessageBox.warning(self, "Invalid Parameters",
                                "Nugget must not exceed total sill.")
            self.tab_widget.setCurrentIndex(0)  # Tab 0: Data & Model
            return False
        # Check drillhole coverage within estimation grid
        try:
            from .base_analysis_panel import check_drillhole_grid_coverage
            gs = self._get_grid_spec()
            if gs and all(k in gs for k in ('xmin', 'ymin', 'zmin', 'nx', 'ny', 'nz')):
                dx = gs.get('xinc', gs.get('dx', 1.0))
                dy = gs.get('yinc', gs.get('dy', 1.0))
                dz = gs.get('zinc', gs.get('dz', 1.0))
                if not check_drillhole_grid_coverage(
                    self, self.drillhole_data,
                    xmin=gs['xmin'], ymin=gs['ymin'], zmin=gs['zmin'],
                    nx=gs['nx'], ny=gs['ny'], nz=gs['nz'],
                    dx=dx, dy=dy, dz=dz,
                    panel_name="FastRBF",
                ):
                    return False
        except Exception:
            pass
        return True

    def _on_run_clicked(self):
        if not self.validate_inputs():
            return
        try:
            params = self.gather_parameters()
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return

        if self.drillhole_data is not None and not self.drillhole_data.empty:
            params["data"] = self.drillhole_data
            logger.info("FastRBF panel: Injected %d samples into params",
                        len(self.drillhole_data))

        self._run_data_checks()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting FastRBF interpolation...")

        try:
            self.controller.signals.task_progress.disconnect(self._on_task_progress)
        except (TypeError, RuntimeError):
            pass
        try:
            self.controller.signals.task_error.disconnect(self._on_task_error)
        except (TypeError, RuntimeError):
            pass
        try:
            self.controller.signals.task_progress.connect(self._on_task_progress)
            self.controller.signals.task_error.connect(self._on_task_error)
        except Exception as e:
            logger.debug("Could not connect task signals: %s", e)

        self.controller.run_fastrbf_interpolation(
            params=params,
            callback=self._on_results,
            progress_callback=self._on_progress,
        )

    def _on_stop_clicked(self):
        try:
            if hasattr(self, "controller") and self.controller:
                self.controller.cancel_task("fastrbf")
        except Exception as e:
            logger.debug("Cancel failed: %s", e)
        self._reset_ui_state()

    def _on_task_progress(self, task_name: str, pct: int, msg: str):
        if task_name != "fastrbf":
            return
        self._on_progress(pct, msg)

    def _on_task_error(self, task_name: str, error_msg: str):
        if task_name != "fastrbf":
            return
        self._reset_ui_state()
        QMessageBox.critical(self, "FastRBF Error", f"Task failed:\n{error_msg}")

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _reset_ui_state(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Ready")
        try:
            self.controller.signals.task_progress.disconnect(self._on_task_progress)
        except Exception:
            pass
        try:
            self.controller.signals.task_error.disconnect(self._on_task_error)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════

    def _on_results(self, result):
        try:
            self._reset_ui_state()
            if result is None:
                QMessageBox.critical(self, "FastRBF Error",
                                     "Task returned no result.")
                return
            if isinstance(result, dict) and "error" in result:
                QMessageBox.critical(self, "FastRBF Error",
                                     f"Failed:\n{result['error']}")
                return

            self.fastrbf_results = result

            # --- Domain masking ---
            if isinstance(self.fastrbf_results, dict):
                self.fastrbf_results = self._apply_domain_masking(
                    self.fastrbf_results,
                    grade_keys=['value', 'grade', 'estimates'],
                    variance_keys=[],
                )
                result = self.fastrbf_results  # Update local ref

            meta = result.get("metadata", {})
            diag = result.get("diagnostics", {})
            self._update_cv_results(diag)
            self._update_classification_display(
                meta.get("classification_counts", {})
            )

            try:
                if self.registry:
                    self.registry.register_fastrbf_results(
                        result, source_panel="FastRBFPanel"
                    )
            except Exception as e:
                logger.warning("Registry: %s", e)

            self.request_visualization.emit({
                "type": "fastrbf_results", "data": result
            })
            self.tab_widget.setCurrentIndex(2)  # Tab 2: Run & Results

            ne = meta.get("n_blocks_estimated", 0)
            nt = meta.get("n_blocks_total", 0)
            pct = 100.0 * ne / nt if nt > 0 else 0
            self.progress_label.setText(
                f"Complete: {ne:,}/{nt:,} blocks ({pct:.1f}%)"
            )
        except Exception as e:
            logger.error("Results error: %s", e)
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")

    def _update_cv_results(self, d: Dict[str, Any]):
        def fmt(k):
            v = d.get(k)
            return f"{v:.4f}" if v is not None else "--"

        self.cv_rmse_label.setText(fmt("rmse"))
        self.cv_mae_label.setText(fmt("mae"))
        self.cv_r2_label.setText(fmt("r_squared"))
        self.cv_me_label.setText(fmt("mean_error"))
        self.cv_corr_label.setText(fmt("correlation"))
        self.cv_slope_label.setText(fmt("slope_of_regression"))

        bp = d.get("global_bias_percent")
        if bp is not None:
            self.bias_pct_label.setText(f"{bp:.2f}%")
            c = ModernColors.ERROR if d.get("bias_flagged") else ModernColors.SUCCESS
            self.bias_pct_label.setStyleSheet(f"color: {c};")
        else:
            self.bias_pct_label.setText("--")

        cb = d.get("conditionally_biased")
        if cb is not None:
            if cb:
                self.bias_flag_label.setText("BIASED")
                self.bias_flag_label.setStyleSheet(
                    f"color: {ModernColors.ERROR};"
                )
            else:
                self.bias_flag_label.setText("UNBIASED")
                self.bias_flag_label.setStyleSheet(
                    f"color: {ModernColors.SUCCESS};"
                )

    def _update_classification_display(self, counts: Dict[str, int]):
        total = sum(counts.values()) or 1
        for i, cat in enumerate(
            ["Measured", "Indicated", "Inferred", "Unclassified"]
        ):
            n = counts.get(cat, 0)
            self.class_table.setItem(i, 1, QTableWidgetItem(f"{n:,}"))
            self.class_table.setItem(
                i, 2, QTableWidgetItem(f"{100 * n / total:.1f}%")
            )
