"""
ARBF Estimation Panel — Adaptive Radial Basis Function Estimator.
=================================================================

6-tab panel for JORC-compliant geostatistical estimation using the
ARBF engine (GPR posterior variance, PUM domain decomposition, LVA,
change-of-support).

Tabs:
  0 — Data       : data source, variable, transform options
  1 — Variogram  : kernel config, anisotropy, LVA toggle
  2 — Estimation : sub-domains, discretisation, search, run button
  3 — Results    : grade/variance statistics, block model registration
  4 — Validation : LOO-CV metrics, swath plots, KNA
  5 — Export     : audit record, JORC Table 1, CSV/JSON export
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QTextEdit, QCheckBox,
    QWidget, QFrame, QTabWidget, QFormLayout, QLineEdit,
    QTableWidget, QTableWidgetItem,
    QProgressBar, QSizePolicy, QScrollArea,
    QRadioButton, QButtonGroup, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    FigureCanvas = None
    Figure = None
    MATPLOTLIB_AVAILABLE = False

from .mixins.coded_domain_filter_mixin import CodedDomainFilterMixin
from .mixins.domain_mask_mixin import DomainMaskMixin
from .base_analysis_panel import BaseAnalysisPanel
from .collapsible_group import CollapsibleGroup
from .design_tokens import tokens
from .modern_styles import ModernColors, get_theme_colors
from .panel_toolkit import (
    section, make_form, form_row, make_combo, make_spin, make_int_spin,
    action_button, hint_label, info_display, separator,
    PANEL_MARGINS, PANEL_SPACING,
)
from .panel_manager import PanelCategory, DockArea
from ..utils.variable_utils import get_grade_columns
from .panel_utils import resolve_variogram_for_variable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Matplotlib Canvas
# ═══════════════════════════════════════════════════════════════════

class _ARBFCanvas(FigureCanvas):
    """Dark-themed matplotlib canvas for ARBF validation plots."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        width: float = 5,
        height: float = 3.5,
        dpi: int = 100,
    ) -> None:
        colors = get_theme_colors()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(colors.CARD_BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(280, 200)
        self.updateGeometry()

    def _apply_theme(self, ax) -> None:
        """Apply current theme colors to a matplotlib axes."""
        colors = get_theme_colors()
        ax.set_facecolor(colors.CARD_BG)
        ax.tick_params(colors=colors.TEXT_PRIMARY, labelsize=8)
        ax.xaxis.label.set_color(colors.TEXT_PRIMARY)
        ax.yaxis.label.set_color(colors.TEXT_PRIMARY)
        ax.title.set_color(colors.TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_color(colors.BORDER)
        ax.grid(True, color=colors.BORDER, linestyle="--", alpha=0.4)

    def _safe_draw(self) -> None:
        """Draw canvas with fallback on RuntimeError."""
        try:
            self.draw_idle()
            self.flush_events()
        except RuntimeError as exc:
            logger.debug("Canvas draw failed: %s", exc)
            try:
                self.update()
                self.repaint()
            except Exception:
                pass
        except Exception as exc:
            logger.debug("Canvas draw failed: %s", exc)
            try:
                self.update()
                self.repaint()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# Tab scroll helper
# ═══════════════════════════════════════════════════════════════════

def _make_tab_scroll(tab_widget: QTabWidget, name: str) -> Tuple[QVBoxLayout, QWidget]:
    """Create a scrollable tab with consistent margins.

    Returns (content_layout, content_widget) — add widgets to content_layout,
    then tab is auto-added to tab_widget.
    """
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    content = QWidget()
    lay = QVBoxLayout(content)
    lay.setContentsMargins(*PANEL_MARGINS)
    lay.setSpacing(PANEL_SPACING)
    scroll.setWidget(content)
    tab_widget.addTab(scroll, name)
    return lay, content


# ═══════════════════════════════════════════════════════════════════
# Export helper
# ═══════════════════════════════════════════════════════════════════

def _export_file(
    parent: QWidget,
    title: str,
    default_name: str,
    file_filter: str,
    write_fn,
    status_label: QLabel,
) -> None:
    """Shared pattern: pick path → write_fn(path) → update status."""
    path, _ = QFileDialog.getSaveFileName(parent, title, default_name, file_filter)
    if not path:
        return
    try:
        write_fn(path)
        status_label.setText(f"Saved to {os.path.basename(path)}")
    except Exception as exc:
        logger.error("%s failed: %s", title, exc)
        status_label.setText(f"Export failed: {exc}")


# ═══════════════════════════════════════════════════════════════════
# ARBF Panel
# ═══════════════════════════════════════════════════════════════════

class ARBFPanel(CodedDomainFilterMixin, DomainMaskMixin, BaseAnalysisPanel):
    """Adaptive RBF Estimation panel with JORC-compliant audit trails.

    Tabs: Data | Variogram | Estimation | Results | Validation | Export
    """

    PANEL_ID = "ARBFPanel"
    PANEL_NAME = "ARBF Estimation"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    task_name = "arbf"
    request_visualization = pyqtSignal(dict)
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Data state
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.arbf_results: Optional[Dict[str, Any]] = None
        self.registry = None

        # Internal state
        self._ui_ready = False
        self._pending_drillhole_data: Optional[pd.DataFrame] = None
        self._registry_data: Optional[Dict] = None
        self._swath_data: Optional[Dict] = None
        self.main_window = None

        super().__init__(parent=parent, panel_id="arbf")

    # ------------------------------------------------------------------
    # Controller binding
    # ------------------------------------------------------------------

    def bind_controller(self, controller):
        """Bind controller and connect to task_progress for real-time progress."""
        super().bind_controller(controller)
        if controller and hasattr(controller, 'signals'):
            controller.signals.task_progress.connect(self._handle_task_progress)

    def _handle_task_progress(self, task_name: str, percent: int, message: str):
        """Route controller task_progress to our progress bar."""
        if task_name == self.task_name:
            self.progress_updated.emit(percent, message)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_base_ui(self) -> None:
        """Build header + 6-tab layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        root.addWidget(self.tab_widget, stretch=1)

        self._build_data_tab()        # 0
        self._build_variogram_tab()   # 1
        self._build_estimation_tab()  # 2
        self._build_results_tab()     # 3
        self._build_validation_tab()  # 4
        self._build_export_tab()      # 5

        self._ui_ready = True
        self._init_registry_connections()
        self.progress_updated.connect(self._on_progress)
        self._process_pending_data()
        QTimer.singleShot(0, self._connect_registry_notice)
        self._is_initialized = True

    def _build_header(self) -> QFrame:
        """Compact header bar with title and status."""
        frame = QFrame()
        frame.setObjectName("Card")
        frame.setStyleSheet(
            f"QFrame#Card {{"
            f"  background-color: {ModernColors.ELEVATED_BG};"
            f"  border-bottom: 1px solid {ModernColors.DIVIDER};"
            f"}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)

        title = QLabel("ARBF Estimation")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        lay.addWidget(title)
        lay.addStretch()

        sub = QLabel("GPR + PUM  |  3D Implicit  |  JORC 2012")
        sub.setStyleSheet(
            f"color: {ModernColors.TEXT_HINT}; font-size: 10px;",
        )
        lay.addWidget(sub)

        lay.addStretch()

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        lay.addWidget(self.status_label)
        return frame

    # ══════════════════════════════════════════════════════════════
    # TAB 0 — DATA
    # ══════════════════════════════════════════════════════════════

    def _build_data_tab(self) -> None:
        """Data source, variable selection, and transforms."""
        lay, content = _make_tab_scroll(self.tab_widget, "Data")

        # Data source
        grp_data = section("Input Data")
        form = make_form()

        self.data_source_group = QButtonGroup(self)
        self.radio_composited = QRadioButton("Composited Drillholes")
        self.radio_raw = QRadioButton("Raw Assays")
        self.radio_composited.setChecked(True)
        self.data_source_group.addButton(self.radio_composited, 0)
        self.data_source_group.addButton(self.radio_raw, 1)

        src_lay = QHBoxLayout()
        src_lay.addWidget(self.radio_composited)
        src_lay.addWidget(self.radio_raw)
        form.addRow("Source:", src_lay)

        self.variable_combo = make_combo(tooltip="Grade variable to estimate")
        form_row(form, "Variable:", self.variable_combo)

        self.domain_combo = make_combo(
            tooltip="Domain selection for coded hard-boundary estimation",
        )
        self.domain_combo.addItem("(none)")
        self.domain_combo.currentTextChanged.connect(self._on_domain_filter_selection_changed)
        self.domain_combo.currentTextChanged.connect(
            lambda _: self._update_domain_variogram_status()
        )
        self.domain_combo.currentTextChanged.connect(
            lambda _: self._reload_variogram_for_domain()
        )
        form_row(form, "Domain:", self.domain_combo)

        self.domain_vario_status = QLabel("")
        self.domain_vario_status.setWordWrap(True)
        form_row(form, "Variogram Coverage:", self.domain_vario_status)

        self.data_status = QLabel("No data loaded")
        self.data_status.setObjectName("statusLabel")
        form_row(form, "Status:", self.data_status)

        btn_load = action_button(
            "Load from Registry", style="secondary",
            tooltip="Pull composited/assay data from the data registry",
        )
        btn_load.clicked.connect(self._on_load_data)
        form.addRow("", btn_load)

        grp_data.add_layout(form)
        lay.addWidget(grp_data)

        # Auto-configure from data
        grp_auto = section("Auto-Configure")
        af = make_form()

        self.auto_recommend_info = QTextEdit()
        self.auto_recommend_info.setReadOnly(True)
        self.auto_recommend_info.setMaximumHeight(200)
        self.auto_recommend_info.setPlaceholderText(
            "Click 'Analyse Data & Recommend Settings' to auto-detect "
            "optimal ARBF parameters from your drillhole data."
        )
        af.addRow(self.auto_recommend_info)

        btn_recommend = action_button(
            "Analyse Data && Recommend Settings", style="secondary",
            tooltip=(
                "Analyse drillhole data statistics (skewness, spacing, "
                "extent) and recommend kernel, variogram, transform, and "
                "clipping settings.\n\n"
                "Does NOT overwrite imported variogram parameters unless "
                "you click 'Apply Recommendations'."
            ),
        )
        btn_recommend.clicked.connect(self._on_auto_recommend)
        af.addRow("", btn_recommend)

        btn_apply = action_button(
            "Apply Recommendations", style="primary",
            tooltip="Apply the recommended settings to all panel controls.",
        )
        btn_apply.clicked.connect(self._on_apply_recommendations)
        af.addRow("", btn_apply)

        grp_auto.add_layout(af)
        lay.addWidget(grp_auto)

        # Transforms
        grp_xform = section("Data Transforms")
        xf = make_form()

        self.chk_normal_score = QCheckBox("Apply Normal-Score Transform")
        self.chk_normal_score.setToolTip(
            "Gaussian anamorphosis for skewed distributions",
        )
        xf.addRow(self.chk_normal_score)

        self.chk_ilr = QCheckBox("ILR Compositional Transform")
        self.chk_ilr.setToolTip(
            "Isometric log-ratio for compositional data",
        )
        self.chk_ilr.setEnabled(False)
        xf.addRow(self.chk_ilr)

        self.chk_clip = QCheckBox("Apply Grade Clipping")
        xf.addRow(self.chk_clip)

        self.clip_min_spin = make_spin(-1e6, 1e6, 0.0, 4, tooltip="Minimum clip value")
        form_row(xf, "Min:", self.clip_min_spin)

        self.clip_max_spin = make_spin(-1e6, 1e6, 1e6, 4, tooltip="Maximum clip value")
        form_row(xf, "Max:", self.clip_max_spin)

        grp_xform.add_layout(xf)
        lay.addWidget(grp_xform)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — VARIOGRAM
    # ══════════════════════════════════════════════════════════════

    def _build_variogram_tab(self) -> None:
        """Kernel config, anisotropy ranges, and LVA toggle."""
        lay, content = _make_tab_scroll(self.tab_widget, "Variogram")

        # RBF Basis Function (primary model input)
        grp_rbf = section("RBF Basis Function")
        rf = make_form()

        self.rbf_basis_combo = make_combo([
            "Wendland C2", "Wendland C4", "Gaussian RBF",
            "Multiquadric", "Inverse Multiquadric",
        ], tooltip=(
            "Radial basis function for the interpolation matrix.\n\n"
            "Wendland C2: compactly supported, C\u00b2 smooth — recommended\n"
            "for most mining deposits. Zero beyond the support radius.\n\n"
            "Wendland C4: smoother (C\u2074), slightly heavier computation.\n\n"
            "Gaussian RBF: infinitely smooth, global support.\n"
            "Good for smooth grade distributions, but needs careful\n"
            "shape parameter selection to avoid ill-conditioning.\n\n"
            "Multiquadric: robust global interpolant. Needs polynomial\n"
            "augmentation (constant or linear drift).\n\n"
            "Inverse Multiquadric: positive definite, global support.\n"
            "Smoother interpolation than Wendland."
        ))
        form_row(rf, "Basis Function:", self.rbf_basis_combo)

        self.shape_param_spin = make_spin(
            0.01, 10.0, 1.0, 3,
            tooltip=(
                "Shape parameter \u03b5 (Gaussian / MQ / IMQ only).\n"
                "Controls smoothness vs localisation in anisotropy-\n"
                "transformed space. \u03b5=1.0 is a sensible default\n"
                "when distances are normalised by variogram range.\n"
                "Smaller \u03b5 = smoother but less stable.\n"
                "Larger \u03b5 = sharper but more localised."
            ),
        )
        form_row(rf, "Shape Parameter (\u03b5):", self.shape_param_spin)

        self.support_radius_spin = make_spin(
            0.1, 10.0, 1.5, 2,
            tooltip=(
                "Support radius R for Wendland kernels.\n"
                "In anisotropy-transformed space where 1 unit = 1\n"
                "variogram range. R=1.5 means the kernel is zero\n"
                "beyond 1.5\u00d7 the variogram range.\n"
                "Larger R = smoother transitions, more samples used.\n"
                "Smaller R = sharper cutoff, faster computation."
            ),
        )
        form_row(rf, "Support Radius:", self.support_radius_spin)

        self.nugget_spin = make_spin(
            0.0, 1e15, 0.05, 4,
            tooltip=(
                "Tikhonov regularisation \u03bb added to the diagonal\n"
                "of the RBF matrix [\u03a6 + \u03bbI]. Controls smoothing\n"
                "vs exact interpolation. Also absorbs measurement noise."
            ),
        )
        form_row(rf, "Regularisation (\u03bb):", self.nugget_spin)

        self.accuracy_spin = make_spin(
            1e-7, 1.0, 1e-6, 8, tooltip="Additional numerical stabilisation",
        )
        form_row(rf, "Accuracy:", self.accuracy_spin)

        self.drift_combo = make_combo(
            ["Constant", "Linear", "None"],
            tooltip=(
                "Polynomial drift augmentation.\n"
                "Constant: p(x) = [1] — removes global mean bias.\n"
                "Linear: p(x) = [1, x, y, z] — handles linear trends.\n"
                "None: no drift — only use when data is clearly stationary."
            ),
        )
        form_row(rf, "Polynomial Drift:", self.drift_combo)

        grp_rbf.add_layout(rf)
        lay.addWidget(grp_rbf)

        # Variogram Guidance (optional — for inferring RBF parameters)
        grp_vario = section("Variogram Guidance (optional)")
        vf = make_form()

        self.kernel_combo = make_combo([
            "Spheroidal", "Gaussian", "Exponential", "Spherical",
        ], tooltip=(
            "Variogram model type — used only as guidance for\n"
            "inferring anisotropy ranges, NOT as the RBF kernel.\n"
            "Import from the Variogram Panel to auto-populate\n"
            "ranges and anisotropy directions."
        ))
        form_row(vf, "Variogram Model:", self.kernel_combo)

        self.alpha_spin = make_spin(
            0.1, 10.0, 1.0, 2, tooltip="Smoothness parameter (guidance only)",
        )
        form_row(vf, "Alpha:", self.alpha_spin)

        self.sill_spin = make_spin(
            0.0, 1e15, 0.0, 4,
            tooltip="Partial sill (C1). 0 = auto. Used for NS back-transform dispersion.",
        )
        form_row(vf, "Sill (C1):", self.sill_spin)

        self.variogram_mode_combo = make_combo(
            ["Hybrid", "Global", "Local"],
            tooltip=(
                "How variogram parameters are used as guidance.\n"
                "Global: imported variogram guides all sub-domains.\n"
                "Hybrid: local variograms with global fallback."
            ),
        )
        form_row(vf, "Guidance Mode:", self.variogram_mode_combo)

        btn_import = action_button(
            "Import from Variogram Panel", style="secondary",
            tooltip="Load variogram parameters as guidance for anisotropy and ranges",
        )
        btn_import.clicked.connect(lambda: self._on_import_variogram(set_mode=True))
        vf.addRow("", btn_import)

        grp_vario.add_layout(vf)
        lay.addWidget(grp_vario)

        # Anisotropy
        grp_aniso = section("Anisotropy & Ranges")
        af = make_form()

        self.range_max_spin = make_spin(
            0.1, 1e6, 100.0, 1, tooltip="Major range (max continuity)",
        )
        form_row(af, "Range Max:", self.range_max_spin)

        self.range_mid_spin = make_spin(
            0.1, 1e6, 100.0, 1, tooltip="Semi-major range",
        )
        form_row(af, "Range Mid:", self.range_mid_spin)

        self.range_min_spin = make_spin(
            0.1, 1e6, 100.0, 1, tooltip="Minor range (min continuity)",
        )
        form_row(af, "Range Min:", self.range_min_spin)

        self.azimuth_spin = make_spin(0.0, 360.0, 0.0, 1, tooltip="Azimuth (degrees)")
        form_row(af, "Azimuth:", self.azimuth_spin)

        self.dip_spin = make_spin(-90.0, 90.0, 0.0, 1, tooltip="Dip (degrees)")
        form_row(af, "Dip:", self.dip_spin)

        self.pitch_spin = make_spin(-90.0, 90.0, 0.0, 1, tooltip="Pitch (degrees)")
        form_row(af, "Pitch:", self.pitch_spin)

        # Rotation convention selector
        self.rotation_convention_combo = make_combo(
            ["GeoX (native)", "Leapfrog / MICROMINE", "Datamine Studio",
             "Vulcan", "Surpac"],
            tooltip=(
                "The software from which azimuth/dip/pitch were measured.\n"
                "CRITICAL: Each package uses a different rotation order and\n"
                "handedness. Entering Leapfrog angles as GeoX angles will\n"
                "produce a mirrored search ellipse — a JORC audit failure.\n"
                "GeoX convention: R = Ry(pitch) @ Rx(dip) @ Rz(-azimuth),\n"
                "X=East, Y=North, Z=Up."
            ),
        )
        form_row(af, "Angle Convention:", self.rotation_convention_combo)

        grp_aniso.add_layout(af)
        lay.addWidget(grp_aniso)

        # LVA
        grp_lva = section("Locally Varying Anisotropy (LVA)")
        lf = make_form()

        self.chk_lva = QCheckBox("Enable LVA")
        self.chk_lva.setToolTip(
            "Spatially varying rotation field inferred from grade data "
            "or structural measurements",
        )
        lf.addRow(self.chk_lva)

        self.lva_source_combo = make_combo([
            "Data-Driven (Boisvert 2009)",
            "Structural Measurements",
            "Identity (None)",
        ], tooltip="Source for orientation field")
        form_row(lf, "LVA Source:", self.lva_source_combo)

        self.chk_geodesic = QCheckBox("Geodesic Distance (Folded Deposits)")
        self.chk_geodesic.setToolTip(
            "Use path-integral distance (Eq. 4.4) for sample pairs where the\n"
            "orientation field changes by more than 15°.\n"
            "Required for strongly folded deposits (Bushveld Complex, Wits Basin)\n"
            "where straight-line distance cuts through a fold hinge.\n"
            "Slower (~10× for N=1000) — leave unchecked for flat or gently dipping deposits."
        )
        lf.addRow(self.chk_geodesic)

        grp_lva.add_layout(lf)
        lay.addWidget(grp_lva)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — ESTIMATION
    # ══════════════════════════════════════════════════════════════

    def _build_estimation_tab(self) -> None:
        """Sub-domains, discretisation, grid, execution, run button."""
        lay, content = _make_tab_scroll(self.tab_widget, "Estimation")

        # Sub-domain config
        grp_sd = section("Sub-Domain Configuration (PUM)")
        sf = make_form()

        self.chk_single_domain = QCheckBox("Single Domain (no sub-domains)")
        self.chk_single_domain.setToolTip(
            "Use one global kernel matrix for all samples.\n"
            "Recommended for N < 3,000 samples.",
        )
        sf.addRow(self.chk_single_domain)

        self.pum_threshold_spin = make_int_spin(
            100, 10000, 3000,
            tooltip="Auto-enable single domain when N < threshold",
        )
        form_row(sf, "Auto Threshold:", self.pum_threshold_spin)

        self.n_subdomains_spin = make_int_spin(
            0, 100, 0, tooltip="Number of sub-domains (0 = auto)",
        )
        form_row(sf, "Sub-domains:", self.n_subdomains_spin)

        self.sd_method_combo = make_combo(
            ["K-Means", "Manual"], tooltip="Domain partitioning method",
        )
        form_row(sf, "Method:", self.sd_method_combo)

        self.overlap_spin = make_spin(
            1.0, 3.0, 1.5, 2, tooltip="Overlap factor for Wendland C2 blending",
        )
        form_row(sf, "Overlap Factor:", self.overlap_spin)

        self.chk_single_domain.toggled.connect(self._on_single_domain_toggled)

        grp_sd.add_layout(sf)
        lay.addWidget(grp_sd)

        # Search
        grp_search = section("Search Parameters")
        sef = make_form()

        self.max_samples_spin = make_int_spin(
            10, 2000, 300, tooltip="Maximum samples per sub-domain",
        )
        form_row(sef, "Max Samples:", self.max_samples_spin)

        self.min_samples_spin = make_int_spin(
            2, 100, 4, tooltip="Minimum samples for estimation",
        )
        form_row(sef, "Min Samples:", self.min_samples_spin)

        grp_search.add_layout(sef)
        lay.addWidget(grp_search)

        # Discretisation
        grp_disc = section("Block Discretisation")
        df = make_form()

        self.disc_mode_combo = make_combo(
            ["Fixed", "Adaptive"],
            tooltip="Adaptive: high-gradient blocks get more points",
        )
        form_row(df, "Mode:", self.disc_mode_combo)

        self.disc_density_combo = make_combo(["8 (2x2x2)", "27 (3x3x3)", "64 (4x4x4)"])
        self.disc_density_combo.setCurrentIndex(0)  # 8 (2x2x2) — fast, sufficient for most block sizes
        form_row(df, "Density:", self.disc_density_combo)

        grp_disc.add_layout(df)
        lay.addWidget(grp_disc)

        # Grid
        grp_grid = section("Grid Specification")
        gf = make_form()

        self.chk_use_block_model = QCheckBox("Use Loaded Block Model Grid")
        self.chk_use_block_model.setChecked(True)
        self.chk_use_block_model.setToolTip(
            "Estimate onto the currently loaded block model grid",
        )
        gf.addRow(self.chk_use_block_model)

        self.grid_nx = make_int_spin(1, 1000, 50, tooltip="Grid cells in X")
        self.grid_ny = make_int_spin(1, 1000, 50, tooltip="Grid cells in Y")
        self.grid_nz = make_int_spin(1, 500, 25, tooltip="Grid cells in Z")
        form_row(gf, "NX:", self.grid_nx)
        form_row(gf, "NY:", self.grid_ny)
        form_row(gf, "NZ:", self.grid_nz)

        self.grid_dx = make_spin(0.1, 1e5, 10.0, 2, tooltip="Block size X")
        self.grid_dy = make_spin(0.1, 1e5, 10.0, 2, tooltip="Block size Y")
        self.grid_dz = make_spin(0.1, 1e5, 10.0, 2, tooltip="Block size Z")
        form_row(gf, "DX:", self.grid_dx)
        form_row(gf, "DY:", self.grid_dy)
        form_row(gf, "DZ:", self.grid_dz)

        self.grid_x0 = make_spin(-1e7, 1e7, 0.0, 2, tooltip="Origin X")
        self.grid_y0 = make_spin(-1e7, 1e7, 0.0, 2, tooltip="Origin Y")
        self.grid_z0 = make_spin(-1e7, 1e7, 0.0, 2, tooltip="Origin Z")
        form_row(gf, "X0:", self.grid_x0)
        form_row(gf, "Y0:", self.grid_y0)
        form_row(gf, "Z0:", self.grid_z0)

        btn_auto = action_button(
            "Auto-Detect from Data", style="secondary",
            tooltip="Set grid extents from drillhole data",
        )
        btn_auto.clicked.connect(self._on_auto_detect_grid)
        gf.addRow("", btn_auto)

        # Spatial clipping — restrict estimation to drilled footprint
        self.chk_clip_to_footprint = QCheckBox(
            "Clip to Drillhole Footprint"
        )
        self.chk_clip_to_footprint.setToolTip(
            "Remove blocks that are further than the buffer distance\n"
            "from the nearest drillhole sample.  Prevents extrapolation\n"
            "into undrilled areas and dramatically improves estimation\n"
            "coverage, CV metrics, and classification.\n\n"
            "Buffer = multiplier x variogram major range."
        )
        self.chk_clip_to_footprint.setChecked(False)
        gf.addRow(self.chk_clip_to_footprint)

        self.footprint_buffer_spin = make_spin(
            0.5, 5.0, 1.5, 1,
            tooltip="Buffer distance as a multiplier of the variogram "
                    "major range.  1.0 = blocks within exactly one range "
                    "of the nearest sample.  1.5 = recommended default.",
        )
        form_row(gf, "Buffer (x range):", self.footprint_buffer_spin)

        grp_grid.add_layout(gf)
        lay.addWidget(grp_grid)

        # Execution
        grp_exec = section("Execution")
        ef = make_form()

        self.chk_cv = QCheckBox("Run Cross-Validation (LOO-CV)")
        self.chk_cv.setChecked(True)
        ef.addRow(self.chk_cv)

        self.chk_cos = QCheckBox("Apply Change-of-Support Correction")
        self.chk_cos.setChecked(True)
        self.chk_cos.setToolTip(
            "Affine correction for volume-variance support effect (Matheron 1976).\n"
            "Automatically skipped when discretisation density > 1 (Block Kriging\n"
            "already accounts for volume-variance — applying CoS on top would\n"
            "double-discount and over-shrink grade estimates)."
        )
        ef.addRow(self.chk_cos)

        self.decluster_cell_size_spin = make_spin(
            0.0, 1e6, 0.0, 1,
            tooltip=(
                "Cell size (metres) for Deutsch (1989) cell declustering when\n"
                "computing the Change-of-Support global mean.\n"
                "0 = auto-select from 5× median nearest-neighbour spacing.\n"
                "Increase for tightly drilled high-grade zones to downweight them more."
            ),
        )
        form_row(ef, "Decluster Cell Size:", self.decluster_cell_size_spin)

        self.operator_edit = QLineEdit()
        self.operator_edit.setPlaceholderText("Geologist name (for JORC audit)")
        form_row(ef, "Operator:", self.operator_edit)

        self.seed_spin = make_int_spin(0, 999999, 42, tooltip="Random seed")
        form_row(ef, "Seed:", self.seed_spin)

        grp_exec.add_layout(ef)
        lay.addWidget(grp_exec)

        # Run button
        self.run_btn = action_button(
            "Run ARBF Estimation", style="primary",
            tooltip="Start ARBF estimation with current parameters",
        )
        self.run_btn.clicked.connect(self._on_run)
        lay.addWidget(self.run_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        lay.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("hintLabel")
        lay.addWidget(self.progress_label)

        lay.addWidget(self._build_domain_mask_group(default_enabled=True))

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — RESULTS
    # ══════════════════════════════════════════════════════════════

    def _build_results_tab(self) -> None:
        """Estimation summary, diagnostics, and registry registration."""
        lay, content = _make_tab_scroll(self.tab_widget, "Results")

        # Summary
        grp_summary = section("Estimation Summary")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Run estimation to see results...")
        sl = QVBoxLayout()
        sl.addWidget(self.results_text)
        grp_summary.add_layout(sl)
        lay.addWidget(grp_summary)

        # Diagnostics
        grp_diag = section("Diagnostics")
        self.diag_table = QTableWidget(0, 2)
        self.diag_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.diag_table.horizontalHeader().setStretchLastSection(True)
        self.diag_table.setMinimumHeight(180)
        dl = QVBoxLayout()
        dl.addWidget(self.diag_table)
        grp_diag.add_layout(dl)
        lay.addWidget(grp_diag)

        # Register
        self.btn_register = action_button(
            "Register Results to Block Model", style="secondary",
            tooltip="Push grades + variances to data registry",
        )
        self.btn_register.setEnabled(False)
        self.btn_register.clicked.connect(self._on_register_results)
        lay.addWidget(self.btn_register)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 4 — VALIDATION
    # ══════════════════════════════════════════════════════════════

    def _build_validation_tab(self) -> None:
        """LOO-CV metrics, swath plots, and scatter plot."""
        lay, content = _make_tab_scroll(self.tab_widget, "Validation")

        # CV metrics
        grp_cv = section("Cross-Validation Metrics (LOO-CV)")
        self.cv_table = QTableWidget(0, 2)
        self.cv_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.cv_table.horizontalHeader().setStretchLastSection(True)
        self.cv_table.setMinimumHeight(200)
        cl = QVBoxLayout()
        cl.addWidget(self.cv_table)
        grp_cv.add_layout(cl)
        lay.addWidget(grp_cv)

        # Swath plot
        grp_swath = section("Swath Plots")
        swath_lay = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.swath_canvas = _ARBFCanvas(parent=content, width=5, height=3)
            swath_lay.addWidget(self.swath_canvas)
        else:
            swath_lay.addWidget(QLabel("Matplotlib not available"))
            self.swath_canvas = None

        self.swath_axis_combo = make_combo(["X", "Y", "Z"])
        self.swath_axis_combo.currentIndexChanged.connect(self._on_swath_axis_changed)
        swath_lay.addWidget(self.swath_axis_combo)
        grp_swath.add_layout(swath_lay)
        lay.addWidget(grp_swath)

        # Scatter plot
        grp_scatter = section("Actual vs Estimated (CV)")
        scatter_lay = QVBoxLayout()
        if MATPLOTLIB_AVAILABLE:
            self.scatter_canvas = _ARBFCanvas(parent=content, width=4, height=4)
            scatter_lay.addWidget(self.scatter_canvas)
        else:
            scatter_lay.addWidget(QLabel("Matplotlib not available"))
            self.scatter_canvas = None
        grp_scatter.add_layout(scatter_lay)
        lay.addWidget(grp_scatter)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 5 — EXPORT
    # ══════════════════════════════════════════════════════════════

    def _build_export_tab(self) -> None:
        """JORC audit record and file export buttons."""
        lay, content = _make_tab_scroll(self.tab_widget, "Export")

        # Audit text
        grp_audit = section("JORC Table 1 Section 3 Audit")
        self.audit_text = QTextEdit()
        self.audit_text.setReadOnly(True)
        self.audit_text.setMinimumHeight(300)
        self.audit_text.setPlaceholderText(
            "Run estimation to generate audit record...",
        )
        al = QVBoxLayout()
        al.addWidget(self.audit_text)
        grp_audit.add_layout(al)
        lay.addWidget(grp_audit)

        # Export buttons
        grp_export = section("Export")
        el = QVBoxLayout()

        btn_jorc = action_button("Export JORC Table 1 Report (.txt)", style="secondary")
        btn_jorc.clicked.connect(self._on_export_jorc)
        el.addWidget(btn_jorc)

        btn_json = action_button("Export Audit Record (.json)", style="secondary")
        btn_json.clicked.connect(self._on_export_audit_json)
        el.addWidget(btn_json)

        btn_csv = action_button("Export Results (.csv)", style="secondary")
        btn_csv.clicked.connect(self._on_export_csv)
        el.addWidget(btn_csv)

        grp_export.add_layout(el)
        lay.addWidget(grp_export)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════

    def _init_registry_connections(self) -> None:
        """Wire registry signals for auto-refresh on data load."""
        try:
            registry = self.get_registry()
            if registry is None:
                return
            self.registry = registry
            registry.drillholeDataLoaded.connect(self._on_registry_data_changed)
            registry.compositesLoaded.connect(self._on_registry_data_changed)
            registry.blockModelLoaded.connect(self._on_registry_data_changed)
            if hasattr(registry, "indicatorRBFDomainLoaded"):
                registry.indicatorRBFDomainLoaded.connect(self._on_indicator_rbf_domain_loaded)
            registry.variogramResultsLoaded.connect(self._on_variogram_updated)
            self._on_registry_data_changed()
            # Load variogram results that were stored before this panel was opened
            self._on_import_variogram()
        except Exception:
            pass

    def _on_variogram_updated(self, results=None) -> None:
        """Auto-import variogram parameters when variogram is re-run."""
        if not self._ui_ready:
            return
        # Update stored results so _on_import_variogram picks up the new ones
        if isinstance(results, dict) and results:
            self.variogram_results = results
        else:
            self.variogram_results = None  # force re-fetch from registry
        self._on_import_variogram()

    def _process_pending_data(self) -> None:
        """Apply data that arrived before UI was ready."""
        if self._pending_drillhole_data is not None:
            self.set_drillhole_data(self._pending_drillhole_data)
            self._pending_drillhole_data = None

    def _on_registry_data_changed(self, _data=None) -> None:
        """Refresh variable list when registry data changes."""
        if not self._ui_ready or self.registry is None:
            return
        try:
            data = self.registry.get_estimation_ready_data()
            if data is None:
                data = self.registry.get_drillhole_data()
            self._registry_data = data
            self._refresh_variable_list(data)
        except Exception as exc:
            logger.debug("Registry refresh failed: %s", exc)

    def _on_load_data(self) -> None:
        """Load data from registry into panel."""
        if self.registry is None:
            try:
                self.registry = self.get_registry()
            except Exception:
                self.data_status.setText("No registry available")
                return

        data = self.registry.get_estimation_ready_data()
        if data is None:
            data = self.registry.get_drillhole_data()
        if data is None:
            self.data_status.setText("No data in registry")
            return

        self._registry_data = data
        self._refresh_variable_list(data)
        self._extract_dataframe(data)

    def _extract_dataframe(self, data) -> None:
        """Extract a usable DataFrame from registry data."""
        self.drillhole_data = None
        if isinstance(data, pd.DataFrame):
            self.drillhole_data = data
        elif isinstance(data, dict):
            for key in ("composites", "composites_df", "assays", "assays_df"):
                df = data.get(key)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    self.drillhole_data = df
                    break

        if self.drillhole_data is not None:
            self.drillhole_data = self._prepare_domain_filter_dataframe(
                self.drillhole_data,
                registry_payload=data,
                populate_combo=False,
                all_label="(none)",
            )
            n = len(self.drillhole_data)
            self.data_status.setText(f"{n} samples loaded")
        else:
            self.data_status.setText("No usable data found")

    def _refresh_variable_list(self, data) -> None:
        """Populate variable combo from data columns."""
        prev_selection = self.variable_combo.currentText()
        self.variable_combo.clear()
        df = None
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            for key in ("composites", "composites_df", "assays", "assays_df"):
                candidate = data.get(key)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    df = candidate
                    break
        if df is not None:
            df = self._prepare_domain_filter_dataframe(
                df,
                registry_payload=data,
                populate_combo=False,
            )
            for col in get_grade_columns(df):
                self.variable_combo.addItem(col)

            self._populate_domain_filter_combo(df, all_label="(none)")

        # Restore previous selection if it still exists in the new list
        if prev_selection:
            idx = self.variable_combo.findText(prev_selection)
            if idx >= 0:
                self.variable_combo.setCurrentIndex(idx)

    def _get_filtered_data(self) -> Optional[pd.DataFrame]:
        if self.drillhole_data is None:
            return None
        filtered_df, metadata = self._get_current_domain_filtered_data(
            df=self.drillhole_data,
            all_label="(none)",
        )
        self._active_domain_filter_metadata = metadata
        return filtered_df

    def _on_import_variogram(self, set_mode: bool = False) -> None:
        """Import variogram parameters as guidance for RBF estimation."""
        try:
            # Always fetch the latest variogram from the registry so that
            # results computed before this panel was opened are picked up.
            selected_var = self.variable_combo.currentText() if hasattr(self, 'variable_combo') else None
            vario = resolve_variogram_for_variable(self.registry, selected_var, self)
            if vario is None:
                # Fall back to cached results (e.g. set via set_variogram_results)
                vario = self.variogram_results
            if vario is None:
                logger.info("No variogram results in registry")
                return

            self.variogram_results = vario
            model = vario.get("combined_3d_model", {})

            if model.get("nugget") is not None:
                self.nugget_spin.setValue(model["nugget"])
            if model.get("major_range") is not None:
                self.range_max_spin.setValue(model["major_range"])
            if model.get("minor_range") is not None:
                self.range_mid_spin.setValue(model["minor_range"])
            elif model.get("major_range") is not None:
                self.range_mid_spin.setValue(model["major_range"])
            if model.get("vertical_range") is not None:
                self.range_min_spin.setValue(model["vertical_range"])
            elif model.get("major_range") is not None:
                self.range_min_spin.setValue(model["major_range"])

            # Azimuth: prefer combined_3d_model (has swap-corrected azimuth),
            # fallback to top-level major_azimuth
            if model.get("azimuth") is not None:
                self.azimuth_spin.setValue(model["azimuth"])
            elif vario.get("major_azimuth") is not None:
                self.azimuth_spin.setValue(vario["major_azimuth"])
            if model.get("dip") is not None:
                self.dip_spin.setValue(model["dip"])
            elif vario.get("major_dip") is not None:
                self.dip_spin.setValue(vario["major_dip"])

            # Sill: ARBF expects partial sill (C1, excluding nugget).
            nugget_val = model.get("nugget", 0.0) or 0.0
            if model.get("total_sill") is not None:
                partial_sill = max(model["total_sill"] - nugget_val, 0.0)
                self.sill_spin.setValue(partial_sill)
                logger.info(
                    "Variogram import: total_sill=%.4f - nugget=%.4f = C1=%.4f",
                    model["total_sill"], nugget_val, partial_sill,
                )
            elif model.get("sill") is not None:
                self.sill_spin.setValue(max(model["sill"], 0.0))
                logger.info(
                    "Variogram import: sill (already partial)=%.4f",
                    model["sill"],
                )

            if set_mode:
                self.variogram_mode_combo.setCurrentText("Global")

            logger.info(
                "Variogram imported: nugget=%.4f, sill=%.4f, "
                "ranges=(%.1f, %.1f, %.1f), azimuth=%.1f, dip=%.1f",
                nugget_val, self.sill_spin.value(),
                self.range_max_spin.value(),
                self.range_mid_spin.value(),
                self.range_min_spin.value(),
                self.azimuth_spin.value(),
                self.dip_spin.value(),
            )
        except Exception as exc:
            logger.warning("Failed to import variogram: %s", exc)

        self._update_domain_variogram_status()

    def _update_domain_variogram_status(self) -> None:
        """Show which domains have per-domain variograms in the registry."""
        label = self.domain_vario_status
        if not self.registry or not hasattr(self, "variable_combo"):
            label.setText("")
            return

        selected_var = self.variable_combo.currentText()
        if not selected_var:
            label.setText("")
            return

        if not hasattr(self.registry, "list_variogram_domains"):
            label.setText("")
            return

        stored_domains = self.registry.list_variogram_domains(selected_var)
        if not stored_domains:
            label.setText("No variograms stored")
            label.setStyleSheet("font-size: 9px; color: grey;")
            return

        # Build a summary
        parts = []
        for d in stored_domains:
            parts.append(d)
        label.setText(", ".join(parts))
        label.setStyleSheet("font-size: 9px; color: green;")

    def _gather_domain_variograms(self) -> Dict[str, Any]:
        """Collect per-domain variograms from the registry for gather_parameters."""
        if not self.registry or not hasattr(self, "variable_combo"):
            return {}
        selected_var = self.variable_combo.currentText()
        if not selected_var:
            return {}
        from .panel_utils import resolve_all_domain_variograms
        return resolve_all_domain_variograms(self.registry, selected_var)

    def _on_auto_detect_grid(self) -> None:
        """Set grid extents from drillhole data."""
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return
        df = filtered_df
        try:
            for cx, cy, cz in [("X", "Y", "Z"), ("x", "y", "z"), ("EAST", "NORTH", "RL")]:
                if cx in df.columns and cy in df.columns and cz in df.columns:
                    x = df[cx].dropna().values
                    y = df[cy].dropna().values
                    z = df[cz].dropna().values
                    break
            else:
                return

            dx, dy, dz = self.grid_dx.value(), self.grid_dy.value(), self.grid_dz.value()
            x0, y0, z0 = x.min() - dx, y.min() - dy, z.min() - dz
            nx = int(np.ceil((x.max() - x0 + dx) / dx))
            ny = int(np.ceil((y.max() - y0 + dy) / dy))
            nz = int(np.ceil((z.max() - z0 + dz) / dz))

            self.grid_x0.setValue(x0)
            self.grid_y0.setValue(y0)
            self.grid_z0.setValue(z0)
            self.grid_nx.setValue(min(nx, 1000))
            self.grid_ny.setValue(min(ny, 1000))
            self.grid_nz.setValue(min(nz, 500))
        except Exception as exc:
            logger.warning("Auto-detect grid failed: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # AUTO-RECOMMEND
    # ══════════════════════════════════════════════════════════════

    def _on_auto_recommend(self) -> None:
        """Analyse loaded data and display recommended settings."""
        coords, values = self._extract_coords_and_values()
        if coords is None:
            self.auto_recommend_info.setPlainText(
                "No data loaded. Load drillhole data first, then "
                "select a variable."
            )
            return

        try:
            from geostats.arbf.recommend import recommend_arbf_settings

            # Get block centroids if a block model is loaded
            block_centroids = None
            block_sizes = None
            ctrl = getattr(self, "controller", None)
            if ctrl is not None:
                bm = getattr(ctrl, "block_model", None)
                if bm is not None:
                    try:
                        grid_spec = {
                            "dx": self.grid_dx.value(),
                            "dy": self.grid_dy.value(),
                            "dz": self.grid_dz.value(),
                        }
                        block_centroids, block_sizes = (
                            ctrl._extract_bm_centroids_and_sizes(bm, grid_spec)
                        )
                    except Exception:
                        pass

            rec = recommend_arbf_settings(
                coords, values, block_centroids, block_sizes,
            )
            self._last_recommendation = rec
            self.auto_recommend_info.setPlainText(rec.summary_text())

        except Exception as exc:
            logger.warning("Auto-recommend failed: %s", exc)
            self.auto_recommend_info.setPlainText(
                f"Analysis failed: {exc}"
            )

    def _on_apply_recommendations(self) -> None:
        """Apply the last computed recommendations to panel controls."""
        rec = getattr(self, "_last_recommendation", None)
        if rec is None or not rec.settings:
            self.auto_recommend_info.setPlainText(
                "No recommendations to apply. Click 'Analyse Data & "
                "Recommend Settings' first."
            )
            return

        s = rec.settings

        # Data transforms
        if "use_normal_score" in s:
            self.chk_normal_score.setChecked(s["use_normal_score"])

        # Kernel
        kernel_map_rev = {
            "spheroidal": "Spheroidal", "gaussian": "Gaussian",
            "matern_32": "Matern 3/2", "matern_52": "Matern 5/2",
            "cubic": "Cubic", "wendland_c2": "Wendland C2",
            "spherical": "Spherical",
        }
        if "kernel_type" in s:
            text = kernel_map_rev.get(s["kernel_type"], "Spheroidal")
            idx = self.kernel_combo.findText(text)
            if idx >= 0:
                self.kernel_combo.setCurrentIndex(idx)

        if "alpha" in s:
            self.alpha_spin.setValue(s["alpha"])

        # Variogram parameters
        if "sill" in s:
            self.sill_spin.setValue(s["sill"])
        if "nugget" in s:
            self.nugget_spin.setValue(s["nugget"])
        if "accuracy" in s:
            self.accuracy_spin.setValue(s["accuracy"])

        # Drift
        drift_map_rev = {"constant": "Constant", "linear": "Linear", "none": "None"}
        if "drift_type" in s:
            text = drift_map_rev.get(s["drift_type"], "Constant")
            idx = self.drift_combo.findText(text)
            if idx >= 0:
                self.drift_combo.setCurrentIndex(idx)

        # Variogram mode
        if "variogram_mode" in s:
            text = s["variogram_mode"].capitalize()
            idx = self.variogram_mode_combo.findText(text)
            if idx >= 0:
                self.variogram_mode_combo.setCurrentIndex(idx)

        # Ranges
        if "range_max" in s:
            self.range_max_spin.setValue(s["range_max"])
        if "range_mid" in s:
            self.range_mid_spin.setValue(s["range_mid"])
        if "range_min" in s:
            self.range_min_spin.setValue(s["range_min"])

        # Search
        if "max_samples" in s:
            self.max_samples_spin.setValue(s["max_samples"])
        if "min_samples" in s:
            self.min_samples_spin.setValue(s["min_samples"])

        # Execution
        if "run_cv" in s:
            self.chk_cv.setChecked(s["run_cv"])
        if "change_of_support" in s:
            self.chk_cos.setChecked(s["change_of_support"])
        if "discretisation_density" in s:
            density_map_rev = {8: "8 (2x2x2)", 27: "27 (3x3x3)", 64: "64 (4x4x4)"}
            text = density_map_rev.get(s["discretisation_density"], "27 (3x3x3)")
            idx = self.disc_density_combo.findText(text)
            if idx >= 0:
                self.disc_density_combo.setCurrentIndex(idx)

        # Footprint clipping
        if "clip_to_drill_footprint" in s:
            self.chk_clip_to_footprint.setChecked(s["clip_to_drill_footprint"])
        if "footprint_buffer_ranges" in s:
            self.footprint_buffer_spin.setValue(s["footprint_buffer_ranges"])

        # Grade clipping
        if "clip_min" in s:
            self.chk_clip.setChecked(True)
            self.clip_min_spin.setValue(s["clip_min"])
        if "clip_max" in s:
            self.chk_clip.setChecked(True)
            self.clip_max_spin.setValue(s["clip_max"])

        self.auto_recommend_info.append(
            "\n--- Settings applied to panel controls ---"
        )

    def _extract_coords_and_values(self):
        """Extract coordinate array and grade values from loaded data.

        Returns (coords, values) or (None, None) if data is unavailable.
        """
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return None, None

        variable = self.variable_combo.currentText()
        if not variable or variable not in filtered_df.columns:
            return None, None

        df = filtered_df
        coord_cols = None
        for cx, cy, cz in [
            ("X", "Y", "Z"), ("x", "y", "z"), ("EAST", "NORTH", "RL"),
            ("MIDX", "MIDY", "MIDZ"), ("XC", "YC", "ZC"),
        ]:
            if cx in df.columns and cy in df.columns and cz in df.columns:
                coord_cols = [cx, cy, cz]
                break

        if coord_cols is None:
            return None, None

        cleaned = df.dropna(subset=coord_cols + [variable])
        if cleaned.empty:
            return None, None

        coords = cleaned[coord_cols].to_numpy(float)
        values = cleaned[variable].to_numpy(float)
        return coords, values

    # ══════════════════════════════════════════════════════════════
    # UI HANDLERS
    # ══════════════════════════════════════════════════════════════

    def _on_single_domain_toggled(self, checked: bool) -> None:
        """Enable/disable PUM controls when single-domain is toggled."""
        for w in (self.n_subdomains_spin, self.sd_method_combo,
                  self.overlap_spin, self.max_samples_spin):
            w.setEnabled(not checked)

    # ══════════════════════════════════════════════════════════════
    # EXECUTION
    # ══════════════════════════════════════════════════════════════

    def _on_run(self) -> None:
        """Validate and dispatch ARBF estimation."""
        if not self.validate_inputs():
            return
        self.run_analysis()

    def validate_inputs(self) -> bool:
        """Check data and variable are available."""
        if self.drillhole_data is None or self.drillhole_data.empty:
            self._on_load_data()
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            self.status_label.setText("No data loaded")
            return False
        if not self.variable_combo.currentText():
            self.status_label.setText("No variable selected")
            return False
        # Check drillhole coverage within estimation grid
        from .base_analysis_panel import check_drillhole_grid_coverage
        if not check_drillhole_grid_coverage(
            self, filtered_df,
            xmin=self.grid_x0.value(), ymin=self.grid_y0.value(), zmin=self.grid_z0.value(),
            nx=self.grid_nx.value(), ny=self.grid_ny.value(), nz=self.grid_nz.value(),
            dx=self.grid_dx.value(), dy=self.grid_dy.value(), dz=self.grid_dz.value(),
            panel_name="ARBF",
        ):
            return False
        return True

    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters for ARBF estimation."""
        filtered_df = self._get_filtered_data()
        variable = self.variable_combo.currentText()
        domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}

        kernel_map = {
            "Spheroidal": "spheroidal", "Gaussian": "gaussian",
            "Exponential": "exponential", "Spherical": "spherical",
        }
        rbf_basis_map = {
            "Wendland C2": "wendland_c2", "Wendland C4": "wendland_c4",
            "Gaussian RBF": "gaussian", "Multiquadric": "multiquadric",
            "Inverse Multiquadric": "inverse_multiquadric",
        }
        drift_map = {"Constant": "constant", "Linear": "linear", "None": "none"}
        disc_density_map = {"8 (2x2x2)": 8, "27 (3x3x3)": 27, "64 (4x4x4)": 64}
        lva_source_map = {
            "Data-Driven (Boisvert 2009)": "data",
            "Structural Measurements": "structural",
            "Identity (None)": "identity",
        }

        return {
            "data": filtered_df,
            "unfiltered_data": self.drillhole_data,
            "variable": variable,
            "domain_column": domain_meta.get("domain_filter_column"),
            "domain_value": domain_meta.get("domain_filter_value"),
            "grid_spec": {
                "nx": self.grid_nx.value(), "ny": self.grid_ny.value(),
                "nz": self.grid_nz.value(), "dx": self.grid_dx.value(),
                "dy": self.grid_dy.value(), "dz": self.grid_dz.value(),
                "x0": self.grid_x0.value(), "y0": self.grid_y0.value(),
                "z0": self.grid_z0.value(),
            },
            "use_block_model_grid": self.chk_use_block_model.isChecked(),
            "clip_to_drill_footprint": self.chk_clip_to_footprint.isChecked(),
            "footprint_buffer_ranges": self.footprint_buffer_spin.value(),

            # RBF Basis Function (primary model)
            "rbf_basis_type": rbf_basis_map.get(
                self.rbf_basis_combo.currentText(), "wendland_c2"),
            "rbf_shape_parameter": self.shape_param_spin.value(),
            "rbf_support_radius": self.support_radius_spin.value(),

            # Variogram guidance (for anisotropy inference and NS back-transform)
            "kernel_type": kernel_map.get(self.kernel_combo.currentText(), "spheroidal"),
            "alpha": self.alpha_spin.value(),
            "nugget": self.nugget_spin.value(),
            "accuracy": self.accuracy_spin.value(),
            "drift_type": drift_map.get(self.drift_combo.currentText(), "constant"),

            # Anisotropy
            "range_max": self.range_max_spin.value(),
            "range_mid": self.range_mid_spin.value(),
            "range_min": self.range_min_spin.value(),
            "azimuth": self.azimuth_spin.value(),
            "dip": self.dip_spin.value(),
            "pitch": self.pitch_spin.value(),

            # Sub-domains
            "n_subdomains": (
                1 if self.chk_single_domain.isChecked()
                else self.n_subdomains_spin.value()
            ),
            "pum_threshold": self.pum_threshold_spin.value(),
            "subdomain_method": (
                "kmeans" if self.sd_method_combo.currentIndex() == 0 else "manual"
            ),
            "overlap_factor": self.overlap_spin.value(),
            "max_samples": self.max_samples_spin.value(),
            "min_samples": self.min_samples_spin.value(),

            # LVA
            "use_lva": self.chk_lva.isChecked(),
            "lva_source": lva_source_map.get(
                self.lva_source_combo.currentText(), "data",
            ),

            # Transforms
            "use_normal_score": self.chk_normal_score.isChecked(),

            # Discretisation
            "discretisation": (
                "adaptive" if self.disc_mode_combo.currentText() == "Adaptive"
                else "fixed"
            ),
            "discretisation_density": disc_density_map.get(
                self.disc_density_combo.currentText(), 27,
            ),

            # Execution
            "run_cv": self.chk_cv.isChecked(),
            "change_of_support": self.chk_cos.isChecked(),
            "operator": self.operator_edit.text(),
            "seed": self.seed_spin.value(),

            # Clipping
            "clip_min": (
                self.clip_min_spin.value() if self.chk_clip.isChecked() else None
            ),
            "clip_max": (
                self.clip_max_spin.value() if self.chk_clip.isChecked() else None
            ),

            # Variogram
            "variogram_mode": self.variogram_mode_combo.currentText().lower(),
            "sill": self.sill_spin.value(),
            "variogram_results": self.variogram_results,
            # Per-domain variograms (looked up from registry)
            "domain_variograms": self._gather_domain_variograms(),

            # JORC compliance — new in Phase 3 audit
            "rotation_convention": {
                "GeoX (native)": "geox",
                "Leapfrog / MICROMINE": "leapfrog",
                "Datamine Studio": "datamine",
                "Vulcan": "vulcan",
                "Surpac": "surpac",
            }.get(self.rotation_convention_combo.currentText(), "geox"),
            "use_geodesic": self.chk_geodesic.isChecked(),
            "decluster_cell_size": self.decluster_cell_size_spin.value(),
        }

    # ══════════════════════════════════════════════════════════════
    # RESULT HANDLING
    # ══════════════════════════════════════════════════════════════

    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle ARBF estimation results from controller -- push mesh to 3D renderer."""
        try:
            # Hide progress, re-enable run
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
            if hasattr(self, 'run_btn'):
                self.run_btn.setEnabled(True)
                self.run_btn.setText("Run ARBF Estimation")

            vis = payload.get("visualization", {})
            mesh = vis.get("mesh")
            if mesh is None:
                logger.warning("on_results: no mesh in payload")
                return

            property_name = payload.get("property_name", "ARBF_estimate")
            layer_name = vis.get("layer_name", "ARBF Result")

            # Walk hierarchy to find renderer
            widget = self
            renderer = None
            for _ in range(20):
                widget = widget.parent()
                if widget is None:
                    break
                if hasattr(widget, 'renderer'):
                    renderer = widget.renderer
                    break
                if hasattr(widget, 'viewer_widget') and hasattr(widget.viewer_widget, 'renderer'):
                    renderer = widget.viewer_widget.renderer
                    break

            if renderer is None:
                logger.warning("on_results: renderer not found")
                return

            renderer.add_block_model_layer(
                mesh, property_name=property_name, layer_name=layer_name,
                source={
                    'kind': 'block_model_layer',
                    'registry_key': 'arbf_results',
                    'mesh_path': 'visualization.mesh',
                    'property_name': property_name,
                    'layer_name': layer_name,
                },
            )

            # Store results
            self.arbf_results = payload

            # Populate results tabs if available
            if hasattr(self, '_populate_results'):
                try:
                    self._populate_results(payload)
                except Exception as exc:
                    logger.debug("_populate_results failed: %s", exc)

            metadata = payload.get("metadata", {})
            msg = metadata.get("message", f"ARBF results loaded: {property_name}")
            if hasattr(self, 'status_label'):
                self.status_label.setText(msg)
            logger.info("on_results: rendered '%s' as '%s'", property_name, layer_name)
        except Exception as e:
            logger.error("on_results failed: %s", e, exc_info=True)

    def handle_results(self, payload: Dict[str, Any]) -> None:
        """Callback from controller (BaseAnalysisPanel interface)."""
        self.on_results(payload)

    def _on_progress(self, percent: int, message: str) -> None:
        """Update progress from engine."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)
        self.status_label.setText(f"{percent}% — {message}")

    # ══════════════════════════════════════════════════════════════
    # RESULTS POPULATION
    # ══════════════════════════════════════════════════════════════

    def _populate_results(self, payload: Dict[str, Any]) -> None:
        """Fill Results tab with estimation summary."""
        metadata = payload.get("metadata", {})
        diagnostics = payload.get("diagnostics", {})
        grades = payload.get("grades")
        variances = payload.get("variances")

        _var = metadata.get('variable', '?')
        from ..utils.unit_utils import var_label as _vl
        _var_label = _vl(self.registry, _var)

        lines = [
            "<h3>ARBF Estimation Results</h3>",
            f"<b>Method:</b> {metadata.get('method', 'ARBF')}",
            f"<b>Variable:</b> {_var_label}",
            f"<b>Samples:</b> {metadata.get('n_samples', '?')}",
            f"<b>Blocks Estimated:</b> {diagnostics.get('n_blocks_estimated', '?')} / "
            f"{diagnostics.get('n_blocks_total', '?')}",
            f"<b>Mode:</b> {'Single Domain' if diagnostics.get('single_domain') else 'PUM (%s sub-domains)' % diagnostics.get('n_subdomains', '?')}",
            f"<b>Elapsed:</b> {diagnostics.get('elapsed_seconds', 0):.1f}s",
        ]

        if grades is not None:
            n_nan = int(np.sum(np.isnan(grades)))
            n_finite = int(np.sum(np.isfinite(grades)))
            lines.extend([
                "",
                "<b>Grade Statistics:</b>",
                f"  Mean: {np.nanmean(grades):.4f}",
                f"  Median: {np.nanmedian(grades):.4f}",
                f"  Std Dev: {np.nanstd(grades):.4f}",
                f"  Min: {np.nanmin(grades):.4f}",
                f"  Max: {np.nanmax(grades):.4f}",
            ])
            if n_nan > 0:
                lines.append(
                    f"  Masked (NaN): {n_nan} / {len(grades)} "
                    f"({100.0 * n_nan / max(len(grades), 1):.1f}%)"
                )

        if variances is not None:
            lines.extend([
                "",
                "<b>Variance Statistics:</b>",
                f"  Mean: {np.nanmean(variances):.6f}",
                f"  Max: {np.nanmax(variances):.6f}",
                f"  Zero-variance blocks: {int(np.sum(variances == 0))}",
            ])

        self.results_text.setHtml("<br>".join(lines))

        # Diagnostics table
        diag_items = list(diagnostics.items())
        self.diag_table.setRowCount(len(diag_items))
        for i, (key, val) in enumerate(diag_items):
            self.diag_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.diag_table.setItem(
                i, 1, QTableWidgetItem(
                    f"{val:.4f}" if isinstance(val, float) else str(val),
                ),
            )

    @staticmethod
    def _get_cv_metric(cv_result, key, default=None):
        """Retrieve a metric from *cv_result* whether it is a dict or object."""
        if isinstance(cv_result, dict):
            return cv_result.get(key, default)
        return getattr(cv_result, key, default)

    def _populate_validation(self, payload: Dict[str, Any]) -> None:
        """Fill Validation tab with CV results and plots."""
        cv_result = payload.get("cv_result")
        if cv_result is None:
            return

        _m = lambda key, fmt, default="N/A": (  # noqa: E731
            fmt.format(v) if (v := self._get_cv_metric(cv_result, key)) is not None else default
        )
        metrics = [
            ("Mean Error (ME)", _m("mean_error", "{:.6f}")),
            ("MAE", _m("mae", "{:.6f}")),
            ("RMSE", _m("rmse", "{:.6f}")),
            ("R²", _m("r_squared", "{:.4f}")),
            ("Correlation", _m("correlation", "{:.4f}")),
            ("Normalised RMSE", _m("normalised_rmse", "{:.4f}")),
            ("Slope of Regression", _m("slope_of_regression", "{:.4f}")),
            ("Intercept", _m("intercept", "{:.4f}")),
            ("N Samples (CV)", _m("n_samples", "{}")),
        ]
        # Add per-domain breakdown if available
        per_domain = (
            cv_result.get("per_domain") if isinstance(cv_result, dict) else None
        )
        if per_domain and isinstance(per_domain, dict):
            metrics.append(("", ""))  # spacer
            metrics.append(("── Per-Domain CV ──", ""))
            for dkey, dcv in per_domain.items():
                r2 = dcv.get("R2", float("nan"))
                rmse = dcv.get("RMSE", float("nan"))
                slope = dcv.get("SLOPE", float("nan"))
                n = len(dcv.get("actual", []))
                metrics.append((
                    f"  {dkey}",
                    f"R²={r2:.3f}  RMSE={rmse:.4f}  Slope={slope:.3f}  (n={n})",
                ))

        self.cv_table.setRowCount(len(metrics))
        for i, (name, val) in enumerate(metrics):
            self.cv_table.setItem(i, 0, QTableWidgetItem(name))
            self.cv_table.setItem(i, 1, QTableWidgetItem(val))

        if self.scatter_canvas is not None and hasattr(cv_result, "actual"):
            self._plot_cv_scatter(cv_result)

        swath_data = payload.get("swath_data")
        if swath_data and self.swath_canvas is not None:
            self._swath_data = swath_data
            self._on_swath_axis_changed()

    def _plot_cv_scatter(self, cv_result) -> None:
        """Plot actual vs estimated cross-validation scatter."""
        if self.scatter_canvas is None:
            return
        self.scatter_canvas.fig.clear()
        ax = self.scatter_canvas.fig.add_subplot(111)
        self.scatter_canvas._apply_theme(ax)

        actual = cv_result.actual
        estimated = cv_result.estimated
        ax.scatter(estimated, actual, s=10, alpha=0.5, c="#4FC3F7", edgecolors="none")

        lo = min(actual.min(), estimated.min())
        hi = max(actual.max(), estimated.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="1:1 line")

        slope = cv_result.slope_of_regression
        intercept = cv_result.intercept
        ax.plot(
            [lo, hi], [intercept + slope * lo, intercept + slope * hi],
            color="#FFB74D", linewidth=1.5, label=f"Slope={slope:.3f}",
        )

        ax.set_xlabel("Estimated", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title("LOO-CV: Actual vs Estimated", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7)
        try:
            self.scatter_canvas.fig.tight_layout()
        except Exception:
            pass
        self.scatter_canvas._safe_draw()

    def _on_swath_axis_changed(self) -> None:
        """Update swath plot for selected axis."""
        if self._swath_data is None or self.swath_canvas is None:
            return

        axis = self.swath_axis_combo.currentText().lower()
        data = self._swath_data.get(axis)
        if data is None:
            return

        self.swath_canvas.fig.clear()
        ax = self.swath_canvas.fig.add_subplot(111)
        self.swath_canvas._apply_theme(ax)

        ax.plot(
            data.slice_positions, data.mean_estimated, "o-",
            color="#4FC3F7", markersize=4, label="Estimated (block mean)",
        )
        ax.plot(
            data.slice_positions, data.mean_actual, "s-",
            color="#FFB74D", markersize=4, label="Actual (composite mean)",
        )

        ax.set_xlabel(f"{axis.upper()} coordinate", fontsize=9)
        ax.set_ylabel("Mean Grade", fontsize=9)
        ax.set_title(
            f"Swath Plot — {axis.upper()} axis", fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=7)
        try:
            self.swath_canvas.fig.tight_layout()
        except Exception:
            pass
        self.swath_canvas._safe_draw()

    def _populate_export(self, payload: Dict[str, Any]) -> None:
        """Fill Export tab with JORC audit record."""
        audit = payload.get("audit_record")
        if audit is None:
            return
        try:
            self.audit_text.setPlainText(audit.to_jorc_table1_section3())
        except Exception as exc:
            self.audit_text.setPlainText(f"Error generating JORC report: {exc}")

    # ══════════════════════════════════════════════════════════════
    # RESULT REGISTRATION
    # ══════════════════════════════════════════════════════════════

    def _on_register_results(self) -> None:
        """Register ARBF results to data registry."""
        if self.arbf_results is None or self.registry is None:
            return

        try:
            payload = self.arbf_results
            grid = payload.get("grid")
            metadata = payload.get("metadata", {})
            variable = metadata.get("variable", "grade")

            if grid is None:
                return

            x_c = payload.get("x_coords")
            y_c = payload.get("y_coords")
            z_c = payload.get("z_coords")
            grades = payload.get("grades")
            variances = payload.get("variances")

            arbf_reg = {
                "grid": grid,
                "metadata": metadata,
                "variable": variable,
                "property_name": f"ARBF_{variable}",
                "variance_property": f"ARBF_{variable}_var",
                "grades": grades,
                "variances": variances,
                "x_coords": x_c,
                "y_coords": y_c,
                "z_coords": z_c,
                "grid_values": grades,
                "block_sizes": payload.get("block_sizes"),
                "classifications": payload.get("classifications"),
                "fail_flags": payload.get("fail_flags"),
                "neff": payload.get("neff"),
                "uncertainty": payload.get("uncertainty"),
                "audit_record": payload.get("audit_record"),
            }
            # Use generic register_results to fire arbfResultsLoaded signal
            self.registry.register_results(
                "arbf_results", arbf_reg, source_panel="ARBF",
                metadata=metadata,
            )

            if x_c is not None and y_c is not None and z_c is not None:
                x_c = np.asarray(x_c)
                y_c = np.asarray(y_c)
                z_c = np.asarray(z_c)
                n_grades = grades.size if grades is not None else 0
                meshgrid_count = len(x_c) * len(y_c) * len(z_c)

                if meshgrid_count > 0 and meshgrid_count == n_grades:
                    # Coords are axis ticks — expand via meshgrid
                    GX, GY, GZ = np.meshgrid(x_c, y_c, z_c, indexing="ij")
                    x_flat, y_flat, z_flat = GX.ravel(), GY.ravel(), GZ.ravel()
                else:
                    # Coords are already per-block (same length as grades)
                    x_flat = x_c.ravel()
                    y_flat = y_c.ravel()
                    z_flat = z_c.ravel()

                grade_flat = grades.ravel() if grades is not None else np.full(len(x_flat), np.nan)
                block_df = pd.DataFrame({
                    "X": x_flat, "Y": y_flat, "Z": z_flat,
                    f"ARBF_{variable}": grade_flat,
                })
                if variances is not None:
                    block_df[f"ARBF_{variable}_var"] = np.asarray(variances).ravel()
                self.registry.register_block_model_generated(
                    block_df,
                    source_panel="ARBF",
                    metadata=metadata,
                )

            self.status_label.setText("Results registered to registry")
            logger.info("ARBF results registered to data registry")
        except Exception as exc:
            logger.error("Failed to register ARBF results: %s", exc)
            self.status_label.setText(f"Registration failed: {exc}")

    # ══════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════

    def _on_export_jorc(self) -> None:
        """Export JORC Table 1 Section 3 report."""
        if self.arbf_results is None:
            return
        audit = self.arbf_results.get("audit_record")
        if audit is None:
            return

        def _write(path: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(audit.to_jorc_table1_section3())

        _export_file(
            self, "Export JORC Report", "ARBF_JORC_Table1.txt",
            "Text Files (*.txt);;All Files (*)", _write, self.status_label,
        )

    def _on_export_audit_json(self) -> None:
        """Export audit record as JSON."""
        if self.arbf_results is None:
            return
        audit = self.arbf_results.get("audit_record")
        if audit is None:
            return

        def _write(path: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(audit.to_json())

        _export_file(
            self, "Export Audit JSON", "ARBF_audit_record.json",
            "JSON Files (*.json);;All Files (*)", _write, self.status_label,
        )

    def _on_export_csv(self) -> None:
        """Export estimation results to CSV."""
        if self.arbf_results is None:
            return

        def _write(path: str) -> None:
            grades = np.asarray(
                self.arbf_results.get("grades", np.array([]))
            ).ravel()
            variances = np.asarray(
                self.arbf_results.get("variances", np.array([]))
            ).ravel()

            df = pd.DataFrame({
                "ARBF_Grade": grades,
                "ARBF_Variance": variances,
            })

            x_c = self.arbf_results.get("x_coords")
            y_c = self.arbf_results.get("y_coords")
            z_c = self.arbf_results.get("z_coords")
            if x_c is not None and y_c is not None and z_c is not None:
                GX, GY, GZ = np.meshgrid(x_c, y_c, z_c, indexing="ij")
                df.insert(0, "Z", GZ.ravel())
                df.insert(0, "Y", GY.ravel())
                df.insert(0, "X", GX.ravel())

            df.to_csv(path, index=False)

        _export_file(
            self, "Export Results CSV", "ARBF_results.csv",
            "CSV Files (*.csv);;All Files (*)", _write, self.status_label,
        )

    # ══════════════════════════════════════════════════════════════
    # PUBLIC SETTERS
    # ══════════════════════════════════════════════════════════════

    def set_drillhole_data(self, data: pd.DataFrame) -> None:
        """Set drillhole data externally."""
        if not self._ui_ready:
            self._pending_drillhole_data = data
            return
        self.drillhole_data = data
        self._refresh_variable_list(data)
        n = len(data) if data is not None else 0
        self.data_status.setText(f"{n} samples loaded")

    def set_variogram_results(self, results: Dict[str, Any]) -> None:
        """Set variogram results externally."""
        self.variogram_results = results
        if self._ui_ready:
            self._on_import_variogram()

    def get_registry(self):
        """Get DataRegistry from parent hierarchy."""
        if self.registry is not None:
            return self.registry
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "registry"):
                self.registry = parent.registry
                return self.registry
            if hasattr(parent, "data_registry"):
                self.registry = parent.data_registry
                return self.registry
            parent = parent.parent()
        try:
            from ..core.data_registry import DataRegistry
            self.registry = DataRegistry.instance()
            return self.registry
        except Exception:
            return None

    def refresh_theme(self) -> None:
        """No-op. Application-level QSS handles theming."""
        pass
