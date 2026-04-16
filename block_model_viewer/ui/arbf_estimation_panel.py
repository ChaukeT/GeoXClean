"""
ARBF Estimation Panel v2 — Guided Workflow with Live Validation.

Replaces the old tabbed ARBFPanel with a split-view design:
  Left:  Scrollable parameter sections (numbered, collapsible, validated)
  Right: Context-sensitive results & diagnostics

Design principles:
  1. Guided workflow — sections are numbered 1→6, expand sequentially.
  2. Live validation — parameter issues shown inline as you type.
  3. Everything visible — no hidden tabs, scrollable single page.
  4. Prominent warnings — anisotropy, nugget ratio, coverage shown prominently.
  5. Results alongside parameters — swath, CV, diagnostics always visible.
"""

from __future__ import annotations

import logging
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .base_analysis_panel import BaseAnalysisPanel
from .collapsible_group import CollapsibleGroup
from .modern_styles import ModernColors
from .mixins.coded_domain_filter_mixin import CodedDomainFilterMixin
from .mixins.domain_mask_mixin import DomainMaskMixin
from .panel_toolkit import (
    action_button,
    form_row,
    hint_label,
    info_display,
    make_combo,
    make_form,
    make_spin,
    section,
    separator,
    PANEL_MARGINS,
    PANEL_SPACING,
)
from .panel_manager import PanelCategory, DockArea
from ..utils.variable_utils import get_grade_columns
from .panel_utils import resolve_variogram_for_variable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Status indicator widget
# ═══════════════════════════════════════════════════════════════════


class _StatusDot(QLabel):
    """Small coloured dot showing validation status."""

    _COLORS = {
        "ok": "#27AE60",
        "warn": "#F39C12",
        "error": "#E74C3C",
        "none": "#BDC3C7",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.set_status("none")

    def set_status(self, status: str, tooltip: str = ""):
        color = self._COLORS.get(status, self._COLORS["none"])
        self.setStyleSheet(
            f"background-color: {color}; border-radius: 6px; "
            f"min-width: 12px; max-width: 12px; "
            f"min-height: 12px; max-height: 12px;"
        )
        self.setToolTip(tooltip)


class _WarningBanner(QFrame):
    """Prominent inline warning banner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("warningBanner")
        self.setStyleSheet(
            "QFrame#warningBanner { "
            "  background-color: #FFF3CD; border: 1px solid #FFEEBA; "
            "  border-radius: 6px; padding: 6px 10px; "
            "}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        self._icon = QLabel("⚠")
        self._icon.setStyleSheet("font-size: 14pt; color: #856404;")
        lay.addWidget(self._icon)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setStyleSheet("color: #856404; font-size: 9pt;")
        lay.addWidget(self._text, 1)
        self.hide()

    def show_warning(self, text: str):
        self._text.setText(text)
        self.show()

    def clear(self):
        self.hide()


# ═══════════════════════════════════════════════════════════════════
# Numbered section header
# ═══════════════════════════════════════════════════════════════════


def _numbered_section(number: int, title: str, collapsed: bool = False) -> CollapsibleGroup:
    """Create a collapsible section with a step number prefix."""
    return section(f"Step {number}: {title}", collapsed=collapsed)


# ═══════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════


class ARBFEstimationPanel(CodedDomainFilterMixin, DomainMaskMixin, BaseAnalysisPanel):
    """ARBF Estimation Panel v2 — Guided workflow with split-view layout.

    Left panel:  Numbered collapsible sections for parameters.
    Right panel: Tabbed results area (Results, Swath, CV, Log).
    Bottom:      Run button + progress bar.
    """

    task_name = "arbf"
    panel_title = "ARBF Estimation"
    request_visualization = pyqtSignal(object, str)
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None, controller=None, **kwargs):
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.transformation_metadata: Optional[Dict[str, Any]] = None
        self.variogram_results: Optional[Dict] = None
        self._variogram_structures: Optional[list] = None
        self.arbf_results: Optional[Dict] = None
        self._geostatistical_gate: Optional[Dict[str, Any]] = None
        self._last_recommendation = None
        self._ui_ready = False
        self._pending_drillhole_data = None
        self._cached_block_df: Optional[pd.DataFrame] = None
        self._cv_fig = None
        self._swath_fig = None

        # BaseAnalysisPanel.__init__ calls _setup_base_ui() → setup_ui()
        super().__init__(parent=parent, panel_id=kwargs.get("panel_id"))

        if controller is not None:
            self.bind_controller(controller)

        self._ui_ready = True
        self._init_registry_connections()
        self._connect_grid_sync()
        self._process_pending_data()

    def showEvent(self, event):
        """Re-import variogram from registry whenever the panel is shown."""
        super().showEvent(event)
        if self._ui_ready and self.variogram_results is None:
            self._on_import_variogram()

    # ══════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════

    def setup_ui(self):
        """Build the complete split-view UI (called by BaseAnalysisPanel)."""
        # ── Splitter: Left (workflow tabs) | Right (results tabs) ──
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter, 1)

        # LEFT: Workflow steps in a tabbed view
        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(0)
        left_container.setMinimumWidth(440)

        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Build each step as a scrollable tab
        self._add_workflow_tab("1. Data", self._build_step1_data)
        self._add_workflow_tab("2. Variogram", self._build_step2_variogram)
        self._add_workflow_tab("3. Grid", self._build_step3_grid)
        self._add_workflow_tab("4. Estimation", self._build_step4_estimation)
        self._add_workflow_tab("5. Run", self._build_step5_execution)
        self._add_workflow_tab("6. Export", self._build_step6_export)

        left_vbox.addWidget(self.workflow_tabs)
        self.splitter.addWidget(left_container)

        # RIGHT: Results tabs
        self._build_right_panel()

        self.splitter.setSizes([480, 520])

    def _add_workflow_tab(self, label: str, builder_fn):
        """Create a scrollable tab, set self._left_layout, call the builder."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        self._left_layout = QVBoxLayout(content)
        self._left_layout.setContentsMargins(*PANEL_MARGINS)
        self._left_layout.setSpacing(PANEL_SPACING)

        builder_fn()
        self._left_layout.addStretch()

        scroll.setWidget(content)
        self.workflow_tabs.addTab(scroll, label)

    # ──────────────────────────────────────────────────────────────
    # Step 1: Data
    # ──────────────────────────────────────────────────────────────

    def _build_step1_data(self):
        grp = section("Input Data")
        f = make_form()

        # Source selector
        self.source_combo = make_combo(
            ["Composited Drillholes", "Raw Assays"],
            tooltip="Which data source to use for estimation",
        )
        self.source_combo.currentTextChanged.connect(self._on_source_selection_changed)
        form_row(f, "Source:", self.source_combo)

        # Variable selector
        self.variable_combo = make_combo(tooltip="Grade variable to estimate")
        self.variable_combo.currentTextChanged.connect(self._on_variable_selection_changed)
        form_row(f, "Variable:", self.variable_combo)

        # Domain column
        self.domain_combo = make_combo(tooltip="Domain selection for coded hard-boundary estimation")
        self.domain_combo.addItem("(none)")
        self.domain_combo.currentTextChanged.connect(self._on_domain_filter_selection_changed)
        form_row(f, "Domain:", self.domain_combo)

        # Status display
        self.data_status = hint_label("No data loaded")
        f.addRow("Status:", self.data_status)

        # Load / Refresh button
        btn_load = action_button("🔄 Refresh from Registry", style="primary")
        btn_load.setToolTip("Reload latest composites/assays from the data registry")
        btn_load.setMinimumWidth(200)
        btn_load.setStyleSheet(
            f"QPushButton {{ background-color: {ModernColors.ACCENT_PRIMARY}; color: white;"
            f" font-weight: bold; font-size: 10pt; border-radius: 4px; padding: 6px 14px; }}"
            f"QPushButton:hover {{ background-color: {ModernColors.ELEVATED_BG};"
            f" color: {ModernColors.ACCENT_PRIMARY}; border: 2px solid {ModernColors.ACCENT_PRIMARY}; }}"
        )
        btn_load.clicked.connect(self._on_load_data)
        f.addRow("", btn_load)

        grp.add_layout(f)
        self._left_layout.addWidget(grp)

        grp_auto = section("Auto-Configure")
        af = make_form()

        self.auto_recommend_info = QTextEdit()
        self.auto_recommend_info.setReadOnly(True)
        self.auto_recommend_info.setMinimumHeight(220)
        self.auto_recommend_info.setPlaceholderText(
            "Click 'Analyse Data & Recommend Settings' to run a deep pre-estimation "
            "analysis of the currently selected data and variable.\n\n"
            "The panel will review distribution shape, clustering, declustering, "
            "directional variography, compositing, stationarity, and block coverage, "
            "then recommend defensible ARBF settings with reasons."
        )
        af.addRow(self.auto_recommend_info)

        btn_recommend = action_button(
            "Analyse Data && Recommend Settings",
            style="secondary",
            tooltip=(
                "Run a detailed geostatistical pre-estimation analysis on the current "
                "source, variable, domain filter, and block/grid setup.\n\n"
                "This only produces recommendations. It does not change the live "
                "controls until you click 'Apply Recommendations'."
            ),
        )
        btn_recommend.clicked.connect(self._on_auto_recommend)
        af.addRow("", btn_recommend)

        btn_apply = action_button(
            "Apply Recommendations",
            style="primary",
            tooltip="Apply the last recommended settings to the live ARBF controls.",
        )
        btn_apply.clicked.connect(self._on_apply_recommendations)
        af.addRow("", btn_apply)

        grp_auto.add_layout(af)
        self._left_layout.addWidget(grp_auto)

    # ──────────────────────────────────────────────────────────────
    # Step 2: Variogram
    # ──────────────────────────────────────────────────────────────

    def _build_step2_variogram(self):
        grp = section("Kernel Configuration")
        f = make_form()

        self.kernel_combo = make_combo(
            ["Spheroidal", "Exponential", "Gaussian", "Cauchy"],
            tooltip="Covariance kernel type",
        )
        form_row(f, "Kernel:", self.kernel_combo)

        self.alpha_spin = make_spin(0.1, 10.0, 1.0, 2, tooltip="Smoothness (spheroidal)")
        form_row(f, "Alpha:", self.alpha_spin)

        self.sill_spin = make_spin(0.0, 1e15, 0.0, 4, tooltip="Partial sill C1. 0 = auto.")
        form_row(f, "Sill (C1):", self.sill_spin)

        self.nugget_spin = make_spin(0.0, 1e15, 0.05, 4, tooltip="Nugget variance C0")
        form_row(f, "Nugget (C0):", self.nugget_spin)

        # Nugget ratio indicator
        self._nugget_ratio_label = hint_label("")
        f.addRow("Nugget ratio:", self._nugget_ratio_label)
        self._nugget_warn = _WarningBanner()
        f.addRow(self._nugget_warn)
        self.sill_spin.valueChanged.connect(self._update_nugget_ratio)
        self.nugget_spin.valueChanged.connect(self._update_nugget_ratio)

        self.accuracy_spin = make_spin(1e-7, 1.0, 1e-6, 8, tooltip="Regularisation")
        form_row(f, "Accuracy:", self.accuracy_spin)

        self.drift_combo = make_combo(
            ["Auto", "Constant", "Linear", "None"],
            tooltip=(
                "Constant drift smooths toward the local mean and can flatten "
                "directional trends; Linear drift follows the local trend and is "
                "usually better for grades with directional structure. "
                "Auto picks per block based on CV slope."
            ),
        )
        self.drift_combo.setCurrentText("Auto")
        form_row(f, "Drift:", self.drift_combo)

        self.cb_auto_background = QCheckBox("Add exponential background for inter-hole continuity")
        self.cb_auto_background.setChecked(True)
        self.cb_auto_background.setToolTip(
            "When the spheroidal model isolates drillholes (>50% of sample pairs "
            "have zero covariance), automatically inject a low-sill exponential "
            "structure (10% of sill, 3× range) to maintain spatial continuity "
            "between holes. Disable for strict compact-support estimation."
        )
        f.addRow("", self.cb_auto_background)

        btn_import = action_button("Import from Variogram Panel", style="secondary")
        btn_import.clicked.connect(lambda: self._on_import_variogram(set_mode=True))
        f.addRow("", btn_import)

        grp.add_layout(f)

        # Anisotropy sub-section
        af = make_form()
        self.range_max_spin = make_spin(0.1, 1e6, 100.0, 1, tooltip="Major range")
        form_row(af, "Range Max:", self.range_max_spin)
        self.range_mid_spin = make_spin(0.1, 1e6, 100.0, 1, tooltip="Semi-major range")
        form_row(af, "Range Mid:", self.range_mid_spin)
        self.range_min_spin = make_spin(0.1, 1e6, 100.0, 1, tooltip="Minor range")
        form_row(af, "Range Min:", self.range_min_spin)

        self.azimuth_spin = make_spin(0.0, 360.0, 0.0, 1, tooltip="Azimuth (degrees)")
        form_row(af, "Azimuth:", self.azimuth_spin)
        self.dip_spin = make_spin(-90.0, 90.0, 0.0, 1, tooltip="Dip (degrees)")
        form_row(af, "Dip:", self.dip_spin)
        self.pitch_spin = make_spin(-90.0, 90.0, 0.0, 1, tooltip="Pitch (degrees)")
        form_row(af, "Pitch:", self.pitch_spin)

        # Isotropic ranges warning
        self._iso_warn = _WarningBanner()
        af.addRow(self._iso_warn)
        self.range_max_spin.valueChanged.connect(self._update_iso_warning)
        self.range_mid_spin.valueChanged.connect(self._update_iso_warning)
        self.range_min_spin.valueChanged.connect(self._update_iso_warning)

        # Anisotropy orientation warning
        self._aniso_warn = _WarningBanner()
        af.addRow(self._aniso_warn)
        self.azimuth_spin.valueChanged.connect(self._update_aniso_warning)
        self.dip_spin.valueChanged.connect(self._update_aniso_warning)
        self.pitch_spin.valueChanged.connect(self._update_aniso_warning)
        self.range_max_spin.valueChanged.connect(self._update_aniso_warning)
        self.range_min_spin.valueChanged.connect(self._update_aniso_warning)

        grp.add_layout(af)
        self._left_layout.addWidget(grp)

    # ──────────────────────────────────────────────────────────────
    # Step 3: Grid / Block Model
    # ──────────────────────────────────────────────────────────────

    def _build_step3_grid(self):
        grp = section("Block Model Grid")
        f = make_form()

        grid_fields = [
            ("NX:", "nx_spin", 1, 9999, 100),
            ("NY:", "ny_spin", 1, 9999, 100),
            ("NZ:", "nz_spin", 1, 9999, 50),
        ]
        for label, attr, mn, mx, default in grid_fields:
            spin = QSpinBox()
            spin.setRange(mn, mx)
            spin.setValue(default)
            setattr(self, attr, spin)
            form_row(f, label, spin)

        for label, attr, default in [("DX:", "dx_spin", 10.0), ("DY:", "dy_spin", 10.0), ("DZ:", "dz_spin", 10.0)]:
            spin = make_spin(0.1, 1000.0, default, 2)
            setattr(self, attr, spin)
            form_row(f, label, spin)

        for label, attr, default in [("X0:", "x0_spin", 0.0), ("Y0:", "y0_spin", 0.0), ("Z0:", "z0_spin", 0.0)]:
            spin = make_spin(-1e8, 1e8, default, 2)
            setattr(self, attr, spin)
            form_row(f, label, spin)

        btn_auto = action_button("Auto-Detect from Drillholes", style="secondary")
        btn_auto.clicked.connect(self._on_auto_detect_grid)
        f.addRow("", btn_auto)

        # Block count indicator
        self._block_count_label = hint_label("")
        f.addRow("Blocks:", self._block_count_label)
        for attr in ["nx_spin", "ny_spin", "nz_spin"]:
            getattr(self, attr).valueChanged.connect(self._update_block_count)
        self._update_block_count()

        self.chk_clip_to_footprint = QCheckBox("Clip to Drillhole Footprint")
        self.chk_clip_to_footprint.setChecked(True)
        self.chk_clip_to_footprint.setToolTip(
            "Limit estimation to blocks within a configurable buffer of the drillholes.\n"
            "Blocks outside the footprint will be NaN (not estimated).\n"
            "Disable only if you need to estimate the entire bounding box."
        )
        f.addRow(self.chk_clip_to_footprint)

        self._footprint_warn = _WarningBanner()
        f.addRow(self._footprint_warn)
        self.chk_clip_to_footprint.toggled.connect(self._update_footprint_warning)

        self.footprint_buffer_spin = make_spin(
            0.5, 5.0, 1.5, 1,
            tooltip="Buffer distance as a multiple of the major range.",
        )
        form_row(f, "Footprint Buffer:", self.footprint_buffer_spin)

        grp.add_layout(f)
        self._left_layout.addWidget(grp)

    # ──────────────────────────────────────────────────────────────
    # Step 4: Estimation Parameters
    # ──────────────────────────────────────────────────────────────

    def _build_step4_estimation(self):
        grp = section("Estimation Settings")
        f = make_form()

        # Transforms
        self.cb_normal_score = QCheckBox("Apply Normal-Score Transform")
        self.cb_normal_score.setChecked(True)
        self.cb_normal_score.setToolTip(
            "STRONGLY RECOMMENDED for skewed variables (Cu, Au).\n"
            "Prevents negative grades and back-transform inflation."
        )
        f.addRow(self.cb_normal_score)

        self.cb_grade_clip = QCheckBox("Clip Grades (prevent negatives)")
        self.cb_grade_clip.setToolTip("Clip estimated grades to a valid range after back-transform.")
        f.addRow(self.cb_grade_clip)
        h_clip = QHBoxLayout()
        self.clip_min_spin = make_spin(-1e12, 1e12, 0.0, 4)
        self.clip_max_spin = make_spin(-1e12, 1e12, 1e6, 4)
        h_clip.addWidget(QLabel("Min:"))
        h_clip.addWidget(self.clip_min_spin)
        h_clip.addWidget(QLabel("Max:"))
        h_clip.addWidget(self.clip_max_spin)
        f.addRow("Clip Range:", h_clip)

        f.addRow(separator())

        # ── Search mode ─────────────────────────────────────────────
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItems([
            "Local (Mining)",
            "Global (Leapfrog-style)",
        ])
        self.search_mode_combo.setToolTip(
            "Local: compact kernel with anisotropic search ellipsoid — "
            "grades constrained to neighbourhood, blocks outside data stay NaN.\n"
            "Global: all composites inform every block (Dual Kriging / Leapfrog RBF) — "
            "smooth continuous field, extrapolates to drift mean everywhere."
        )
        self.search_mode_combo.setCurrentIndex(0)
        self.search_mode_combo.currentIndexChanged.connect(self._on_search_mode_changed)
        form_row(f, "Search Mode:", self.search_mode_combo)

        # Search parameters (local mode)
        self._local_search_widgets = []  # widgets to show/hide with mode

        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(4, 5000)
        # Default is 12 — enough samples for a stable local fit but not
        # so many that a wide search becomes a pure local average. The
        # old default of 300 was appropriate for short-range deposits
        # where the effective neighbourhood is tight, but caused
        # catastrophic over-smoothing on long-range (~300 m) deposits.
        self.max_samples_spin.setValue(12)
        self.max_samples_spin.setToolTip(
            "Maximum composites per search neighbourhood. "
            "Keep this low (8-16) when the variogram range is large "
            "relative to drill spacing — too many samples with a flat "
            "kernel produces a moving average, destroying local structure."
        )
        form_row(f, "Max Samples:", self.max_samples_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 100)
        self.min_samples_spin.setValue(4)
        form_row(f, "Min Samples:", self.min_samples_spin)

        # Search-radius multipliers tightened for long-range deposits.
        # Previous defaults (0.75 / 1.50 / 2.00) produced 261 / 522 /
        # 696 m radii when multiplied against a ~348 m variogram range
        # — far larger than typical drill spacing and the immediate
        # cause of the block_std / point_std ≈ 0.17 / 1.03 collapse.
        # New defaults give 87 / 174 / 348 m for the same variogram,
        # which is close to one hole spacing for the first pass.
        self.search_radius_1_spin = make_spin(
            0.01,
            100.0,
            0.25,
            2,
            tooltip=(
                "Pass 1 search radius as multiple of variogram range. "
                "Keep this small (0.1-0.3) so the local RBF actually "
                "weights by distance instead of averaging a large "
                "moving window. Check the '(\u2192 X m)' label to see "
                "the resolved radius against your variogram range."
            ),
        )
        self._lbl_sr1 = QLabel("Search Radius 1:")
        f.addRow(self._lbl_sr1, self.search_radius_1_spin)
        self._local_search_widgets.extend([self._lbl_sr1, self.search_radius_1_spin])

        self.search_radius_2_spin = make_spin(
            0.01,
            100.0,
            0.50,
            2,
            tooltip="Pass 2 (fallback) search radius as multiple of variogram range.",
        )
        self._lbl_sr2 = QLabel("Search Radius 2:")
        f.addRow(self._lbl_sr2, self.search_radius_2_spin)
        self._local_search_widgets.extend([self._lbl_sr2, self.search_radius_2_spin])

        self.search_radius_3_spin = make_spin(
            0.01,
            100.0,
            1.00,
            2,
            tooltip="Pass 3 (outer safety net) search radius as multiple of variogram range.",
        )
        self._lbl_sr3 = QLabel("Search Radius 3:")
        f.addRow(self._lbl_sr3, self.search_radius_3_spin)
        self._local_search_widgets.extend([self._lbl_sr3, self.search_radius_3_spin])

        # Live "→ X m" display so users can see what the multiplier
        # actually resolves to in metres given the current variogram
        # ranges. Fires when either the range, multiplier, max_samples
        # or block-size spinboxes change.
        for _spin in (
            self.search_radius_1_spin,
            self.search_radius_2_spin,
            self.search_radius_3_spin,
            self.range_max_spin,
            self.range_mid_spin,
            self.range_min_spin,
            self.max_samples_spin,
        ):
            try:
                _spin.valueChanged.connect(self._update_search_radius_labels)
            except Exception:
                pass
        self._update_search_radius_labels()

        self.cb_balanced_search = QCheckBox("Balanced multi-octant selection")
        self.cb_balanced_search.setChecked(True)
        self.cb_balanced_search.setToolTip(
            "Spread the neighbourhood across octants instead of taking only the nearest clustered samples.",
        )
        f.addRow(self.cb_balanced_search)
        self._local_search_widgets.append(self.cb_balanced_search)

        self.search_min_octants_spin = QSpinBox()
        self.search_min_octants_spin.setRange(1, 8)
        self.search_min_octants_spin.setValue(3)
        self.search_min_octants_spin.setToolTip(
            "Minimum occupied octants required for constant-drift neighbourhoods.",
        )
        self._lbl_mo = QLabel("Min Octants:")
        f.addRow(self._lbl_mo, self.search_min_octants_spin)
        self._local_search_widgets.extend([self._lbl_mo, self.search_min_octants_spin])

        self.search_min_octants_linear_spin = QSpinBox()
        self.search_min_octants_linear_spin.setRange(1, 8)
        self.search_min_octants_linear_spin.setValue(4)
        self.search_min_octants_linear_spin.setToolTip(
            "Minimum occupied octants required before local linear drift is allowed.",
        )
        form_row(f, "Min Octants (Linear):", self.search_min_octants_linear_spin)

        self.max_samples_per_octant_spin = QSpinBox()
        self.max_samples_per_octant_spin.setRange(0, 500)
        self.max_samples_per_octant_spin.setValue(4)
        self.max_samples_per_octant_spin.setToolTip("0 = no per-octant cap.")
        form_row(f, "Max / Octant:", self.max_samples_per_octant_spin)

        btn_tight_preset = action_button(
            "Preset: Tight neighbourhood (recommended)", style="secondary",
        )
        btn_tight_preset.setToolTip(
            "Sets search radii 0.25 / 0.50 / 1.00, max samples 12, "
            "max/octant 2, and drift = Linear. Good starting point for "
            "grades with directional trend and moderate clustering."
        )
        btn_tight_preset.clicked.connect(self._apply_tight_preset)
        f.addRow("", btn_tight_preset)

        self.auto_drift_slope_spin = make_spin(
            0.0,
            1.0,
            0.20,
            2,
            tooltip=(
                "Maximum extra CV slope deviation tolerated before auto-drift "
                "rejects linear drift."
            ),
        )
        form_row(f, "Auto Drift Slope Tol.:", self.auto_drift_slope_spin)

        f.addRow(separator())

        # Discretisation
        self.disc_mode_combo = make_combo(["Fixed", "Adaptive"], tooltip="Block discretisation mode")
        form_row(f, "Discretisation:", self.disc_mode_combo)

        self.disc_density_combo = make_combo(
            ["8 (2x2x2)", "27 (3x3x3)", "64 (4x4x4)"],
            tooltip="Points per block for volume averaging",
        )
        self.disc_density_combo.setCurrentIndex(1)  # Default 27
        form_row(f, "Density:", self.disc_density_combo)

        grp.add_layout(f)
        self._left_layout.addWidget(grp)

    # ──────────────────────────────────────────────────────────────
    # Step 5: Execution
    # ──────────────────────────────────────────────────────────────

    def _build_step5_execution(self):
        grp = section("Execution")
        lay = QVBoxLayout()

        # Estimation mode selector (preview / standard / final)
        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("Mode:"))
        self.estimation_mode_combo = make_combo(
            ["Preview (fast)", "Standard", "Final (full uncertainty)"],
            tooltip=(
                "Preview: 1 subpoint, 32 neighbours, no variance — seconds\n"
                "Standard: 8 subpoints, 48 neighbours, block variance — minutes\n"
                "Final: 27 subpoints, 64 neighbours, full diagnostics — slower"
            ),
        )
        self.estimation_mode_combo.setCurrentIndex(1)  # Standard default
        h_mode.addWidget(self.estimation_mode_combo)
        lay.addLayout(h_mode)

        # Active block count display (populated after grid is configured)
        self._active_block_label = hint_label("")
        lay.addWidget(self._active_block_label)

        lay.addWidget(separator())

        self.cb_run_cv = QCheckBox("Run Spatial Cross-Validation")
        self.cb_run_cv.setToolTip("Runs the backend spatial k-fold CV, not ordinary LOO interpolation CV.")
        self.cb_run_cv.setChecked(True)
        lay.addWidget(self.cb_run_cv)

        h_cv_mode = QHBoxLayout()
        h_cv_mode.addWidget(QLabel("CV Mode:"))
        self.cv_mode_combo = make_combo(
            ["Spatial K-Fold", "Fast LOO"],
            tooltip="Spatial K-fold is the defensible default. Fast LOO is a faster diagnostic path.",
        )
        h_cv_mode.addWidget(self.cv_mode_combo)
        lay.addLayout(h_cv_mode)

        h_cv_folds = QHBoxLayout()
        h_cv_folds.addWidget(QLabel("CV Folds:"))
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 20)
        self.cv_folds_spin.setValue(5)
        self.cv_folds_spin.setToolTip("Number of folds for spatial K-fold cross-validation.")
        h_cv_folds.addWidget(self.cv_folds_spin)
        lay.addLayout(h_cv_folds)

        h_seed = QHBoxLayout()
        h_seed.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        h_seed.addWidget(self.seed_spin)
        lay.addLayout(h_seed)

        lay.addWidget(separator())

        # Domain masking
        lay.addWidget(self._build_domain_mask_group(default_enabled=True))

        lay.addWidget(separator())

        # Run button
        self.run_btn = action_button("Run ARBF Estimation", style="primary")
        self.run_btn.clicked.connect(self._on_run)
        lay.addWidget(self.run_btn)

        # Progress (local to Run tab)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% — %v of 100")
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)
        self.progress_label = hint_label("")
        lay.addWidget(self.progress_label)

        grp.add_layout(lay)
        self._left_layout.addWidget(grp)

        self.cb_run_cv.toggled.connect(self._update_execution_controls)
        self.cv_mode_combo.currentTextChanged.connect(self._update_execution_controls)
        self._update_execution_controls()

    # ──────────────────────────────────────────────────────────────
    # Step 6: Export
    # ──────────────────────────────────────────────────────────────

    def _build_step6_export(self):
        grp = section("Export & Reporting")
        lay = QVBoxLayout()

        self.btn_visualise = action_button("Visualise Estimated Blocks", style="primary")
        self.btn_visualise.setEnabled(False)
        self.btn_visualise.setToolTip(
            "Register the ARBF block model and display it in the 3D viewer.\n"
            "Adds grade, variance, and classification as properties.\n"
            "Uses P2/P98 percentile color limits (not min/max)."
        )
        self.btn_visualise.clicked.connect(self._on_visualise_blocks)
        lay.addWidget(self.btn_visualise)

        self.btn_register = action_button("Register Block Model (no render)", style="secondary")
        self.btn_register.setEnabled(False)
        self.btn_register.clicked.connect(self._on_register_model)
        lay.addWidget(self.btn_register)

        self.btn_jorc = action_button("Export JORC Table 1 (.txt)", style="secondary")
        self.btn_jorc.setEnabled(False)
        self.btn_jorc.clicked.connect(self._on_export_jorc)
        lay.addWidget(self.btn_jorc)

        self.btn_json = action_button("Export Audit Record (.json)", style="secondary")
        self.btn_json.setEnabled(False)
        self.btn_json.clicked.connect(self._on_export_audit_json)
        lay.addWidget(self.btn_json)

        self.btn_csv = action_button("Export Results (.csv)", style="secondary")
        self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self._on_export_csv)
        lay.addWidget(self.btn_csv)

        grp.add_layout(lay)
        self._left_layout.addWidget(grp)

    # ──────────────────────────────────────────────────────────────
    # Right panel: Results & Diagnostics
    # ──────────────────────────────────────────────────────────────

    def _build_right_panel(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self._verdict_banner = QLabel("No ARBF result yet.")
        self._verdict_banner.setWordWrap(True)
        self._verdict_banner.setStyleSheet(
            "background: #F5F5F5; border: 1px solid #D9D9D9; border-radius: 6px; "
            "padding: 8px; color: #444;"
        )
        right_layout.addWidget(self._verdict_banner)

        self.results_tabs = QTabWidget()

        # Tab 1: Defensibility
        self._gate_text = QTextEdit()
        self._gate_text.setReadOnly(True)
        self._gate_text.setPlaceholderText(
            "Run an estimation to see the geostatistical verdict here."
        )
        self.results_tabs.addTab(self._gate_text, "Defensibility")

        # Tab 2: Summary
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setPlaceholderText(
            "Run an estimation to see results here.\n\n"
            "The summary will include:\n"
            "  • Cross-validation metrics (NS and original units)\n"
            "  • Grade statistics\n"
            "  • Classification summary\n"
            "  • Parameter audit trail"
        )
        self.results_tabs.addTab(self._summary_text, "Summary")

        # Tab 3: CV Scatter (placeholder for matplotlib)
        self._cv_widget = QWidget()
        self._cv_layout = QVBoxLayout(self._cv_widget)
        self._cv_layout.setContentsMargins(0, 0, 0, 0)
        self._cv_layout.addWidget(QLabel("Cross-validation scatter plot will appear here after estimation."))
        self.results_tabs.addTab(self._cv_widget, "CV Scatter")

        # Tab 4: Swath Plots — single plot with axis toggle
        self._swath_widget = QWidget()
        swath_outer = QVBoxLayout(self._swath_widget)
        swath_outer.setContentsMargins(4, 4, 4, 4)
        swath_outer.setSpacing(4)

        # Direction toggle bar — styled segmented control
        swath_btn_bar = QHBoxLayout()
        swath_btn_bar.setSpacing(2)
        self._swath_axis_btns = {}
        self._swath_payload_cache = None
        _toggle_style = (
            "QPushButton { "
            "  background: transparent; color: #a0a0a0; border: 1px solid #3c3c3c;"
            "  border-radius: 4px; padding: 4px 16px; font-weight: bold; font-size: 11px;"
            "}"
            "QPushButton:checked { "
            f"  background: {ModernColors.ACCENT_PRIMARY}; color: white;"
            f"  border-color: {ModernColors.ACCENT_PRIMARY};"
            "}"
            "QPushButton:hover:!checked { background: #333337; color: #d4d4d4; }"
        )
        for axis_label in ("X", "Y", "Z"):
            btn = QPushButton(axis_label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(52)
            btn.setStyleSheet(_toggle_style)
            btn.clicked.connect(lambda checked, a=axis_label: self._on_swath_axis_toggle(a))
            swath_btn_bar.addWidget(btn)
            self._swath_axis_btns[axis_label] = btn
        self._swath_axis_btns["X"].setChecked(True)
        self._current_swath_axis = "X"
        swath_btn_bar.addSpacing(12)

        # Grade / Metal sub-mode toggle — same segmented-control style
        # as the axis buttons. Grade is the primary QA curve; Metal
        # converts the panels to contained-metal bars so the user can
        # see whether a grade mismatch survives the volume weighting.
        self._swath_mode_btns = {}
        for mode_label in ("Grade", "Metal"):
            btn = QPushButton(mode_label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(64)
            btn.setStyleSheet(_toggle_style)
            btn.clicked.connect(
                lambda checked, m=mode_label: self._on_swath_mode_toggle(m)
            )
            swath_btn_bar.addWidget(btn)
            self._swath_mode_btns[mode_label] = btn
        self._swath_mode_btns["Grade"].setChecked(True)
        self._current_swath_mode = "Grade"
        swath_btn_bar.addStretch()
        swath_outer.addLayout(swath_btn_bar)

        # Panel-width spinner — 0 = auto (3-SMU rule).
        panel_row = QHBoxLayout()
        panel_row.setContentsMargins(0, 0, 0, 0)
        panel_row.addWidget(QLabel("Panel width (m):"))
        self.panel_width_spin = QDoubleSpinBox()
        self.panel_width_spin.setRange(0.0, 10000.0)
        self.panel_width_spin.setDecimals(1)
        self.panel_width_spin.setSingleStep(5.0)
        self.panel_width_spin.setValue(0.0)
        self.panel_width_spin.setToolTip(
            "Swath panel width in metres. 0 = auto (3 × SMU or 10% of "
            "range_max, whichever is larger — the broad-panel rule for "
            "support-aware QA). Do not set this finer than ~3 block "
            "widths: thin panels make composite means noisy and "
            "produce false bias in Gate 3."
        )
        panel_row.addWidget(self.panel_width_spin)
        panel_row.addStretch()
        swath_outer.addLayout(panel_row)

        # Plot area
        self._swath_layout = QVBoxLayout()
        self._swath_layout.setContentsMargins(0, 0, 0, 0)
        self._swath_layout.addWidget(QLabel("Support swath plots will appear here after estimation."))
        swath_outer.addLayout(self._swath_layout, 1)
        self.results_tabs.addTab(self._swath_widget, "Support Swaths")

        # Tab 5: Log
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Consolas", 9))
        self.results_tabs.addTab(self._log_text, "Engine Log")

        right_layout.addWidget(self.results_tabs, 1)

        # ── Global progress bar (always visible at bottom of right panel) ──
        progress_frame = QFrame()
        progress_frame.setObjectName("globalProgressFrame")
        pf_lay = QVBoxLayout(progress_frame)
        pf_lay.setContentsMargins(4, 4, 4, 4)
        pf_lay.setSpacing(2)

        self._global_progress_bar = QProgressBar()
        self._global_progress_bar.setRange(0, 100)
        self._global_progress_bar.setValue(0)
        self._global_progress_bar.setFormat("%p%")
        self._global_progress_bar.setFixedHeight(18)
        self._global_progress_bar.setVisible(False)
        pf_lay.addWidget(self._global_progress_bar)

        self._global_progress_label = QLabel("")
        self._global_progress_label.setStyleSheet("color: #666; font-size: 9pt;")
        pf_lay.addWidget(self._global_progress_label)

        right_layout.addWidget(progress_frame)
        self.splitter.addWidget(right_widget)

    # ══════════════════════════════════════════════════════════════
    # LIVE VALIDATION
    # ══════════════════════════════════════════════════════════════

    def _update_nugget_ratio(self):
        sill = self.sill_spin.value()
        nugget = self.nugget_spin.value()
        total = sill + nugget
        if total > 0:
            ratio = nugget / total * 100
            self._nugget_ratio_label.setText(f"{ratio:.1f}%")
            if ratio > 40:
                self._nugget_warn.show_warning(
                    f"Nugget ratio is {ratio:.0f}% — very high. "
                    f"The estimator will treat {ratio:.0f}% of grade variation as random noise. "
                    f"Check your variogram fit or re-import from the variogram panel."
                )
            elif ratio > 25:
                self._nugget_warn.show_warning(
                    f"Nugget ratio is {ratio:.0f}% — moderately high. "
                    f"Consider whether this reflects true measurement noise."
                )
            else:
                self._nugget_warn.clear()
        else:
            self._nugget_ratio_label.setText("—")
            self._nugget_warn.clear()

    def _on_search_mode_changed(self, index: int):
        """Toggle local search widgets visibility based on search mode."""
        is_local = index == 0
        for w in self._local_search_widgets:
            w.setVisible(is_local)

    def _apply_tight_preset(self):
        """Apply the 'tight neighbourhood' recommended defaults.

        These are the settings that avoid the over-smoothing failure
        mode on long-range deposits: small search radii, low max-samples,
        strict octant cap, and linear drift to follow local trend.
        """
        try:
            self.search_radius_1_spin.setValue(0.25)
            self.search_radius_2_spin.setValue(0.50)
            self.search_radius_3_spin.setValue(1.00)
            self.max_samples_spin.setValue(12)
            self.max_samples_per_octant_spin.setValue(2)
            self.cb_balanced_search.setChecked(True)
            idx = self.drift_combo.findText("Linear")
            if idx >= 0:
                self.drift_combo.setCurrentIndex(idx)
        except Exception:
            logger.exception("Failed to apply tight preset")

    def _update_search_radius_labels(self, *_):
        """Refresh the search-radius labels so they show the resolved
        radius in metres next to the multiplier. The panel stores
        multipliers of the major variogram range, but users need to see
        the resulting radius in real units to judge whether the
        neighbourhood is sensible for their block size.
        """
        try:
            r_major = float(self.range_max_spin.value() or 0.0)
        except Exception:
            r_major = 0.0
        mults = {}
        for key, spin in (
            ("sr1", getattr(self, "search_radius_1_spin", None)),
            ("sr2", getattr(self, "search_radius_2_spin", None)),
            ("sr3", getattr(self, "search_radius_3_spin", None)),
        ):
            try:
                mults[key] = float(spin.value()) if spin is not None else 0.0
            except Exception:
                mults[key] = 0.0

        def _fmt(label_base: str, mult: float) -> str:
            if r_major <= 0:
                return f"{label_base}:"
            meters = mult * r_major
            return f"{label_base}  (\u2192 {meters:,.0f} m):"

        lbl1 = getattr(self, "_lbl_sr1", None)
        lbl2 = getattr(self, "_lbl_sr2", None)
        lbl3 = getattr(self, "_lbl_sr3", None)
        if lbl1 is not None:
            lbl1.setText(_fmt("Search Radius 1", mults["sr1"]))
        if lbl2 is not None:
            lbl2.setText(_fmt("Search Radius 2", mults["sr2"]))
        if lbl3 is not None:
            lbl3.setText(_fmt("Search Radius 3", mults["sr3"]))

        # Flag over-smoothing risk: when Pass 1 resolved radius is
        # more than ~20x the smallest block dimension, the RBF kernel
        # is effectively flat over the block and ``max_samples``
        # samples get averaged with near-uniform weights, producing
        # a moving-average field with ~1/sqrt(N) variance reduction
        # instead of a proper support-corrected estimate. Colour the
        # Pass 1 label red as a warning.
        def _spin_val(attr: str) -> float:
            w = getattr(self, attr, None)
            if w is None or not hasattr(w, "value"):
                return 0.0
            try:
                return float(w.value())
            except Exception:
                return 0.0

        try:
            dx = _spin_val("dx_spin")
            dy = _spin_val("dy_spin")
            dz = _spin_val("dz_spin")
            block_sizes = [d for d in (dx, dy, dz) if d > 0]
            block_min = max(1e-6, min(block_sizes)) if block_sizes else 1.0
            pass1_m = mults["sr1"] * r_major
            max_samples_val = int(_spin_val("max_samples_spin"))
            if r_major > 0 and lbl1 is not None:
                if pass1_m >= 20.0 * block_min and max_samples_val >= 16:
                    lbl1.setStyleSheet("color: #e74c3c; font-weight: bold;")
                    lbl1.setToolTip(
                        f"Pass 1 radius {pass1_m:.0f} m is {pass1_m/block_min:.0f}x the "
                        f"block size ({block_min:.0f} m) with {max_samples_val} max samples. "
                        f"The RBF kernel will be nearly flat over each block and the "
                        f"estimate will collapse to a moving average. Try Search Radius 1 "
                        f"≤ {0.1 * r_major / max(r_major, 1e-6):.2f} and Max Samples ≤ 12."
                    )
                else:
                    lbl1.setStyleSheet("")
                    lbl1.setToolTip("")
        except Exception:
            pass

    def _update_footprint_warning(self):
        clip_on = self.chk_clip_to_footprint.isChecked()
        if not clip_on:
            self._footprint_warn.show_warning(
                "Footprint clipping is OFF \u2014 the entire bounding box will be "
                "estimated, including barren rock and air blocks far from "
                "drillhole data. Enable footprint clipping or use a domain mask."
            )
        else:
            self._footprint_warn.clear()

    def _update_iso_warning(self):
        r_max = max(self.range_max_spin.value(), 1e-6)
        r_mid = max(self.range_mid_spin.value(), 1e-6)
        r_min = max(self.range_min_spin.value(), 1e-6)
        ratio = max(r_max, r_mid, r_min) / min(r_max, r_mid, r_min)
        if ratio < 1.1:
            self._iso_warn.show_warning(
                "Ranges are isotropic (all within 10%) \u2014 the search ellipsoid "
                "is a sphere. Set different ranges from your variography to "
                "honour geological anisotropy (e.g. 100 / 50 / 25)."
            )
        else:
            self._iso_warn.clear()

    def _update_aniso_warning(self):
        max_r = max(self.range_max_spin.value(), 1e-6)
        min_r = max(self.range_min_spin.value(), 1e-6)
        ratio = max_r / min_r
        az = abs(self.azimuth_spin.value())
        dip = abs(self.dip_spin.value())
        pitch = abs(self.pitch_spin.value())
        all_zero = az < 1e-6 and dip < 1e-6 and pitch < 1e-6

        if ratio > 1.5 and all_zero:
            self._aniso_warn.show_warning(
                f"Anisotropy ratio is {ratio:.1f}:1 but all angles are 0\u00b0. "
                f"The search ellipsoid is aligned with the grid axes, not the geology. "
                f"Fit directional variograms to find the correct orientation."
            )
        else:
            self._aniso_warn.clear()

    def _update_block_count(self):
        nx = self.nx_spin.value()
        ny = self.ny_spin.value()
        nz = self.nz_spin.value()
        total = nx * ny * nz
        if total > 2_000_000:
            self._block_count_label.setText(f"{total:,} blocks (VERY LARGE — may be slow)")
            self._block_count_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
        elif total > 500_000:
            self._block_count_label.setText(f"{total:,} blocks")
            self._block_count_label.setStyleSheet("color: #F39C12;")
        else:
            self._block_count_label.setText(f"{total:,} blocks")
            self._block_count_label.setStyleSheet("")

        # Update estimated active block count (guard: created in Step 5, called from Step 3)
        label = getattr(self, "_active_block_label", None)
        if label is not None:
            clip = getattr(self, "chk_clip_to_footprint", None)
            if clip and clip.isChecked():
                label.setText("Footprint clipping ON — active blocks computed at run time")
                label.setStyleSheet("")
            elif total > 500_000:
                label.setText(f"All {total:,} blocks active — consider enabling footprint clipping")
                label.setStyleSheet("color: #F39C12;")
            else:
                label.setText(f"All {total:,} blocks active")
                label.setStyleSheet("")

    # ══════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════

    def _init_registry_connections(self):
        try:
            registry = self.get_registry()
            if registry is None:
                logger.debug("ARBF: No registry found during init")
                return
            self.registry = registry

            # Prevent duplicate signal connections on repeated calls
            for sig, slot in (
                (registry.drillholeDataLoaded, self._on_registry_data_changed),
                (registry.blockModelLoaded, self._on_registry_data_changed),
                (registry.variogramResultsLoaded, self._on_variogram_updated),
                (registry.transformationMetadataLoaded, self._on_transformation_loaded),
                (registry.compositesLoaded, self._on_composites_refreshed),
                (registry.indicatorRBFDomainLoaded, self._on_indicator_rbf_domain_loaded),
            ):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
                sig.connect(slot)

            self._refresh_transformation_metadata()
            self._on_registry_data_changed()
            # Load variogram results that were stored before this panel was opened
            self._on_import_variogram()
        except Exception as exc:
            logger.warning("ARBF: Registry connection failed: %s", exc, exc_info=True)

    def _on_variogram_updated(self, results=None):
        if not self._ui_ready:
            return
        if isinstance(results, dict) and results:
            self.variogram_results = results
        else:
            self.variogram_results = None
        self._on_import_variogram()

    def _refresh_transformation_metadata(self) -> None:
        if self.registry is None or not hasattr(self.registry, "get_transformation_metadata"):
            self.transformation_metadata = None
            return
        try:
            self.transformation_metadata = self.registry.get_transformation_metadata()
        except Exception as exc:
            logger.debug("ARBF: Failed to refresh transformation metadata: %s", exc)
            self.transformation_metadata = None

    def _on_transformation_loaded(self, metadata):
        self.transformation_metadata = metadata if isinstance(metadata, dict) else None

    def _process_pending_data(self):
        if hasattr(self, '_pending_drillhole_data') and self._pending_drillhole_data is not None:
            self.set_drillhole_data(self._pending_drillhole_data)
            self._pending_drillhole_data = None

    def _on_registry_data_changed(self, _data=None):
        if not self._ui_ready or self.registry is None:
            return
        try:
            self._refresh_transformation_metadata()
            data = self._get_registry_payload_for_source()
            self._registry_data = data
            self._refresh_variable_list(data)
        except Exception as exc:
            logger.debug("Registry refresh failed: %s", exc)

    def _on_load_data(self):
        if self.registry is None:
            try:
                self.registry = self.get_registry()
            except Exception:
                self.data_status.setText("No registry available")
                return

        self._refresh_transformation_metadata()
        data = self._get_registry_payload_for_source()
        if data is None:
            self.data_status.setText("No data in registry")
            return

        self._registry_data = data
        self._refresh_variable_list(data)
        self._extract_dataframe(data)

    def _preferred_source_registry_keys(self) -> Tuple[str, ...]:
        source_text = self.source_combo.currentText() if hasattr(self, "source_combo") else ""
        if source_text == "Raw Assays":
            return ("assays", "assays_df", "composites", "composites_df")
        return ("composites", "composites_df", "assays", "assays_df")

    def _get_registry_payload_for_source(self):
        if self.registry is None:
            return None

        raw_data = self.registry.get_drillhole_data()
        if isinstance(raw_data, dict):
            preferred_keys = self._preferred_source_registry_keys()
            if any(isinstance(raw_data.get(key), pd.DataFrame) for key in preferred_keys):
                return raw_data

        data = self.registry.get_estimation_ready_data()
        if data is None:
            data = raw_data
        return data

    def _on_source_selection_changed(self, _text: str) -> None:
        self._last_recommendation = None
        if hasattr(self, "auto_recommend_info"):
            self.auto_recommend_info.clear()
        if self._registry_data is None:
            return
        self._refresh_variable_list(self._registry_data)
        self._extract_dataframe(self._registry_data)

    def _extract_dataframe(self, data):
        self.drillhole_data = None
        if isinstance(data, pd.DataFrame):
            self.drillhole_data = data
        elif isinstance(data, dict):
            for key in self._preferred_source_registry_keys():
                candidate = data.get(key)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    self.drillhole_data = candidate
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

    def _refresh_variable_list(self, data):
        prev_selection = self.variable_combo.currentText()
        self.variable_combo.clear()
        df = None
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            for key in self._preferred_source_registry_keys():
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

        # NOTE: Collocated-Z auto-fix is handled ONCE in the controller's
        # _prepare_arbf_payload().  A duplicate fix here would corrupt
        # already-desurveyed coordinates (Z = Z_desurvey - 2*mid_depth).
        # Removed in favour of the single controller-side fix.

        filtered_df, metadata = self._get_current_domain_filtered_data(
            df=self.drillhole_data,
            all_label="(none)",
        )
        self._active_domain_filter_metadata = metadata
        return filtered_df

    def _extract_coords_and_values(
        self,
        variable_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return None, None, None

        variable = variable_name or self.variable_combo.currentText()
        if not variable or variable not in filtered_df.columns:
            return None, None, None

        coord_cols = None
        for cx, cy, cz in [
            ("X", "Y", "Z"),
            ("x", "y", "z"),
            ("EAST", "NORTH", "RL"),
            ("MIDX", "MIDY", "MIDZ"),
            ("XC", "YC", "ZC"),
        ]:
            if cx in filtered_df.columns and cy in filtered_df.columns and cz in filtered_df.columns:
                coord_cols = [cx, cy, cz]
                break

        if coord_cols is None:
            return None, None, None

        cleaned = filtered_df.dropna(subset=coord_cols + [variable])
        if cleaned.empty:
            return None, None, None

        coords = cleaned[coord_cols].to_numpy(dtype=float)
        values = cleaned[variable].to_numpy(dtype=float)
        return coords, values, cleaned

    def _extract_recommendation_block_geometry(
        self,
        sample_coords: Optional[np.ndarray] = None,
        cleaned_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        grid_spec = {
            "nx": self.nx_spin.value(),
            "ny": self.ny_spin.value(),
            "nz": self.nz_spin.value(),
            "dx": self.dx_spin.value(),
            "dy": self.dy_spin.value(),
            "dz": self.dz_spin.value(),
            "x0": self.x0_spin.value(),
            "y0": self.y0_spin.value(),
            "z0": self.z0_spin.value(),
        }

        ctrl = getattr(self, "controller", None)
        if sample_coords is not None and ctrl is not None and hasattr(ctrl, "_build_arbf_block_geometry"):
            try:
                params = {
                    "grid_spec": grid_spec,
                    "use_block_model_grid": True,
                    "clip_to_drill_footprint": self.chk_clip_to_footprint.isChecked(),
                    "footprint_buffer_ranges": self.footprint_buffer_spin.value(),
                    "range_max": self.range_max_spin.value(),
                }
                block_centroids, block_sizes, _ = ctrl._build_arbf_block_geometry(
                    params,
                    np.asarray(sample_coords, dtype=float),
                )

                domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}
                domain_col = domain_meta.get("domain_filter_column")
                domain_value = domain_meta.get("domain_filter_value")
                if (
                    cleaned_df is not None
                    and domain_col
                    and domain_value is not None
                    and domain_col in cleaned_df.columns
                    and hasattr(ctrl, "_extract_arbf_block_domains")
                    and len(block_centroids) > 0
                ):
                    block_domains, _ = ctrl._extract_arbf_block_domains(
                        domain_col,
                        len(block_centroids),
                        composite_coords=np.asarray(sample_coords, dtype=float),
                        composite_domains=cleaned_df[domain_col].to_numpy(),
                        block_centroids=block_centroids,
                    )
                    if block_domains is not None and len(block_domains) == len(block_centroids):
                        domain_mask = np.asarray(block_domains) == domain_value
                        if not np.any(domain_mask):
                            domain_mask = np.asarray(block_domains).astype(str) == str(domain_value)
                        if np.any(domain_mask):
                            block_centroids = block_centroids[domain_mask]
                            if (
                                isinstance(block_sizes, np.ndarray)
                                and block_sizes.ndim == 2
                                and len(block_sizes) == len(block_domains)
                            ):
                                block_sizes = block_sizes[domain_mask]
                        else:
                            logger.warning(
                                "ARBF auto-configure: domain filter '%s=%s' matched no recommendation blocks.",
                                domain_col,
                                domain_value,
                            )
                return block_centroids, block_sizes
            except Exception as exc:
                logger.warning("ARBF auto-configure: failed to inspect estimation geometry: %s", exc)

        try:
            block_centroids = self._get_block_centroids()
            block_sizes = np.array(
                [self.dx_spin.value(), self.dy_spin.value(), self.dz_spin.value()],
                dtype=float,
            )
            return block_centroids, block_sizes
        except Exception as exc:
            logger.warning("ARBF auto-configure: failed to build grid centroids: %s", exc)
            return None, None

    def _on_auto_recommend(self) -> None:
        selected_variable = self.variable_combo.currentText()
        coords, values, cleaned_df = self._extract_coords_and_values(selected_variable)
        if coords is None or values is None:
            self.auto_recommend_info.setPlainText(
                "No usable data available. Load data, pick the source/variable, "
                "and ensure coordinates plus grade values are present."
            )
            return

        self.auto_recommend_info.setPlainText("Running deep ARBF pre-estimation analysis...")

        try:
            from geostats.arbf.recommend import recommend_arbf_settings

            analysis_variable = selected_variable
            original_variable, stored_transformer, has_metadata_ns = (
                self._resolve_external_normal_score_variable(selected_variable)
            )
            exact_external_ns = bool(has_metadata_ns and stored_transformer is not None)
            if (
                not exact_external_ns
                and self._looks_like_normal_score_variable(selected_variable)
                and original_variable
            ):
                raw_coords, raw_values, raw_cleaned = self._extract_coords_and_values(original_variable)
                if raw_coords is not None and raw_values is not None:
                    analysis_variable = original_variable
                    coords, values, cleaned_df = raw_coords, raw_values, raw_cleaned

            block_centroids, block_sizes = self._extract_recommendation_block_geometry(
                sample_coords=coords,
                cleaned_df=cleaned_df,
            )
            rec = recommend_arbf_settings(coords, values, block_centroids, block_sizes)
            if exact_external_ns and original_variable:
                rec.settings["use_normal_score"] = False
                rec.reasons.insert(
                    0,
                    f"Variable: keep transformed '{selected_variable}'. ARBF will reuse the stored "
                    f"Grade Transform normal-score transformer and back-transform to raw '{original_variable}'.",
                )
            elif analysis_variable != selected_variable:
                rec.settings["recommended_variable"] = analysis_variable
                rec.reasons.insert(
                    0,
                    f"Variable: switch from '{selected_variable}' to raw '{analysis_variable}' for estimation.",
                )
                rec.warnings.insert(
                    0,
                    f"'{selected_variable}' is already normal-scored. Recommendations were built from raw "
                    f"'{analysis_variable}' so ARBF can estimate in raw-grade space with a single internal transform.",
                )
            self._last_recommendation = rec
            self.auto_recommend_info.setPlainText(rec.summary_text())
        except Exception as exc:
            logger.warning("Auto-recommend failed: %s", exc, exc_info=True)
            self.auto_recommend_info.setPlainText(f"Analysis failed: {exc}")

    def _on_apply_recommendations(self) -> None:
        rec = self._last_recommendation
        if rec is None or not rec.settings:
            self.auto_recommend_info.setPlainText(
                "No recommendations to apply. Click 'Analyse Data & Recommend Settings' first."
            )
            return

        settings = rec.settings

        if "recommended_variable" in settings:
            idx = self.variable_combo.findText(str(settings["recommended_variable"]))
            if idx >= 0:
                self.variable_combo.setCurrentIndex(idx)

        if "use_normal_score" in settings:
            self.cb_normal_score.setChecked(bool(settings["use_normal_score"]))

        kernel_map = {
            "spheroidal": "Spheroidal",
            "spherical": "Spherical",
            "gaussian": "Gaussian",
            "matern_32": "Matern-3/2",
            "matern_52": "Matern-5/2",
            "cubic": "Cubic",
        }
        if "kernel_type" in settings:
            kernel_text = kernel_map.get(settings["kernel_type"])
            if kernel_text:
                idx = self.kernel_combo.findText(kernel_text)
                if idx >= 0:
                    self.kernel_combo.setCurrentIndex(idx)

        if "alpha" in settings:
            self.alpha_spin.setValue(float(settings["alpha"]))
        if "sill" in settings:
            self.sill_spin.setValue(float(settings["sill"]))
        if "nugget" in settings:
            self.nugget_spin.setValue(float(settings["nugget"]))
        if "accuracy" in settings:
            self.accuracy_spin.setValue(float(settings["accuracy"]))

        drift_map = {
            "auto": "Auto",
            "constant": "Constant",
            "linear": "Linear",
            "none": "None",
        }
        if "drift_type" in settings:
            drift_text = drift_map.get(settings["drift_type"], "Auto")
            idx = self.drift_combo.findText(drift_text)
            if idx >= 0:
                self.drift_combo.setCurrentIndex(idx)

        if "range_max" in settings:
            self.range_max_spin.setValue(float(settings["range_max"]))
        if "range_mid" in settings:
            self.range_mid_spin.setValue(float(settings["range_mid"]))
        if "range_min" in settings:
            self.range_min_spin.setValue(float(settings["range_min"]))
        if "azimuth" in settings:
            self.azimuth_spin.setValue(float(settings["azimuth"]))
        if "dip" in settings:
            self.dip_spin.setValue(float(settings["dip"]))
        if "pitch" in settings:
            self.pitch_spin.setValue(float(settings["pitch"]))

        if "search_mode" in settings:
            idx = 0 if settings["search_mode"] == "local" else 1
            self.search_mode_combo.setCurrentIndex(idx)

        if "max_samples" in settings:
            self.max_samples_spin.setValue(int(settings["max_samples"]))
        if "min_samples" in settings:
            self.min_samples_spin.setValue(int(settings["min_samples"]))

        # Search radii (multi-pass)
        if "search_radius_1" in settings:
            self.search_radius_1_spin.setValue(float(settings["search_radius_1"]))
        if "search_radius_2" in settings:
            self.search_radius_2_spin.setValue(float(settings["search_radius_2"]))
        if "search_radius_3" in settings:
            self.search_radius_3_spin.setValue(float(settings["search_radius_3"]))

        # Octant settings
        if "balanced_octant" in settings:
            self.cb_balanced_search.setChecked(bool(settings["balanced_octant"]))
        if "min_octants" in settings:
            self.search_min_octants_spin.setValue(int(settings["min_octants"]))
        if "min_octants_linear" in settings:
            self.search_min_octants_linear_spin.setValue(int(settings["min_octants_linear"]))
        if "max_per_octant" in settings:
            self.max_samples_per_octant_spin.setValue(int(settings["max_per_octant"]))

        # Auto drift slope tolerance
        if "auto_drift_slope_tol" in settings:
            self.auto_drift_slope_spin.setValue(float(settings["auto_drift_slope_tol"]))

        # Discretisation mode and density
        if "discretisation_mode" in settings:
            idx = self.disc_mode_combo.findText(settings["discretisation_mode"])
            if idx >= 0:
                self.disc_mode_combo.setCurrentIndex(idx)

        if "discretisation_density" in settings:
            density_text = {
                8: "8 (2x2x2)",
                27: "27 (3x3x3)",
                64: "64 (4x4x4)",
            }.get(int(settings["discretisation_density"]))
            if density_text:
                idx = self.disc_density_combo.findText(density_text)
                if idx >= 0:
                    self.disc_density_combo.setCurrentIndex(idx)

        # Grade clipping
        if "clip_min" in settings or "clip_max" in settings:
            self.cb_grade_clip.setChecked(True)
            if "clip_min" in settings:
                self.clip_min_spin.setValue(float(settings["clip_min"]))
            if "clip_max" in settings:
                self.clip_max_spin.setValue(float(settings["clip_max"]))

        if "run_cv" in settings:
            self.cb_run_cv.setChecked(bool(settings["run_cv"]))
        if "clip_to_drill_footprint" in settings:
            self.chk_clip_to_footprint.setChecked(bool(settings["clip_to_drill_footprint"]))
        if "footprint_buffer_ranges" in settings:
            self.footprint_buffer_spin.setValue(float(settings["footprint_buffer_ranges"]))

        self.auto_recommend_info.append("\n--- Settings applied to panel controls ---")

    # ══════════════════════════════════════════════════════════════
    # VARIOGRAM IMPORT
    # ══════════════════════════════════════════════════════════════

    def _on_import_variogram(self, set_mode=False):
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
                sill_val = model["sill"]
                if sill_val > nugget_val and nugget_val > 0:
                    # 'sill' might be total sill (legacy convention) — compute partial
                    partial_sill = max(sill_val - nugget_val, 0.0)
                    logger.warning(
                        "Variogram import fallback: 'sill' (%.4f) > nugget (%.4f), "
                        "assuming total sill → partial C1=%.4f",
                        sill_val, nugget_val, partial_sill,
                    )
                    self.sill_spin.setValue(partial_sill)
                else:
                    self.sill_spin.setValue(max(sill_val, 0.0))

            # Import model type (variogram panel → ARBF kernel combo)
            _VARIO_TO_KERNEL = {
                "spherical": "Spheroidal",
                "exponential": "Exponential",
                "gaussian": "Gaussian",
            }
            mt = model.get("model_type", "")
            kernel_text = _VARIO_TO_KERNEL.get(mt.lower())
            if kernel_text:
                idx = self.kernel_combo.findText(kernel_text)
                if idx >= 0:
                    self.kernel_combo.setCurrentIndex(idx)

            # Store nested structures for the adapter (if available)
            self._variogram_structures = model.get("structures")
            if self._variogram_structures:
                logger.info(
                    "Variogram imported: %d nested structures",
                    len(self._variogram_structures),
                )

            logger.info(
                "Variogram imported: nugget=%.4f, sill=%.4f, "
                "ranges=(%.1f, %.1f, %.1f), azimuth=%.1f, dip=%.1f, "
                "model_type=%s → kernel=%s, nested=%s",
                nugget_val, self.sill_spin.value(),
                self.range_max_spin.value(), self.range_mid_spin.value(),
                self.range_min_spin.value(), self.azimuth_spin.value(),
                self.dip_spin.value(), mt, self.kernel_combo.currentText(),
                len(self._variogram_structures) if self._variogram_structures else 0,
            )
        except Exception as exc:
            logger.warning("Failed to import variogram: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # GRID AUTO-DETECT
    # ══════════════════════════════════════════════════════════════

    def _on_auto_detect_grid(self):
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

            pad = 0.05
            dx = self.dx_spin.value()
            dy = self.dy_spin.value()
            dz = self.dz_spin.value()

            x0 = np.floor((x.min() - pad * (np.max(x) - np.min(x))) / dx) * dx
            y0 = np.floor((y.min() - pad * (np.max(y) - np.min(y))) / dy) * dy
            z0 = np.floor((z.min() - pad * (np.max(z) - np.min(z))) / dz) * dz

            x1 = np.ceil((x.max() + pad * (np.max(x) - np.min(x))) / dx) * dx
            y1 = np.ceil((y.max() + pad * (np.max(y) - np.min(y))) / dy) * dy
            z1 = np.ceil((z.max() + pad * (np.max(z) - np.min(z))) / dz) * dz

            self.nx_spin.setValue(max(1, int((x1 - x0) / dx)))
            self.ny_spin.setValue(max(1, int((y1 - y0) / dy)))
            self.nz_spin.setValue(max(1, int((z1 - z0) / dz)))
            self.x0_spin.setValue(x0)
            self.y0_spin.setValue(y0)
            self.z0_spin.setValue(z0)
            self._update_block_count()

        except Exception as exc:
            logger.warning("Auto-detect grid failed: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # RUN ESTIMATION
    # ══════════════════════════════════════════════════════════════

    def bind_controller(self, controller):
        """Bind controller and connect task_progress for real-time updates."""
        super().bind_controller(controller)
        if controller and hasattr(controller, 'signals'):
            try:
                controller.signals.task_progress.connect(self._handle_task_progress)
            except Exception:
                pass

    def _handle_task_progress(self, task_name: str, percent: int, message: str):
        """Route controller task_progress signal to our progress bars."""
        if task_name == (self._current_task or self.task_name):
            self._on_progress(percent, message)

    def _on_run(self):
        if self._current_task is not None:
            logger.warning("ARBF estimation already running — ignoring duplicate run request")
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.run_analysis()

    def run_analysis(self) -> None:
        if self._current_task is not None:
            logger.warning("ARBF: run_analysis() called while task '%s' is active — skipping", self._current_task)
            return
        self._dispatch_task(self.task_name)

    def _dispatch_task(
        self,
        task_name: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> None:
        if self._current_task is not None:
            logger.warning("ARBF: _dispatch_task('%s') blocked — '%s' already running", task_name, self._current_task)
            return
        if not self.controller:
            self.show_warning("Unavailable", "Controller is not connected; cannot run analysis.")
            return

        if not self.validate_inputs():
            return

        if params is None:
            params = self.gather_parameters()

        self.show_progress(label or f"Running {task_name.replace('_', ' ').title()}...")
        self._current_task = task_name

        try:
            self.controller.run_analysis_task(
                task=task_name,
                params=params,
                callback=self.handle_results,
            )
            if (
                hasattr(self.controller, "_active_workers")
                and task_name in self.controller._active_workers
            ):
                self._current_worker = self.controller._active_workers[task_name]
        except Exception as exc:
            logger.error("Dispatch failed for '%s': %s", task_name, exc, exc_info=True)
            self.hide_progress()
            self.show_error("Analysis Error", str(exc))
            self._current_task = None
            self._current_worker = None

    def _update_execution_controls(self) -> None:
        """Keep CV controls aligned with the main toggle."""
        run_cv = self.cb_run_cv.isChecked()
        self.cv_mode_combo.setEnabled(run_cv)
        self.cv_folds_spin.setEnabled(
            run_cv and self.cv_mode_combo.currentText() == "Spatial K-Fold",
        )

    def _auto_fix_grid_origin_if_needed(self):
        """Auto-detect grid origin from drillhole data if spinboxes are at default (0,0,0).

        When the user hasn't clicked 'Auto-Detect from Drillholes' and the
        origin spinboxes are still at their initial value of (0,0,0), the grid
        will be ~500 km away from UTM-coordinate drillholes, causing the
        footprint clip to remove ALL blocks.  Detect this and silently run
        auto-detect.
        """
        x0 = self.x0_spin.value()
        y0 = self.y0_spin.value()
        z0 = self.z0_spin.value()

        # Only intervene when all three origins are exactly at default
        if x0 != 0.0 or y0 != 0.0 or z0 != 0.0:
            return

        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            return

        # Find coordinate columns
        for cx, cy, cz in [("X", "Y", "Z"), ("x", "y", "z"), ("EAST", "NORTH", "RL")]:
            if cx in filtered_df.columns and cy in filtered_df.columns and cz in filtered_df.columns:
                x = filtered_df[cx].dropna().values
                y = filtered_df[cy].dropna().values
                z = filtered_df[cz].dropna().values
                break
        else:
            return

        data_center = (np.mean(x), np.mean(y), np.mean(z))
        # If data centroid is far from origin, the user forgot to auto-detect
        if abs(data_center[0]) < 1000 and abs(data_center[1]) < 1000:
            return  # Data is already near origin — no fix needed

        logger.info(
            "ARBF grid origin is (0,0,0) but data centroid is (%.0f, %.0f, %.0f). "
            "Auto-detecting grid from drillhole extents.",
            *data_center,
        )
        self._on_auto_detect_grid()

    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters for the ARBF estimation task."""
        # Auto-detect grid from data if origin is still at default (0,0,0)
        # and data coordinates are far from origin (UTM vs local mismatch)
        self._auto_fix_grid_origin_if_needed()

        self._publish_grid_to_registry()

        filtered_df = self._get_filtered_data()
        disc_map = {"8 (2x2x2)": 8, "27 (3x3x3)": 27, "64 (4x4x4)": 64}
        disc_density = disc_map.get(self.disc_density_combo.currentText(), 27)
        cv_mode_map = {
            "Spatial K-Fold": "spatial_kfold",
            "Fast LOO": "fast_loo",
        }

        domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}
        domain_col = domain_meta.get("domain_filter_column")
        domain_policy = "require" if domain_col else "warn"

        # Extract declustering weights if the filtered composite frame
        # has a 'declust_weight' column (populated by the Declustering
        # panel when that step has been run). Previously ARBF was
        # completely weight-blind — even when declustering ran, ARBF
        # fit on the unweighted composite distribution, biasing the
        # estimate toward clustered high-grade zones.
        declustering_weights = None
        try:
            if isinstance(filtered_df, pd.DataFrame) and "declust_weight" in filtered_df.columns:
                _w = filtered_df["declust_weight"].to_numpy(dtype=float)
                if np.all(np.isfinite(_w)) and _w.sum() > 0:
                    declustering_weights = _w
                    logger.info(
                        "ARBF: using %d declustering weights from composites "
                        "(sum=%.2f, effective N=%.1f).",
                        len(_w), float(_w.sum()),
                        float(_w.sum() ** 2 / max((_w ** 2).sum(), 1e-12)),
                    )
        except Exception as _exc:
            logger.debug("ARBF: declustering weight extraction failed: %s", _exc)

        # Fetch raw IRBF domain dict so the worker can resample onto the
        # (possibly re-computed) estimation grid. Matches the pattern used by
        # kriging/SGSIM workers wired into the rebuilt renderer pipeline.
        irbf_domain_raw = None
        try:
            reg = getattr(self, "registry", None)
            if reg is None and hasattr(self, "_get_any_registry"):
                reg = self._get_any_registry()
            if reg is not None:
                if hasattr(reg, "get_indicator_rbf_domain"):
                    irbf_domain_raw = reg.get_indicator_rbf_domain()
                if not irbf_domain_raw and hasattr(reg, "get_data"):
                    irbf_domain_raw = reg.get_data("indicator_rbf_domain", copy_data=False)
        except Exception as _exc:
            logger.debug("ARBF: failed to fetch IRBF domain from registry: %s", _exc)
            irbf_domain_raw = None

        return {
            "data": filtered_df,
            "variable": self.variable_combo.currentText(),
            "domain_column": domain_col,
            "domain_value": domain_meta.get("domain_filter_value"),
            "domain_policy": domain_policy,
            "irbf_domain_raw": irbf_domain_raw,
            "declustering_weights": declustering_weights,
            "grid_spec": {
                "nx": self.nx_spin.value(),
                "ny": self.ny_spin.value(),
                "nz": self.nz_spin.value(),
                "dx": self.dx_spin.value(),
                "dy": self.dy_spin.value(),
                "dz": self.dz_spin.value(),
                "x0": self.x0_spin.value(),
                "y0": self.y0_spin.value(),
                "z0": self.z0_spin.value(),
            },
            "use_block_model_grid": True,
            "clip_to_drill_footprint": self.chk_clip_to_footprint.isChecked(),
            "footprint_buffer_ranges": self.footprint_buffer_spin.value(),
            "kernel_type": self.kernel_combo.currentText().lower(),
            "alpha": self.alpha_spin.value(),
            "sill": self.sill_spin.value(),
            "nugget": self.nugget_spin.value(),
            "accuracy": self.accuracy_spin.value(),
            "panel_width": float(self.panel_width_spin.value()),
            "drift_type": self.drift_combo.currentText().lower(),
            "estimation_mode": "local_neighbourhood_gpr",
            "search_mode": "local" if self.search_mode_combo.currentIndex() == 0 else "global",
            "range_max": self.range_max_spin.value(),
            "range_mid": self.range_mid_spin.value(),
            "range_min": self.range_min_spin.value(),
            "azimuth": self.azimuth_spin.value(),
            "dip": self.dip_spin.value(),
            "pitch": self.pitch_spin.value(),
            "variogram_structures": getattr(self, '_variogram_structures', None),
            "auto_exponential_background": getattr(self, 'cb_auto_background', None) is None or getattr(self, 'cb_auto_background').isChecked(),
            "local_search_radii": (
                self.search_radius_1_spin.value(),
                self.search_radius_2_spin.value(),
                self.search_radius_3_spin.value(),
            ),
            "balanced_neighbourhood_selection": self.cb_balanced_search.isChecked(),
            "search_min_octants": self.search_min_octants_spin.value(),
            "search_min_octants_linear": self.search_min_octants_linear_spin.value(),
            "max_samples_per_octant": self.max_samples_per_octant_spin.value(),
            "auto_drift_max_slope_deviation": self.auto_drift_slope_spin.value(),
            "n_subdomains": 0,
            "pum_threshold": 3000,
            "subdomain_method": "kmeans",
            "overlap_factor": 2.0,
            "max_samples": self.max_samples_spin.value(),
            "min_samples": self.min_samples_spin.value(),
            "use_normal_score": self._resolve_normal_score(),
            "discretisation": self.disc_mode_combo.currentText().lower(),
            "discretisation_density": disc_density,
            "run_cv": self.cb_run_cv.isChecked(),
            "cv_mode": cv_mode_map.get(self.cv_mode_combo.currentText(), "spatial_kfold"),
            "cv_folds": self.cv_folds_spin.value(),
            "seed": self.seed_spin.value(),
            "variogram_results": self.variogram_results,
            "external_ns_transformer": getattr(self, '_external_ns_transformer', None),
            "external_ns_original_var": getattr(self, '_external_ns_original_var', None),
            "clip_min": self.clip_min_spin.value() if self.cb_grade_clip.isChecked() else None,
            "clip_max": self.clip_max_spin.value() if self.cb_grade_clip.isChecked() else None,
            "estimation_mode_name": {
                "Preview (fast)": "preview",
                "Standard": "standard",
                "Final (full uncertainty)": "final",
            }.get(self.estimation_mode_combo.currentText(), "standard"),
        }

    @staticmethod
    def _normalise_transform_method(method: Any) -> str:
        return "".join(ch.lower() for ch in str(method or "") if ch.isalnum())

    @classmethod
    def _is_normal_score_method(cls, method: Any) -> bool:
        normalised = cls._normalise_transform_method(method)
        return (
            "normalscore" in normalised
            or normalised in {"nscore", "normaltransform", "normaltransformation"}
        )

    @staticmethod
    def _looks_like_normal_score_variable(variable: str) -> bool:
        ns_suffixes = ("_NS", "_ns", "_NSCORE", "_nscore")
        return bool(variable) and any(variable.endswith(s) for s in ns_suffixes)

    @staticmethod
    def _infer_original_from_ns_suffix(variable: str) -> str:
        ns_suffixes = ("_NS", "_ns", "_NSCORE", "_nscore")
        for suffix in ns_suffixes:
            if variable.endswith(suffix):
                return variable[:-len(suffix)]
        return variable

    def _get_transformation_entries(self) -> Dict[str, Dict[str, Any]]:
        metadata = self.transformation_metadata
        if not isinstance(metadata, dict):
            return {}
        transformations = metadata.get("transformations", {})
        if isinstance(transformations, dict) and transformations:
            return {
                str(original): meta
                for original, meta in transformations.items()
                if isinstance(meta, dict)
            }
        return {
            str(original): meta
            for original, meta in metadata.items()
            if isinstance(meta, dict)
        }

    def _get_transformation_entry_for_output(
        self,
        transformed_col: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        for original_col, meta in self._get_transformation_entries().items():
            new_col = meta.get("new_col") or meta.get("transformed_col_name")
            if new_col == transformed_col:
                return original_col, meta
        return None, None

    def _find_original_column(self, transformed_col: str) -> Optional[str]:
        original_col, _meta = self._get_transformation_entry_for_output(transformed_col)
        return original_col

    def _get_registered_transformer(self, original_col: str):
        if not original_col or self.registry is None:
            return None

        try:
            if self.drillhole_data is not None:
                self._get_filtered_data()
        except Exception:
            pass

        domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}
        domain_val = domain_meta.get("domain_filter_value")

        try:
            if domain_val and hasattr(self.registry, "get_transformer_for_domain"):
                transformer = self.registry.get_transformer_for_domain(original_col, domain=domain_val)
                if transformer is not None:
                    logger.info(
                        "ARBF: Using domain-specific Grade Transform transformer for '%s' domain '%s'.",
                        original_col,
                        domain_val,
                    )
                    return transformer
            if hasattr(self.registry, "get_transformers"):
                transformers = self.registry.get_transformers()
                if transformers and original_col in transformers:
                    logger.info(
                        "ARBF: Using global Grade Transform transformer for '%s'.",
                        original_col,
                    )
                    return transformers[original_col]
        except Exception as exc:
            logger.warning(
                "ARBF: Failed to retrieve stored transformer for '%s': %s",
                original_col,
                exc,
            )
        return None

    def _resolve_external_normal_score_variable(
        self,
        variable: str,
    ) -> Tuple[Optional[str], Optional[Any], bool]:
        original_col, transform_meta = self._get_transformation_entry_for_output(variable)
        if original_col and transform_meta and self._is_normal_score_method(transform_meta.get("method")):
            return original_col, self._get_registered_transformer(original_col), True
        if self._looks_like_normal_score_variable(variable):
            return self._infer_original_from_ns_suffix(variable), None, False
        return None, None, False

    def _get_back_transformer(self, transformed_variable: str):
        original_col, _stored_transformer, has_metadata_ns = (
            self._resolve_external_normal_score_variable(transformed_variable)
        )
        if not original_col:
            if self._looks_like_normal_score_variable(transformed_variable):
                original_col = self._infer_original_from_ns_suffix(transformed_variable)
            else:
                return None

        transformer = self._get_registered_transformer(original_col)
        if transformer is None:
            source_label = "Grade Transform metadata" if has_metadata_ns else "suffix inference"
            logger.warning(
                "ARBF: No stored back-transformer found for '%s' -> '%s' (%s).",
                transformed_variable,
                original_col,
                source_label,
            )
        return transformer

    def _on_variable_selection_changed(self, variable: str):
        """Auto-detect already-transformed variables and warn/adjust NS checkbox."""
        if not variable:
            return
        original_var, stored_transformer, has_metadata_ns = (
            self._resolve_external_normal_score_variable(variable)
        )
        is_ns = has_metadata_ns or self._looks_like_normal_score_variable(variable)

        if is_ns and hasattr(self, 'cb_normal_score') and self.cb_normal_score.isChecked():
            # Auto-uncheck to prevent double-transform
            self.cb_normal_score.setChecked(False)
            original = original_var or self._infer_original_from_ns_suffix(variable)
            if has_metadata_ns and stored_transformer is not None:
                message = (
                    f"'{variable}' is already normal-scored. "
                    f"'Apply Normal-Score Transform' has been unchecked to prevent "
                    f"double-transformation. ARBF will reuse the stored Grade Transform "
                    f"normal-score transformer and back-transform to '{original}'."
                )
            elif has_metadata_ns:
                message = (
                    f"'{variable}' is already normal-scored. "
                    f"'Apply Normal-Score Transform' has been unchecked to prevent "
                    f"double-transformation. Grade Transform metadata points to raw "
                    f"'{original}', but no stored transformer was found. Re-run Grade "
                    f"Transformation or switch to raw '{original}'."
                )
            else:
                message = (
                    f"'{variable}' is already normal-scored. "
                    f"'Apply Normal-Score Transform' has been unchecked to prevent "
                    f"double-transformation. For best results, select the raw "
                    f"variable '{original}' and enable the transform."
                )
            if hasattr(self, 'toast'):
                try:
                    self.toast(message, duration=8000)
                except Exception:
                    pass
            logger.info(
                "Auto-unchecked 'Apply Normal-Score Transform' for already-NS "
                "variable '%s' (raw '%s', exact_transformer=%s).",
                variable,
                original,
                bool(stored_transformer),
            )

    def _resolve_normal_score(self) -> bool:
        """Decide whether ARBF should apply its internal normal-score transform.

        If the selected variable is already normal-scored, enabling the
        internal NS again would double-transform the data. In that case we
        resolve the stored Grade Transform transformer so the controller can
        back-transform results to the original grade units.
        """
        self._external_ns_transformer = None
        self._external_ns_original_var = None

        variable = self.variable_combo.currentText()
        if not variable:
            return self.cb_normal_score.isChecked()

        original_var, stored_transformer, has_metadata_ns = (
            self._resolve_external_normal_score_variable(variable)
        )
        is_ns_variable = has_metadata_ns or self._looks_like_normal_score_variable(variable)
        if not is_ns_variable:
            return self.cb_normal_score.isChecked()

        original_var = original_var or self._infer_original_from_ns_suffix(variable)

        if stored_transformer is not None:
            self._external_ns_transformer = stored_transformer
            self._external_ns_original_var = original_var
            logger.info(
                "ARBF: Retrieved Grade Transform panel's back-transformer "
                "for '%s' -> '%s' from registry.",
                variable,
                original_var,
            )
        elif has_metadata_ns:
            logger.warning(
                "ARBF: Grade Transform metadata identifies '%s' as a normal-score "
                "transform of '%s', but no stored transformer was found.",
                variable,
                original_var,
            )

        if self._external_ns_transformer is None:
            df = self._get_filtered_data()
            if df is not None and original_var in df.columns:
                try:
                    from ..models.transform import NormalScoreTransformer
                    import numpy as np
                    raw_vals = df[original_var].values
                    valid = np.isfinite(raw_vals)
                    if valid.sum() >= 10:
                        nst = NormalScoreTransformer()
                        nst.fit(raw_vals[valid])
                        self._external_ns_transformer = nst
                        self._external_ns_original_var = original_var
                        logger.info(
                            "ARBF: Re-fitted back-transformer from raw '%s' "
                            "(%d samples). NOTE: This may differ slightly from "
                            "the Grade Transform panel's transformer.",
                            original_var, int(valid.sum()),
                        )
                except Exception as exc:
                    logger.warning(
                        "ARBF: Failed to build back-transformer for '%s': %s",
                        original_var, exc,
                    )

        if self._external_ns_transformer is None:
            logger.warning(
                "ARBF: Variable '%s' is already NS but no back-transformer "
                "could be obtained (original '%s' not in registry or data). "
                "Results will remain in NS-space.",
                variable, original_var,
            )
        else:
            logger.info(
                "ARBF: Back-transformer ready for '%s'. "
                "Results will be in original '%s' units.",
                variable, original_var,
            )

        return False

    def validate_inputs(self) -> bool:
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            self.show_warning("No Data", "Load data first (Step 1).")
            return False
        if not self.variable_combo.currentText():
            self.show_warning("No Variable", "Select a grade variable (Step 1).")
            return False

        # Check for collocated composites — log info only.
        # The controller auto-fixes this using FROM/TO depth columns
        # (Z_fixed = collar_Z - midpoint_depth), so no user intervention
        # is needed.  Only warn if FROM/TO columns are missing (no auto-fix).
        try:
            import numpy as np
            from scipy.spatial import cKDTree
            coord_cols = [c for c in ('X', 'Y', 'Z') if c in filtered_df.columns]
            if len(coord_cols) == 3:
                coords = filtered_df[coord_cols].values
                tree = cKDTree(coords)
                n_collocated = len(tree.query_pairs(r=0.1))
                pct = n_collocated / max(len(filtered_df), 1) * 100
                if pct > 20:
                    has_from_to = any(
                        fc in filtered_df.columns and tc in filtered_df.columns
                        for fc, tc in [("From", "To"), ("FROM", "TO"), ("from", "to")]
                    )
                    if has_from_to:
                        logger.info(
                            "Collocated composites detected (%d pairs, %.0f%%) — "
                            "will auto-fix using FROM/TO depth columns.",
                            n_collocated, pct,
                        )
                    else:
                        QMessageBox.warning(
                            self, "Collocated Composites",
                            f"{n_collocated} composite pairs ({pct:.0f}%) share the "
                            f"same XYZ location, and no FROM/TO columns are available "
                            f"for automatic correction.\n\n"
                            f"Results may be unreliable. Consider desurveying your "
                            f"drillholes before estimation.",
                        )
        except Exception:
            pass

        # ── Parameter quality warnings (dismissable) ─────────────────
        warnings = []

        # 1. Isotropic ranges
        r_max = self.range_max_spin.value()
        r_mid = self.range_mid_spin.value()
        r_min = self.range_min_spin.value()
        r_hi = max(r_max, r_mid, r_min)
        r_lo = max(min(r_max, r_mid, r_min), 1e-6)
        if r_hi / r_lo < 1.1:
            warnings.append(
                "\u2022 Ranges are isotropic (all within 10%) \u2014 the search "
                "ellipsoid is a sphere. Set different ranges from your "
                "variography to honour geological anisotropy."
            )

        # 2. Zero rotation with anisotropic ranges
        if r_hi / r_lo > 1.5:
            az = abs(self.azimuth_spin.value())
            dip = abs(self.dip_spin.value())
            pitch = abs(self.pitch_spin.value())
            if az < 1e-6 and dip < 1e-6 and pitch < 1e-6:
                warnings.append(
                    f"\u2022 Anisotropy ratio is {r_hi / r_lo:.1f}:1 but all "
                    f"rotation angles are 0\u00b0. The ellipsoid is aligned with "
                    f"grid axes, not geology."
                )

        # 3. Nugget-to-sill ratio
        sill_val = self.sill_spin.value()
        nugget_val = self.nugget_spin.value()
        if sill_val > 0:
            nug_ratio = nugget_val / (sill_val + nugget_val)
            if nug_ratio > 0.5:
                warnings.append(
                    f"\u2022 Nugget/sill ratio is {nug_ratio * 100:.0f}% \u2014 "
                    f"the estimator will treat most variation as noise, "
                    f"producing over-smoothed grades."
                )
            elif nugget_val == 0:
                warnings.append(
                    "\u2022 Nugget is exactly 0 with sill > 0. This forces "
                    "exact interpolation through every data point, which "
                    "may overfit noisy assays."
                )

        # 4. No spatial filtering
        if not self.chk_clip_to_footprint.isChecked():
            warnings.append(
                "\u2022 Footprint clipping is OFF \u2014 the entire bounding box "
                "will be estimated, including barren rock and air."
            )

        if warnings:
            msg = (
                "These parameters may produce geologically "
                "unreliable results:\n\n"
                + "\n\n".join(warnings)
                + "\n\nContinue anyway?"
            )
            reply = QMessageBox.warning(
                self,
                "Parameter Quality Warnings",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        return True

    # Override base class progress dialog — use our inline progress bars instead
    def show_progress(self, message: str) -> None:
        """Show progress via inline bars, NOT the modal QProgressDialog."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(message)
        self._global_progress_bar.setVisible(True)
        self._global_progress_bar.setValue(0)
        self._global_progress_label.setText(message)

    def hide_progress(self) -> None:
        """Hide inline progress bars."""
        self.progress_bar.setVisible(False)
        self._global_progress_bar.setVisible(False)

    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle ARBF estimation results — route mesh through the
        request_visualization signal so the main window's rebuilt
        visualize_sgsim_results path handles the block-model add,
        property-panel combo refresh, and clim/colormap sync."""
        try:
            self.progress_bar.setVisible(False)
            self._global_progress_bar.setVisible(False)

            vis = payload.get("visualization", {}) or {}
            mesh = vis.get("mesh")
            if mesh is None:
                logger.warning("on_results: no mesh in payload")
                return

            property_name = payload.get("property_name", "ARBF_estimate")
            layer_name = vis.get("layer_name", f"ARBF: {property_name}")

            # Store results FIRST so downstream handlers can read them
            self.arbf_results = payload

            # Primary path: emit request_visualization → main_window routes
            # through the rebuilt add_block_model_layer pipeline.
            emitted = False
            try:
                self.request_visualization.emit(mesh, property_name)
                emitted = True
            except Exception as exc:
                logger.warning(
                    "on_results: request_visualization failed (%s), falling back to direct renderer",
                    exc,
                )

            # Fallback: walk parent hierarchy to find the renderer.
            if not emitted:
                widget = self
                renderer = None
                for _ in range(20):
                    widget = widget.parent()
                    if widget is None:
                        break
                    if hasattr(widget, "renderer"):
                        renderer = widget.renderer
                        break
                    if hasattr(widget, "viewer_widget") and hasattr(widget.viewer_widget, "renderer"):
                        renderer = widget.viewer_widget.renderer
                        break
                if renderer is None:
                    logger.warning("on_results: renderer not found (signal+fallback both failed)")
                    return
                renderer.add_block_model_layer(
                    mesh,
                    property_name=property_name,
                    layer_name=layer_name,
                    source={
                        'kind': 'block_model_layer',
                        'registry_key': 'arbf_results',
                        'property_name': property_name,
                        'layer_name': layer_name,
                    },
                )

            # Populate result tabs (Summary / Defensibility / CV Scatter / Swaths / Log)
            for _populate_fn, _tab_name in (
                (self._apply_geostatistical_gate, "verdict banner"),
                (self._populate_summary, "Summary"),
                (self._populate_defensibility, "Defensibility"),
                (self._populate_cv_scatter, "CV Scatter"),
                (self._populate_swath, "Support Swaths"),
                (self._populate_log, "Engine Log"),
            ):
                try:
                    _populate_fn(payload)
                except Exception as _exc:
                    logger.debug("on_results: %s populate failed: %s", _tab_name, _exc)

            # Persist to registry so re-opening the panel restores results
            try:
                reg = getattr(self, "registry", None)
                if reg is not None:
                    self._register_arbf_to_registry(payload)
            except Exception as exc:
                logger.debug("on_results: failed to persist ARBF results to registry: %s", exc)

            metadata = payload.get("metadata", {}) or {}
            msg = metadata.get("message", f"ARBF results loaded: {property_name}")
            if hasattr(self, "progress_label"):
                self.progress_label.setText(msg)
            logger.info("on_results: rendered '%s' as '%s'", property_name, layer_name)
        except Exception as e:
            logger.error("on_results failed: %s", e, exc_info=True)

    def handle_results(self, payload: Dict[str, Any]) -> None:
        # Reset task state FIRST to prevent re-dispatch loops
        self._current_task = None
        self._current_worker = None
        # Re-enable the Run button
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run ARBF Estimation")
        self.on_results(payload)

    def _on_progress(self, percent: int, message: str):
        # Update both the local (Run tab) and global (right panel) progress bars
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

        self._global_progress_bar.setVisible(True)
        self._global_progress_bar.setValue(percent)
        self._global_progress_label.setText(f"{percent}% — {message}")

    def _populate_summary(self, payload: Dict[str, Any]):
        """Build a text summary of the estimation results."""
        lines = []
        lines.append("=" * 60)
        lines.append(str(payload.get("method", "ARBF Estimation")).upper() + " SUMMARY")
        lines.append("=" * 60)

        lines.append(f"Method:         {payload.get('method', 'ARBF Estimation')}")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        diagnostics = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
        if metadata:
            workflow_kind = str(metadata.get("workflow_kind", "estimation")).replace("_", " ").title()
            lines.append(f"Workflow:       {workflow_kind}")
        if diagnostics:
            lines.append(
                f"Estimator:      {diagnostics.get('estimation_mode', metadata.get('estimation_mode', '?'))}"
            )
            lines.append(
                f"Drift used:     {diagnostics.get('effective_drift_type', metadata.get('drift_type', '?'))}"
            )
            if diagnostics.get("block_domain_assignment"):
                lines.append(f"Block domains:  {diagnostics.get('block_domain_assignment')}")

        gate = payload.get("geostatistical_gate") or {}
        if gate:
            lines.append("")
            lines.append("--- Geostatistical Verdict ---")
            lines.append(f"Status:         {str(gate.get('overall_status', 'unknown')).upper()}")
            lines.append(f"Headline:       {gate.get('headline', '')}")
            lines.append(f"Summary:        {gate.get('summary', '')}")

        audit = payload.get("audit_record")
        if audit:
            rec = audit if isinstance(audit, dict) else (audit.__dict__ if hasattr(audit, '__dict__') else {})
            lines.append(f"\nComposites:     {rec.get('num_composites', '?')}")
            lines.append(f"Blocks est'd:   {rec.get('n_blocks_estimated', '?')} / {rec.get('n_blocks_total', '?')}")
            lines.append(f"Elapsed:        {rec.get('elapsed_seconds', 0):.1f}s")

            lines.append(f"\n--- Cross-Validation ---")
            lines.append(f"Slope:          {rec.get('cv_slope_of_regression', 0):.4f}")
            lines.append(f"R²:             {rec.get('cv_r_squared', 0):.4f}")
            lines.append(f"RMSE:           {rec.get('cv_rmse', 0):.4f}")
            lines.append(f"Mean Error:     {rec.get('cv_mean_error', 0):.6f}")
            lines.append(f"Cond. Bias:     {rec.get('conditional_bias_binned_slope', 0):.4f}")
            lines.append(f"Support RMSE:   {rec.get('support_swath_mean_rmse', 0):.4f}")

            lines.append(f"\n--- Classification Guidance ---")
            lines.append(f"High conf:      {rec.get('measured_blocks', 0):,}")
            lines.append(f"Moderate conf:  {rec.get('indicated_blocks', 0):,}")
            lines.append(f"Low conf:       {rec.get('inferred_blocks', 0):,}")
            lines.append(f"Very low conf:  {rec.get('unclassified_blocks', 0):,}")

            lines.append(f"\n--- Block Support ---")
            lines.append(f"Support ratio:  {rec.get('support_ratio', 0):.4f}")
            lines.append(f"Data std:       {rec.get('sigma_point', 0):.4f}")
            lines.append(f"Block std:      {rec.get('sigma_block', 0):.4f}")

            # ── Gate 3: Panel Reproduction (support-aware) ──────────────
            g3_status = str(rec.get("gate3_status", "unknown")).upper()
            g3_gb = rec.get("gate3_panel_grade_bias")
            g3_mb = rec.get("gate3_panel_metal_bias")
            g3_nv = rec.get("gate3_valid_panels", 0)
            g3_nt = rec.get("gate3_total_panels", 0)
            lines.append(f"\n--- Gate 3: Panel Reproduction ---")
            lines.append(f"Status:         {g3_status}")
            if g3_gb is not None and np.isfinite(g3_gb):
                lines.append(f"Grade bias:     {g3_gb:+.1%}")
            if g3_mb is not None and np.isfinite(g3_mb):
                lines.append(f"Metal bias:     {g3_mb:+.1%}")
            lines.append(f"Valid panels:   {g3_nv} / {g3_nt}")
            if g3_status in ("WARN", "FAIL"):
                lines.append(
                    "[Advisory] Panel reproduction is biased — try the "
                    "Tight Neighbourhood preset (linear drift + tighter "
                    "radii). Persistent bias after that points to heavy-"
                    "tail / top-cut review or NS back-transform issues."
                )

            # Over-smoothing advisory: block std should not collapse
            # more than ~70 % below point std. Constant drift + wide
            # neighbourhood is the usual cause.
            try:
                _sp = float(rec.get("sigma_point", 0) or 0)
                _sb = float(rec.get("sigma_block", 0) or 0)
                _dr = str(rec.get("effective_drift_type", "") or "").lower()
                if _sp > 0 and (_sb / _sp) < 0.30 and _dr == "constant":
                    lines.append(
                        "\n[Advisory] Block variance collapsed to "
                        f"{_sb / _sp:.0%} of point variance with constant "
                        "drift — try the 'Tight neighbourhood' preset "
                        "(linear drift + tighter radii) to reduce "
                        "over-smoothing."
                    )
            except Exception:
                pass

        stitching_variance = payload.get("stitching_variance")
        if stitching_variance is not None:
            sv = np.asarray(stitching_variance)
            valid_sv = sv[np.isfinite(sv)]
            if valid_sv.size:
                lines.append(f"\n--- RBF Uncertainty ---")
                lines.append(f"Residual Var:   {np.nanmean(np.asarray(payload.get('variances'))):.4f}")
                lines.append(f"Stitching Var:  {np.mean(valid_sv):.4f}")
                total_blending = payload.get("total_blending_variance")
                if total_blending is not None:
                    tb = np.asarray(total_blending)
                    valid_tb = tb[np.isfinite(tb)]
                    if valid_tb.size:
                        lines.append(f"Total Blend:    {np.mean(valid_tb):.4f}")

        simulation_diagnostics = payload.get("simulation_diagnostics")
        if simulation_diagnostics:
            lines.append(f"\n--- Conditional Simulation ---")
            lines.append(f"Realizations:   {simulation_diagnostics.get('n_realizations', '?')}")
            lines.append(f"Sim Seed:       {simulation_diagnostics.get('simulation_seed', '?')}")
            lines.append(
                f"Cond. Mean N:   {simulation_diagnostics.get('mean_conditioning_size', 0):.2f}"
            )
            lines.append(
                f"Uncond. Nodes:  {simulation_diagnostics.get('mean_unconditional_nodes', 0):.2f}"
            )

        grades = payload.get("grades")
        if grades is not None:
            g = np.asarray(grades)
            valid = g[~np.isnan(g)] if np.any(np.isnan(g)) else g
            lines.append(f"\n--- Grade Statistics ---")
            lines.append(f"N informed:     {len(valid):,}")
            lines.append(f"Min:            {np.min(valid):.2f}")
            lines.append(f"Mean:           {np.mean(valid):.2f}")
            lines.append(f"Median:         {np.median(valid):.2f}")
            lines.append(f"Max:            {np.max(valid):.2f}")
            lines.append(f"Std:            {np.std(valid):.2f}")

        lines.append("\n" + "=" * 60)
        self._summary_text.setPlainText("\n".join(lines))

    def _populate_cv_scatter(self, payload: Dict[str, Any]):
        """Populate CV scatter tab with a modern actual-vs-estimated plot."""
        cv = payload.get("cv_result")
        if cv is None:
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            # Close previous figure to free matplotlib resources
            if self._cv_fig is not None:
                plt.close(self._cv_fig)
                self._cv_fig = None

            # Clear old content
            while self._cv_layout.count():
                w = self._cv_layout.takeAt(0).widget()
                if w:
                    w.deleteLater()

            # ── Extract data ────────────────────────────────────────
            actual = np.asarray(
                cv.actual if hasattr(cv, 'actual') else cv.get('actual', []),
                dtype=float,
            )
            estimated = np.asarray(
                cv.estimated if hasattr(cv, 'estimated') else cv.get('estimated', []),
                dtype=float,
            )
            slope = float(
                cv.slope_of_regression if hasattr(cv, 'slope_of_regression')
                else cv.get('slope_of_regression', 1.0)
            )
            intercept = float(
                cv.intercept if hasattr(cv, 'intercept')
                else cv.get('intercept', 0.0)
            )
            r2 = float(
                cv.r_squared if hasattr(cv, 'r_squared')
                else cv.get('r_squared', np.nan)
            )
            rmse = float(
                cv.rmse if hasattr(cv, 'rmse')
                else cv.get('rmse', np.nan)
            )

            mask = np.isfinite(actual) & np.isfinite(estimated)
            actual, estimated = actual[mask], estimated[mask]
            n_pts = len(actual)
            if n_pts == 0:
                return

            # ── Theme-aware colours ─────────────────────────────────
            try:
                from .design_tokens import tokens
                c = tokens.colors()
                bg = c.BG_SURFACE
                txt = c.TEXT_PRIMARY
                txt2 = c.TEXT_SECONDARY
                border = c.BORDER_SUBTLE
                accent = c.ACCENT
            except Exception:
                bg, txt, txt2, border, accent = (
                    "#252526", "#d4d4d4", "#a0a0a0", "#3c3c3c", "#4FC3F7"
                )

            fig = Figure(figsize=(7, 6), dpi=110, facecolor=bg)
            self._cv_fig = fig
            ax = fig.add_subplot(111)
            ax.set_facecolor(bg)

            # ── Density scatter (hexbin) ────────────────────────────
            if n_pts > 200:
                hb = ax.hexbin(
                    estimated, actual,
                    gridsize=40, cmap="inferno", mincnt=1,
                    linewidths=0.2, edgecolors="none",
                )
                cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
                cb.set_label("Point density", fontsize=8, color=txt2)
                cb.ax.tick_params(labelsize=7, colors=txt2)
                cb.outline.set_edgecolor(border)
            else:
                ax.scatter(
                    estimated, actual,
                    s=18, alpha=0.6, c=accent,
                    edgecolors="white", linewidths=0.3,
                )

            # ── Reference lines ─────────────────────────────────────
            lo = min(np.min(actual), np.min(estimated))
            hi = max(np.max(actual), np.max(estimated))
            pad = (hi - lo) * 0.04
            lo, hi = lo - pad, hi + pad

            ax.plot(
                [lo, hi], [lo, hi], color="#888888",
                linewidth=1.2, linestyle="--", alpha=0.7, label="1:1",
                zorder=3,
            )
            ax.plot(
                [lo, hi],
                [intercept + slope * lo, intercept + slope * hi],
                color="#FF6B6B", linewidth=2, alpha=0.9,
                label=f"Regression  (slope {slope:.3f})",
                zorder=4,
            )

            # ── Axis styling ────────────────────────────────────────
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Estimated", fontsize=10, color=txt, labelpad=6)
            ax.set_ylabel("Actual", fontsize=10, color=txt, labelpad=6)
            ax.set_title(
                "Cross-Validation: Actual vs Estimated",
                fontsize=11, fontweight="bold", color=txt, pad=10,
            )
            ax.tick_params(colors=txt2, labelsize=8)
            ax.grid(True, linestyle=":", alpha=0.15, color=txt2)
            for spine in ax.spines.values():
                spine.set_color(border)

            ax.legend(
                fontsize=8, loc="upper left",
                framealpha=0.8, edgecolor=border,
                facecolor=bg, labelcolor=txt2,
            )

            # ── Stats annotation box ────────────────────────────────
            stats_lines = [f"n = {n_pts:,}"]
            if np.isfinite(r2):
                stats_lines.append(f"R\u00b2 = {r2:.4f}")
            if np.isfinite(rmse):
                stats_lines.append(f"RMSE = {rmse:.4f}")
            stats_lines.append(f"Slope = {slope:.4f}")
            stats_text = "\n".join(stats_lines)
            ax.text(
                0.97, 0.03, stats_text,
                transform=ax.transAxes,
                fontsize=8, fontfamily="monospace",
                verticalalignment="bottom",
                horizontalalignment="right",
                color=txt2,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=bg, edgecolor=border, alpha=0.9,
                ),
            )

            fig.tight_layout()
            canvas = FigureCanvasQTAgg(fig)
            self._cv_layout.addWidget(canvas)
        except Exception as exc:
            logger.warning("CV scatter plot failed: %s", exc, exc_info=True)

    def _populate_swath(self, payload: Dict[str, Any]):
        """Cache swath data and draw for the currently selected axis."""
        swath_data = (
            payload.get("support_swath_data")
            or payload.get("swath_plots")
            or payload.get("swath_data")
        )
        if swath_data is None:
            return

        # Normalise into a dict keyed by lower-case axis name
        if hasattr(swath_data, "axes"):
            self._swath_payload_cache = dict(swath_data.axes)
        elif isinstance(swath_data, dict):
            self._swath_payload_cache = dict(swath_data)
        elif isinstance(swath_data, list):
            self._swath_payload_cache = {
                (sd.axis if hasattr(sd, 'axis') else sd.get('axis', f'{i}')).lower(): sd
                for i, sd in enumerate(swath_data)
            }
        else:
            return

        # Ensure the default button is active and draw
        if self._current_swath_axis.lower() not in self._swath_payload_cache:
            self._current_swath_axis = next(iter(self._swath_payload_cache)).upper()
            for k, btn in self._swath_axis_btns.items():
                btn.setChecked(k == self._current_swath_axis)
        self._draw_swath(self._current_swath_axis)

    def _on_swath_axis_toggle(self, axis: str):
        """Handle X / Y / Z button click."""
        self._current_swath_axis = axis
        for k, btn in self._swath_axis_btns.items():
            btn.setChecked(k == axis)
        if self._swath_payload_cache is not None:
            self._draw_swath(axis)

    def _on_swath_mode_toggle(self, mode: str):
        """Handle Grade / Metal sub-mode button click."""
        self._current_swath_mode = mode
        for k, btn in self._swath_mode_btns.items():
            btn.setChecked(k == mode)
        if self._swath_payload_cache is not None:
            self._draw_swath(self._current_swath_axis)

    def _draw_swath(self, axis: str):
        """Dispatch to the Grade or Metal sub-view for this axis."""
        sd = self._swath_payload_cache.get(axis.lower())
        if sd is None:
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            import matplotlib.pyplot as plt

            if self._swath_fig is not None:
                plt.close(self._swath_fig)
                self._swath_fig = None

            while self._swath_layout.count():
                w = self._swath_layout.takeAt(0).widget()
                if w:
                    w.deleteLater()

            def _field(name, default):
                if hasattr(sd, name):
                    return getattr(sd, name)
                if isinstance(sd, dict):
                    return sd.get(name, default)
                return default

            try:
                from .design_tokens import tokens
                c = tokens.colors()
                bg = c.BG_SURFACE
                txt = c.TEXT_PRIMARY
                txt2 = c.TEXT_SECONDARY
                border = c.BORDER_SUBTLE
            except Exception:
                bg, txt, txt2, border = "#252526", "#d4d4d4", "#a0a0a0", "#3c3c3c"

            theme = dict(bg=bg, txt=txt, txt2=txt2, border=border)
            fig = Figure(figsize=(9, 5), dpi=110, facecolor=bg)
            self._swath_fig = fig
            ax = fig.add_subplot(111)
            ax.set_facecolor(bg)

            mode = getattr(self, "_current_swath_mode", "Grade")
            if mode == "Metal":
                ARBFEstimationPanel._render_swath_metal(ax, _field, theme)
            else:
                ARBFEstimationPanel._render_swath_grade(ax, _field, theme)

            fig.tight_layout()
            canvas = FigureCanvasQTAgg(fig)
            self._swath_layout.addWidget(canvas)
        except Exception as exc:
            logger.warning("Swath plot failed: %s", exc, exc_info=True)

    @staticmethod
    def _render_swath_grade(ax, _field, theme: dict) -> None:
        """Primary ARBF QA plot: volume-weighted block panel mean vs
        declustered composite panel mean. Raw composite shown as thin
        faint context line only.
        """
        bg = theme["bg"]; txt = theme["txt"]; txt2 = theme["txt2"]; border = theme["border"]
        pos = np.asarray(_field('slice_positions', []), dtype=float)
        # Prefer volume-weighted block mean when the engine supplies it
        est_vw = np.asarray(_field('mean_estimated_volume_weighted', []), dtype=float)
        est_fallback = np.asarray(_field('mean_estimated', []), dtype=float)
        if est_vw.size == pos.size and np.any(np.isfinite(est_vw)):
            est = est_vw
            est_label = "Block panel mean (volume-weighted)"
        else:
            est = est_fallback
            est_label = "Estimated (block mean)"

        # Field semantics differ between the two swath sources:
        #   * CP-patched SupportSwathData: ``mean_actual`` is the
        #     declustered primary reference; ``mean_actual_raw`` is the
        #     raw composite context line.
        #   * Basic _build_swath_data dict: ``mean_actual`` is the raw
        #     composite and ``mean_actual_declustered`` is the green
        #     primary.
        # We detect the SupportSwathData path by the presence of
        # ``mean_actual_raw`` and remap accordingly.
        raw_from_support = _field('mean_actual_raw', None)
        if raw_from_support is not None:
            raw_act = np.asarray(raw_from_support, dtype=float)
            decl_act = np.asarray(_field('mean_actual', []), dtype=float)
            # Prefer an explicit declustered field if present.
            _decl_explicit = _field('mean_actual_declustered', None)
            if _decl_explicit is not None:
                decl_act = np.asarray(_decl_explicit, dtype=float)
        else:
            raw_act = np.asarray(_field('mean_actual', []), dtype=float)
            decl_act = np.asarray(
                _field('mean_actual_declustered', _field('mean_declustered', [])),
                dtype=float,
            )

        # Panel-support bars: prefer the CP-patched support fields, fall
        # back to the per-panel counts from the basic swath.
        block_vol_slice = np.asarray(
            _field('block_volume_per_slice', []), dtype=float,
        )
        comp_weight_slice = np.asarray(
            _field('composite_weight_per_slice', []), dtype=float,
        )
        n_block = np.asarray(
            _field('n_block', _field('n_panels_per_slice', [])),
            dtype=float,
        )
        n_comp = np.asarray(
            _field('n_composite',
                   _field('n_composites_per_slice', _field('n_comp', []))),
            dtype=float,
        )
        axis_name = _field('axis', "")
        panel_width = _field('panel_width', None)

        est_color = "#4FC3F7"  # blue (block)
        act_color = "#FFB74D"  # orange (raw composite, context)
        decl_color = "#66BB6A"  # green (declustered composite, primary reference)

        # Raw composite — demoted to faint dotted context line.
        if raw_act.size == pos.size and np.any(np.isfinite(raw_act)):
            ax.plot(
                pos, raw_act,
                color=act_color, linewidth=1.2, marker="s",
                markersize=4, alpha=0.55, linestyle=":",
                label="Raw composite (context)", zorder=2,
            )

        # Declustered composite — primary reference.
        if decl_act.size == pos.size and np.any(np.isfinite(decl_act)):
            ax.plot(
                pos, decl_act,
                color=decl_color, linewidth=2.2, marker="^",
                markersize=6, markerfacecolor=bg,
                markeredgecolor=decl_color, markeredgewidth=1.5,
                linestyle="--",
                label="Declustered composite (primary reference)", zorder=3,
            )

        # Block panel mean — primary result.
        ax.plot(
            pos, est,
            color=est_color, linewidth=2.4, marker="o",
            markersize=6, markerfacecolor="white",
            markeredgecolor=est_color, markeredgewidth=1.6,
            label=est_label, zorder=4,
        )

        # Counts on twinx so the user can see panel support.
        if n_comp.size == pos.size and np.any(n_comp > 0):
            ax2 = ax.twinx()
            ax2.set_facecolor(bg)
            bw = (pos[1] - pos[0]) * 0.3 if pos.size > 1 else 1.0
            ax2.bar(pos, n_comp, width=bw, color=act_color, alpha=0.18,
                    zorder=0, edgecolor="none", label="Composite count")
            if n_block.size == pos.size and np.any(n_block > 0):
                ax2.bar(pos + bw / 2, n_block, width=bw, color=est_color,
                        alpha=0.18, zorder=0, edgecolor="none",
                        label="Informed block count")
            ax2.set_ylabel("Count per panel", fontsize=9, color=txt2, labelpad=4)
            ax2.tick_params(colors=txt2, labelsize=7)
            for spine in ax2.spines.values():
                spine.set_color(border)

        # Primary bias: block vs declustered.
        valid = np.isfinite(est) & np.isfinite(decl_act)
        if not np.any(valid) and decl_act.size != pos.size:
            valid = np.isfinite(est) & np.isfinite(raw_act)
            ref_for_bias = raw_act
        else:
            ref_for_bias = decl_act

        pw_suffix = f", panel width {float(panel_width):.0f} m" if panel_width else ""
        ax.set_xlabel(f"{axis_name} coordinate", fontsize=10, color=txt, labelpad=6)
        ax.set_ylabel("Cu grade", fontsize=10, color=txt, labelpad=6)
        ax.set_title(
            f"Panel Grade Swath — {axis_name} axis{pw_suffix}",
            fontsize=12, fontweight="bold", color=txt, pad=10,
        )
        ax.tick_params(colors=txt2, labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.15, color=txt2)
        for spine in ax.spines.values():
            spine.set_color(border)
        ax.legend(
            fontsize=8, loc="best",
            framealpha=0.85, edgecolor=border, facecolor=bg,
            labelcolor=txt2,
        )

        if np.any(valid):
            mean_est = float(np.nanmean(est[valid]))
            mean_ref = float(np.nanmean(ref_for_bias[valid]))
            bias = mean_est - mean_ref
            rel_bias = (bias / mean_ref * 100) if abs(mean_ref) > 1e-12 else 0
            ref_name = "decl" if ref_for_bias is decl_act else "raw"
            info = (
                f"Mean block:   {mean_est:.2f}\n"
                f"Mean {ref_name}:    {mean_ref:.2f}\n"
                f"Bias ({ref_name}):  {rel_bias:+.1f}%\n"
                f"Valid panels: {int(np.sum(valid))}/{pos.size}"
            )
            ax.text(
                0.98, 0.97, info, transform=ax.transAxes,
                fontsize=8, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="right",
                color=txt2,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=bg, edgecolor=border, alpha=0.9,
                ),
            )

    @staticmethod
    def _render_swath_metal(ax, _field, theme: dict) -> None:
        """Contained-metal view: paired bars per panel plus a bias line.
        Both sides use the same volumetric support (block panel volume).
        """
        bg = theme["bg"]; txt = theme["txt"]; txt2 = theme["txt2"]; border = theme["border"]
        pos = np.asarray(_field('slice_positions', []), dtype=float)

        # Two possible metal sources:
        #   * Basic _build_swath_data dict provides per-slice ``metal_block``
        #     and ``metal_composite_declustered`` directly.
        #   * CP-patched SupportSwathData provides per-slice ``mean_estimated``
        #     + ``mean_actual`` (declustered) + ``block_volume_per_slice``.
        #     Synthesise metal = grade × block_volume_per_slice so both
        #     sources give the same volumetric quantity.
        met_blk = np.asarray(_field('metal_block', []), dtype=float)
        met_dec = np.asarray(_field('metal_composite_declustered', []), dtype=float)
        if met_blk.size != pos.size or met_dec.size != pos.size:
            vol_slice = np.asarray(
                _field('block_volume_per_slice', []), dtype=float,
            )
            if vol_slice.size == pos.size:
                est_slice = np.asarray(_field('mean_estimated', []), dtype=float)
                # mean_actual on SupportSwathData is already the declustered
                # primary reference; mean_actual_declustered is an alias.
                dec_slice = np.asarray(
                    _field('mean_actual_declustered',
                           _field('mean_actual', [])),
                    dtype=float,
                )
                if est_slice.size == pos.size and dec_slice.size == pos.size:
                    met_blk = est_slice * vol_slice
                    met_dec = dec_slice * vol_slice
        axis_name = _field('axis', "")
        panel_width = _field('panel_width', None)

        est_color = "#4FC3F7"
        decl_color = "#66BB6A"

        if met_blk.size != pos.size or met_dec.size != pos.size:
            ax.text(
                0.5, 0.5,
                "No contained-metal data available.\n"
                "Run ARBF on a block model with volumes to populate.",
                ha="center", va="center",
                transform=ax.transAxes, color=txt2, fontsize=10,
            )
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(border)
            return

        valid = np.isfinite(met_blk) & np.isfinite(met_dec)
        bw = (pos[1] - pos[0]) * 0.35 if pos.size > 1 else 1.0

        ax.bar(
            pos[valid] - bw / 2, met_dec[valid], width=bw,
            color=decl_color, alpha=0.85, edgecolor="none",
            label="Composite metal (decl × panel volume)",
        )
        ax.bar(
            pos[valid] + bw / 2, met_blk[valid], width=bw,
            color=est_color, alpha=0.85, edgecolor="none",
            label="Block metal (Σ volume × grade)",
        )

        # Per-panel bias line on twinx so the user can see which
        # panels carry the aggregate Gate 3 bias.
        if np.any(valid):
            ax2 = ax.twinx()
            ax2.set_facecolor(bg)
            ratio = np.full(pos.size, np.nan, dtype=float)
            denom = met_dec[valid]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_v = (met_blk[valid] - denom) / np.where(
                    np.abs(denom) > 1e-12, denom, np.nan,
                )
            ratio[valid] = ratio_v
            ax2.axhline(0.0, color=txt2, linewidth=0.8, linestyle=":", alpha=0.6)
            ax2.plot(
                pos, ratio * 100.0,
                color="#E57373", linewidth=1.6, marker="d",
                markersize=5, markerfacecolor=bg,
                markeredgecolor="#E57373", markeredgewidth=1.2,
                label="Per-panel bias %", zorder=5,
            )
            ax2.set_ylabel("Bias (%)", fontsize=9, color=txt2, labelpad=4)
            ax2.tick_params(colors=txt2, labelsize=7)
            for spine in ax2.spines.values():
                spine.set_color(border)

        pw_suffix = f", panel width {float(panel_width):.0f} m" if panel_width else ""
        ax.set_xlabel(f"{axis_name} coordinate", fontsize=10, color=txt, labelpad=6)
        ax.set_ylabel("Contained metal (grade × volume)",
                      fontsize=10, color=txt, labelpad=6)
        ax.set_title(
            f"Panel Metal — {axis_name} axis{pw_suffix}",
            fontsize=12, fontweight="bold", color=txt, pad=10,
        )
        ax.tick_params(colors=txt2, labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.15, color=txt2, axis="y")
        for spine in ax.spines.values():
            spine.set_color(border)
        ax.legend(
            fontsize=8, loc="upper left",
            framealpha=0.85, edgecolor=border, facecolor=bg,
            labelcolor=txt2,
        )

        # Aggregate-bias annotation (total block / total decl ratio).
        if np.any(valid):
            tot_blk = float(np.sum(met_blk[valid]))
            tot_dec = float(np.sum(met_dec[valid]))
            if abs(tot_dec) > 1e-12:
                agg_bias = (tot_blk - tot_dec) / tot_dec * 100.0
            else:
                agg_bias = float("nan")
            info = (
                f"Σ block metal:  {tot_blk:.2e}\n"
                f"Σ decl metal:   {tot_dec:.2e}\n"
                f"Total bias:     {agg_bias:+.1f}%\n"
                f"Valid panels:   {int(np.sum(valid))}/{pos.size}"
            )
            ax.text(
                0.98, 0.97, info, transform=ax.transAxes,
                fontsize=8, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="right",
                color=txt2,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=bg, edgecolor=border, alpha=0.9,
                ),
            )

    def _gate_allows(self, action: str) -> bool:
        gate = self._geostatistical_gate or {}
        actions = gate.get("actions", {}) if isinstance(gate, dict) else {}
        return bool(actions.get(action, False))

    def _apply_geostatistical_gate(self, payload: Dict[str, Any]) -> None:
        gate = payload.get("geostatistical_gate") or {}
        self._geostatistical_gate = gate

        overall = str(gate.get("overall_status", "unknown")).lower()
        headline = gate.get("headline", "No geostatistical verdict available.")
        summary = gate.get("summary", "")

        style_map = {
            "pass": "background: #EAF7EE; border: 1px solid #B8E0C2; color: #1E6B3A;",
            "warn": "background: #FFF6E5; border: 1px solid #F3CF8A; color: #8A5A00;",
            "fail": "background: #FDECEC; border: 1px solid #F2B6B6; color: #8C1D18;",
        }
        style = style_map.get(
            overall,
            "background: #F5F5F5; border: 1px solid #D9D9D9; color: #444;",
        )
        self._verdict_banner.setStyleSheet(
            style + " border-radius: 6px; padding: 8px;"
        )
        self._verdict_banner.setText(f"{headline}\n{summary}".strip())

        self.btn_visualise.setEnabled(self._gate_allows("allow_visualise"))
        self.btn_register.setEnabled(self._gate_allows("allow_register"))
        self.btn_jorc.setEnabled(self._gate_allows("allow_jorc_export"))
        self.btn_json.setEnabled(self._gate_allows("allow_audit_export"))
        self.btn_csv.setEnabled(self._gate_allows("allow_csv_export"))

        if overall == "fail":
            self.btn_register.setToolTip(
                "Registration is blocked because the current ARBF run failed "
                "the geostatistical gate."
            )
            self.btn_jorc.setToolTip(
                "JORC export is blocked because the current ARBF run is not "
                "geostatistically defensible."
            )
        elif overall == "warn":
            self.btn_jorc.setToolTip(
                "JORC export stays disabled until all geostatistical warnings are cleared."
            )

    def _populate_defensibility(self, payload: Dict[str, Any]) -> None:
        gate = payload.get("geostatistical_gate") or {}
        if not gate:
            self._gate_text.setPlainText("No geostatistical gate result available.")
            return

        lines = [
            gate.get("headline", "No verdict"),
            "",
            gate.get("summary", ""),
            "",
            f"Pass: {gate.get('pass_count', 0)}",
            f"Warn: {gate.get('warn_count', 0)}",
            f"Fail: {gate.get('fail_count', 0)}",
            "",
            "Checks:",
        ]

        for check in gate.get("checks", []):
            status = str(check.get("status", "info")).upper()
            label = check.get("label", check.get("code", "check"))
            detail = check.get("detail", "")
            threshold = check.get("threshold", "")
            lines.append(f"[{status}] {label}")
            if detail:
                lines.append(f"  {detail}")
            if threshold:
                lines.append(f"  Threshold: {threshold}")
            lines.append("")

        self._gate_text.setPlainText("\n".join(lines).strip())

    def _populate_log(self, payload: Dict[str, Any]):
        """Populate engine log tab from the audit record."""
        audit = payload.get("audit_record")
        if audit is None:
            self._log_text.setPlainText("No audit record available.")
            return
        try:
            rec = audit.__dict__ if hasattr(audit, '__dict__') else audit
            if payload.get("geostatistical_gate"):
                rec = dict(rec)
                rec["geostatistical_gate"] = payload.get("geostatistical_gate")
            if payload.get("simulation_diagnostics"):
                rec = dict(rec)
                rec["simulation_diagnostics"] = payload.get("simulation_diagnostics")
            if payload.get("metadata"):
                rec = dict(rec)
                rec["metadata"] = payload.get("metadata")
            import json as _json
            self._log_text.setPlainText(_json.dumps(rec, indent=2, default=str))
        except Exception as exc:
            self._log_text.setPlainText(f"Error formatting log: {exc}")

    # ══════════════════════════════════════════════════════════════
    # ARBF RESULTS REGISTRATION
    # ══════════════════════════════════════════════════════════════

    def _register_arbf_to_registry(self, payload: Dict[str, Any]) -> None:
        """Register ARBF results to the data registry signal chain.

        This fires arbfResultsLoaded so BlockModelBuilderPanel picks up
        the ARBF estimation in its property control combo.
        """
        if self.registry is None:
            return
        try:
            variable = (payload.get("metadata", {}).get("variable")
                        or self.variable_combo.currentText() or "Grade")
            prop_name = payload.get("property_name") or f"ARBF_{variable}"
            variance_name = payload.get("variance_property") or f"{prop_name}_var"
            arbf_reg = {
                "grid": payload.get("grid"),
                "variable": variable,
                "property_name": prop_name,
                "variance_property": variance_name,
                "stitching_variance_property": payload.get("stitching_variance_property"),
                "total_blending_variance_property": payload.get("total_blending_variance_property"),
                "p10_property": payload.get("p10_property"),
                "p50_property": payload.get("p50_property"),
                "p90_property": payload.get("p90_property"),
                "grades": payload.get("grades"),
                "variances": payload.get("variances"),
                "stitching_variance": payload.get("stitching_variance"),
                "total_blending_variance": payload.get("total_blending_variance"),
                "p10": payload.get("p10"),
                "p50": payload.get("p50"),
                "p90": payload.get("p90"),
                "x_coords": payload.get("x_coords"),
                "y_coords": payload.get("y_coords"),
                "z_coords": payload.get("z_coords"),
                "grid_values": payload.get("grades"),
                "block_sizes": payload.get("block_sizes"),
                "classifications": payload.get("classifications"),
                "fail_flags": payload.get("fail_flags"),
                "neff": payload.get("neff"),
                "uncertainty": payload.get("uncertainty"),
                "audit_record": payload.get("audit_record"),
                "metadata": payload.get("metadata", {}),
            }
            arbf_reg["geostatistical_gate"] = payload.get("geostatistical_gate")
            self.registry.register_results(
                "arbf_results", arbf_reg, source_panel="ARBF",
                metadata={
                    **payload.get("metadata", {}),
                    "geostatistical_gate": payload.get("geostatistical_gate"),
                },
            )
            logger.info("ARBF results registered to signal chain (arbfResultsLoaded)")
        except Exception as exc:
            logger.warning("Failed to register ARBF results to signal chain: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _dataframe_to_block_model(df: pd.DataFrame):
        """Convert a DataFrame to a BlockModel so the renderer gets ImageData.

        Without this conversion the registry stores a raw DataFrame.  The
        viewer_widget will convert it, but the renderer then sees an
        incomplete grid (NaN-dropped rows) and falls back to
        UnstructuredGrid which is 50-100× slower.
        """
        from ..models.block_model import BlockModel

        bm = BlockModel()
        bm.update_from_dataframe(df)
        return bm

    def _build_block_dataframe(self) -> Optional[pd.DataFrame]:
        """Build a DataFrame with X, Y, Z + grade/variance/classification columns.

        Caches the result so repeated calls (register then visualise) avoid
        redundant meshgrid computation.
        """
        if self._cached_block_df is not None:
            return self._cached_block_df

        if self.arbf_results is None:
            return None

        grades = self.arbf_results.get("grades")
        variances = self.arbf_results.get("variances")
        x_c = self.arbf_results.get("x_coords")
        y_c = self.arbf_results.get("y_coords")
        z_c = self.arbf_results.get("z_coords")
        block_sizes = self.arbf_results.get("block_sizes")

        if x_c is None or y_c is None or z_c is None or grades is None:
            logger.warning("ARBF results missing coordinates or grades")
            return None

        x_c = np.asarray(x_c)
        y_c = np.asarray(y_c)
        z_c = np.asarray(z_c)
        n_grades = grades.size if grades is not None else 0
        meshgrid_count = len(x_c) * len(y_c) * len(z_c)

        if meshgrid_count > 0 and meshgrid_count == n_grades:
            GX, GY, GZ = np.meshgrid(x_c, y_c, z_c, indexing="ij")
            x_flat, y_flat, z_flat = GX.ravel(), GY.ravel(), GZ.ravel()
        else:
            x_flat = x_c.ravel()
            y_flat = y_c.ravel()
            z_flat = z_c.ravel()

        variable = self.variable_combo.currentText() or "Grade"
        prop_name = self.arbf_results.get("property_name") or f"ARBF_{variable}"
        variance_name = self.arbf_results.get("variance_property") or f"{prop_name}_var"
        grade_flat = np.asarray(grades).ravel()

        df = pd.DataFrame({
            "X": x_flat,
            "Y": y_flat,
            "Z": z_flat,
            prop_name: grade_flat,
        })

        # Add block dimensions
        if block_sizes is not None:
            block_sizes_arr = np.asarray(block_sizes)
            if block_sizes_arr.ndim == 2 and len(block_sizes_arr) == len(df):
                df["DX"] = block_sizes_arr[:, 0]
                df["DY"] = block_sizes_arr[:, 1]
                df["DZ"] = block_sizes_arr[:, 2]
            elif block_sizes_arr.ndim == 1 and len(block_sizes_arr) == 3:
                df["DX"] = block_sizes_arr[0]
                df["DY"] = block_sizes_arr[1]
                df["DZ"] = block_sizes_arr[2]
            else:
                df["DX"] = self.dx_spin.value()
                df["DY"] = self.dy_spin.value()
                df["DZ"] = self.dz_spin.value()
        else:
            df["DX"] = self.dx_spin.value()
            df["DY"] = self.dy_spin.value()
            df["DZ"] = self.dz_spin.value()

        if variances is not None:
            df[variance_name] = np.asarray(variances).ravel()

        stitching_variance = self.arbf_results.get("stitching_variance")
        stitching_name = self.arbf_results.get("stitching_variance_property")
        if stitching_variance is not None and stitching_name:
            df[stitching_name] = np.asarray(stitching_variance).ravel()

        total_blending = self.arbf_results.get("total_blending_variance")
        total_blending_name = self.arbf_results.get("total_blending_variance_property")
        if total_blending is not None and total_blending_name:
            df[total_blending_name] = np.asarray(total_blending).ravel()

        for key, property_key in (("p10", "p10_property"), ("p50", "p50_property"), ("p90", "p90_property")):
            values = self.arbf_results.get(key)
            property_name = self.arbf_results.get(property_key)
            if values is not None and property_name:
                df[property_name] = np.asarray(values).ravel()

        # Classification column
        classification = self.arbf_results.get("classification_result")
        if classification is not None:
            if hasattr(classification, 'classes'):
                df["ARBF_Class"] = classification.classes
            elif hasattr(classification, 'block_classes'):
                df["ARBF_Class"] = classification.block_classes
            elif isinstance(classification, dict) and "classes" in classification:
                df["ARBF_Class"] = classification["classes"]
            elif isinstance(classification, dict) and "block_classes" in classification:
                df["ARBF_Class"] = classification["block_classes"]
        elif self.arbf_results.get("classifications") is not None:
            df["ARBF_Class"] = np.asarray(self.arbf_results.get("classifications")).ravel()

        # Keep NaN-grade rows so the grid stays complete (nx×ny×nz).
        # A complete grid qualifies for ImageData (fast) instead of
        # UnstructuredGrid (50-100× slower).  NaN blocks are masked via
        # domain_mask transparency in the renderer.
        nan_count = int(df[prop_name].isna().sum())
        if nan_count:
            # Add domain_mask: 1 = valid, 0 = uninformed (hidden via
            # apply_domain_mask_transparency in the renderer).
            df["domain_mask"] = (~df[prop_name].isna()).astype(np.int8)
            # Fill NaN grades with the global MEAN of valid grades so
            # they don't compress the colour range to zero.  The old
            # fillna(0.0) made these cells dominate the dark end of the
            # colourmap, turning the entire block model uniformly dark.
            valid_grades = df[prop_name].dropna()
            fill_value = float(valid_grades.mean()) if len(valid_grades) > 0 else 0.0
            df[prop_name] = df[prop_name].fillna(fill_value)
            logger.info(
                "ARBF grid has %d uninformed blocks (%.1f%%) — kept for "
                "ImageData eligibility, masked via domain_mask, "
                "filled with mean=%.2f",
                nan_count, 100.0 * nan_count / len(df), fill_value,
            )

        logger.info(
            "Built ARBF block DataFrame: %d blocks, columns=%s",
            len(df), list(df.columns),
        )
        self._cached_block_df = df
        return df

    def _on_register_model(self):
        """Register block model in the DataRegistry (no rendering)."""
        if not self._gate_allows("allow_register"):
            self.show_warning(
                "Registration Blocked",
                "This ARBF run failed the geostatistical gate. Fix the failed checks first.",
            )
            return
        df = self._build_block_dataframe()
        if df is None:
            self.progress_label.setText("No results to register")
            return
        try:
            variable = self.variable_combo.currentText() or "Grade"
            metadata = {
                "source": "ARBF",
                "variable": variable,
                "primary_property": self.arbf_results.get("property_name") or f"ARBF_{variable}",
                "audit_record": self.arbf_results.get("audit_record"),
                "geostatistical_gate": self.arbf_results.get("geostatistical_gate"),
            }
            bm = self._dataframe_to_block_model(df)
            self.registry.register_block_model_generated(
                bm,
                source_panel="ARBF",
                metadata=metadata,
            )
            self.progress_label.setText("Block model registered")
            logger.info("ARBF block model registered to DataRegistry")
        except Exception as exc:
            logger.warning("Failed to register model: %s", exc)
            self.progress_label.setText(f"Registration failed: {exc}")

    def _on_visualise_blocks(self):
        """Render the ARBF block model in the 3D viewer.

        Always creates pv.ImageData — the same approach SGSIM uses — giving
        ~4 MB implicit geometry and 40-120 FPS even for 1.8 M+ blocks.

        For clipped / footprint-limited results the active grades are
        scattered into a full-size NaN grid; ``extract_cells`` then strips
        the uninformed blocks before sending to the 3-D viewer.
        """
        import pyvista as pv
        from PyQt6.QtCore import QTimer

        if self.arbf_results is None:
            self.progress_label.setText("No results to visualise")
            return

        variable = self.variable_combo.currentText() or "Grade"
        prop_name = self.arbf_results.get("property_name") or f"ARBF_{variable}"

        try:
            grades = self.arbf_results.get("grades")
            variances = self.arbf_results.get("variances")
            if grades is None:
                self.progress_label.setText("No grade data to visualise")
                return

            grades = np.asarray(grades).ravel()

            # ── Read grid parameters ────────────────────────────────
            nx = self.nx_spin.value()
            ny = self.ny_spin.value()
            nz = self.nz_spin.value()
            dx = self.dx_spin.value()
            dy = self.dy_spin.value()
            dz = self.dz_spin.value()
            x0 = self.x0_spin.value()
            y0 = self.y0_spin.value()
            z0 = self.z0_spin.value()
            n_grid = nx * ny * nz

            # ── Fast path: use the controller's pre-built grid ──────
            # The controller (geostats_controller._prepare_arbf_payload)
            # already builds an ImageData grid with correctly VTK-ordered
            # cell_data when the estimation ran on a regular meshgrid.
            # Using it avoids any centroid-order assumptions.
            prebuilt_grid = self.arbf_results.get("grid")
            if prebuilt_grid is not None and prop_name in prebuilt_grid.cell_data:
                grid = prebuilt_grid
                logger.info(
                    "ARBF: using pre-built ImageData grid from controller "
                    "(%d cells, property='%s')",
                    grid.n_cells, prop_name,
                )
            else:
                # ── Build VTK-ordered grade array via coordinate scatter ─
                # ALWAYS use the coordinate-based scatter approach to map
                # each grade to its correct VTK cell index.  The previous
                # reshape shortcut (grades.reshape((nx,ny,nz)).ravel("F"))
                # assumed grades were in meshgrid("ij") C-order, which is
                # WRONG when centroids come from a loaded block model
                # (arbitrary row order) — causing spatial inversion of
                # the estimated grades in the 3D viewer.
                x_c = self.arbf_results.get("x_coords")
                y_c = self.arbf_results.get("y_coords")
                z_c = self.arbf_results.get("z_coords")
                if x_c is None or y_c is None or z_c is None:
                    self.progress_label.setText("Missing coordinate data")
                    return

                x_c = np.asarray(x_c).ravel()
                y_c = np.asarray(y_c).ravel()
                z_c = np.asarray(z_c).ravel()

                # Centroids may be 1-D unique arrays (meshgrid needed)
                # or flat per-block arrays (already expanded)
                meshgrid_count = len(x_c) * len(y_c) * len(z_c)
                if meshgrid_count > 0 and meshgrid_count == grades.size:
                    GX, GY, GZ = np.meshgrid(x_c, y_c, z_c, indexing="ij")
                    xf = GX.ravel()
                    yf = GY.ravel()
                    zf = GZ.ravel()
                else:
                    xf, yf, zf = x_c, y_c, z_c

                # Map each block centroid → integer grid index
                ix = np.clip(
                    np.round((xf - x0) / dx - 0.5).astype(np.intp), 0, nx - 1
                )
                iy = np.clip(
                    np.round((yf - y0) / dy - 0.5).astype(np.intp), 0, ny - 1
                )
                iz = np.clip(
                    np.round((zf - z0) / dz - 0.5).astype(np.intp), 0, nz - 1
                )
                # VTK cell index: X varies fastest
                vtk_idx = ix + iy * nx + iz * nx * ny

                vtk_grades = np.full(n_grid, np.nan, dtype=np.float64)
                vtk_grades[vtk_idx] = grades[: len(vtk_idx)]

                logger.info(
                    "ARBF: scattered %d active grades into %d-cell "
                    "ImageData (%.1f%% filled)",
                    len(vtk_idx), n_grid,
                    100.0 * len(vtk_idx) / n_grid,
                )

                # ── Create ImageData ────────────────────────────────
                grid = pv.ImageData(
                    dimensions=(nx + 1, ny + 1, nz + 1),
                    spacing=(dx, dy, dz),
                    origin=(x0, y0, z0),
                )
                grid.cell_data[prop_name] = vtk_grades

                # Variance
                if variances is not None:
                    var_name = (
                        self.arbf_results.get("variance_property")
                        or f"{prop_name}_var"
                    )
                    var_arr = np.asarray(variances).ravel()
                    if var_arr.size == grades.size:
                        vtk_var = np.full(n_grid, np.nan, dtype=np.float64)
                        vtk_var[vtk_idx] = var_arr[: len(vtk_idx)]
                    else:
                        vtk_var = None
                    if vtk_var is not None:
                        grid.cell_data[var_name] = vtk_var

                # Extra properties (stitching var, blending var)
                for key, prop_key in (
                    ("stitching_variance", "stitching_variance_property"),
                    ("total_blending_variance", "total_blending_variance_property"),
                ):
                    vals = self.arbf_results.get(key)
                    pname = self.arbf_results.get(prop_key)
                    if vals is not None and pname:
                        ea = np.asarray(vals).ravel()
                        if ea.size == grades.size:
                            tmp = np.full(n_grid, np.nan, dtype=np.float64)
                            tmp[vtk_idx] = ea[: len(vtk_idx)]
                            grid.cell_data[pname] = tmp

            # ── Strip only truly unestimated blocks (NaN from masking) ──
            grade_data = grid.cell_data[prop_name]
            valid_mask = np.isfinite(grade_data)
            n_valid = int(valid_mask.sum())
            n_total = len(grade_data)

            if n_valid == 0:
                self.progress_label.setText("All blocks are NaN — nothing to show")
                return

            if n_valid < n_total and n_valid > 0:
                cell_ids = np.where(valid_mask)[0]
                grid = grid.extract_cells(cell_ids)
                logger.info(
                    "ARBF ImageData: stripped %d unestimated NaN blocks, "
                    "%d estimated blocks remain (%.1f%%)",
                    n_total - n_valid, n_valid, 100.0 * n_valid / n_total,
                )

            n_cells = grid.n_cells
            self.progress_label.setText(
                f"Sending ARBF to 3D viewer ({n_cells:,} blocks)..."
            )
            logger.info(
                "ARBF ImageData created: %d cells, property=%s, "
                "grid=%dx%dx%d, spacing=(%.1f,%.1f,%.1f), "
                "origin=(%.1f,%.1f,%.1f)",
                n_cells, prop_name, nx, ny, nz, dx, dy, dz, x0, y0, z0,
            )

            # Emit through the same fast path SGSIM uses
            QTimer.singleShot(
                50,
                lambda g=grid, p=prop_name: self._emit_viz_safe(g, p),
            )

        except Exception as exc:
            logger.warning("Failed to visualise blocks: %s", exc, exc_info=True)
            self.progress_label.setText(f"Visualisation failed: {exc}")

    # ------------------------------------------------------------------
    def _emit_viz_safe(self, grid, property_name: str):
        """Emit the ImageData grid via request_visualization signal.

        Also auto-register the grid with the DataRegistry so downstream
        panels (Statistics, Charts, Classification, Grade-Tonnage,
        Resource Reporting, Swath) can pick it up as a selectable source
        without the user needing to click "Register Model" separately.
        """
        try:
            try:
                if self.registry is not None and grid is not None:
                    from ..models.block_model import BlockModel
                    bm = BlockModel.from_pyvista_grid(grid)
                    variable = self.variable_combo.currentText() or "Grade"
                    model_id = f"ARBF_{variable}"
                    # Replace any prior run with the same variable instead of
                    # accumulating ARBF_Cu_NS_2, _3, ...
                    if hasattr(self.registry, 'clear_block_model'):
                        try:
                            self.registry.clear_block_model(model_id)
                        except Exception:
                            pass
                    self.registry.register_block_model_generated(
                        bm,
                        source_panel="ARBF",
                        metadata={
                            "source": "ARBF",
                            "variable": variable,
                            "primary_property": property_name,
                        },
                        model_id=model_id,
                    )
                    logger.info(
                        "ARBF: auto-registered block model "
                        "(variable=%s, property=%s, %d cells)",
                        variable, property_name, grid.n_cells,
                    )
            except Exception as reg_exc:
                logger.warning(
                    "ARBF: auto-register on visualise failed: %s", reg_exc,
                )

            self.request_visualization.emit(grid, property_name)
            n_cells = grid.n_cells
            self.progress_label.setText(
                f"Block model visualised ({n_cells:,} blocks)"
            )
            self._global_progress_label.setText(
                f"ARBF blocks loaded in 3D viewer ({n_cells:,} blocks)"
            )
            if (self._geostatistical_gate or {}).get("overall_status") == "fail":
                self.progress_label.setText(
                    f"Preview visualised ({n_cells:,} blocks) — not defensible"
                )
            logger.info(
                "ARBF ImageData sent to 3D viewer: %d cells, property=%s",
                n_cells, property_name,
            )
        except Exception as exc:
            logger.warning("ARBF visualisation emit failed: %s", exc, exc_info=True)
            self.progress_label.setText(f"Visualisation failed: {exc}")

    def _on_export_jorc(self):
        if self.arbf_results is None:
            return
        if not self._gate_allows("allow_jorc_export"):
            self.show_warning(
                "JORC Export Blocked",
                "JORC export is disabled until the ARBF run passes the geostatistical gate without warnings.",
            )
            return
        audit = self.arbf_results.get("audit_record")
        if audit is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JORC Table 1", "ARBF_JORC_Table1.txt", "Text Files (*.txt)"
        )
        if path:
            try:
                jorc_text = audit.to_jorc_table1() if hasattr(audit, 'to_jorc_table1') else str(audit)
                with open(path, "w") as f:
                    f.write(jorc_text)
                self.progress_label.setText(f"Exported JORC to {path}")
            except Exception as exc:
                logger.warning("JORC export failed: %s", exc)

    def _on_export_audit_json(self):
        if self.arbf_results is None:
            return
        audit = self.arbf_results.get("audit_record")
        if audit is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit JSON", "ARBF_audit_record.json", "JSON Files (*.json)"
        )
        if path:
            try:
                d = audit.__dict__ if hasattr(audit, '__dict__') else audit
                if self.arbf_results.get("geostatistical_gate"):
                    d = dict(d)
                    d["geostatistical_gate"] = self.arbf_results.get("geostatistical_gate")
                if self.arbf_results.get("simulation_diagnostics"):
                    d = dict(d)
                    d["simulation_diagnostics"] = self.arbf_results.get("simulation_diagnostics")
                if self.arbf_results.get("metadata"):
                    d = dict(d)
                    d["metadata"] = self.arbf_results.get("metadata")
                with open(path, "w") as f:
                    json.dump(d, f, indent=2, default=str)
                self.progress_label.setText(f"Exported audit to {path}")
            except Exception as exc:
                logger.warning("Audit export failed: %s", exc)

    def _on_export_csv(self):
        if self.arbf_results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", "ARBF_results.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                df = self._build_block_dataframe()
                if df is None:
                    raise ValueError("No ARBF block data available for CSV export.")
                df.to_csv(path, index=False)
                self.progress_label.setText(f"Exported results to {path}")
            except Exception as exc:
                logger.warning("CSV export failed: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # PUBLIC SETTERS (for controller integration)
    # ══════════════════════════════════════════════════════════════

    def set_drillhole_data(self, data) -> None:
        if not self._ui_ready:
            self._pending_drillhole_data = data
            return
        if isinstance(data, pd.DataFrame):
            self.drillhole_data = data
        elif isinstance(data, dict):
            self._extract_dataframe(data)
        self._refresh_variable_list(data)
        n = len(self.drillhole_data) if self.drillhole_data is not None else 0
        self.data_status.setText(f"{n} samples loaded")

    def set_variogram_results(self, results: Dict[str, Any]) -> None:
        self.variogram_results = results
        if self._ui_ready:
            self._on_import_variogram()

    def get_registry(self):
        if hasattr(self, 'registry') and self.registry is not None:
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

    # ══════════════════════════════════════════════════════════════
    # DomainMaskMixin overrides — the mixin's default attribute
    # lookups don't match our spinbox names, so we override.
    # ══════════════════════════════════════════════════════════════

    def _get_block_centroids(self):
        """Build block centroids from grid spinboxes for domain masking."""
        import numpy as _np
        ox = self.x0_spin.value()
        oy = self.y0_spin.value()
        oz = self.z0_spin.value()
        ddx = self.dx_spin.value()
        ddy = self.dy_spin.value()
        ddz = self.dz_spin.value()
        nnx = self.nx_spin.value()
        nny = self.ny_spin.value()
        nnz = self.nz_spin.value()
        xs = ox + (_np.arange(nnx) + 0.5) * ddx
        ys = oy + (_np.arange(nny) + 0.5) * ddy
        zs = oz + (_np.arange(nnz) + 0.5) * ddz
        gx, gy, gz = _np.meshgrid(xs, ys, zs, indexing="ij")
        return _np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def _get_search_radii(self):
        """Return anisotropic search radii = 2× variogram ranges."""
        rM = self.range_max_spin.value()
        rm = self.range_mid_spin.value()
        rv = self.range_min_spin.value()
        if rM > 0 and rm > 0 and rv > 0:
            return (rM * 2.0, rm * 2.0, rv * 2.0)
        return (200.0, 200.0, 200.0)

    def _get_search_azimuth(self):
        return self.azimuth_spin.value()

    def _get_search_dip(self):
        return self.dip_spin.value()

    def refresh_theme(self) -> None:
        pass

    def show_warning(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)


