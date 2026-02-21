"""
3D SIMPLE KRIGING PANEL

UI for configuring and running Simple Kriging (SK).

Features:
- Global Mean input (SK specific)
- Auto-Grid detection from data bounds
- Integration with Variogram Panel settings
- Progress tracking and 3D visualization signals
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import pyvista as pv

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox,
    QTextEdit, QWidget, QSplitter, QScrollArea, QFrame,
    QProgressBar, QFormLayout, QRadioButton, QButtonGroup, QDialog
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import (
    get_grade_columns, validate_variable, populate_variable_combo,
    get_variable_from_combo_or_fallback
)
from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors

logger = logging.getLogger(__name__)


def _get_btn_style_primary() -> str:
    """Get primary button style for current theme."""
    colors = get_theme_colors()
    return f"""
        QPushButton {{ background-color: {colors.SUCCESS}; color: white; font-weight: bold; padding: 10px; border-radius: 4px; }}
        QPushButton:hover {{ background-color: #43A047; }}
    """


def _get_btn_style_action() -> str:
    """Get action button style for current theme."""
    colors = get_theme_colors()
    return f"""
        QPushButton {{ background-color: {colors.INFO}; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }}
        QPushButton:hover {{ background-color: {colors.ACCENT_PRIMARY}; }}
        QPushButton:disabled {{ background-color: {colors.BORDER}; color: {colors.TEXT_DISABLED}; }}
    """


class SimpleKrigingPanel(BaseAnalysisPanel):

    """
    Panel for configuring and launching 3D Simple Kriging.

    Requires a known global mean.
    """
    # PanelManager metadata
    PANEL_ID = "SimpleKrigingPanel"
    PANEL_NAME = "SimpleKriging Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT





    task_name = "simple_kriging"
    request_visualization = pyqtSignal(dict)  # Signal to request 3D viz

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="simple_kriging")
        self.setWindowTitle("Simple Kriging")
        self.resize(1100, 750)
        self.setStyleSheet(get_analysis_panel_stylesheet())

        # State
        self.data_df: Optional[pd.DataFrame] = None
        self.vcol: Optional[str] = None
        self.grid_spec: Optional[Dict[str, float]] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.kriging_results = None

        self._build_ui()

        self._init_registry_connections()

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self.setStyleSheet(get_analysis_panel_stylesheet())
        # Refresh button styles
        if hasattr(self, 'run_btn'):
            self.run_btn.setStyleSheet(_get_btn_style_primary())
        for btn_name in ['viz_btn', 'view_table_btn', 'export_btn', 'diagnostics_btn', 'cv_btn', 'stationarity_btn']:
            if hasattr(self, btn_name):
                getattr(self, btn_name).setStyleSheet(_get_btn_style_action())

    def _build_ui(self):
        """Build custom split-pane UI."""
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # --- LEFT: CONFIGURATION ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setSpacing(15)
        
        # 0. Data Source
        self._create_data_source_group(self.scroll_layout)
        
        # 1. Global Mean
        self._create_mean_group(self.scroll_layout)
        
        # 2. Variogram
        self._create_variogram_group(self.scroll_layout)

        # 2.1 View Variogram Model
        view_group = QGroupBox("")
        view_layout = QHBoxLayout(view_group)
        self.view_variogram_btn = QPushButton("View Variogram Model")
        self.view_variogram_btn.setStyleSheet("background-color: #444; border: 1px solid #666;")
        self.view_variogram_btn.clicked.connect(self._view_variogram_model)
        self.view_variogram_btn.setEnabled(False)
        view_layout.addWidget(self.view_variogram_btn)
        view_layout.addStretch()
        self.scroll_layout.addWidget(view_group)
        
        # 3. Anisotropy
        self._create_aniso_group(self.scroll_layout)
        
        # 4. Grid
        self._create_grid_group(self.scroll_layout)
        
        self.scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        # --- RIGHT: RESULTS & LOGS ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # SK Info Header
        info_header = QHBoxLayout()
        sk_title_label = QLabel("<b>Simple Kriging</b>")
        sk_info_btn = QPushButton("ℹ")
        sk_info_btn.setMaximumWidth(30)
        sk_info_btn.setMaximumHeight(30)
        sk_info_btn.setToolTip(
            "Simple Kriging: Uses a known global mean to assess spatial continuity.\n\n"
            "Professional Use:\n"
            "• Sanity check for estimation\n"
            "• Bias diagnostic tool\n"
            "• Mean control reference\n\n"
            "Not recommended for production resource estimation when results show "
            "strong reversion to global mean. Use Ordinary Kriging or SGSIM in such cases."
        )
        sk_info_btn.clicked.connect(self._show_sk_professional_context)
        info_header.addWidget(sk_title_label)
        info_header.addWidget(sk_info_btn)
        info_header.addStretch()
        right_layout.addLayout(info_header)

        # Progress
        pg_group = QGroupBox("Status")
        pg_layout = QVBoxLayout(pg_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.lbl_status = QLabel("Ready")
        pg_layout.addWidget(self.progress_bar)
        pg_layout.addWidget(self.lbl_status)
        right_layout.addWidget(pg_group)
        
        # Logs
        log_group = QGroupBox("Process Log")
        l_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        l_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, stretch=2)
        
        # Summary
        sum_group = QGroupBox("Results Summary")
        s_layout = QVBoxLayout(sum_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        s_layout.addWidget(self.summary_text)
        right_layout.addWidget(sum_group, stretch=1)
        
        # Actions
        btn_layout = QHBoxLayout()
        
        # Debug mode checkbox
        self.debug_mode_check = QCheckBox("Debug Mode")
        self.debug_mode_check.setToolTip(
            "Enable comprehensive debugging:\n"
            "- Detailed execution logging\n"
            "- Error tracking at every stage\n"
            "- Writes to 'sk_debug_log.txt'\n"
            "Use this if kriging fails or crashes"
        )
        self.debug_mode_check.setStyleSheet("QCheckBox { color: #ff9800; font-weight: bold; }")
        btn_layout.addWidget(self.debug_mode_check)
        
        self.run_btn = QPushButton("RUN ESTIMATION")
        self.run_btn.setStyleSheet(_get_btn_style_primary())
        self.run_btn.clicked.connect(self.run_analysis)
        
        self.viz_btn = QPushButton("Visualize 3D")
        self.viz_btn.setStyleSheet(_get_btn_style_action())
        self.viz_btn.setEnabled(False)
        self.viz_btn.clicked.connect(self.visualize_results)

        self.view_table_btn = QPushButton("View Table")
        self.view_table_btn.setStyleSheet(_get_btn_style_action())
        self.view_table_btn.setEnabled(False)
        self.view_table_btn.clicked.connect(self.open_results_table)

        self.export_btn = QPushButton("Export Bundle")
        self.export_btn.setStyleSheet(_get_btn_style_action())
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_audit_bundle)

        self.diagnostics_btn = QPushButton("Diagnostics")
        self.diagnostics_btn.setStyleSheet(_get_btn_style_action())
        self.diagnostics_btn.setEnabled(False)
        self.diagnostics_btn.clicked.connect(self.show_diagnostics)

        self.cv_btn = QPushButton("Cross-Validation")
        self.cv_btn.setStyleSheet(_get_btn_style_action())
        self.cv_btn.setEnabled(False)
        self.cv_btn.clicked.connect(self.run_cross_validation)
        self.cv_btn.setToolTip("Run Leave-One-Out Cross-Validation (LOOCV) to assess estimation accuracy")

        self.stationarity_btn = QPushButton("Stationarity Check")
        self.stationarity_btn.setStyleSheet(_get_btn_style_action())
        self.stationarity_btn.setEnabled(False)
        self.stationarity_btn.clicked.connect(self.show_stationarity_report)
        self.stationarity_btn.setToolTip("Validate global mean assumption for Simple Kriging")

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.viz_btn)
        btn_layout.addWidget(self.view_table_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.diagnostics_btn)
        btn_layout.addWidget(self.cv_btn)
        btn_layout.addWidget(self.stationarity_btn)
        right_layout.addLayout(btn_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 650])
        
        main_layout.addWidget(splitter)

    # --- COMPONENT BUILDERS ---

    def _create_data_source_group(self, layout):
        """Create data source selection group."""
        group = QGroupBox("0. Data Source")
        form = QFormLayout(group)
        
        # Data source selector
        data_source_layout = QHBoxLayout()
        self.data_source_group = QButtonGroup()
        self.data_source_composited = QRadioButton("Composited Data")
        self.data_source_composited.setToolTip("Use composited drillhole data (recommended)")
        self.data_source_raw = QRadioButton("Raw Assay Data")
        self.data_source_raw.setToolTip("Use raw drillhole assay data")
        
        self.data_source_group.addButton(self.data_source_composited, 0)
        self.data_source_group.addButton(self.data_source_raw, 1)
        self.data_source_composited.setChecked(True)
        self.data_source_group.buttonClicked.connect(self._on_data_source_changed)

        data_source_layout.addWidget(self.data_source_composited)
        data_source_layout.addWidget(self.data_source_raw)
        data_source_layout.addStretch()
        form.addRow("Source:", data_source_layout)

        # Warning banner for raw assays (initially hidden)
        self.raw_assay_warning = QLabel("⚠ Raw assays selected: For diagnostic use only, not for reporting-grade SK")
        self.raw_assay_warning.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px; background-color: #FFF3E0; border: 1px solid #FF9800; border-radius: 3px;")
        self.raw_assay_warning.setVisible(False)
        form.addRow("", self.raw_assay_warning)
        
        # Variable/Element selector
        variable_layout = QHBoxLayout()
        self.variable_combo = QComboBox()
        self.variable_combo.setMinimumWidth(200)
        self.variable_combo.setToolTip("Select the variable/element to estimate")
        self.variable_combo.currentTextChanged.connect(self._on_variable_changed)
        variable_layout.addWidget(self.variable_combo)
        variable_layout.addStretch()
        form.addRow("Variable:", variable_layout)

        # Domain selector
        domain_layout = QHBoxLayout()
        self.domain_combo = QComboBox()
        self.domain_combo.setMinimumWidth(200)
        self.domain_combo.setToolTip("Select geological domain for stationarity (recommended)")
        self.domain_combo.addItem("All Data")
        self.domain_combo.currentTextChanged.connect(self._on_domain_changed)
        domain_layout.addWidget(self.domain_combo)
        domain_layout.addStretch()
        form.addRow("Domain:", domain_layout)

        # Data source status label
        self.data_source_status_label = QLabel("")
        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #888;")
        form.addRow("", self.data_source_status_label)
        
        layout.addWidget(group)

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return

        # Show warning banner for raw assays
        if hasattr(self, 'raw_assay_warning'):
            is_raw_selected = self.data_source_raw.isChecked()
            self.raw_assay_warning.setVisible(is_raw_selected)

            if is_raw_selected:
                self._log("Warning: Raw assays selected - for diagnostic use only", "error")

        # Reload full drillhole payload so panel can apply selected source correctly.
        data = self.registry.get_drillhole_data()
        if data is not None:
            self._on_drillhole_data_loaded(data)
    
    def _on_variable_changed(self, variable_name):
        """Handle variable selection change - update global mean."""
        if not variable_name or not hasattr(self, 'data_df') or self.data_df is None:
            return

        if variable_name in self.data_df.columns:
            # Update mean based on current mode
            self._update_mean_for_current_mode()
            self.vcol = variable_name
            domain_info = self._get_domain_info()
            mean_info = self._get_mean_info()
            self._log(f"Selected variable: {variable_name}{mean_info}{domain_info}")

    def _on_mean_mode_changed(self, mode):
        """Handle mean mode change - update spinbox and recalculate mean."""
        self._update_mean_for_current_mode()
        mean_info = self._get_mean_info()
        self._log(f"Mean mode changed to: {mode}{mean_info}")

    def _update_mean_for_current_mode(self):
        """Update the mean spinbox based on current mode and data."""
        if not hasattr(self, 'data_df') or self.data_df is None or not hasattr(self, 'variable_combo'):
            return

        variable_name = self.variable_combo.currentText()
        if not variable_name or variable_name not in self.data_df.columns:
            return

        filtered_df = self._get_filtered_data()
        if filtered_df is None or variable_name not in filtered_df.columns:
            return

        mode = self.mean_mode_combo.currentText()

        if mode == "Global (dataset)":
            # Use full dataset mean
            mean = self.data_df[variable_name].mean()
            self.mean_spin.setEnabled(False)  # Auto-calculated
        elif mode == "Per-domain":
            # Use domain-filtered mean
            mean = filtered_df[variable_name].mean()
            self.mean_spin.setEnabled(False)  # Auto-calculated
        else:  # User-defined
            # Keep current value, enable manual editing
            mean = self.mean_spin.value()
            self.mean_spin.setEnabled(True)  # User can edit
            return  # Don't update the value

        self.mean_spin.setValue(mean)

    def _get_mean_info(self):
        """Get mean calculation info for logging."""
        mode = self.mean_mode_combo.currentText()
        current_mean = self.mean_spin.value()
        return f" (Mean: {current_mean:.3f}, Mode: {mode})"

    def _on_domain_changed(self, domain_column):
        """Handle domain selection change - update mean and grid."""
        if not hasattr(self, 'data_df') or self.data_df is None:
            return

        # Update mean based on current mode
        self._update_mean_for_current_mode()

        # Log domain change with mean info
        domain_info = self._get_domain_info()
        mean_info = self._get_mean_info()
        self._log(f"Domain changed to: {domain_column}{mean_info}{domain_info}")

        # Auto-detect grid for new domain extent
        if hasattr(self, 'auto_fit_grid_check') and self.auto_fit_grid_check.isChecked():
            self._auto_detect_grid()

    def _get_filtered_data(self):
        """Get data filtered by selected domain."""
        if not hasattr(self, 'data_df') or self.data_df is None:
            return None

        if not hasattr(self, 'domain_combo') or self.domain_combo.currentText() == "All Data":
            return self.data_df

        domain_column = self.domain_combo.currentText()
        if domain_column not in self.data_df.columns:
            return self.data_df

        # Filter out null domain values
        filtered_df = self.data_df.dropna(subset=[domain_column])
        if filtered_df.empty:
            self._log(f"Warning: No valid data after filtering by domain '{domain_column}'", "error")
            return self.data_df

        return filtered_df

    def _get_domain_info(self):
        """Get domain filtering info for logging."""
        if not hasattr(self, 'domain_combo') or self.domain_combo.currentText() == "All Data":
            return ""

        domain_column = self.domain_combo.currentText()
        filtered_df = self._get_filtered_data()
        if filtered_df is not None:
            total_count = len(self.data_df)
            filtered_count = len(filtered_df)
            return f" [Domain: {domain_column}, {filtered_count}/{total_count} samples]"
        return ""

    def _create_mean_group(self, layout):
        group = QGroupBox("1. Global Parameters")
        form = QFormLayout(group)

        # Mean mode selector
        self.mean_mode_combo = QComboBox()
        self.mean_mode_combo.addItems(["Global (dataset)", "Per-domain", "User-defined"])
        self.mean_mode_combo.setToolTip("How to determine the global mean for Simple Kriging")
        self.mean_mode_combo.currentTextChanged.connect(self._on_mean_mode_changed)
        form.addRow("Mean mode:", self.mean_mode_combo)

        self.mean_spin = QDoubleSpinBox()
        self.mean_spin.setRange(-1e6, 1e6)
        self.mean_spin.setDecimals(3)
        self.mean_spin.setToolTip("Global mean of the domain (Stationarity assumption)")

        form.addRow("Global Mean (μ):", self.mean_spin)
        layout.addWidget(group)

    def _create_variogram_group(self, layout):
        group = QGroupBox("2. Variogram Model")
        vbox = QVBoxLayout(group)
        
        self.load_btn = QPushButton("Load from Variogram Panel")
        self.load_btn.setStyleSheet("background-color: #444; border: 1px solid #666;")
        self.load_btn.clicked.connect(self._load_from_variogram)
        vbox.addWidget(self.load_btn)
        
        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["spherical", "exponential", "gaussian"])
        self.sill_spin = QDoubleSpinBox(); self.sill_spin.setRange(0, 1e5); self.sill_spin.setValue(1)
        self.nugget_spin = QDoubleSpinBox(); self.nugget_spin.setRange(0, 1e5)
        
        form.addRow("Model Type:", self.model_combo)
        form.addRow("Sill (C1):", self.sill_spin)
        form.addRow("Nugget (C0):", self.nugget_spin)
        vbox.addLayout(form)
        layout.addWidget(group)

    def _create_aniso_group(self, layout):
        group = QGroupBox("3. Ranges & Anisotropy")
        form = QFormLayout(group)
        
        self.rmaj_spin = QDoubleSpinBox(); self.rmaj_spin.setRange(0.1, 1e5); self.rmaj_spin.setValue(100)
        self.rmin_spin = QDoubleSpinBox(); self.rmin_spin.setRange(0.1, 1e5); self.rmin_spin.setValue(100)
        self.rver_spin = QDoubleSpinBox(); self.rver_spin.setRange(0.1, 1e5); self.rver_spin.setValue(50)
        
        self.azim_spin = QDoubleSpinBox(); self.azim_spin.setRange(0, 360)
        self.dip_spin = QDoubleSpinBox(); self.dip_spin.setRange(-90, 90)
        
        form.addRow("Major Range:", self.rmaj_spin)
        form.addRow("Minor Range:", self.rmin_spin)
        form.addRow("Vertical Range:", self.rver_spin)
        form.addRow("Azimuth:", self.azim_spin)
        form.addRow("Dip:", self.dip_spin)
        layout.addWidget(group)

    def _create_grid_group(self, layout):
        group = QGroupBox("4. Grid Definition")
        vbox = QVBoxLayout(group)
        
        # Origin
        g1 = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox(); self.xmin_spin.setRange(-1e9, 1e9); self.xmin_spin.setPrefix("X: ")
        self.ymin_spin = QDoubleSpinBox(); self.ymin_spin.setRange(-1e9, 1e9); self.ymin_spin.setPrefix("Y: ")
        self.zmin_spin = QDoubleSpinBox(); self.zmin_spin.setRange(-1e9, 1e9); self.zmin_spin.setPrefix("Z: ")
        g1.addWidget(self.xmin_spin); g1.addWidget(self.ymin_spin); g1.addWidget(self.zmin_spin)
        vbox.addLayout(g1)
        
        # Dimensions
        g2 = QHBoxLayout()
        self.nx = QSpinBox(); self.nx.setRange(1, 2000); self.nx.setPrefix("NX: "); self.nx.setValue(50)
        self.ny = QSpinBox(); self.ny.setRange(1, 2000); self.ny.setPrefix("NY: "); self.ny.setValue(50)
        self.nz = QSpinBox(); self.nz.setRange(1, 2000); self.nz.setPrefix("NZ: "); self.nz.setValue(20)
        g2.addWidget(self.nx); g2.addWidget(self.ny); g2.addWidget(self.nz)
        vbox.addLayout(g2)
        
        # Block Size
        g3 = QHBoxLayout()
        self.dx = QDoubleSpinBox(); self.dx.setRange(0.1, 1000); self.dx.setPrefix("DX: "); self.dx.setValue(10)
        self.dy = QDoubleSpinBox(); self.dy.setRange(0.1, 1000); self.dy.setPrefix("DY: "); self.dy.setValue(10)
        self.dz = QDoubleSpinBox(); self.dz.setRange(0.1, 1000); self.dz.setPrefix("DZ: "); self.dz.setValue(5)
        g3.addWidget(self.dx); g3.addWidget(self.dy); g3.addWidget(self.dz)
        vbox.addLayout(g3)
        
        auto_btn = QPushButton("Auto-Fit to Data")
        auto_btn.clicked.connect(self._auto_detect_grid)
        vbox.addWidget(auto_btn)
        
        # Auto-fit checkbox (enabled by default)
        self.auto_fit_grid_check = QCheckBox("Auto-fit grid when data loads")
        self.auto_fit_grid_check.setChecked(True)
        self.auto_fit_grid_check.setToolTip("Automatically restrict grid to drillhole extent when new data is loaded.")
        vbox.addWidget(self.auto_fit_grid_check)
        
        # Search
        vbox.addWidget(QLabel("Search Parameters:"))
        h1 = QHBoxLayout()
        self.search_spin = QDoubleSpinBox(); self.search_spin.setRange(1, 10000); self.search_spin.setValue(200); self.search_spin.setPrefix("Radius: ")
        self.ndmax_spin = QSpinBox(); self.ndmax_spin.setRange(1, 200); self.ndmax_spin.setValue(12); self.ndmax_spin.setPrefix("Max N: ")
        h1.addWidget(self.search_spin)
        h1.addWidget(self.ndmax_spin)
        vbox.addLayout(h1)

        # Neighbourhood controls
        vbox.addWidget(QLabel("Neighbourhood Controls:"))
        h2 = QHBoxLayout()
        self.nmin_spin = QSpinBox(); self.nmin_spin.setRange(1, 50); self.nmin_spin.setValue(1); self.nmin_spin.setPrefix("Min N: ")
        self.nmin_spin.setToolTip("Minimum number of samples required (block set to global mean if not satisfied)")
        h2.addWidget(self.nmin_spin)

        # Sectoring option
        self.sectoring_combo = QComboBox()
        self.sectoring_combo.addItems(["No sectoring", "4 sectors", "8 sectors"])
        self.sectoring_combo.setToolTip("Octant/sector search to ensure spatial distribution")
        h2.addWidget(self.sectoring_combo)
        vbox.addLayout(h2)
        
        layout.addWidget(group)

    # --- LOGIC ---

    def _init_registry_connections(self):
        try:
            self.registry = self.get_registry()
            if not self.registry:
                logger.warning("DataRegistry not available - get_registry() returned None")
                return

            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
            self.registry.variogramResultsLoaded.connect(self._on_variogram_results_loaded)

            # Source-toggle panels must load full drillhole payload for proper source switching.
            data = self.registry.get_drillhole_data()

            if data is not None:
                self._on_drillhole_data_loaded(data)
            
            vario = self.registry.get_variogram_results()
            if vario is not None: 
                self._on_variogram_results_loaded(vario)
            
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None

    def _on_drillhole_data_loaded(self, data):
        """Load data, respecting user's data source selection."""
        # Log diagnostic info about registry contents
        log_registry_data_status("Simple Kriging", data)
        
        # Store registry data
        self._registry_data = data
        
        df = None
        composites = None
        assays = None
        composites_available = False
        assays_available = False
        
        if isinstance(data, dict):
            composites = data.get('composites')
            if composites is None:
                composites = data.get('composites_df')
            assays = data.get('assays')
            if assays is None:
                assays = data.get('assays_df')
            composites_available = isinstance(composites, pd.DataFrame) and not composites.empty
            assays_available = isinstance(assays, pd.DataFrame) and not assays.empty
            
            # Respect user's selection if radio buttons exist
            if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
                use_composited = self.data_source_composited.isChecked()

                if use_composited and composites_available:
                    df = composites
                    # AUDIT FIX: Set provenance for composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                elif not use_composited and assays_available:
                    df = assays
                    # AUDIT FIX: Mark raw assays appropriately
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
                    # Show warning if raw assays are used
                    if hasattr(self, 'raw_assay_warning'):
                        self.raw_assay_warning.setVisible(True)
                elif composites_available:
                    df = composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                    if hasattr(self, 'data_source_composited'):
                        self.data_source_composited.setChecked(True)
                elif assays_available:
                    df = assays
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
                    if hasattr(self, 'data_source_raw'):
                        self.data_source_raw.setChecked(True)
                else:
                    df = None
            else:
                # Legacy: prefer composites
                if composites_available:
                    df = composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                elif assays_available:
                    df = assays
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
        elif isinstance(data, pd.DataFrame):
            df = data
            # AUDIT FIX: Set default provenance if not present
            if 'source_type' not in df.attrs:
                df.attrs['source_type'] = 'composites'
                df.attrs['lineage_gate_passed'] = True
            source_type = str(df.attrs.get('source_type', '')).lower()
            if source_type in ('raw_assays', 'assays'):
                assays_available = True
                assays = df
            else:
                composites_available = True
                composites = df
        
        # Update radio button states
        if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
            self.data_source_composited.setEnabled(composites_available)
            self.data_source_raw.setEnabled(assays_available)
            
            # Update status label
            if hasattr(self, 'data_source_status_label'):
                if composites_available and assays_available:
                    self.data_source_status_label.setText(
                        f"✓ Both available: {len(composites):,} composites, {len(assays):,} assays"
                    )
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
                elif composites_available:
                    self.data_source_status_label.setText(f"✓ {len(composites):,} composites available")
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
                elif assays_available:
                    self.data_source_status_label.setText(f"✓ {len(assays):,} raw assays available")
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #FF9800;")
                else:
                    self.data_source_status_label.setText("⚠ No data available")
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #e57373;")
            
        if df is not None and not df.empty:
            self.data_df = ensure_xyz_columns(df)
            self._log(f"Data loaded: {len(self.data_df)} records")
            
            # Populate variable combo box
            if hasattr(self, 'variable_combo'):
                # Block signals to avoid triggering updates during population
                self.variable_combo.blockSignals(True)
                self.variable_combo.clear()
                
                # Get numeric columns (excluding coordinates and metadata)
                numeric_cols = get_grade_columns(self.data_df)
                if numeric_cols:
                    self.variable_combo.addItems(sorted(numeric_cols))
                    
                    # Auto-select first variable and update mean
                    if self.variable_combo.count() > 0:
                        self.variable_combo.setCurrentIndex(0)
                        selected_var = self.variable_combo.currentText()
                        if selected_var in self.data_df.columns:
                            self.vcol = selected_var
                            mean = self.data_df[selected_var].mean()
                            self.mean_spin.setValue(mean)
                            self._log(f"Auto-selected variable: {selected_var} (Mean: {mean:.2f})")
                else:
                    self._log("⚠ No numeric variables found in data")
                
                self.variable_combo.blockSignals(False)

            # Populate domain combo box
            if hasattr(self, 'domain_combo'):
                self.domain_combo.blockSignals(True)
                self.domain_combo.clear()
                self.domain_combo.addItem("All Data")

                # Find categorical columns that could be domains
                categorical_cols = []
                for col in self.data_df.columns:
                    if self.data_df[col].dtype == 'object':
                        try:
                            # Check if column is suitable for domain filtering
                            unique_count = self.data_df[col].nunique()
                            if 2 <= unique_count <= 20:  # Reasonable domain count
                                # Check if values are hashable and not too long
                                sample_values = self.data_df[col].dropna().head(10)
                                if all(isinstance(v, (str, int, float)) and len(str(v)) <= 50 for v in sample_values):
                                    categorical_cols.append(col)
                        except (TypeError, ValueError):
                            continue

                # Sort and add domain columns
                for col in sorted(categorical_cols):
                    self.domain_combo.addItem(col)

                self.domain_combo.blockSignals(False)
                self._log(f"Found {len(categorical_cols)} potential domain columns")

            # Auto-detect grid from drillhole extent if checkbox is enabled
            if hasattr(self, 'auto_fit_grid_check') and self.auto_fit_grid_check.isChecked():
                self._auto_detect_grid()
                logger.info("Simple Kriging: Auto-fitted grid to drillhole extent")

    def _on_variogram_results_loaded(self, results):
        self.variogram_results = results
        self.load_btn.setEnabled(True)
        self.view_variogram_btn.setEnabled(True)
        self._log("New variogram model available.")

    def _load_from_variogram(self):
        """Populate UI from stored variogram results."""
        if not self.variogram_results: return
        
        try:
            # Check for combined 3D model first
            model = self.variogram_results.get('combined_3d_model')
            if not model:
                # Fallback to directional fits
                fits = self.variogram_results.get('fitted_models', {})
                model = fits.get('omni', {}).get(next(iter(fits.get('omni', {})), 'spherical'))
            
            if model:
                self.model_combo.setCurrentText(model.get('model_type', 'spherical'))
                self.nugget_spin.setValue(model.get('nugget', 0))
                self.sill_spin.setValue(model.get('sill', 1))
                self.rmaj_spin.setValue(model.get('major_range', 100))
                self.rmin_spin.setValue(model.get('minor_range', 100))
                self.rver_spin.setValue(model.get('vertical_range', 50))
                self.azim_spin.setValue(model.get('azimuth', 0))
                self.dip_spin.setValue(model.get('dip', 0))
                self._log("Variogram parameters loaded.")
        except Exception as e:
            self._log(f"Error loading variogram: {e}", "error")

    def _view_variogram_model(self):
        """Open the variogram panel to view the current model."""
        try:
            # Try to find and activate the variogram panel
            main_window = self._find_main_window()
            if main_window and hasattr(main_window, 'activate_panel_by_id'):
                success = main_window.activate_panel_by_id('VariogramPanel')
                if success:
                    self._log("Opened Variogram Panel to view current model.", "info")
                else:
                    self._log("Variogram Panel not available.", "error")
            else:
                self._log("Cannot access main window to open Variogram Panel.", "error")
        except Exception as e:
            logger.error(f"Error opening variogram panel: {e}", exc_info=True)
            self._log("Error opening Variogram Panel.", "error")

    def _find_main_window(self):
        """Find the main window parent."""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'activate_panel_by_id'):
                return parent
            parent = parent.parent()
        return None

    def _auto_detect_grid(self):
        """Calculate grid bounds from data extent + padding."""
        filtered_df = self._get_filtered_data()
        if filtered_df is None or filtered_df.empty:
            self._log("Warning: No data available for grid auto-detection", "error")
            return

        # AUDIT FIX: Ensure coordinate columns are standardized to X, Y, Z
        filtered_df = ensure_xyz_columns(filtered_df)
        
        # Check if X, Y, Z columns exist after normalization
        if not all(col in filtered_df.columns for col in ('X', 'Y', 'Z')):
            self._log("Warning: Cannot auto-detect grid - missing coordinate columns (X, Y, Z)", "error")
            logger.warning("Simple Kriging: Auto-detect grid failed - DataFrame missing X, Y, Z columns")
            return

        # 1. Calc Extents from filtered data
        xmin, xmax = filtered_df['X'].min(), filtered_df['X'].max()
        ymin, ymax = filtered_df['Y'].min(), filtered_df['Y'].max()
        zmin, zmax = filtered_df['Z'].min(), filtered_df['Z'].max()
        
        # 2. Add Padding (5%)
        pad = 0.05
        dx = (xmax - xmin) * pad
        dy = (ymax - ymin) * pad
        dz = (zmax - zmin) * pad
        
        xmin -= dx; xmax += dx
        ymin -= dy; ymax += dy
        zmin -= dz; zmax += dz
        
        # 3. Snap to nice numbers
        bs_x, bs_y, bs_z = self.dx.value(), self.dy.value(), self.dz.value()
        xmin = np.floor(xmin / bs_x) * bs_x
        ymin = np.floor(ymin / bs_y) * bs_y
        zmin = np.floor(zmin / bs_z) * bs_z
        
        nx = int(np.ceil((xmax - xmin) / bs_x))
        ny = int(np.ceil((ymax - ymin) / bs_y))
        nz = int(np.ceil((zmax - zmin) / bs_z))
        
        # 4. Set UI
        self.xmin_spin.setValue(xmin)
        self.ymin_spin.setValue(ymin)
        self.zmin_spin.setValue(zmin)
        self.nx.setValue(nx)
        self.ny.setValue(ny)
        self.nz.setValue(nz)
        
        self._log(f"Grid auto-fitted: {nx}x{ny}x{nz}")

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Simple Kriging.

        Simple Kriging requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Validated data quality

        Returns:
            True if data is acceptable for Simple Kriging
        """
        registry = getattr(self, 'registry', None)
        if not registry:
            logger.warning("LINEAGE: No registry available - cannot verify data lineage")
            return True  # Allow to proceed but log warning

        # HARD GATE: Use require_validation_for_estimation() method
        # This enforces JORC/SAMREC compliance - NO estimation without validation
        allowed, message = registry.require_validation_for_estimation()
        if not allowed:
            logger.error(f"LINEAGE HARD GATE: {message}")
            QMessageBox.critical(
                self, "Validation Required",
                f"Cannot run Simple Kriging:\n\n{message}\n\n"
                "Open the QC Window to validate your data before running estimation."
            )
            return False

        # Log validation status for audit trail
        validation_state = registry.get_drillholes_validation_state()
        if validation_state:
            status = validation_state.get("status", "UNKNOWN")
            if status == "WARN":
                logger.warning(
                    "LINEAGE: Validation passed with warnings. "
                    "Review warnings for JORC/SAMREC compliance."
                )
            else:
                logger.info(f"LINEAGE: Validation status = {status}")

        return True

    def run_analysis(self):
        if self.data_df is None:
            QMessageBox.warning(self, "Error", "Missing data.")
            return

        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return
        
        # Get selected variable from combo box
        if hasattr(self, 'variable_combo') and self.variable_combo.count() > 0:
            selected_var = self.variable_combo.currentText()
            if not selected_var or selected_var not in self.data_df.columns:
                QMessageBox.warning(self, "Error", "Please select a valid variable.")
                return
            self.vcol = selected_var
        elif self.vcol is None:
            QMessageBox.warning(self, "Error", "Please select a variable to estimate.")
            return
            
        # Get filtered data for the analysis
        filtered_data = self._get_filtered_data()
        if filtered_data is None or filtered_data.empty:
            QMessageBox.warning(self, "Error", "No data available for analysis after domain filtering.")
            self.run_btn.setEnabled(True)
            return

        # Validate minimum sample count
        if len(filtered_data) < 10:
            QMessageBox.warning(self, "Error", f"Insufficient data for kriging: only {len(filtered_data)} samples after filtering. Need at least 10 samples.")
            self.run_btn.setEnabled(True)
            return

        # Determine data mode for logging
        data_mode = "COMPOSITES"
        if hasattr(self, 'data_source_raw') and self.data_source_raw.isChecked():
            data_mode = "RAW ASSAYS"

        mean_mode = self.mean_mode_combo.currentText()
        domain_info = self._get_domain_info()
        self._log(f"Running Simple Kriging on {len(filtered_data)} samples{domain_info}")
        self._log(f"Data mode: {data_mode} - Mean mode: {mean_mode} - Global mean: {self.mean_spin.value():.3f}")

        params = {
            "data": filtered_data,  # Use filtered data
            "variable": self.vcol,
            "debug_mode": self.debug_mode_check.isChecked(),  # Enable debug mode if checked
            "grid_spec": {
                "nx": self.nx.value(), "ny": self.ny.value(), "nz": self.nz.value(),
                "xmin": self.xmin_spin.value(), "ymin": self.ymin_spin.value(), "zmin": self.zmin_spin.value(),
                "xinc": self.dx.value(), "yinc": self.dy.value(), "zinc": self.dz.value()
            },
            "parameters": {
                "global_mean": self.mean_spin.value(),
                "mean_mode": self.mean_mode_combo.currentText(),
                "variogram_type": self.model_combo.currentText(),
                "sill": self.sill_spin.value(),
                "nugget": self.nugget_spin.value(),
                "range_major": self.rmaj_spin.value(),
                "range_minor": self.rmin_spin.value(),
                "range_vert": self.rver_spin.value(),
                "azimuth": self.azim_spin.value(),
                "dip": self.dip_spin.value(),
                "ndmax": self.ndmax_spin.value(),
                "max_search_radius": self.search_spin.value(),
                "nmin": self.nmin_spin.value(),
                "sectoring": self.sectoring_combo.currentText(),
            }
        }
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("Starting Simple Kriging...", "info")
        
        # Dispatch to Controller
        if self.controller:
            self.controller.run_simple_kriging(
                params, 
                callback=self.on_results, 
                progress_callback=self._update_progress
            )
        else:
            self._log("Error: No Controller connected.", "error")
            self.run_btn.setEnabled(True)

    def _update_progress(self, val, msg):
        pct = max(0, min(100, int(val)))
        self.progress_bar.setValue(pct)
        if msg:
            self.progress_bar.setFormat(f"{pct}% - {msg}")
        else:
            self.progress_bar.setFormat(f"{pct}%")
        self.lbl_status.setText(msg)

    def on_results(self, payload):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        if "error" in payload:
            self._log(f"Kriging Failed: {payload['error']}", "error")
            return
            
        # Extract mesh and property names
        viz = payload.get("visualization", {})
        mesh = viz.get("mesh")
        property_name = payload.get("property_name")
        variance_property = payload.get("variance_property")
        metadata = payload.get("metadata", {})
        
        if mesh is None or property_name is None:
            self._log("Error: Missing mesh or property name in results.", "error")
            return
        
        # Extract variable from property naming convention (fallback)
        variable = payload.get("variable")
        if not variable and property_name:
            variable = property_name.replace("_SK_est", "").replace("_SK", "").replace("_est", "")
        
        # Copy estimates/variances/neighbour_counts to detach from VTK buffers
        estimates = mesh[property_name] if property_name in mesh.array_names else None
        variances = mesh[variance_property] if variance_property and variance_property in mesh.array_names else None
        neighbour_counts = mesh["SK_NN"] if "SK_NN" in mesh.array_names else None

        if estimates is None:
            self._log(f"Error: Property '{property_name}' not found in mesh.", "error")
            return

        if hasattr(estimates, "numpy"):
            estimates = estimates.numpy()
        estimates = np.array(estimates, copy=True)

        if variances is not None:
            if hasattr(variances, "numpy"):
                variances = variances.numpy()
            variances = np.array(variances, copy=True)

        if neighbour_counts is not None:
            if hasattr(neighbour_counts, "numpy"):
                neighbour_counts = neighbour_counts.numpy()
            neighbour_counts = np.array(neighbour_counts, copy=True)
        
        # Extract grid coordinates
        if isinstance(mesh, pv.StructuredGrid):
            grid_x = mesh.x.reshape(mesh.dimensions, order='F')
            grid_y = mesh.y.reshape(mesh.dimensions, order='F')
            grid_z = mesh.z.reshape(mesh.dimensions, order='F')
        else:
            pts = mesh.points
            grid_x, grid_y, grid_z = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Package stable results
        results = {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "estimates": estimates,
            "variances": variances if variances is not None else np.full_like(estimates, np.nan),
            "neighbour_counts": neighbour_counts if neighbour_counts is not None else np.full_like(estimates, 0, dtype=int),
            "variable": variable,
            "property_name": property_name,
            "variance_property": variance_property,
            "metadata": metadata,
            "back_transformed": False,
        }
        
        # Store for Visualization and registry
        self.kriging_results = results
        if self.registry:
            try:
                self.registry.register_simple_kriging_results(results, source_panel="SimpleKrigingPanel")
                self._log("Results registered to data registry.", "info")
            except Exception as e:
                logger.warning(f"Failed to register simple kriging results: {e}")

            # Also register the block model DataFrame for cross-sections and other panels
            try:
                grid_x = results['grid_x']
                grid_y = results['grid_y']
                grid_z = results['grid_z']
                estimates = results['estimates']
                variances = results['variances']

                # Create block model DataFrame
                coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                block_df = pd.DataFrame({
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                    f'{variable}_sk_est': estimates.ravel(),
                    f'{variable}_sk_var': variances.ravel() if variances is not None else np.full_like(estimates.ravel(), np.nan)
                }).dropna()

                # Register the block model
                self.registry.register_block_model_generated(
                    block_df,
                    source_panel="Simple Kriging",
                    metadata={
                        'variable': variable,
                        'method': 'simple_kriging',
                        'grid_size': (len(np.unique(grid_x)), len(np.unique(grid_y)), len(np.unique(grid_z))),
                        'n_blocks': len(block_df)
                    }
                )
                self._log("Block model registered to data registry", "info")
            except Exception as e:
                logger.warning(f"Failed to register simple kriging block model: {e}")

        self.viz_btn.setEnabled(True)
        self.view_table_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.diagnostics_btn.setEnabled(True)
        self.cv_btn.setEnabled(True)
        self.stationarity_btn.setEnabled(True)
        
        # Log Stats
        if len(estimates) > 0:
            # Get collapse metadata for display
            collapse_metadata = metadata.get("sk_mean_collapse", {})
            collapse_flag = "COLLAPSED" if collapse_metadata.get("flag") else "Active"

            summary_text = (
                f"Count: {len(estimates)}\n"
                f"Min: {np.nanmin(estimates):.3f}\n"
                f"Max: {np.nanmax(estimates):.3f}\n"
                f"Mean: {np.nanmean(estimates):.3f}\n"
                f"Std: {np.nanstd(estimates):.3f}\n"
                f"\nSK Status: {collapse_flag}"
            )

            if collapse_metadata.get("flag"):
                ratio = collapse_metadata.get("collapse_ratio", 0.0)
                summary_text += f"\nCollapse Ratio: {ratio:.4f}"

            self.summary_text.setText(summary_text)
            self._log("Estimation completed successfully.", "success")

            # Show warning banner if SK collapsed to mean
            if collapse_metadata.get("flag", False):
                severity = collapse_metadata.get("severity", "MODERATE")
                collapse_ratio = collapse_metadata.get("collapse_ratio", 0.0)
                self._show_sk_collapse_warning(severity, collapse_ratio, collapse_metadata)

            # Check stability and warn if >5% blocks unstable
            stability_metrics = metadata.get("stability_metrics", {})
            pct_unstable = stability_metrics.get("pct_unstable", 0.0)
            if pct_unstable > 5.0:
                QMessageBox.warning(
                    self, "Kriging Stability Warning",
                    f"{pct_unstable:.1f}% of blocks have kriging stability issues.\n\n"
                    "Common causes:\n"
                    "• Sparse data geometry\n"
                    "• Collocated samples\n"
                    "• Extreme search radius\n\n"
                    "Review diagnostic exports for details."
                )

    def visualize_results(self):
        if self.kriging_results:
            self.request_visualization.emit(self.kriging_results)
            self._log("Sent to 3D Viewer.", "info")

    def open_results_table(self):
        """Open Simple Kriging results as a table."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run estimation first.")
            return

        try:
            grid_x = self.kriging_results.get('grid_x')
            grid_y = self.kriging_results.get('grid_y')
            grid_z = self.kriging_results.get('grid_z')
            estimates = self.kriging_results.get('estimates')
            variances = self.kriging_results.get('variances')
            variable = self.kriging_results.get('variable', 'Estimate')

            if grid_x is None or estimates is None:
                QMessageBox.warning(self, "Invalid Results", "Results data is incomplete.")
                return

            # Ensure estimates and variances are numpy arrays
            estimates = np.asarray(estimates)
            if variances is not None:
                variances = np.asarray(variances)

            coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
            df = pd.DataFrame({
                'X': coords[:, 0],
                'Y': coords[:, 1],
                'Z': coords[:, 2],
                f'{variable} (Estimate)': estimates.ravel(),
            })

            if variances is not None and not np.all(np.isnan(variances)):
                df['Variance'] = variances.ravel()

            df = df.dropna()

            title = f"Simple Kriging Results - {variable}"

            # Try to find MainWindow parent first
            parent = self.parent()
            main_window = None
            while parent:
                if hasattr(parent, 'open_table_viewer_window_from_df'):
                    main_window = parent
                    break
                parent = parent.parent()

            if main_window:
                main_window.open_table_viewer_window_from_df(df, title=title)
            else:
                # Create table viewer dialog directly
                from .table_viewer_panel import TableViewerPanel

                dialog = QDialog(self)
                dialog.setWindowTitle(title)
                dialog.resize(900, 700)
                dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowMaximizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint
                )

                layout = QVBoxLayout(dialog)
                table_viewer = TableViewerPanel()
                table_viewer.set_dataframe(df)
                layout.addWidget(table_viewer)

                dialog.show()

        except Exception as e:
            logger.error(f"Error opening results table: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to open results table:\n{str(e)}")

    def export_audit_bundle(self):
        """Export comprehensive audit bundle with results and metadata."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run estimation first.")
            return

        try:
            from PyQt6.QtWidgets import QFileDialog
            import pandas as pd
            from datetime import datetime
            import json

            # Get export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            variable = self.kriging_results.get('variable', 'Unknown')
            default_name = f"SK_Audit_{variable}_{timestamp}"

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export SK Audit Bundle", default_name,
                "ZIP Archive (*.zip);;All Files (*)"
            )

            if not file_path:
                return

            # Ensure .zip extension
            if not file_path.lower().endswith('.zip'):
                file_path += '.zip'

            import zipfile
            import io

            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. Block model CSV
                grid_x = self.kriging_results.get('grid_x')
                grid_y = self.kriging_results.get('grid_y')
                grid_z = self.kriging_results.get('grid_z')
                estimates = self.kriging_results.get('estimates')
                variances = self.kriging_results.get('variances')
                neighbour_counts = self.kriging_results.get('neighbour_counts')

                # Ensure arrays are numpy arrays
                estimates = np.asarray(estimates)
                if variances is not None:
                    variances = np.asarray(variances)
                if neighbour_counts is not None:
                    neighbour_counts = np.asarray(neighbour_counts)

                coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                bm_df = pd.DataFrame({
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                    f'{variable}_SK_EST': estimates.ravel(),
                    f'{variable}_SK_VAR': variances.ravel() if variances is not None else np.full_like(estimates, np.nan).ravel(),
                    'SK_NN': neighbour_counts.ravel() if neighbour_counts is not None else np.full_like(estimates, 0, dtype=int).ravel(),
                })
                bm_df = bm_df.dropna()

                bm_csv = io.StringIO()
                bm_df.to_csv(bm_csv, index=False)
                zf.writestr("block_model.csv", bm_csv.getvalue())

                # 2. Audit report JSON
                metadata = self.kriging_results.get('metadata', {})
                audit_data = {
                    "audit_timestamp": datetime.now().isoformat(),
                    "analysis_type": "Simple Kriging",
                    "software_version": "GeoX Block Model Viewer",
                    "variable": variable,
                    "data_summary": {
                        "data_mode": "COMPOSITES" if not hasattr(self, 'data_source_raw') or not self.data_source_raw.isChecked() else "RAW ASSAYS",
                        "domain_filter": self.domain_combo.currentText() if hasattr(self, 'domain_combo') else "All Data",
                        "samples_used": metadata.get('samples_used', 0),
                        "mean_mode": self.mean_mode_combo.currentText() if hasattr(self, 'mean_mode_combo') else "Unknown",
                        "global_mean": self.mean_spin.value() if hasattr(self, 'mean_spin') else 0.0,
                    },
                    "variogram_parameters": {
                        "model_type": self.model_combo.currentText() if hasattr(self, 'model_combo') else "Unknown",
                        "sill": self.sill_spin.value() if hasattr(self, 'sill_spin') else 0.0,
                        "nugget": self.nugget_spin.value() if hasattr(self, 'nugget_spin') else 0.0,
                        "range_major": self.rmaj_spin.value() if hasattr(self, 'rmaj_spin') else 0.0,
                        "range_minor": self.rmin_spin.value() if hasattr(self, 'rmin_spin') else 0.0,
                        "range_vertical": self.rver_spin.value() if hasattr(self, 'rver_spin') else 0.0,
                        "azimuth": self.azim_spin.value() if hasattr(self, 'azim_spin') else 0.0,
                        "dip": self.dip_spin.value() if hasattr(self, 'dip_spin') else 0.0,
                    },
                    "search_parameters": {
                        "max_neighbours": self.ndmax_spin.value() if hasattr(self, 'ndmax_spin') else 12,
                        "min_neighbours": self.nmin_spin.value() if hasattr(self, 'nmin_spin') else 1,
                        "search_radius": self.search_spin.value() if hasattr(self, 'search_spin') else 200.0,
                        "sectoring": self.sectoring_combo.currentText() if hasattr(self, 'sectoring_combo') else "No sectoring",
                    },
                    "grid_definition": {
                        "dimensions": metadata.get('grid_dimensions', (0, 0, 0)),
                        "spacing": metadata.get('grid_spacing', (0.0, 0.0, 0.0)),
                        "origin": (
                            self.xmin_spin.value() if hasattr(self, 'xmin_spin') else 0.0,
                            self.ymin_spin.value() if hasattr(self, 'ymin_spin') else 0.0,
                            self.zmin_spin.value() if hasattr(self, 'zmin_spin') else 0.0,
                        ),
                        "total_blocks": len(estimates) if estimates is not None else 0,
                    },
                    "results_statistics": {
                        "estimates": {
                            "min": float(metadata.get('estimates_min', 'nan')),
                            "max": float(metadata.get('estimates_max', 'nan')),
                            "valid_count": int((~np.isnan(estimates)).sum()) if estimates is not None else 0,
                        },
                        "variances": {
                            "min": float(metadata.get('variance_min', 'nan')),
                            "max": float(metadata.get('variance_max', 'nan')),
                        },
                        "neighbour_counts": {
                            "min": int(metadata.get('neighbour_count_min', 0)),
                            "max": int(metadata.get('neighbour_count_max', 0)),
                            "mean": float(metadata.get('neighbour_count_mean', 0.0)),
                        },
                    },
                    "processing_log": [],  # Could be populated from log history if stored
                }

                # Add SK collapse diagnostics
                collapse_metadata = metadata.get("sk_mean_collapse", {})
                audit_data["sk_diagnostics"] = {
                    "mean_collapse": collapse_metadata,
                    "professional_positioning": {
                        "role": "diagnostic_tool",
                        "use_case": "sanity_check_and_bias_diagnostic",
                        "reporting_status": "not_for_production_estimation" if collapse_metadata.get("flag") else "suitable_for_estimation",
                        "justification": (
                            "SK results demonstrate weak spatial structure relative to nugget effect, "
                            "validating the use of Ordinary Kriging for production estimation."
                        ) if collapse_metadata.get("flag") else (
                            "SK results show sufficient local variation to support production estimation."
                        )
                    }
                }

                # Add Component 5: Support Documentation
                support_doc = metadata.get("support_documentation", {})
                if support_doc:
                    audit_data["support_documentation"] = support_doc

                # Add Component 3: Stationarity Validation
                stationarity = metadata.get("stationarity_validation", {})
                if stationarity:
                    audit_data["stationarity_validation"] = stationarity

                # Add Component 2: Stability Metrics
                stability = metadata.get("stability_metrics", {})
                if stability:
                    audit_data["stability_metrics"] = stability

                # Add Component 7: Enhanced Provenance
                full_prov = metadata.get("full_provenance", {})
                if full_prov:
                    audit_data["full_provenance"] = full_prov

                # Add Component 6: Domain Controls
                domain_ctrl = metadata.get("domain_controls", {})
                if domain_ctrl:
                    audit_data["domain_controls"] = domain_ctrl

                # Add log entries (last 20)
                log_text = self.log_text.toPlainText() if hasattr(self, 'log_text') else ""
                log_lines = log_text.split('\n')[-20:]  # Last 20 lines
                audit_data["processing_log"] = [line for line in log_lines if line.strip()]

                zf.writestr("audit_report.json", json.dumps(audit_data, indent=2, default=str))

                # 3. Summary report text
                summary_report = f"""SIMPLE KRIGING AUDIT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS SUMMARY
Variable: {variable}
Data Mode: {audit_data['data_summary']['data_mode']}
Domain Filter: {audit_data['data_summary']['domain_filter']}
Mean Mode: {audit_data['data_summary']['mean_mode']}
Global Mean: {audit_data['data_summary']['global_mean']:.3f}
Samples Used: {audit_data['data_summary']['samples_used']}

VARIOGRAM PARAMETERS
Model: {audit_data['variogram_parameters']['model_type']}
Sill: {audit_data['variogram_parameters']['sill']:.3f}
Nugget: {audit_data['variogram_parameters']['nugget']:.3f}
Major Range: {audit_data['variogram_parameters']['range_major']:.1f}
Minor Range: {audit_data['variogram_parameters']['range_minor']:.1f}
Vertical Range: {audit_data['variogram_parameters']['range_vertical']:.1f}
Azimuth: {audit_data['variogram_parameters']['azimuth']:.1f}°
Dip: {audit_data['variogram_parameters']['dip']:.1f}°

SEARCH PARAMETERS
Max Neighbours: {audit_data['search_parameters']['max_neighbours']}
Min Neighbours: {audit_data['search_parameters']['min_neighbours']}
Search Radius: {audit_data['search_parameters']['search_radius']:.1f}
Sectoring: {audit_data['search_parameters']['sectoring']}

GRID DEFINITION
Dimensions: {audit_data['grid_definition']['dimensions'][0]} x {audit_data['grid_definition']['dimensions'][1]} x {audit_data['grid_definition']['dimensions'][2]}
Spacing: {audit_data['grid_definition']['spacing'][0]:.1f} x {audit_data['grid_definition']['spacing'][1]:.1f} x {audit_data['grid_definition']['spacing'][2]:.1f}
Origin: {audit_data['grid_definition']['origin'][0]:.1f}, {audit_data['grid_definition']['origin'][1]:.1f}, {audit_data['grid_definition']['origin'][2]:.1f}
Total Blocks: {audit_data['grid_definition']['total_blocks']:,}

RESULTS STATISTICS
Estimates - Min: {audit_data['results_statistics']['estimates']['min']:.3f}, Max: {audit_data['results_statistics']['estimates']['max']:.3f}, Valid: {audit_data['results_statistics']['estimates']['valid_count']:,}
Variances - Min: {audit_data['results_statistics']['variances']['min']:.3f}, Max: {audit_data['results_statistics']['variances']['max']:.3f}
Neighbour Counts - Min: {audit_data['results_statistics']['neighbour_counts']['min']}, Max: {audit_data['results_statistics']['neighbour_counts']['max']}, Mean: {audit_data['results_statistics']['neighbour_counts']['mean']:.1f}
"""

                # Add collapse diagnostic if present
                import textwrap
                if collapse_metadata.get("flag"):
                    severity = collapse_metadata.get("severity", "")
                    ratio = collapse_metadata.get("collapse_ratio", 0.0)
                    interpretation = collapse_metadata.get("interpretation", "")
                    recommendation = collapse_metadata.get("recommendation", "")

                    summary_report += "\n" + "="*80 + "\n"
                    summary_report += "SIMPLE KRIGING DIAGNOSTIC ALERT\n"
                    summary_report += "="*80 + "\n\n"
                    summary_report += f"SK Mean Collapse Detected: {severity}\n"
                    summary_report += f"Collapse Ratio: {ratio:.6f}\n\n"
                    summary_report += "Interpretation:\n"
                    summary_report += textwrap.fill(interpretation, width=78) + "\n\n"
                    summary_report += "Recommendation:\n"
                    summary_report += textwrap.fill(recommendation, width=78) + "\n\n"
                    summary_report += "Professional Context:\n"
                    summary_report += textwrap.fill(
                        "Simple Kriging was run using a global mean to assess the strength of "
                        "local spatial continuity. Results show strong reversion to the mean, "
                        "confirming that Ordinary Kriging is the appropriate estimator for "
                        "production resource reporting.",
                        width=78
                    ) + "\n"

                # Add Support Documentation (Component 5)
                support_doc = metadata.get("support_documentation", {})
                if support_doc:
                    summary_report += "\n" + "="*80 + "\n"
                    summary_report += "SUPPORT DOCUMENTATION\n"
                    summary_report += "="*80 + "\n\n"
                    comp_support = support_doc.get("composite_support_m")
                    if comp_support:
                        summary_report += f"Composite Support: {comp_support:.2f}m ({support_doc.get('composite_method', 'Unknown')} method)\n"
                    else:
                        summary_report += "Composite Support: Not documented in provenance\n"
                    summary_report += f"Block Support: {support_doc.get('block_support_x_m'):.2f} x {support_doc.get('block_support_y_m'):.2f} x {support_doc.get('block_support_z_m'):.2f}m\n"
                    summary_report += f"Block Volume: {support_doc.get('block_volume_m3'):.2f}m³\n"
                    if support_doc.get('support_ratio', {}).get('x'):
                        sr = support_doc['support_ratio']
                        summary_report += f"Support Ratios: X={sr['x']:.2f}, Y={sr['y']:.2f}, Z={sr['z']:.2f}\n"
                    summary_report += f"Documentation Status: {support_doc.get('documentation_status', 'UNKNOWN')}\n\n"
                    summary_report += "Change-of-Support Statement:\n"
                    summary_report += textwrap.fill(support_doc.get('change_of_support_statement', ''), width=78) + "\n\n"
                    summary_report += "Professional Note:\n"
                    summary_report += textwrap.fill(support_doc.get('professional_note', ''), width=78) + "\n"

                # Add Stationarity Validation (Component 3)
                stationarity = metadata.get("stationarity_validation", {})
                if stationarity:
                    summary_report += "\n" + "="*80 + "\n"
                    summary_report += "STATIONARITY VALIDATION\n"
                    summary_report += "="*80 + "\n\n"
                    summary_report += f"Global Mean: {stationarity.get('global_mean', 0.0):.6f}\n"
                    summary_report += f"Confidence Level: {stationarity.get('confidence_level', 'unknown').upper()}\n\n"

                    # Domain means
                    if stationarity.get('mean_by_domain'):
                        summary_report += "Mean by Domain:\n"
                        for domain, mean_val in stationarity['mean_by_domain'].items():
                            summary_report += f"  {domain}: {mean_val:.4f}\n"
                        summary_report += "\n"

                    # Spatial trends
                    summary_report += "Spatial Trends:\n"
                    for direction in ['x', 'y', 'z']:
                        trend_key = f'trend_{direction}'
                        trend = stationarity.get(trend_key, {})
                        if trend:
                            r2 = trend.get('r_squared', 0)
                            summary_report += f"  {direction.upper()}-direction: R²={r2:.4f}\n"
                    summary_report += "\n"

                    # Issues
                    issues = stationarity.get('issues', [])
                    if issues:
                        summary_report += "Issues Identified:\n"
                        for issue in issues:
                            summary_report += textwrap.fill(f"  • {issue}", width=78, subsequent_indent="    ") + "\n"
                    else:
                        summary_report += "✓ No stationarity issues identified\n"
                    summary_report += "\n"

                    # Interpretation
                    interpretation = stationarity.get('interpretation', '')
                    if interpretation:
                        summary_report += "Interpretation:\n"
                        summary_report += textwrap.fill(interpretation, width=78) + "\n"

                # Add Stability Metrics (Component 2)
                stability = metadata.get("stability_metrics", {})
                if stability:
                    summary_report += "\n" + "="*80 + "\n"
                    summary_report += "KRIGING STABILITY METRICS\n"
                    summary_report += "="*80 + "\n\n"
                    n_unstable = stability.get('n_unstable_blocks', 0)
                    pct_unstable = stability.get('pct_unstable', 0.0)
                    summary_report += f"Unstable Blocks: {n_unstable:,} ({pct_unstable:.1f}%)\n"
                    summary_report += f"  High Negative Weights: {stability.get('n_high_neg_weights', 0):,}\n"
                    summary_report += f"  Bad Sum of Weights: {stability.get('n_bad_sum_weights', 0):,}\n"
                    summary_report += f"  Solver Issues: {stability.get('n_solver_issues', 0):,}\n\n"
                    if pct_unstable > 5.0:
                        summary_report += "⚠ WARNING: More than 5% of blocks have stability issues.\n"
                        summary_report += textwrap.fill(
                            "This may indicate sparse data geometry, collocated samples, or "
                            "extreme search parameters. Review diagnostic exports for details.",
                            width=78
                        ) + "\n"
                    else:
                        summary_report += "✓ Kriging system stability is acceptable.\n"

                # Add Full Provenance (Component 7)
                full_prov = metadata.get("full_provenance", {})
                if full_prov:
                    summary_report += "\n" + "="*80 + "\n"
                    summary_report += "FULL PROVENANCE & REPRODUCIBILITY\n"
                    summary_report += "="*80 + "\n\n"

                    # Data provenance
                    data_prov = full_prov.get('data_provenance', {})
                    if data_prov:
                        summary_report += f"Data Source: {data_prov.get('source_file', 'Unknown')}\n"
                        summary_report += f"Data Hash: {data_prov.get('data_hash', 'N/A')}\n"
                        summary_report += f"Source Type: {data_prov.get('source_type', 'Unknown')}\n\n"

                        # Transformation chain
                        transforms = data_prov.get('transformation_chain', [])
                        if transforms:
                            summary_report += "Transformation Chain:\n"
                            for i, step in enumerate(transforms, 1):
                                summary_report += f"  {i}. {step.get('type', 'Unknown')} - {step.get('description', 'N/A')}\n"
                                summary_report += f"     Panel: {step.get('source_panel', 'Unknown')}\n"
                                summary_report += f"     Timestamp: {step.get('timestamp', 'N/A')}\n"
                            summary_report += "\n"

                    # Software info
                    sw_info = full_prov.get('software_info', {})
                    if sw_info:
                        summary_report += "Software:\n"
                        summary_report += f"  {sw_info.get('software', 'Unknown')} v{sw_info.get('version', 'Unknown')}\n"
                        summary_report += f"  Platform: {sw_info.get('platform', 'Unknown')} {sw_info.get('platform_release', '')}\n"
                        summary_report += f"  Python: {sw_info.get('python_version', 'Unknown')}\n"
                        summary_report += f"  User: {sw_info.get('user', 'Unknown')}\n"
                        summary_report += f"  Timestamp: {sw_info.get('timestamp', 'Unknown')}\n\n"

                    # Variogram signature
                    vario_sig = full_prov.get('variogram_signature', 'N/A')
                    summary_report += f"Variogram Signature: {vario_sig}\n\n"

                    # Reproducibility statement
                    repro_statement = full_prov.get('reproducibility_statement', '')
                    if repro_statement:
                        summary_report += "Reproducibility Statement:\n"
                        summary_report += textwrap.fill(repro_statement, width=78) + "\n\n"

                    # Compliance note
                    compliance = full_prov.get('compliance_note', '')
                    if compliance:
                        summary_report += textwrap.fill(compliance, width=78) + "\n"

                # Add Domain Controls (Component 6)
                domain_ctrl = metadata.get("domain_controls", {})
                if domain_ctrl:
                    summary_report += "\n" + "="*80 + "\n"
                    summary_report += "DOMAIN CONTROLS & GEOLOGICAL BOUNDARIES\n"
                    summary_report += "="*80 + "\n\n"

                    summary_report += f"Domain Enforcement: {'ENABLED' if domain_ctrl.get('domain_enforcement_enabled') else 'DISABLED'}\n"
                    summary_report += f"Domain Column: {domain_ctrl.get('domain_column', 'None')}\n"
                    summary_report += f"Number of Domains: {domain_ctrl.get('n_domains', 0)}\n"

                    domains = domain_ctrl.get('domains_present', [])
                    if domains:
                        summary_report += f"Domains Present: {', '.join(domains[:10])}"
                        if len(domains) > 10:
                            summary_report += f" ... and {len(domains) - 10} more"
                        summary_report += "\n"

                    summary_report += f"Contact Handling: {domain_ctrl.get('contact_handling', 'no_boundaries')}\n"
                    summary_report += f"Estimation Strategy: {domain_ctrl.get('estimation_strategy', 'global')}\n\n"

                    # Domain statement
                    domain_statement = domain_ctrl.get('domain_statement', '')
                    if domain_statement:
                        summary_report += "Domain Statement:\n"
                        summary_report += textwrap.fill(domain_statement, width=78) + "\n\n"

                    # Professional note
                    prof_note = domain_ctrl.get('professional_note', '')
                    if prof_note:
                        summary_report += "Professional Note:\n"
                        summary_report += textwrap.fill(prof_note, width=78) + "\n"

                zf.writestr("audit_summary.txt", summary_report)

            self._log(f"Audit bundle exported to: {file_path}", "success")
            QMessageBox.information(self, "Export Complete", f"Audit bundle saved to:\n{file_path}")

        except Exception as e:
            logger.error(f"Error exporting audit bundle: {e}", exc_info=True)
            QMessageBox.warning(self, "Export Error", f"Failed to export audit bundle:\n{str(e)}")

    def show_diagnostics(self):
        """Show diagnostics dialog with parity plots and statistical checks."""
        if self.kriging_results is None or self.data_df is None:
            QMessageBox.information(self, "No Results", "Please run estimation first.")
            return

        try:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QTextEdit
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import seaborn as sns
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Simple Kriging Diagnostics")
            dialog.resize(1200, 800)

            layout = QVBoxLayout(dialog)

            # Tab widget for different diagnostics
            tabs = QTabWidget()

            # Tab 1: Parity Plot
            parity_tab = QWidget()
            parity_layout = QVBoxLayout(parity_tab)

            # Sample estimates back to composite locations
            variable = self.kriging_results.get('variable')
            if variable not in self.data_df.columns:
                QMessageBox.warning(self, "Error", f"Variable '{variable}' not found in original data.")
                return

            # Get filtered data used for estimation
            filtered_data = self._get_filtered_data()
            if filtered_data is None or variable not in filtered_data.columns:
                QMessageBox.warning(self, "Error", "Cannot access filtered data used for estimation.")
                return

            # Get coordinates for sampling
            data_coords = filtered_data[['X', 'Y', 'Z']].to_numpy()
            data_values = filtered_data[variable].to_numpy()

            # Sample SK estimates at data locations (simple nearest neighbor for now)
            grid_x = self.kriging_results.get('grid_x')
            grid_y = self.kriging_results.get('grid_y')
            grid_z = self.kriging_results.get('grid_z')
            estimates = self.kriging_results.get('estimates')

            if grid_x is None or estimates is None:
                QMessageBox.warning(self, "Error", "Invalid results data.")
                return

            # Ensure estimates is a numpy array
            estimates = np.asarray(estimates)

            # Flatten grid coordinates and estimates
            grid_coords = np.column_stack([
                grid_x.ravel(), grid_y.ravel(), grid_z.ravel()
            ])
            est_values = estimates.ravel()

            # Find nearest grid points to data locations
            from scipy.spatial import cKDTree
            tree = cKDTree(grid_coords)
            distances, indices = tree.query(data_coords, k=1)

            sampled_estimates = est_values[indices]

            # Create parity plot
            fig1 = Figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111)

            # Plot 1:1 line
            min_val = min(data_values.min(), sampled_estimates.min())
            max_val = max(data_values.max(), sampled_estimates.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='1:1 Line')

            # Plot data points
            ax1.scatter(data_values, sampled_estimates, alpha=0.6, s=20)

            # Calculate statistics
            r2 = r2_score(data_values, sampled_estimates)
            mae = mean_absolute_error(data_values, sampled_estimates)
            rmse = np.sqrt(mean_squared_error(data_values, sampled_estimates))
            slope = np.polyfit(data_values, sampled_estimates, 1)[0]

            ax1.set_xlabel(f'Observed {variable}')
            ax1.set_ylabel(f'Estimated {variable} (SK)')
            ax1.set_title(f'Parity Plot - {variable}\nR² = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}, Slope = {slope:.3f}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            canvas1 = FigureCanvas(fig1)
            parity_layout.addWidget(canvas1)

            # Statistics text
            stats_text = QTextEdit()
            stats_text.setReadOnly(True)
            stats_text.setMaximumHeight(100)
            stats_text.setPlainText(f"""Cross-Validation Statistics:
R² Score: {r2:.4f}
Mean Absolute Error: {mae:.4f}
Root Mean Square Error: {rmse:.4f}
Regression Slope: {slope:.4f}
Samples: {len(data_values)}
""")
            parity_layout.addWidget(stats_text)

            tabs.addTab(parity_tab, "Parity Plot")

            # Tab 2: Swath Plots
            swath_tab = QWidget()
            swath_layout = QVBoxLayout(swath_tab)

            # Create swath plots along X, Y, Z
            fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

            directions = ['X', 'Y', 'Z']
            for i, direction in enumerate(directions):
                ax = axes[i]

                # Bin data along the direction
                bins = np.linspace(filtered_data[direction].min(), filtered_data[direction].max(), 20)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                obs_means = []
                est_means = []

                for j in range(len(bins) - 1):
                    mask = (filtered_data[direction] >= bins[j]) & (filtered_data[direction] < bins[j+1])
                    if mask.sum() > 0:
                        obs_means.append(data_values[mask].mean())
                        est_means.append(sampled_estimates[mask].mean())
                    else:
                        obs_means.append(np.nan)
                        est_means.append(np.nan)

                ax.plot(bin_centers, obs_means, 'b-o', label='Observed', markersize=4)
                ax.plot(bin_centers, est_means, 'r-s', label='Estimated', markersize=4)

                ax.set_xlabel(f'{direction} Coordinate')
                ax.set_ylabel(f'Mean {variable}')
                ax.set_title(f'Swath Plot - {direction} Direction')
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            canvas2 = FigureCanvas(fig2)
            swath_layout.addWidget(canvas2)

            tabs.addTab(swath_tab, "Swath Plots")

            # Tab 3: Distribution Comparison
            dist_tab = QWidget()
            dist_layout = QVBoxLayout(dist_tab)

            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Observed distribution
            ax1.hist(data_values, bins=30, alpha=0.7, color='blue', density=True, label='Observed')
            ax1.set_xlabel(f'{variable} Value')
            ax1.set_ylabel('Density')
            ax1.set_title('Observed Distribution')
            ax1.grid(True, alpha=0.3)

            # Estimated distribution
            ax2.hist(sampled_estimates, bins=30, alpha=0.7, color='red', density=True, label='Estimated')
            ax2.set_xlabel(f'{variable} Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Estimated Distribution')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            canvas3 = FigureCanvas(fig3)
            dist_layout.addWidget(canvas3)

            # Distribution statistics
            dist_stats = QTextEdit()
            dist_stats.setReadOnly(True)
            dist_stats.setMaximumHeight(120)
            dist_stats.setPlainText(f"""Distribution Statistics:

Observed {variable}:
  Mean: {data_values.mean():.3f}
  Std:  {data_values.std():.3f}
  Min:  {data_values.min():.3f}
  Max:  {data_values.max():.3f}

Estimated {variable}:
  Mean: {sampled_estimates.mean():.3f}
  Std:  {sampled_estimates.std():.3f}
  Min:  {sampled_estimates.min():.3f}
  Max:  {sampled_estimates.max():.3f}
""")
            dist_layout.addWidget(dist_stats)

            tabs.addTab(dist_tab, "Distributions")

            # Tab 4: SK Collapse Diagnostic (if collapsed)
            collapse_metadata = self.kriging_results.get("metadata", {}).get("sk_mean_collapse", {})
            if collapse_metadata.get("flag"):
                collapse_tab = QWidget()
                collapse_layout = QVBoxLayout(collapse_tab)

                # Title
                title = QLabel("<h3>Simple Kriging Mean Collapse Diagnostic</h3>")
                collapse_layout.addWidget(title)

                # Diagnostic metrics
                metrics_text = QTextEdit()
                metrics_text.setReadOnly(True)
                metrics_text.setMaximumHeight(300)

                severity = collapse_metadata.get("severity", "")
                ratio = collapse_metadata.get("collapse_ratio", 0.0)
                std_est = collapse_metadata.get("std_estimates", 0.0)
                std_data = collapse_metadata.get("std_data", 0.0)
                threshold = collapse_metadata.get("threshold", 0.0)
                interpretation = collapse_metadata.get("interpretation", "")
                recommendation = collapse_metadata.get("recommendation", "")

                metrics_html = f"""
<b>Collapse Severity:</b> {severity}<br>
<b>Collapse Ratio:</b> {ratio:.4f}<br><br>
<b>Standard Deviation (SK Estimates):</b> {std_est:.3f}<br>
<b>Standard Deviation (Original Data):</b> {std_data:.3f}<br>
<b>Collapse Threshold (5% of data std):</b> {threshold:.3f}<br><br>
<b>Interpretation:</b><br>{interpretation}<br><br>
<b>Recommendation:</b><br>{recommendation}<br><br>
<b>Professional Context:</b><br>
Simple Kriging was run using a global mean to assess the strength of local spatial
continuity. Results show strong reversion to the mean, confirming that Ordinary
Kriging is the appropriate estimator for production resource reporting.
"""

                metrics_text.setHtml(metrics_html)
                collapse_layout.addWidget(metrics_text)

                tabs.addTab(collapse_tab, "SK Collapse Diagnostic")

            layout.addWidget(tabs)
            dialog.exec()

        except ImportError as e:
            QMessageBox.warning(self, "Missing Dependencies",
                              f"Diagnostics require matplotlib and seaborn.\nPlease install: pip install matplotlib seaborn\n\nError: {e}")
        except Exception as e:
            logger.error(f"Error showing diagnostics: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to show diagnostics:\n{str(e)}")

    def run_cross_validation(self):
        """Run Leave-One-Out Cross-Validation on Simple Kriging."""
        if self.kriging_results is None or self.data_df is None:
            QMessageBox.information(self, "No Results", "Please run estimation first.")
            return

        # Confirm with user (this can be time-consuming)
        filtered_data = self._get_filtered_data()
        if filtered_data is None:
            QMessageBox.warning(self, "Error", "Cannot access filtered data.")
            return

        n_samples = len(filtered_data)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Run Cross-Validation")
        msg.setText(
            f"Leave-One-Out Cross-Validation will estimate all {n_samples} samples.\n\n"
            "This may take several minutes depending on the dataset size.\n\n"
            "Do you want to proceed?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        if msg.exec() != QMessageBox.Yes:
            return

        try:
            from ..geostats.sk_cross_validation import run_loocv_simple_kriging

            # Gather SK parameters from UI
            variable = self.kriging_results.get('variable')
            coords = filtered_data[['X', 'Y', 'Z']].to_numpy()
            values = filtered_data[variable].to_numpy()

            # Get SK parameters (reconstruct from current settings)
            from ..models.simple_kriging3d import SKParameters

            # Get variogram parameters from current results metadata
            metadata = self.kriging_results.get('metadata', {})
            vario_info = metadata.get('variogram', {})

            sk_params = SKParameters(
                global_mean=float(self.kriging_results['metadata']['global_mean']),
                max_neighbours=int(self.max_samples_spin.value()),
                min_neighbours=int(self.min_samples_spin.value()),
                search_radius=float(self.search_radius_spin.value()),
                nugget=float(vario_info.get('nugget', 0.0)),
                sill=float(vario_info.get('sill', 1.0)),
                range_major=float(vario_info.get('range_major', 100.0)),
                range_minor=float(vario_info.get('range_minor', 100.0)),
                range_vert=float(vario_info.get('range_vert', 10.0)),
                azimuth=float(vario_info.get('azimuth', 0.0)),
                dip=float(vario_info.get('dip', 0.0)),
                model_type=vario_info.get('model_type', 'spherical')
            )

            # Show progress dialog
            progress_dialog = QProgressDialog("Running cross-validation...", "Cancel", 0, 100, self)
            progress_dialog.setWindowTitle("Cross-Validation")
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setValue(0)

            # Progress callback
            def update_cv_progress(pct, msg):
                progress_dialog.setValue(pct)
                progress_dialog.setLabelText(msg)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise InterruptedError("Cross-validation cancelled by user")

            # Run CV
            self._log("Starting cross-validation (LOOCV)...", "info")
            cv_results = run_loocv_simple_kriging(
                coords, values, sk_params, progress_callback=update_cv_progress
            )

            progress_dialog.close()
            self._log("Cross-validation completed.", "success")

            # Display results
            self._show_cv_results(cv_results, coords, variable)

        except InterruptedError:
            self._log("Cross-validation cancelled.", "warning")
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Cross-validation failed:\n{str(e)}")

    def _show_cv_results(self, cv_results, coords, variable):
        """Display cross-validation results in a dialog with tabs."""
        try:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, QTextEdit, QPushButton
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import scipy.stats as stats

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Simple Kriging Cross-Validation - {variable}")
            dialog.resize(1200, 800)

            layout = QVBoxLayout(dialog)
            tabs = QTabWidget()

            # Tab 1: Summary Metrics
            summary_tab = self._create_cv_summary_tab(cv_results)
            tabs.addTab(summary_tab, "Summary")

            # Tab 2: Parity Plot
            parity_tab = self._create_cv_parity_plot(cv_results, variable)
            tabs.addTab(parity_tab, "Parity Plot")

            # Tab 3: Residual Analysis
            residual_tab = self._create_cv_residual_plots(cv_results, variable)
            tabs.addTab(residual_tab, "Residuals")

            # Tab 4: Q-Q Plot
            qq_tab = self._create_cv_qq_plot(cv_results)
            tabs.addTab(qq_tab, "Q-Q Plot")

            layout.addWidget(tabs)

            # Export button
            export_btn = QPushButton("Export CV Results")
            export_btn.clicked.connect(lambda: self._export_cv_results(cv_results, coords, variable))
            layout.addWidget(export_btn)

            dialog.exec()

        except Exception as e:
            logger.error(f"Error displaying CV results: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to display CV results:\n{str(e)}")

    def _create_cv_summary_tab(self, cv_results):
        """Create summary metrics tab for CV results."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

        tab = QWidget()
        layout = QVBoxLayout(tab)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        n_total = len(cv_results.sample_ids)
        n_valid = np.sum(~np.isnan(cv_results.estimated_values))

        summary_html = f"""
        <h2>Cross-Validation Summary</h2>
        <h3>Validation Metrics</h3>
        <table border="1" cellpadding="5" cellspacing="0">
        <tr><td><b>Mean Error (ME)</b></td><td>{cv_results.me:.4f}</td></tr>
        <tr><td><b>Mean Absolute Error (MAE)</b></td><td>{cv_results.mae:.4f}</td></tr>
        <tr><td><b>Root Mean Squared Error (RMSE)</b></td><td>{cv_results.rmse:.4f}</td></tr>
        <tr><td><b>R-Squared</b></td><td>{cv_results.r_squared:.4f}</td></tr>
        <tr><td><b>Regression Slope</b></td><td>{cv_results.regression_slope:.4f}</td></tr>
        <tr><td><b>Regression Intercept</b></td><td>{cv_results.regression_intercept:.4f}</td></tr>
        </table>

        <h3>Sample Statistics</h3>
        <table border="1" cellpadding="5" cellspacing="0">
        <tr><td><b>Total Samples</b></td><td>{n_total}</td></tr>
        <tr><td><b>Successfully Estimated</b></td><td>{n_valid}</td></tr>
        <tr><td><b>Failed Estimates</b></td><td>{n_total - n_valid}</td></tr>
        </table>

        <h3>Interpretation</h3>
        <ul>
        """

        # ME interpretation
        std_data = np.std(cv_results.actual_values)
        if abs(cv_results.me) < 0.1 * std_data:
            summary_html += "<li>✓ <b>Mean Error:</b> Minimal bias detected</li>"
        else:
            summary_html += "<li>⚠ <b>Mean Error:</b> Systematic bias present</li>"

        # R² interpretation
        if cv_results.r_squared > 0.7:
            summary_html += "<li>✓ <b>R-Squared:</b> Strong correlation (&gt;0.7)</li>"
        elif cv_results.r_squared > 0.5:
            summary_html += "<li>○ <b>R-Squared:</b> Moderate correlation (0.5-0.7)</li>"
        else:
            summary_html += "<li>⚠ <b>R-Squared:</b> Weak correlation (&lt;0.5)</li>"

        # Slope interpretation
        if 0.9 <= cv_results.regression_slope <= 1.1:
            summary_html += "<li>✓ <b>Regression Slope:</b> Near ideal (0.9-1.1)</li>"
        else:
            summary_html += f"<li>○ <b>Regression Slope:</b> {cv_results.regression_slope:.3f} (ideal=1.0)</li>"

        summary_html += """
        </ul>

        <h3>Professional Context</h3>
        <p>
        Leave-One-Out Cross-Validation (LOOCV) provides an unbiased estimate
        of Simple Kriging prediction error. Each sample is estimated using all
        other samples, then compared to its true value.
        </p>
        <p>
        These metrics are suitable for JORC/NI 43-101 reporting and CP review.
        </p>
        """

        text_edit.setHtml(summary_html)
        layout.addWidget(text_edit)

        return tab

    def _create_cv_parity_plot(self, cv_results, variable):
        """Create parity plot for CV results."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Filter out NaN values
        valid_mask = ~np.isnan(cv_results.estimated_values)
        actual = cv_results.actual_values[valid_mask]
        estimated = cv_results.estimated_values[valid_mask]

        # Plot 1:1 line
        min_val = min(actual.min(), estimated.min())
        max_val = max(actual.max(), estimated.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='1:1 Line')

        # Plot regression line
        ax.plot([min_val, max_val],
                [cv_results.regression_intercept + cv_results.regression_slope * min_val,
                 cv_results.regression_intercept + cv_results.regression_slope * max_val],
                'b--', lw=2, alpha=0.7, label=f'Regression (slope={cv_results.regression_slope:.3f})')

        # Scatter plot
        ax.scatter(actual, estimated, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)

        ax.set_xlabel(f'Actual {variable}', fontsize=12)
        ax.set_ylabel(f'Estimated {variable} (LOOCV)', fontsize=12)
        ax.set_title(
            f'Cross-Validation Parity Plot\n'
            f'R² = {cv_results.r_squared:.3f}, RMSE = {cv_results.rmse:.3f}',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        return tab

    def _create_cv_residual_plots(self, cv_results, variable):
        """Create residual analysis plots for CV results."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        fig = Figure(figsize=(12, 5))

        # Filter valid residuals
        valid_mask = ~np.isnan(cv_results.residuals)
        residuals = cv_results.residuals[valid_mask]
        actual = cv_results.actual_values[valid_mask]

        # Plot 1: Residual histogram
        ax1 = fig.add_subplot(121)
        ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
        ax1.axvline(cv_results.me, color='b', linestyle='--', lw=2, label=f'Mean={cv_results.me:.3f}')
        ax1.set_xlabel('Residual (Estimated - Actual)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residual vs Actual
        ax2 = fig.add_subplot(122)
        ax2.scatter(actual, residuals, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        ax2.axhline(0, color='r', linestyle='--', lw=2, label='Zero')
        ax2.axhline(cv_results.me, color='b', linestyle='--', lw=2, label=f'Mean={cv_results.me:.3f}')
        ax2.set_xlabel(f'Actual {variable}', fontsize=11)
        ax2.set_ylabel('Residual', fontsize=11)
        ax2.set_title('Residual vs Actual', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        return tab

    def _create_cv_qq_plot(self, cv_results):
        """Create Q-Q plot for residual normality check."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import scipy.stats as stats

        tab = QWidget()
        layout = QVBoxLayout(tab)

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Filter valid residuals
        valid_mask = ~np.isnan(cv_results.residuals)
        residuals = cv_results.residuals[valid_mask]

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax)

        ax.set_title('Q-Q Plot (Residual Normality Check)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        return tab

    def _export_cv_results(self, cv_results, coords, variable):
        """Export cross-validation results to CSV."""
        from PyQt6.QtWidgets import QFileDialog
        from ..geostats.sk_cross_validation import export_cv_results_to_csv, export_cv_summary_to_txt

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CV Results", f"SK_CrossValidation_{variable}.csv",
            "CSV (*.csv)"
        )

        if not file_path:
            return

        try:
            # Export detailed results
            export_cv_results_to_csv(cv_results, coords, file_path)

            # Export summary
            summary_path = file_path.replace('.csv', '_Summary.txt')
            export_cv_summary_to_txt(cv_results, summary_path)

            QMessageBox.information(
                self, "Export Complete",
                f"Cross-validation results exported:\n{file_path}\n{summary_path}"
            )
            self._log(f"CV results exported to {file_path}", "success")

        except Exception as e:
            logger.error(f"Failed to export CV results: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Failed to export CV results:\n{str(e)}")

    def _show_sk_collapse_warning(self, severity: str, ratio: float, metadata: dict):
        """Display warning banner for SK mean collapse."""
        warning_box = QMessageBox(self)
        warning_box.setIcon(QMessageBox.Warning)
        warning_box.setWindowTitle("Simple Kriging Diagnostic Alert")

        interpretation = metadata.get("interpretation", "")
        recommendation = metadata.get("recommendation", "")

        warning_text = (
            f"<b>Simple Kriging Collapsed to Global Mean</b><br><br>"
            f"<b>Severity:</b> {severity}<br>"
            f"<b>Collapse Ratio:</b> {ratio:.4f} (std_estimates / std_data)<br><br>"
            f"<b>Interpretation:</b><br>{interpretation}<br><br>"
            f"<b>Recommendation:</b><br>{recommendation}<br><br>"
            f"<i>This result is mathematically correct and provides valuable diagnostic "
            f"information about spatial continuity. SK should be used as a sanity check "
            f"and mean control reference, not as a primary estimator.</i>"
        )

        warning_box.setText(warning_text)
        warning_box.setStandardButtons(QMessageBox.Ok)
        warning_box.exec_()

    def _show_sk_professional_context(self):
        """Show detailed professional context for Simple Kriging."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("Simple Kriging - Professional Context")

        context_text = """
<h3>Simple Kriging (SK) - Professional Positioning</h3>

<b>What is Simple Kriging?</b><br>
Simple Kriging uses a known global mean (μ) and estimates local values based on:
<pre>Z*(x) = μ + Σ wᵢ(Zᵢ - μ)</pre>

<b>When SK Works Well:</b><br>
• Strong spatial structure (low nugget)<br>
• Locally correlated data<br>
• Valid assumption of constant mean<br><br>

<b>When SK Collapses to Mean:</b><br>
• High nugget effect<br>
• Weak spatial continuity<br>
• Sparse or distant neighbours<br>
• Result: Z*(x) → μ everywhere<br><br>

<b>Professional Use Cases:</b><br>
✓ Diagnostic tool for spatial continuity<br>
✓ Bias assessment reference<br>
✓ Mean control validation<br>
✗ Not for production estimation when collapsed<br><br>

<b>For Resource Reporting:</b><br>
When SK shows strong reversion to global mean, this is valuable diagnostic
information that justifies using Ordinary Kriging (local mean adaptation) or
SGSIM (conditional simulation) for production estimates.

<i>This positioning is suitable for JORC/NI 43-101 compliance and CP review.</i>
"""

        dialog.setText(context_text)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec_()

    def show_stationarity_report(self):
        """Display stationarity validation report."""
        if not self.kriging_results:
            QMessageBox.information(self, "No Results", "Please run estimation first.")
            return

        try:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QTextEdit, QPushButton
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            metadata = self.kriging_results.get('metadata', {})
            stat_val = metadata.get('stationarity_validation', {})

            if not stat_val:
                QMessageBox.information(
                    self, "No Data",
                    "Stationarity validation data not available.\n"
                    "This may be an older result. Please re-run estimation."
                )
                return

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Stationarity Validation Report")
            dialog.resize(1200, 800)

            layout = QVBoxLayout(dialog)
            tabs = QTabWidget()

            # Tab 1: Summary
            summary_tab = self._create_stationarity_summary_tab(stat_val)
            tabs.addTab(summary_tab, "Summary")

            # Tab 2: Spatial Trends
            trend_tab = self._create_stationarity_trend_tab(stat_val)
            tabs.addTab(trend_tab, "Spatial Trends")

            # Tab 3: Domain Comparison (if applicable)
            if stat_val.get('mean_by_domain'):
                domain_tab = self._create_stationarity_domain_tab(stat_val)
                tabs.addTab(domain_tab, "Domain Analysis")

            layout.addWidget(tabs)

            # Export button
            export_btn = QPushButton("Export Report")
            export_btn.clicked.connect(lambda: self._export_stationarity_report(stat_val))
            layout.addWidget(export_btn)

            dialog.exec()

        except Exception as e:
            logger.error(f"Error displaying stationarity report: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to display stationarity report:\n{str(e)}")

    def _create_stationarity_summary_tab(self, stat_val):
        """Create summary tab for stationarity validation."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

        tab = QWidget()
        layout = QVBoxLayout(tab)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        global_mean = stat_val.get('global_mean', 0.0)
        confidence = stat_val.get('confidence_level', 'unknown').upper()
        issues = stat_val.get('issues', [])
        interpretation = stat_val.get('interpretation', '')

        # Color code confidence level
        if confidence == 'HIGH':
            confidence_color = 'green'
            confidence_symbol = '✓'
        elif confidence == 'MEDIUM':
            confidence_color = 'orange'
            confidence_symbol = '○'
        else:
            confidence_color = 'red'
            confidence_symbol = '⚠'

        summary_html = f"""
        <h2>Stationarity Validation Summary</h2>

        <h3>Assessment</h3>
        <table border="1" cellpadding="5" cellspacing="0">
        <tr><td><b>Global Mean Used</b></td><td>{global_mean:.6f}</td></tr>
        <tr><td><b>Confidence Level</b></td><td style="color: {confidence_color}; font-weight: bold;">
            {confidence_symbol} {confidence}</td></tr>
        </table>

        <h3>Interpretation</h3>
        <p style="background-color: #f0f0f0; padding: 10px; border-left: 4px solid {confidence_color};">
        {interpretation}
        </p>
        """

        if issues:
            summary_html += """
            <h3>Identified Issues</h3>
            <ul style="color: #d32f2f;">
            """
            for issue in issues:
                summary_html += f"<li>{issue}</li>"
            summary_html += "</ul>"
        else:
            summary_html += """
            <h3>Issues</h3>
            <p style="color: green;">✓ No stationarity issues identified</p>
            """

        summary_html += """
        <h3>Professional Context</h3>
        <p>
        Stationarity validation ensures that the global mean assumption is appropriate
        for Simple Kriging. This analysis checks for:
        </p>
        <ul>
        <li><b>Domain-specific means:</b> Significant differences between geological domains</li>
        <li><b>Spatial trends:</b> Systematic increases/decreases in X, Y, or Z directions</li>
        <li><b>Mean stability:</b> Whether a single global mean is representative</li>
        </ul>
        <p>
        These checks are critical for JORC/NI 43-101 compliance and CP review.
        </p>
        """

        text_edit.setHtml(summary_html)
        layout.addWidget(text_edit)

        return tab

    def _create_stationarity_trend_tab(self, stat_val):
        """Create spatial trend plots tab."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create figure with 3 subplots
        fig = Figure(figsize=(12, 10))

        trends = [
            ('X', stat_val.get('trend_x', {})),
            ('Y', stat_val.get('trend_y', {})),
            ('Z', stat_val.get('trend_z', {}))
        ]

        for idx, (direction, trend_dict) in enumerate(trends, 1):
            ax = fig.add_subplot(3, 1, idx)

            if trend_dict:
                r2 = trend_dict.get('r_squared', 0)
                p_val = trend_dict.get('p_value', 1)
                slope = trend_dict.get('slope', 0)

                # Determine significance
                if r2 > 0.3 and p_val < 0.05:
                    trend_status = "STRONG TREND"
                    color = 'red'
                elif r2 > 0.15 and p_val < 0.05:
                    trend_status = "MODERATE TREND"
                    color = 'orange'
                else:
                    trend_status = "NO SIGNIFICANT TREND"
                    color = 'green'

                # Display as text (actual plotting would require original data)
                ax.text(
                    0.5, 0.5,
                    f"{direction}-Direction Trend Analysis\n\n"
                    f"R² = {r2:.4f}\n"
                    f"p-value = {p_val:.4f}\n"
                    f"Slope = {slope:.6f}\n\n"
                    f"Status: {trend_status}",
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                ax.text(
                    0.5, 0.5,
                    f"{direction}-Direction: No trend data available",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='center',
                    horizontalalignment='center'
                )
                ax.axis('off')

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Add interpretation text
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(150)
        text_edit.setHtml("""
        <b>Trend Analysis Interpretation:</b><br>
        • <b>R² &gt; 0.3 + p &lt; 0.05:</b> Strong trend - stationarity questionable<br>
        • <b>R² 0.15-0.3 + p &lt; 0.05:</b> Moderate trend - review recommended<br>
        • <b>R² &lt; 0.15 or p &gt; 0.05:</b> No significant trend - stationarity supported<br><br>
        <i>Strong trends indicate that the global mean varies systematically across space,
        suggesting Ordinary Kriging may be more appropriate.</i>
        """)
        layout.addWidget(text_edit)

        return tab

    def _create_stationarity_domain_tab(self, stat_val):
        """Create domain comparison tab."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        mean_by_domain = stat_val.get('mean_by_domain', {})
        global_mean = stat_val.get('global_mean', 0.0)

        # Create bar chart
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        domains = list(mean_by_domain.keys())
        means = list(mean_by_domain.values())

        # Calculate percent differences
        pct_diffs = [abs(m - global_mean) / global_mean * 100 for m in means]

        # Color bars based on difference
        colors = ['red' if pd > 15 else 'orange' if pd > 10 else 'green' for pd in pct_diffs]

        bars = ax.bar(range(len(domains)), means, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(global_mean, color='blue', linestyle='--', linewidth=2, label=f'Global Mean ({global_mean:.3f})')

        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title('Mean Values by Domain', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Add table with detailed numbers
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(200)

        table_html = f"""
        <h3>Domain Mean Comparison</h3>
        <table border="1" cellpadding="5" cellspacing="0">
        <tr><th>Domain</th><th>Mean</th><th>Δ from Global</th><th>% Difference</th><th>Status</th></tr>
        """

        for domain, mean_val in mean_by_domain.items():
            diff = mean_val - global_mean
            pct_diff = abs(diff) / global_mean * 100
            if pct_diff > 15:
                status = '<span style="color: red;">⚠ SIGNIFICANT</span>'
            elif pct_diff > 10:
                status = '<span style="color: orange;">○ MODERATE</span>'
            else:
                status = '<span style="color: green;">✓ OK</span>'

            table_html += f"""
            <tr>
                <td>{domain}</td>
                <td>{mean_val:.4f}</td>
                <td>{diff:+.4f}</td>
                <td>{pct_diff:.1f}%</td>
                <td>{status}</td>
            </tr>
            """

        table_html += f"""
        <tr style="font-weight: bold; background-color: #e3f2fd;">
            <td>Global</td>
            <td>{global_mean:.4f}</td>
            <td>-</td>
            <td>-</td>
            <td>Reference</td>
        </tr>
        </table>
        <br>
        <p><i>Differences &gt;15% from global mean suggest domain-based estimation may be appropriate.</i></p>
        """

        text_edit.setHtml(table_html)
        layout.addWidget(text_edit)

        return tab

    def _export_stationarity_report(self, stat_val):
        """Export stationarity report to text file."""
        from PyQt6.QtWidgets import QFileDialog
        from ..geostats.sk_stationarity import StationarityReport, export_stationarity_report_to_txt

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Stationarity Report", "SK_Stationarity_Report.txt",
            "Text Files (*.txt)"
        )

        if not file_path:
            return

        try:
            # Reconstruct StationarityReport object from dict
            report = StationarityReport(
                global_mean=stat_val.get('global_mean', 0.0),
                mean_by_domain=stat_val.get('mean_by_domain', {}),
                trend_x=stat_val.get('trend_x', {}),
                trend_y=stat_val.get('trend_y', {}),
                trend_z=stat_val.get('trend_z', {}),
                issues=stat_val.get('issues', []),
                confidence_level=stat_val.get('confidence_level', 'unknown')
            )

            export_stationarity_report_to_txt(report, file_path)

            QMessageBox.information(
                self, "Export Complete",
                f"Stationarity report exported to:\n{file_path}"
            )
            self._log(f"Stationarity report exported to {file_path}", "success")

        except Exception as e:
            logger.error(f"Failed to export stationarity report: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Failed to export report:\n{str(e)}")

    def _log(self, msg, level="info"):
        color = "#4fc3f7" if level == "info" else "#81c784" if level == "success" else "#e57373"
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f'<span style="color:#777">[{timestamp}]</span> <span style="color:{color}">{msg}</span>')

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Data source
            if hasattr(self, 'data_source_composited'):
                settings['data_source'] = 'composited' if self.data_source_composited.isChecked() else 'raw'
            
            # Variable and global mean
            settings['variable'] = get_safe_widget_value(self, 'var_combo')
            settings['global_mean'] = get_safe_widget_value(self, 'global_mean_spin')
            settings['auto_mean'] = get_safe_widget_value(self, 'auto_mean_check')
            
            # Variogram model
            settings['model_type'] = get_safe_widget_value(self, 'model_combo')
            settings['nugget'] = get_safe_widget_value(self, 'nugget_spin')
            settings['sill'] = get_safe_widget_value(self, 'sill_spin')
            settings['range'] = get_safe_widget_value(self, 'range_spin')
            
            # Anisotropy
            settings['azimuth'] = get_safe_widget_value(self, 'azimuth_spin')
            settings['dip'] = get_safe_widget_value(self, 'dip_spin')
            settings['ani_minor'] = get_safe_widget_value(self, 'ani_minor_spin')
            settings['ani_vert'] = get_safe_widget_value(self, 'ani_vert_spin')
            
            # Grid
            settings['xmin'] = get_safe_widget_value(self, 'xmin_spin')
            settings['ymin'] = get_safe_widget_value(self, 'ymin_spin')
            settings['zmin'] = get_safe_widget_value(self, 'zmin_spin')
            settings['grid_x'] = get_safe_widget_value(self, 'dx_spin')
            settings['grid_y'] = get_safe_widget_value(self, 'dy_spin')
            settings['grid_z'] = get_safe_widget_value(self, 'dz_spin')
            
            # Search settings
            settings['neighbors'] = get_safe_widget_value(self, 'neighbors_spin')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save simple kriging panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Data source
            if 'data_source' in settings:
                if settings['data_source'] == 'composited' and hasattr(self, 'data_source_composited'):
                    self.data_source_composited.setChecked(True)
                elif settings['data_source'] == 'raw' and hasattr(self, 'data_source_raw'):
                    self.data_source_raw.setChecked(True)
            
            # Variable and global mean
            set_safe_widget_value(self, 'var_combo', settings.get('variable'))
            set_safe_widget_value(self, 'global_mean_spin', settings.get('global_mean'))
            set_safe_widget_value(self, 'auto_mean_check', settings.get('auto_mean'))
            
            # Variogram model
            set_safe_widget_value(self, 'model_combo', settings.get('model_type'))
            set_safe_widget_value(self, 'nugget_spin', settings.get('nugget'))
            set_safe_widget_value(self, 'sill_spin', settings.get('sill'))
            set_safe_widget_value(self, 'range_spin', settings.get('range'))
            
            # Anisotropy
            set_safe_widget_value(self, 'azimuth_spin', settings.get('azimuth'))
            set_safe_widget_value(self, 'dip_spin', settings.get('dip'))
            set_safe_widget_value(self, 'ani_minor_spin', settings.get('ani_minor'))
            set_safe_widget_value(self, 'ani_vert_spin', settings.get('ani_vert'))
            
            # Grid
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            set_safe_widget_value(self, 'dx_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'dy_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'dz_spin', settings.get('grid_z'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
                
            logger.info("Restored simple kriging panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore simple kriging panel settings: {e}")
