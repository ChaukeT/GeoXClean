"""
UNIVERSAL KRIGING PANEL

UI for configuring and running Universal Kriging (UK) with Trend Models.

Features:
- Linear/Quadratic Drift Selection
- Auto-Grid detection from data bounds
- Integration with Variogram Panel (Residual Variograms)
- Progress tracking and 3D visualization signals
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QWidget, QSplitter, QScrollArea, QFrame, QMessageBox, QProgressBar,
    QRadioButton, QButtonGroup, QCheckBox, QDialog
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import populate_variable_combo
from .base_analysis_panel import BaseAnalysisPanel
from ..geostats.uk_validation_utils import UKValidationDashboard
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

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


class UniversalKrigingPanel(BaseAnalysisPanel):

    """
    Panel for configuring and launching 3D Universal Kriging.

    Supports Linear/Quadratic drift models.
    """
    # PanelManager metadata
    PANEL_ID = "UniversalKrigingPanel"
    PANEL_NAME = "UniversalKriging Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT





    task_name = "universal_kriging"
    request_visualization = pyqtSignal(dict)  # Signal to request 3D viz
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="universal_kriging")
        self.setWindowTitle("Universal Kriging")
        self.resize(1100, 750)
        self.setStyleSheet(get_analysis_panel_stylesheet())

        # State
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.grid_spec: Optional[Dict[str, float]] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.kriging_results = None

        self._build_ui()

        self._init_registry()

        # Connect progress signal to update method
        self.progress_updated.connect(self._update_progress)

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
        
        # 1. Input Data
        self._create_data_group(self.scroll_layout)
        
        # 2. Drift Model
        self._create_drift_group(self.scroll_layout)
        
        # 3. Variogram (Residuals)
        self._create_vario_group(self.scroll_layout)
        
        # 4. Grid
        self._create_grid_group(self.scroll_layout)

        # 5. Post-Processing Filters
        self._create_post_processing_group(self.scroll_layout)

        self.scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        # --- RIGHT: RESULTS & LOGS ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
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

        self.validation_btn = QPushButton("Validation Dashboard")
        self.validation_btn.setStyleSheet(_get_btn_style_action())
        self.validation_btn.setEnabled(False)
        self.validation_btn.clicked.connect(self.open_validation_dashboard)

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.viz_btn)
        btn_layout.addWidget(self.view_table_btn)
        btn_layout.addWidget(self.validation_btn)
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
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setVerticalSpacing(12)
        form.setContentsMargins(12, 10, 12, 12)
        
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
        
        # Data source status label
        self.data_source_status_label = QLabel("")
        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #888;")
        form.addRow("", self.data_source_status_label)
        
        layout.addWidget(group)

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return

        # Reload full drillhole payload so panel can apply selected source correctly.
        data = self.registry.get_drillhole_data()
        if data is not None:
            self._on_data_loaded(data)

    def _create_data_group(self, layout):
        group = QGroupBox("1. Input Data")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setVerticalSpacing(12)
        form.setContentsMargins(12, 10, 12, 12)

        self.var_combo = QComboBox()
        self.var_combo.setToolTip(
            "Select the grade variable to estimate using Universal Kriging.\n"
            "Variable should be from validated drillhole data (composites or assays)."
        )
        form.addRow("Variable:", self.var_combo)
        layout.addWidget(group)

    def _create_drift_group(self, layout):
        group = QGroupBox("2. Drift Model")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(8)
        vbox.setContentsMargins(12, 10, 12, 12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setVerticalSpacing(12)
        form.setContentsMargins(0, 0, 0, 0)  # Inner form, no extra margins

        self.drift_combo = QComboBox()
        self.drift_combo.addItems(["Linear", "Quadratic"])
        self.drift_combo.currentTextChanged.connect(self._update_drift_info)
        self.drift_combo.setToolTip(
            "Drift model type for spatial trend:\n"
            "• Linear: β₀ + β₁x + β₂y + β₃z (4 coefficients)\n"
            "• Quadratic: Includes x², y², z² and cross-terms (10 coefficients)\n\n"
            "Use Linear for gradual trends, Quadratic for complex curved trends."
        )

        self.info_lbl = QLabel("Linear: β₀ + β₁x + β₂y + β₃z")
        self.info_lbl.setStyleSheet("color: #aaa; font-style: italic;")

        form.addRow("Trend Type:", self.drift_combo)
        vbox.addLayout(form)
        vbox.addWidget(self.info_lbl)
        layout.addWidget(group)

    def _create_vario_group(self, layout):
        group = QGroupBox("3. Residual Variogram")
        group.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)
        vbox = QVBoxLayout(group)
        vbox.setSpacing(8)
        vbox.setContentsMargins(12, 10, 12, 12)

        # Auto-load buttons
        btn_layout = QHBoxLayout()
        self.auto_vario_btn = QPushButton("Load from Variogram Panel")
        self.auto_vario_btn.clicked.connect(self.load_variogram_parameters)
        self.auto_vario_btn.setEnabled(False)

        self.use_assisted_btn = QPushButton("Load from Assistant")
        self.use_assisted_btn.clicked.connect(self._load_assisted_variogram)
        self.use_assisted_btn.setEnabled(False)

        btn_layout.addWidget(self.auto_vario_btn)
        btn_layout.addWidget(self.use_assisted_btn)
        vbox.addLayout(btn_layout)

        # Parameters Grid
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setVerticalSpacing(12)
        form.setContentsMargins(0, 0, 0, 0)  # Inner form, no extra margins

        self.model_combo = QComboBox()
        self.model_combo.addItems(["spherical", "exponential", "gaussian"])
        self.model_combo.setToolTip(
            "Residual variogram model type:\n"
            "• Spherical: Most common, reaches sill at range\n"
            "• Exponential: Gradual approach to sill, effective range ≈ 3×range\n"
            "• Gaussian: Very smooth, use for highly continuous phenomena\n\n"
            "Load from Variogram Panel to use pre-fit parameters."
        )

        self.range = QDoubleSpinBox(); self.range.setRange(1, 1e5); self.range.setValue(100)
        self.range.setToolTip(
            "Range of spatial continuity (meters or project units).\n"
            "Distance at which values become spatially independent.\n"
            "Should match the range from residual variogram analysis."
        )

        self.sill = QDoubleSpinBox(); self.sill.setRange(0, 1e5); self.sill.setValue(1)
        self.sill.setToolTip(
            "Partial sill of residual variogram (variance units).\n"
            "Total variability = nugget + sill.\n"
            "Should be derived from residual variogram after drift removal."
        )

        self.nugget = QDoubleSpinBox(); self.nugget.setRange(0, 1e5)
        self.nugget.setToolTip(
            "Nugget effect (variance units at distance = 0).\n"
            "Represents measurement error and micro-scale variability.\n"
            "High nugget (>40% of total sill) indicates poor spatial continuity."
        )

        form.addRow("Model:", self.model_combo)
        form.addRow("Range:", self.range)
        form.addRow("Sill:", self.sill)
        form.addRow("Nugget:", self.nugget)
        vbox.addLayout(form)

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
        
        # Auto-detect button
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
        h = QHBoxLayout()
        self.search_spin = QDoubleSpinBox(); self.search_spin.setRange(1, 10000); self.search_spin.setValue(200); self.search_spin.setPrefix("Rad: ")
        self.ndmax_spin = QSpinBox(); self.ndmax_spin.setRange(1, 200); self.ndmax_spin.setValue(12); self.ndmax_spin.setPrefix("Max: ")
        h.addWidget(self.search_spin)
        h.addWidget(self.ndmax_spin)
        vbox.addLayout(h)
        
        layout.addWidget(group)

    def _create_post_processing_group(self, layout):
        """Create post-processing options group."""
        group = QGroupBox("5. Post-Processing Filters")
        group.setStyleSheet("QGroupBox { font-weight: bold; color: #ff9800; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        vbox = QVBoxLayout(group)

        # Percentile clipping
        self.percentile_clip_check = QCheckBox("Apply percentile clipping")
        self.percentile_clip_check.setChecked(True)
        self.percentile_clip_check.setToolTip("Remove extreme values using percentiles")
        vbox.addWidget(self.percentile_clip_check)

        # Percentile settings
        h1 = QHBoxLayout()
        self.lower_percentile_spin = QDoubleSpinBox()
        self.lower_percentile_spin.setRange(0.1, 5.0)
        self.lower_percentile_spin.setValue(1.0)
        self.lower_percentile_spin.setSingleStep(0.1)
        self.lower_percentile_spin.setPrefix("Lower %: ")

        self.upper_percentile_spin = QDoubleSpinBox()
        self.upper_percentile_spin.setRange(95.0, 99.9)
        self.upper_percentile_spin.setValue(99.0)
        self.upper_percentile_spin.setSingleStep(0.1)
        self.upper_percentile_spin.setPrefix("Upper %: ")

        h1.addWidget(self.lower_percentile_spin)
        h1.addWidget(self.upper_percentile_spin)
        vbox.addLayout(h1)

        # Mean reversion
        self.mean_reversion_check = QCheckBox("Apply mean reversion damping")
        self.mean_reversion_check.setChecked(True)
        self.mean_reversion_check.setToolTip("Damp extrapolation using mean reversion")
        vbox.addWidget(self.mean_reversion_check)

        # Mean reversion factor
        h2 = QHBoxLayout()
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.6, 0.95)
        self.lambda_spin.setValue(0.8)
        self.lambda_spin.setSingleStep(0.05)
        self.lambda_spin.setPrefix("λ: ")
        h2.addWidget(QLabel("Damping factor:"))
        h2.addWidget(self.lambda_spin)
        h2.addStretch()
        vbox.addLayout(h2)

        # Positivity constraint
        self.positivity_check = QCheckBox("Enforce positivity constraint")
        self.positivity_check.setChecked(True)
        self.positivity_check.setToolTip("Clamp negative values to zero")
        vbox.addWidget(self.positivity_check)

        layout.addWidget(group)

    # --- LOGIC ---

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if not self.registry:
                logger.warning("DataRegistry not available - get_registry() returned None")
                return
            
            self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
            self.registry.variogramResultsLoaded.connect(self._on_variogram_results_loaded)
            
            # Source-toggle panels must load full drillhole payload for proper source switching.
            d = self.registry.get_drillhole_data()
            if d is not None:
                self._on_data_loaded(d)

            # Check for existing variogram results
            vario = self.registry.get_variogram_results()
            if vario is not None:
                self._on_variogram_results_loaded(vario)
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None

    def _on_variogram_results_loaded(self, results):
        """Store variogram results and enable loading buttons."""
        self.variogram_results = results
        # Enable variogram loading buttons if they exist
        if hasattr(self, 'auto_vario_btn'):
            self.auto_vario_btn.setEnabled(results is not None)
        if hasattr(self, 'use_assisted_btn'):
            self.use_assisted_btn.setEnabled(results is not None)

    def _on_data_loaded(self, data):
        """Load data, respecting user's data source selection."""
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
                elif composites_available:
                    # Fallback to composites
                    df = composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                    if hasattr(self, 'data_source_composited'):
                        self.data_source_composited.setChecked(True)
                elif assays_available:
                    # Fallback to assays
                    df = assays
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
                    if hasattr(self, 'data_source_raw'):
                        self.data_source_raw.setChecked(True)
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
            self.drillhole_data = ensure_xyz_columns(df)
            data_source = "composited" if (hasattr(self, 'data_source_composited') and self.data_source_composited.isChecked()) else "raw"
            logger.info(f"Universal Kriging: Using {data_source} data ({len(df)} samples)")
            self._log_event(f"Data loaded: {len(self.drillhole_data)} records from {data_source} data")
            
            # Populate variable combo using standardized method
            populate_variable_combo(self.var_combo, self.drillhole_data)
            
            # Auto-detect grid from drillhole extent if checkbox is enabled
            if hasattr(self, 'auto_fit_grid_check') and self.auto_fit_grid_check.isChecked():
                self._auto_detect_grid()
                logger.info("Universal Kriging: Auto-fitted grid to drillhole extent")

    def _update_drift_info(self, txt):
        if "Linear" in txt:
            self.info_lbl.setText("Linear: β₀ + β₁x + β₂y + β₃z")
        else:
            self.info_lbl.setText("Quadratic: Includes x², y², z² and cross-terms")

    def load_variogram_parameters(self) -> bool:
        """Populate fields from variogram results.

        Priority order:
        1. combined_3d_model (what's displayed in variogram panel) - PREFERRED
        2. major variogram (for anisotropic models)
        3. omni variogram (fallback for isotropic)
        """
        if not self.variogram_results:
            return False

        # Only update UI if it's been built
        if not (hasattr(self, 'model_combo') and hasattr(self, 'range') and
                hasattr(self, 'sill') and hasattr(self, 'nugget')):
            logger.debug("Universal Kriging panel: UI not ready for variogram parameter loading")
            return False

        try:
            model_type = self.model_combo.currentText().lower()

            # PRIORITY 1: Use combined_3d_model (matches what variogram panel displays)
            combined = self.variogram_results.get('combined_3d_model', {})
            if combined and combined.get('model_type', '').lower() == model_type:
                nugget = combined.get('nugget', 0.0)
                total_sill = combined.get('sill', 0.0)
                partial_sill = total_sill - nugget
                major_range = combined.get('major_range', 100.0)

                self.range.setValue(major_range)
                self.sill.setValue(max(0.001, partial_sill))
                self.nugget.setValue(nugget)

                self._log(f"Loaded variogram from combined model: range={major_range:.1f}, "
                         f"sill={partial_sill:.3f}, nugget={nugget:.3f}")
                return True

            # PRIORITY 2: Use major directional variogram
            major = self.variogram_results.get('major', {})
            if major and major.get('model_type', '').lower() == model_type:
                nugget = major.get('nugget', 0.0)
                total_sill = major.get('sill', 0.0)
                partial_sill = total_sill - nugget
                major_range = major.get('range', 100.0)

                self.range.setValue(major_range)
                self.sill.setValue(max(0.001, partial_sill))
                self.nugget.setValue(nugget)

                self._log(f"Loaded variogram from major direction: range={major_range:.1f}, "
                         f"sill={partial_sill:.3f}, nugget={nugget:.3f}")
                return True

            # PRIORITY 3: Use omni-directional variogram
            omni = self.variogram_results.get('omni', {})
            if omni and omni.get('model_type', '').lower() == model_type:
                nugget = omni.get('nugget', 0.0)
                total_sill = omni.get('sill', 0.0)
                partial_sill = total_sill - nugget
                omni_range = omni.get('range', 100.0)

                self.range.setValue(omni_range)
                self.sill.setValue(max(0.001, partial_sill))
                self.nugget.setValue(nugget)

                self._log(f"Loaded variogram from omni direction: range={omni_range:.1f}, "
                         f"sill={partial_sill:.3f}, nugget={nugget:.3f}")
                return True

            # If no matching model type found, show warning
            available_models = []
            for key in ['combined_3d_model', 'major', 'omni']:
                if key in self.variogram_results:
                    model = self.variogram_results[key]
                    if isinstance(model, dict) and 'model_type' in model:
                        available_models.append(f"{key}: {model['model_type']}")

            QMessageBox.warning(self, "Model Type Mismatch",
                               f"No {model_type} model found in variogram results.\n\n"
                               f"Available models:\n" + "\n".join(available_models) +
                               f"\n\nChange the model type to match available results.")
            return False

        except Exception as e:
            logger.error(f"Error loading variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load variogram parameters:\n{str(e)}")
            return False

    def _load_assisted_variogram(self):
        """Load variogram parameters from Variogram Assistant."""
        # Check if UI is ready
        if not (hasattr(self, 'model_combo') and hasattr(self, 'range') and
                hasattr(self, 'sill') and hasattr(self, 'nugget')):
            QMessageBox.warning(self, "UI Not Ready", "Please wait for the panel to finish loading.")
            return

        if not self.controller or not hasattr(self.controller, '_assisted_variogram_models'):
            QMessageBox.warning(self, "No Model", "No assisted variogram model available.")
            return

        try:
            assisted_models = self.controller._assisted_variogram_models
            if not assisted_models:
                QMessageBox.warning(self, "No Model", "No assisted variogram models available.")
                return

            # Get current model type
            model_type = self.model_combo.currentText().lower()

            # Find matching assisted model
            assisted_model = None
            for model in assisted_models:
                if model.get('model_type', '').lower() == model_type:
                    assisted_model = model
                    break

            if not assisted_model:
                available_types = [m.get('model_type', 'unknown') for m in assisted_models]
                QMessageBox.warning(self, "No Matching Model",
                                   f"No assisted {model_type} model found.\n\n"
                                   f"Available types: {', '.join(set(available_types))}")
                return

            # Apply parameters
            nugget = assisted_model.get('nugget', 0.0)
            sill = assisted_model.get('sill', 1.0)
            range_val = assisted_model.get('range', 100.0)

            self.range.setValue(range_val)
            self.sill.setValue(max(0.001, sill))
            self.nugget.setValue(nugget)

            self._log(f"Loaded assisted variogram: range={range_val:.1f}, "
                     f"sill={sill:.3f}, nugget={nugget:.3f}")

        except Exception as e:
            logger.error(f"Error loading assisted variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load assisted variogram:\n{str(e)}")

    def _auto_detect_grid(self):
        """Robust grid detection from DataFrame bounds."""
        if self.drillhole_data is None: return
        
        df = self.drillhole_data
        xmin, xmax = df['X'].min(), df['X'].max()
        ymin, ymax = df['Y'].min(), df['Y'].max()
        zmin, zmax = df['Z'].min(), df['Z'].max()
        
        # 5% Padding
        pad = 0.05
        dx = (xmax - xmin) * pad
        dy = (ymax - ymin) * pad
        dz = (zmax - zmin) * pad
        
        xmin -= dx; xmax += dx
        ymin -= dy; ymax += dy
        zmin -= dz; zmax += dz
        
        # Block sizes
        bs_x, bs_y, bs_z = self.dx.value(), self.dy.value(), self.dz.value()
        
        # Snap origin
        xmin = np.floor(xmin / bs_x) * bs_x
        ymin = np.floor(ymin / bs_y) * bs_y
        zmin = np.floor(zmin / bs_z) * bs_z
        
        # Count blocks
        nx = int(np.ceil((xmax - xmin) / bs_x))
        ny = int(np.ceil((ymax - ymin) / bs_y))
        nz = int(np.ceil((zmax - zmin) / bs_z))
        
        # Update UI
        self.xmin_spin.setValue(xmin)
        self.ymin_spin.setValue(ymin)
        self.zmin_spin.setValue(zmin)
        self.nx.setValue(nx)
        self.ny.setValue(ny)
        self.nz.setValue(nz)
        
        self._log_event(f"Grid auto-fitted: {nx}x{ny}x{nz}", "success")

    def gather_parameters(self) -> Dict[str, Any]:
        return {
            "data_df": self.drillhole_data,
            "variable": self.var_combo.currentText(),
            "drift_type": self.drift_combo.currentText().lower(),
            "variogram_params": {
                "model_type": self.model_combo.currentText().lower(),
                "range": self.range.value(),
                "sill": self.sill.value(),
                "nugget": self.nugget.value()
            },
            "grid_origin": (self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value()),
            "grid_spacing": (self.dx.value(), self.dy.value(), self.dz.value()),
            "grid_counts": (self.nx.value(), self.ny.value(), self.nz.value()),
            "n_neighbors": self.ndmax_spin.value(),
            "max_distance": self.search_spin.value(),
            "post_processing_config": {
                "percentile_clip": self.percentile_clip_check.isChecked(),
                "lower_percentile": self.lower_percentile_spin.value(),
                "upper_percentile": self.upper_percentile_spin.value(),
                "mean_reversion": self.mean_reversion_check.isChecked(),
                "lambda_factor": self.lambda_spin.value(),
                "positivity_constraint": self.positivity_check.isChecked(),
                "replacement_value": 0.0
            }
        }

    def validate_inputs(self) -> bool:
        if self.drillhole_data is None:
            QMessageBox.warning(self, "Error", "No Data Loaded")
            return False
        if not self.var_combo.currentText():
            QMessageBox.warning(self, "Error", "No Variable Selected")
            return False
        return True

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Universal Kriging.

        Universal Kriging requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Validated data quality

        Returns:
            True if data is acceptable for Universal Kriging
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
                f"Cannot run Universal Kriging:\n\n{message}\n\n"
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

    # ---------------- PROGRESS -----------------
    def _log_event(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"info": f"{ModernColors.TEXT_PRIMARY}", "success": "#81c784", "warning": "#ffb74d", "error": "#e57373"}
        self.log_text.append(f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {colors.get(level, f"{ModernColors.TEXT_PRIMARY}")};">{message}</span>')
    
    def _log(self, msg, level="info"):
        """Log message (compatibility method that wraps _log_event)."""
        self._log_event(msg, level)
    
    def _update_progress(self, percent: int, message: str = ""):
        pct = max(0, min(100, int(percent)))
        self.progress_bar.setValue(pct)
        if message:
            self.progress_bar.setFormat(f"{pct}% - {message}")
        else:
            self.progress_bar.setFormat(f"{pct}%")
        self.lbl_status.setText(message if message else f"{pct}% complete")
    
    def show_progress(self, message: str) -> None:
        self._update_progress(0, message)
        self._log_event(f"Starting: {message}", "info")
        self.run_btn.setEnabled(False)
    
    def hide_progress(self) -> None:
        self.run_btn.setEnabled(True)
    
    def run_analysis(self) -> None:
        if not self.controller:
            self._log_event("Error: No Controller connected.", "error")
            return

        if not self.validate_inputs():
            return

        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return

        params = self.gather_parameters()
        self.show_progress("Starting Universal Kriging...")

        def progress_callback(percent: int, message: str):
            """Update progress from worker thread using signals."""
            # Emit signal to update UI from main thread
            self.progress_updated.emit(percent, message)
        
        try:
            # Inject callback into params so controller logic can use it
            params['_progress_callback'] = progress_callback
            
            self.controller.run_universal_kriging(
                config=params,
                callback=self.on_results,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Dispatch failed: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def on_results(self, payload):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        if "error" in payload:
            self._log_event(f"Kriging Failed: {payload['error']}", "error")
            return
            
        self.kriging_results = payload

        # Register results and block model to DataRegistry
        if self.registry:
            try:
                # Register universal kriging results
                self.registry.register_universal_kriging_results(payload, source_panel="Universal Kriging")
                self._log_event("Results registered to data registry", "info")

                # Also register the block model DataFrame for cross-sections and other panels
                grid_x = payload.get('grid_x')
                grid_y = payload.get('grid_y')
                grid_z = payload.get('grid_z')
                estimates = payload.get('estimates')
                variances = payload.get('variances')

                if grid_x is not None and estimates is not None:
                    coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                    block_df = pd.DataFrame({
                        'X': coords[:, 0],
                        'Y': coords[:, 1],
                        'Z': coords[:, 2],
                        f'{payload.get("property_name", "uk")}_est': estimates.ravel(),
                        f'{payload.get("property_name", "uk")}_var': variances.ravel() if variances is not None else np.full_like(estimates.ravel(), np.nan)
                    }).dropna()

                    # Register the block model
                    self.registry.register_block_model_generated(
                        block_df,
                        source_panel="Universal Kriging",
                        metadata={
                            'variable': payload.get('property_name', 'unknown'),
                            'method': 'universal_kriging',
                            'grid_size': (len(np.unique(grid_x)), len(np.unique(grid_y)), len(np.unique(grid_z))),
                            'n_blocks': len(block_df)
                        }
                    )
                    self._log_event("Block model registered to data registry", "info")
            except Exception as e:
                logger.warning(f"Failed to register universal kriging results/block model: {e}")

        self.viz_btn.setEnabled(True)
        self.view_table_btn.setEnabled(True)
        self.validation_btn.setEnabled(True)
        
        est = payload.get('estimates', [])
        self.summary_text.setText(
            f"Universal Kriging Complete.\n"
            f"Count: {len(est.flatten())}\n"
            f"Min: {np.nanmin(est):.3f}\n"
            f"Max: {np.nanmax(est):.3f}\n"
            f"Mean: {np.nanmean(est):.3f}"
        )
        self._log_event("✓ Estimation completed successfully.", "success")
    
    def visualize_results(self):
        if self.kriging_results:
            self.request_visualization.emit(self.kriging_results)
            self._log_event("Sent to 3D Viewer.", "info")

    def open_uk_validation_dashboard(self):
        """Open comprehensive UK validation dashboard."""
        if self.kriging_results is None or self.drillhole_data is None:
            QMessageBox.warning(self, "Validation Error",
                              "No UK results or data available for validation.")
            return

        try:
            self._log_event("Opening UK validation dashboard...", "info")

            # Ensure kriging_results is in the right format
            if hasattr(self.kriging_results, 'estimates'):
                # It's a UniversalKrigingResults object, convert to dict
                kriging_data = {
                    'estimates': self.kriging_results.estimates,
                    'kriging_variance': self.kriging_results.kriging_variance,
                    'status': self.kriging_results.status,
                    'metadata': getattr(self.kriging_results, 'metadata', {})
                }
            elif isinstance(self.kriging_results, dict):
                kriging_data = self.kriging_results
            else:
                QMessageBox.warning(self, "Validation Error",
                                  "UK results are in an unexpected format.")
                return

            # Create validation dialog
            try:
                dialog = UKValidationDialog(kriging_data, self.drillhole_data, self)
                dialog.exec()
            except ValueError as e:
                QMessageBox.warning(self, "Validation Error", str(e))
            except Exception as e:
                self._log_event(f"UK validation dashboard failed: {e}", "error")
                QMessageBox.critical(self, "Validation Error", f"Failed to open validation dashboard: {e}")

        except Exception as e:
            self._log_event(f"UK validation dashboard failed: {e}", "error")
            QMessageBox.critical(self, "Validation Error", f"Failed to open validation dashboard: {e}")

    def open_validation_dashboard(self):
        """Open UK validation dashboard with comprehensive diagnostics."""
        if self.kriging_results is None or self.drillhole_data is None:
            QMessageBox.warning(self, "Validation Error",
                              "No UK results or data available for validation.")
            return

        try:
            self._log_event("Running UK validation workflow...", "info")

            # Get variable name
            variable = self.var_combo.currentText()
            if not variable:
                QMessageBox.warning(self, "Validation Error", "No variable selected.")
                return

            # Run validation
            dashboard = UKValidationDashboard()
            validation_report = dashboard.run_full_validation(
                self.kriging_results, self.drillhole_data, variable
            )

            # Display results
            self._show_validation_dashboard(validation_report)

            self._log_event("UK validation completed", "success")

        except Exception as e:
            logger.error(f"Validation dashboard failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Validation Error", f"Validation failed: {str(e)}")

    def _show_validation_dashboard(self, report):
        """Display validation dashboard in a comprehensive dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QTabWidget, QWidget, QLabel, QFormLayout, QGroupBox, QScrollArea

        dialog = QDialog(self)
        dialog.setWindowTitle("Universal Kriging Validation Dashboard")
        dialog.resize(900, 700)

        layout = QVBoxLayout(dialog)

        # Create tab widget for different validation sections
        tab_widget = QTabWidget()

        # Overview tab
        overview_tab = self._create_overview_tab(report)
        tab_widget.addTab(overview_tab, "Overview")

        # Drift Analysis tab
        drift_tab = self._create_drift_analysis_tab(report)
        tab_widget.addTab(drift_tab, "Drift Analysis")

        # Residual Analysis tab
        residual_tab = self._create_residual_analysis_tab(report)
        tab_widget.addTab(residual_tab, "Residual Analysis")

        # Edge Stability tab
        edge_tab = self._create_edge_stability_tab(report)
        tab_widget.addTab(edge_tab, "Edge Stability")

        # Recommendations tab
        rec_tab = self._create_recommendations_tab(report)
        tab_widget.addTab(rec_tab, "Recommendations")

        layout.addWidget(tab_widget)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(lambda: self._export_validation_report(report))
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _create_overview_tab(self, report):
        """Create overview tab with summary statistics."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        assessment = report.get('overall_assessment', {})

        # Overall score
        score_group = QGroupBox("Overall Assessment")
        score_layout = QFormLayout(score_group)

        score = assessment.get('overall_score', 0)
        confidence = assessment.get('confidence_level', 'unknown')

        score_layout.addRow("Confidence Level:", QLabel(confidence.upper()))
        score_layout.addRow("Validation Score:", QLabel(f"{score:.2f}"))
        score_layout.addRow("Total Issues:", QLabel(str(assessment.get('total_issues', 0))))

        layout.addWidget(score_group)

        # Key metrics
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QFormLayout(metrics_group)

        drift_section = report['validation_sections'].get('drift_fit', {})
        residual_section = report['validation_sections'].get('residual_analysis', {})

        r2 = drift_section.get('r_squared', 0)
        rmse = drift_section.get('rmse', 0)
        mean_residual = residual_section.get('statistics', {}).get('mean', 0)

        metrics_layout.addRow("Drift R²:", QLabel(f"{r2:.3f}"))
        metrics_layout.addRow("Drift RMSE:", QLabel(f"{rmse:.3f}"))
        metrics_layout.addRow("Mean Residual:", QLabel(f"{mean_residual:.3f}"))

        layout.addWidget(metrics_group)

        # Residual sill correction info
        metadata = report.get('metadata', {})
        correction_info = metadata.get('residual_sill_correction')
        if correction_info:
            correction_group = QGroupBox("Residual Sill Correction")
            correction_layout = QFormLayout(correction_group)

            orig_sill = correction_info.get('original_sill', 0)
            corr_sill = correction_info.get('corrected_sill', 0)
            residual_var = correction_info.get('residual_variance', 0)

            correction_layout.addRow("Original Sill:", QLabel(f"{orig_sill:.3f}"))
            correction_layout.addRow("Corrected Sill:", QLabel(f"{corr_sill:.3f}"))
            correction_layout.addRow("Residual Variance:", QLabel(f"{residual_var:.3f}"))

            layout.addWidget(correction_group)

        layout.addStretch()
        return widget

    def _create_drift_analysis_tab(self, report):
        """Create drift analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        drift_section = report['validation_sections'].get('drift_fit', {})

        # Drift fit statistics
        stats_group = QGroupBox("Drift Fit Statistics")
        stats_layout = QFormLayout(stats_group)

        stats_layout.addRow("Drift Type:", QLabel(drift_section.get('drift_type', 'unknown')))
        stats_layout.addRow("R²:", QLabel(f"{drift_section.get('r_squared', 0):.4f}" if 'r_squared' in drift_section else "N/A"))
        stats_layout.addRow("RMSE:", QLabel(f"{drift_section.get('rmse', 0):.4f}" if 'rmse' in drift_section else "N/A"))
        stats_layout.addRow("Mean Residual:", QLabel(f"{drift_section.get('mean_residual', 0):.4f}" if 'mean_residual' in drift_section else "N/A"))

        variance = drift_section.get('variance_breakdown', {})
        if variance:
            drift_contrib = variance.get('drift_contribution_pct', 0)
            stats_layout.addRow("Drift Contribution %:", QLabel(f"{drift_contrib:.1f}"))

        layout.addWidget(stats_group)

        # Issues
        issues = drift_section.get('issues', [])
        if issues:
            issues_group = QGroupBox("Issues Detected")
            issues_layout = QVBoxLayout(issues_group)

            for issue in issues:
                issues_layout.addWidget(QLabel(f"• {issue}"))

            layout.addWidget(issues_group)

        # Recommendations
        recommendations = drift_section.get('recommendations', [])
        if recommendations:
            rec_group = QGroupBox("Recommendations")
            rec_layout = QVBoxLayout(rec_group)

            for rec in recommendations:
                rec_layout.addWidget(QLabel(f"• {rec}"))

            layout.addWidget(rec_group)

        layout.addStretch()
        return widget

    def _create_residual_analysis_tab(self, report):
        """Create residual analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        residual_section = report['validation_sections'].get('residual_analysis', {})

        # Residual statistics
        stats_group = QGroupBox("Residual Statistics")
        stats_layout = QFormLayout(stats_group)

        statistics = residual_section.get('statistics', {})
        stats_layout.addRow("Mean:", QLabel(f"{statistics.get('mean', 0):.4f}" if 'mean' in statistics else "N/A"))
        stats_layout.addRow("Std Dev:", QLabel(f"{statistics.get('std', 0):.4f}" if 'std' in statistics else "N/A"))
        stats_layout.addRow("Skewness:", QLabel(f"{statistics.get('skewness', 0):.4f}" if 'skewness' in statistics else "N/A"))
        stats_layout.addRow("Kurtosis:", QLabel(f"{statistics.get('kurtosis', 0):.4f}" if 'kurtosis' in statistics else "N/A"))

        distribution = residual_section.get('distribution_check', {})
        is_normal = distribution.get('is_normal', False)
        stats_layout.addRow("Normal Distribution:", QLabel("Yes" if is_normal else "No"))

        layout.addWidget(stats_group)

        # Issues
        issues = residual_section.get('issues', [])
        if issues:
            issues_group = QGroupBox("Issues Detected")
            issues_layout = QVBoxLayout(issues_group)

            for issue in issues:
                issues_layout.addWidget(QLabel(f"• {issue}"))

            layout.addWidget(issues_group)

        layout.addStretch()
        return widget

    def _create_edge_stability_tab(self, report):
        """Create edge stability analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        edge_section = report['validation_sections'].get('edge_instability', {})

        # Edge statistics
        stats_group = QGroupBox("Edge Stability Statistics")
        stats_layout = QFormLayout(stats_group)

        stats_layout.addRow("Extreme Low Values:", QLabel(str(edge_section.get('n_extreme_low', 0))))
        stats_layout.addRow("Extreme High Values:", QLabel(str(edge_section.get('n_extreme_high', 0))))
        stats_layout.addRow("Percent Extreme:", QLabel(f"{edge_section.get('percent_extreme', 0):.1f}"))
        stats_layout.addRow("Data Std Dev Threshold:", QLabel(f"{edge_section.get('data_std_threshold', 0):.3f}"))

        layout.addWidget(stats_group)

        # Issues
        issues = edge_section.get('issues', [])
        if issues:
            issues_group = QGroupBox("Stability Issues")
            issues_layout = QVBoxLayout(issues_group)

            for issue in issues:
                issues_layout.addWidget(QLabel(f"• {issue}"))

            layout.addWidget(issues_group)

        # Recommendations
        recommendations = edge_section.get('recommendations', [])
        if recommendations:
            rec_group = QGroupBox("Recommendations")
            rec_layout = QVBoxLayout(rec_group)

            for rec in recommendations:
                rec_layout.addWidget(QLabel(f"• {rec}"))

            layout.addWidget(rec_group)

        layout.addStretch()
        return widget

    def _create_recommendations_tab(self, report):
        """Create recommendations tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        assessment = report.get('overall_assessment', {})

        # Overall recommendations
        rec_group = QGroupBox("Overall Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        recommendations = assessment.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                rec_layout.addWidget(QLabel(f"• {rec}"))
        else:
            rec_layout.addWidget(QLabel("No major issues detected. UK results appear reliable."))

        layout.addWidget(rec_group)

        # Action items
        action_group = QGroupBox("Suggested Actions")
        action_layout = QVBoxLayout(action_group)

        actions = []
        if assessment.get('confidence_level') == 'low':
            actions.append("Consider switching to Ordinary Kriging")
            actions.append("Review drift model selection")
            actions.append("Apply post-processing filters")

        if assessment.get('total_issues', 0) > 2:
            actions.append("Re-run with different variogram parameters")
            actions.append("Consider data transformations")

        if not actions:
            actions.append("UK configuration appears appropriate")
            actions.append("Results can be used with confidence")

        for action in actions:
            action_layout.addWidget(QLabel(f"• {action}"))

        layout.addWidget(action_group)

        layout.addStretch()
        return widget

    def _export_validation_report(self, report):
        """Export validation report to file."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Validation Report", "", "JSON files (*.json);;All files (*)"
            )

            if filename:
                import json
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                QMessageBox.information(self, "Export Complete",
                                      f"Validation report saved to {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export report: {str(e)}")

    def open_results_table(self):
        """Open universal kriging results as a table."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run estimation first.")
            return

        try:
            grid_x = self.kriging_results.get('grid_x')
            grid_y = self.kriging_results.get('grid_y')
            grid_z = self.kriging_results.get('grid_z')
            estimates = self.kriging_results.get('estimates')
            variances = self.kriging_results.get('variances')
            variable = self.kriging_results.get('property_name', 'UK_estimate').replace('UK_', '')

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

            if variances is not None:
                df['Variance'] = variances.ravel()

            df = df.dropna()

            title = f"Universal Kriging Results - {variable}"

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


class UKValidationDialog(QDialog):
    """Comprehensive UK validation dashboard with diagnostics and quality metrics."""

    def __init__(self, kriging_results, drillhole_data, parent=None):
        super().__init__(parent)
        self.kriging_results = kriging_results

        # Ensure drillhole_data is a DataFrame
        if isinstance(drillhole_data, dict):
            # Try to extract the DataFrame from common keys
            if 'composites_df' in drillhole_data:
                self.drillhole_data = drillhole_data['composites_df']
            elif 'composites' in drillhole_data:
                self.drillhole_data = drillhole_data['composites']
            elif 'assays' in drillhole_data:
                self.drillhole_data = drillhole_data['assays']
            else:
                # Convert dict to DataFrame if possible
                try:
                    self.drillhole_data = pd.DataFrame(drillhole_data)
                except:
                    self.drillhole_data = None
        elif hasattr(drillhole_data, 'values'):  # DataFrame-like
            self.drillhole_data = pd.DataFrame(drillhole_data)
        else:
            self.drillhole_data = drillhole_data

        # Validate that we have proper data
        if self.drillhole_data is None or (hasattr(self.drillhole_data, 'empty') and self.drillhole_data.empty):
            raise ValueError("No valid drillhole data available for validation")

        self.setWindowTitle("Universal Kriging Validation Dashboard")
        self.resize(1200, 800)
        self.setStyleSheet(get_analysis_panel_stylesheet())

        self._setup_ui()
        self._run_validation()

    def _setup_ui(self):
        """Set up the validation dashboard UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("UK Model Validation & Diagnostics")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Create tab widget for different validation views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Summary tab
        self._create_summary_tab()

        # Drift analysis tab
        self._create_drift_tab()

        # Variogram analysis tab
        self._create_variogram_tab()

        # Residual analysis tab
        self._create_residual_tab()

        # Quality metrics tab
        self._create_quality_tab()

        # Buttons
        button_layout = QHBoxLayout()

        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self._export_report)
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _create_summary_tab(self):
        """Create the summary validation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Status indicators
        status_group = QGroupBox("Model Status")
        status_layout = QFormLayout(status_group)

        self.status_labels = {}
        status_items = [
            ("Data Validation", "⏳"),
            ("Drift Fit Quality", "⏳"),
            ("Residual Variogram", "⏳"),
            ("UK Stability", "⏳"),
            ("Grade Distribution", "⏳"),
            ("Overall Quality", "⏳")
        ]

        for label_text, status in status_items:
            label = QLabel(status)
            label.setStyleSheet("font-size: 14px; font-weight: bold;")
            status_layout.addRow(label_text, label)
            self.status_labels[label_text] = label

        layout.addWidget(status_group)

        # Key metrics
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QFormLayout(metrics_group)

        self.metrics_labels = {}
        metric_items = [
            ("Mean Preservation", "—"),
            ("Min/Max Range", "—"),
            ("Drift R²", "—"),
            ("Residual Variance", "—"),
            ("Extreme Values", "—")
        ]

        for label_text, value in metric_items:
            label = QLabel(value)
            metrics_layout.addRow(label_text, label)
            self.metrics_labels[label_text] = label

        layout.addWidget(metrics_group)
        layout.addStretch()

        self.tab_widget.addTab(widget, "Summary")

    def _create_drift_tab(self):
        """Create drift analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Drift fit statistics
        drift_group = QGroupBox("Drift Model Fit")
        drift_layout = QFormLayout(drift_group)

        self.drift_labels = {}
        drift_items = [
            ("Drift Type", "—"),
            ("R² Score", "—"),
            ("RMSE", "—"),
            ("Variance Explained", "—"),
            ("Drift Contribution %", "—")
        ]

        for label_text, value in drift_items:
            label = QLabel(value)
            drift_layout.addRow(label_text, label)
            self.drift_labels[label_text] = label

        layout.addWidget(drift_group)

        # Placeholder for drift visualization
        viz_placeholder = QLabel("Drift surface visualization would go here")
        viz_placeholder.setStyleSheet("background-color: #333; padding: 20px; border-radius: 5px;")
        viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(viz_placeholder)

        layout.addStretch()
        self.tab_widget.addTab(widget, "Drift Analysis")

    def _create_variogram_tab(self):
        """Create variogram analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Variogram parameters
        vario_group = QGroupBox("Variogram Parameters")
        vario_layout = QFormLayout(vario_group)

        self.vario_labels = {}
        vario_items = [
            ("Model Type", "—"),
            ("Sill (Residual)", "—"),
            ("Range", "—"),
            ("Nugget", "—"),
            ("Fit Quality (R²)", "—")
        ]

        for label_text, value in vario_items:
            label = QLabel(value)
            vario_layout.addRow(label_text, label)
            self.vario_labels[label_text] = label

        layout.addWidget(vario_group)

        # Variogram comparison
        comparison_group = QGroupBox("Variogram Comparison")
        comparison_layout = QVBoxLayout(comparison_group)

        comparison_text = QLabel(
            "This tab would show:\n"
            "• Experimental vs Fitted Residual Variogram\n"
            "• Raw Grade Variogram (for comparison)\n"
            "• Variogram model parameters\n"
            "• Fit quality metrics"
        )
        comparison_text.setStyleSheet("padding: 10px;")
        comparison_layout.addWidget(comparison_text)

        layout.addWidget(comparison_group)
        layout.addStretch()

        self.tab_widget.addTab(widget, "Variogram Analysis")

    def _create_residual_tab(self):
        """Create residual analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Residual statistics
        residual_group = QGroupBox("Residual Statistics")
        residual_layout = QFormLayout(residual_group)

        self.residual_labels = {}
        residual_items = [
            ("Mean Residual", "—"),
            ("Residual Variance", "—"),
            ("Residual Skewness", "—"),
            ("Residual Kurtosis", "—"),
            ("Normality Test", "—")
        ]

        for label_text, value in residual_items:
            label = QLabel(value)
            residual_layout.addRow(label_text, label)
            self.residual_labels[label_text] = label

        layout.addWidget(residual_group)

        # Residual distribution plot placeholder
        plot_placeholder = QLabel("Residual histogram and Q-Q plot would go here")
        plot_placeholder.setStyleSheet("background-color: #333; padding: 20px; border-radius: 5px;")
        plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(plot_placeholder)

        layout.addStretch()
        self.tab_widget.addTab(widget, "Residual Analysis")

    def _create_quality_tab(self):
        """Create quality metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # UK acceptance rules
        rules_group = QGroupBox("UK Model Acceptance Rules")
        rules_layout = QVBoxLayout(rules_group)

        self.rule_checks = {}
        rules = [
            ("Mean within 1% of composite mean", False),
            ("No values < 0 or > 65% (for Fe)", False),
            ("Residual histogram ≈ normal", False),
            ("Drift explains >5% variance", False),
            ("Extreme values controlled", False),
            ("UK range < 2x data range", False)
        ]

        for rule_text, passed in rules:
            check_label = QLabel(f"{'✅' if passed else '❌'} {rule_text}")
            rules_layout.addWidget(check_label)
            self.rule_checks[rule_text] = check_label

        layout.addWidget(rules_group)

        # Recommendations
        rec_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        self.recommendations_text = QLabel("Recommendations will be generated after validation...")
        self.recommendations_text.setWordWrap(True)
        self.recommendations_text.setStyleSheet("padding: 10px;")
        rec_layout.addWidget(self.recommendations_text)

        layout.addWidget(rec_group)
        layout.addStretch()

        self.tab_widget.addTab(widget, "Quality Metrics")

    def _run_validation(self):
        """Run comprehensive UK validation."""
        try:
            # Extract data
            estimates = self.kriging_results.get('estimates', np.array([]))

            # Get the selected variable from the parent panel
            variable_name = None
            if hasattr(self.parent(), 'var_combo'):
                variable_name = self.parent().var_combo.currentText()

            if variable_name and variable_name in self.drillhole_data.columns:
                data_values = self.drillhole_data[variable_name].values
            else:
                # Fallback: try to find a numeric column that might be grades
                numeric_cols = self.drillhole_data.select_dtypes(include=[np.number]).columns
                grade_cols = [col for col in numeric_cols if col not in ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH']]
                if grade_cols:
                    data_values = self.drillhole_data[grade_cols[0]].values
                else:
                    self._show_validation_error("Could not identify grade variable column")
                    return

            if len(estimates) == 0 or len(data_values) == 0:
                self._show_validation_error("No data available for validation")
                return

            # Run validation checks
            self._validate_data_integrity(estimates, data_values)
            self._validate_drift_fit()
            self._validate_variogram()
            self._validate_residuals(estimates, data_values)
            self._validate_overall_quality(estimates, data_values)
            self._generate_recommendations()

        except Exception as e:
            self._show_validation_error(f"Validation failed: {e}")

    def _validate_data_integrity(self, estimates, data_values):
        """Validate basic data integrity."""
        # Mean preservation
        data_mean = np.mean(data_values)
        estimate_mean = np.mean(estimates)
        mean_diff_pct = abs(estimate_mean - data_mean) / data_mean * 100

        if mean_diff_pct < 1.0:
            self.metrics_labels["Mean Preservation"].setText(f"{mean_diff_pct:.2f}% diff (✅)")
            self.status_labels["Data Validation"].setText("✅")
        else:
            self.metrics_labels["Mean Preservation"].setText(f"{mean_diff_pct:.2f}% diff (❌)")
            self.status_labels["Data Validation"].setText("❌")

        # Min/Max range
        min_val, max_val = np.min(estimates), np.max(estimates)
        self.metrics_labels["Min/Max Range"].setText(f"[{min_val:.2f}, {max_val:.2f}]")

    def _validate_drift_fit(self):
        """Validate drift model fit."""
        # Check if drift information is available in results
        metadata = self.kriging_results.get('metadata', {})

        if 'drift_analysis' in metadata:
            drift_info = metadata['drift_analysis']
            r_squared = drift_info.get('r_squared', 0)
            drift_contrib = drift_info.get('drift_contribution_pct', 0)

            self.drift_labels["R² Score"].setText(f"{r_squared:.3f}")
            self.drift_labels["Drift Contribution %"].setText(f"{drift_contrib:.1f}%")

            if r_squared > 0.3:
                self.status_labels["Drift Fit Quality"].setText("✅")
            else:
                self.status_labels["Drift Fit Quality"].setText("⚠️")
        else:
            self.status_labels["Drift Fit Quality"].setText("❌ (No drift info)")

    def _validate_variogram(self):
        """Validate variogram fit."""
        metadata = self.kriging_results.get('metadata', {})

        if 'residual_variogram' in metadata:
            vario_info = metadata['residual_variogram']
            fit_quality = vario_info.get('validation', {}).get('fit_quality', 'unknown')
            vario_params = vario_info.get('variogram_params', {})

            model_type = vario_params.get('model_type', 'unknown')
            sill_val = vario_params.get('sill', 0)
            range_val = vario_params.get('range', 0)

            self.vario_labels["Model Type"].setText(model_type)
            self.vario_labels["Sill (Residual)"].setText(f"{sill_val:.3f}")
            self.vario_labels["Range"].setText(f"{range_val:.1f}")

            if fit_quality in ['good', 'fair']:
                self.status_labels["Residual Variogram"].setText("✅")
            else:
                self.status_labels["Residual Variogram"].setText("⚠️")
        else:
            self.status_labels["Residual Variogram"].setText("❌ (No residual variogram)")

    def _validate_residuals(self, estimates, data_values):
        """Validate residual properties."""
        # This would compute actual residuals if drift info available
        # For now, show placeholder
        self.status_labels["Grade Distribution"].setText("⏳ (Analysis pending)")

    def _validate_overall_quality(self, estimates, data_values):
        """Run overall quality checks."""
        # Check UK acceptance rules
        data_mean = np.mean(data_values)
        estimate_mean = np.mean(estimates)
        mean_diff_pct = abs(estimate_mean - data_mean) / data_mean * 100

        min_val, max_val = np.min(estimates), np.max(estimates)

        # Rule 1: Mean within 1%
        rule1_pass = mean_diff_pct < 1.0
        self._update_rule_check("Mean within 1% of composite mean", rule1_pass)

        # Rule 2: No impossible values (for Fe)
        rule2_pass = min_val >= 0 and max_val <= 65.0
        self._update_rule_check("No values < 0 or > 65% (for Fe)", rule2_pass)

        # Rule 3: Controlled extremes
        data_p01, data_p99 = np.percentile(data_values, [1, 99])
        extreme_count = np.sum((estimates < data_p01) | (estimates > data_p99))
        rule3_pass = extreme_count < len(estimates) * 0.01  # Less than 1% extremes
        self._update_rule_check("Extreme values controlled", rule3_pass)

        # Overall status
        if rule1_pass and rule2_pass and rule3_pass:
            self.status_labels["Overall Quality"].setText("✅ PASS")
        else:
            self.status_labels["Overall Quality"].setText("❌ FAIL")

    def _update_rule_check(self, rule_text, passed):
        """Update rule check display."""
        if rule_text in self.rule_checks:
            status = "✅" if passed else "❌"
            current_text = self.rule_checks[rule_text].text()
            # Remove old status and add new
            clean_text = current_text[1:] if current_text.startswith(('✅', '❌')) else current_text
            self.rule_checks[rule_text].setText(f"{status} {clean_text}")

    def _generate_recommendations(self):
        """Generate validation recommendations."""
        recommendations = []

        # Check various conditions and provide recommendations
        estimates = self.kriging_results.get('estimates', np.array([]))
        if len(estimates) > 0:
            min_val, max_val = np.min(estimates), np.max(estimates)

            if max_val > 100:  # Assuming percentage units
                recommendations.append("• Extreme high values detected - consider stronger drift contraction")
            if min_val < 0:
                recommendations.append("• Negative values present - ensure positivity constraint is applied")
            if abs(max_val - min_val) > 100:
                recommendations.append("• Very wide value range - review variogram parameters")

        if not recommendations:
            recommendations.append("• UK results look reasonable - monitor for stability in production use")

        self.recommendations_text.setText("\n".join(recommendations))

    def _show_validation_error(self, message):
        """Show validation error message."""
        for label in self.status_labels.values():
            label.setText("❌")

        error_msg = f"Validation Error: {message}"
        self.recommendations_text.setText(error_msg)

        QMessageBox.warning(self, "Validation Error", message)

    def _export_report(self):
        """Export validation report."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export UK Validation Report", "", "JSON files (*.json);;All files (*)"
            )

            if filename:
                # Collect all validation results
                report = {
                    'validation_timestamp': str(pd.Timestamp.now()),
                    'status_summary': {k: v.text() for k, v in self.status_labels.items()},
                    'key_metrics': {k: v.text() for k, v in self.metrics_labels.items()},
                    'acceptance_rules': {k: v.text() for k, v in self.rule_checks.items()},
                    'recommendations': self.recommendations_text.text()
                }

                import json
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                QMessageBox.information(self, "Export Complete",
                                      f"UK validation report saved to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export report: {e}")

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
            
            # Variable
            settings['variable'] = get_safe_widget_value(self, 'var_combo')
            
            # Drift model
            settings['drift_order'] = get_safe_widget_value(self, 'drift_combo')
            settings['drift_limit'] = get_safe_widget_value(self, 'drift_limit_spin')
            settings['cap_extreme_values'] = get_safe_widget_value(self, 'cap_check')
            settings['contract_to_mean'] = get_safe_widget_value(self, 'contract_check')
            settings['contraction_factor'] = get_safe_widget_value(self, 'contract_spin')
            
            # Variogram model
            settings['model_type'] = get_safe_widget_value(self, 'model_combo')
            settings['nugget'] = get_safe_widget_value(self, 'nugget_spin')
            settings['sill'] = get_safe_widget_value(self, 'sill_spin')
            settings['range'] = get_safe_widget_value(self, 'range_spin')
            
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
            logger.warning(f"Could not save universal kriging panel settings: {e}")
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
            
            # Variable
            set_safe_widget_value(self, 'var_combo', settings.get('variable'))
            
            # Drift model
            set_safe_widget_value(self, 'drift_combo', settings.get('drift_order'))
            set_safe_widget_value(self, 'drift_limit_spin', settings.get('drift_limit'))
            set_safe_widget_value(self, 'cap_check', settings.get('cap_extreme_values'))
            set_safe_widget_value(self, 'contract_check', settings.get('contract_to_mean'))
            set_safe_widget_value(self, 'contract_spin', settings.get('contraction_factor'))
            
            # Variogram model
            set_safe_widget_value(self, 'model_combo', settings.get('model_type'))
            set_safe_widget_value(self, 'nugget_spin', settings.get('nugget'))
            set_safe_widget_value(self, 'sill_spin', settings.get('sill'))
            set_safe_widget_value(self, 'range_spin', settings.get('range'))
            
            # Grid
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            set_safe_widget_value(self, 'dx_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'dy_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'dz_spin', settings.get('grid_z'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
                
            logger.info("Restored universal kriging panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore universal kriging panel settings: {e}")
