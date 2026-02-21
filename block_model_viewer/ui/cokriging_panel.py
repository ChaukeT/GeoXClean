"""
Co-Kriging Panel

UI panel for configuring and running Co-Kriging estimation.
Refactored for modern UX/UI.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import populate_variable_combo
from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit, QCheckBox,
    QWidget, QSplitter, QScrollArea, QFrame, QMessageBox, QProgressBar,
    QRadioButton, QButtonGroup, QDialog
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from datetime import datetime
from .base_analysis_panel import BaseAnalysisPanel
from .modern_styles import get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class CoKrigingPanel(BaseAnalysisPanel):
    """
    Co-Kriging Analysis Panel.
    """
    # PanelManager metadata
    PANEL_ID = "CoKrigingPanel"
    PANEL_NAME = "CoKriging Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "cokriging"
    request_visualization = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        # Initialize state BEFORE super().__init__
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.drillhole_data: Optional[pd.DataFrame] = None

        super().__init__(parent=parent, panel_id="cokriging")

        # Build the UI immediately after base initialization
        self._build_ui()

        # Initialize registry connections
        self._init_registry()

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
        self.setWindowTitle("Co-Kriging")
        self.resize(1100, 800)
    
    def _build_ui(self):
        """Build custom split-pane UI. Called by base class."""
        self._setup_ui()

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if not self.registry:
                logger.warning("DataRegistry not available - get_registry() returned None")
                return
            
            # FIX: Check if signals are available before connecting
            dh_signal = self.registry.drillholeDataLoaded
            if dh_signal is not None:
                dh_signal.connect(self._on_data_loaded)
                logger.debug("CokrigingPanel: Connected to drillholeDataLoaded signal")
            
            vario_signal = self.registry.variogramResultsLoaded
            if vario_signal is not None:
                vario_signal.connect(self._on_vario_loaded)
                logger.debug("CokrigingPanel: Connected to variogramResultsLoaded signal")
            
            # Source-toggle panels must load full drillhole payload for proper source switching.
            d = self.registry.get_drillhole_data()
            if d is not None:
                self._on_data_loaded(d)
            v = self.registry.get_variogram_results()
            if v is not None:
                self._on_vario_loaded(v)
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None

    def _setup_ui(self):
        # Use the main_layout provided by BaseAnalysisPanel
        layout = self.main_layout
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT: CONFIGURATION ---
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        
        self._create_data_source_group(s_lay)
        self._create_var_method_group(s_lay)
        self._create_variograms_group(s_lay)
        self._create_search_grid_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # --- RIGHT: RESULTS ---
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Progress Bar
        prog_group = QGroupBox("Progress")
        colors = get_theme_colors()
        prog_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; }}")
        prog_lay = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - %v/%m")
        self.progress_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid {colors.BORDER}; border-radius: 4px; background-color: {colors.CARD_BG}; text-align: center; color: {colors.TEXT_PRIMARY}; height: 22px; }} QProgressBar::chunk {{ background-color: #4CAF50; border-radius: 3px; }}")
        prog_lay.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY};")
        prog_lay.addWidget(self.progress_label)
        r_lay.addWidget(prog_group)

        # Event Log
        log_group = QGroupBox("Event Log")
        log_lay = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas; font-size: 9pt;")
        log_lay.addWidget(self.log_text)
        r_lay.addWidget(log_group)

        res_box = QGroupBox("Results Summary")
        res_l = QVBoxLayout(res_box)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas;")
        res_l.addWidget(self.results_text)
        r_lay.addWidget(res_box, stretch=1)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("RUN CO-KRIGING")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.run_btn)
        
        self.visualize_btn = QPushButton("Visualize 3D")
        self.visualize_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 12px;")
        self.visualize_btn.clicked.connect(self.visualize_results)
        self.visualize_btn.setEnabled(False)
        btn_layout.addWidget(self.visualize_btn)

        self.view_table_btn = QPushButton("View Table")
        self.view_table_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 12px;")
        self.view_table_btn.clicked.connect(self.open_results_table)
        self.view_table_btn.setEnabled(False)
        btn_layout.addWidget(self.view_table_btn)

        self.diagnostics_btn = QPushButton("Scaling Diagnostics")
        self.diagnostics_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; padding: 12px;")
        self.diagnostics_btn.clicked.connect(self.open_scaling_diagnostics)
        self.diagnostics_btn.setEnabled(False)
        btn_layout.addWidget(self.diagnostics_btn)
        
        r_lay.addLayout(btn_layout)
        
        # Store results
        self.kriging_results = None
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_data_source_group(self, layout):
        """Create data source selection group."""
        g = QGroupBox("0. Data Source")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #81c784; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QFormLayout(g)
        l.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        l.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
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
        l.addRow("Source:", data_source_layout)
        
        # Data source status label
        self.data_source_status_label = QLabel("")
        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #888;")
        l.addRow("", self.data_source_status_label)
        
        layout.addWidget(g)

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return

        # Reload full drillhole payload so panel can apply selected source correctly.
        data = self.registry.get_drillhole_data()
        if data is not None:
            self._on_data_loaded(data)

    def _create_var_method_group(self, layout):
        g = QGroupBox("1. Variables & Method")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #4fc3f7; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QFormLayout(g)
        l.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        l.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.primary_combo = QComboBox()
        self.primary_combo.setToolTip(
            "Primary variable (target to estimate).\n"
            "This is the variable of interest (e.g., gold grade).\n"
            "Should be sparsely sampled compared to secondary."
        )

        self.secondary_combo = QComboBox()
        self.secondary_combo.setToolTip(
            "Secondary variable (auxiliary information).\n"
            "Helps improve primary estimates via cross-correlation.\n"
            "Should be densely sampled and well-correlated (|r| > 0.5)."
        )

        self.method_combo = QComboBox()
        self.method_combo.addItems(["collocated", "full"])
        self.method_combo.setToolTip(
            "Co-kriging method:\n"
            "• Collocated: Uses only secondary data at target location (faster, stable)\n"
            "• Full: Uses all secondary data in neighborhood (slower, potentially unstable)\n\n"
            "Collocated is recommended for production use."
        )

        # Connect variable selection change to correlation check
        self.primary_combo.currentTextChanged.connect(self._check_correlation)
        self.secondary_combo.currentTextChanged.connect(self._check_correlation)

        l.addRow("Primary Var:", self.primary_combo)
        l.addRow("Secondary Var:", self.secondary_combo)
        l.addRow("Method:", self.method_combo)
        
        # Correlation indicator (professional audit feature)
        self.correlation_label = QLabel("Select variables to check correlation")
        self.correlation_label.setStyleSheet("font-size: 9px; color: #888; padding: 4px;")
        self.correlation_label.setWordWrap(True)
        l.addRow("Correlation:", self.correlation_label)
        
        layout.addWidget(g)

    def _create_variograms_group(self, layout):
        g = QGroupBox("2. Variogram Models")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)

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
        l.addLayout(btn_layout)

        # Primary
        l.addWidget(QLabel("<b>Primary Variogram</b>"))
        h1 = QHBoxLayout()
        self.model_primary = QComboBox()
        self.model_primary.addItems(["spherical", "exponential"])
        self.range_primary = QDoubleSpinBox()
        self.range_primary.setRange(1, 10000)
        self.range_primary.setValue(100)
        self.sill_primary = QDoubleSpinBox()
        self.sill_primary.setValue(1)
        self.nug_primary = QDoubleSpinBox()
        self.nug_primary.setValue(0)
        h1.addWidget(self.model_primary)
        h1.addWidget(QLabel("R:"))
        h1.addWidget(self.range_primary)
        h1.addWidget(QLabel("S:"))
        h1.addWidget(self.sill_primary)
        h1.addWidget(QLabel("N:"))
        h1.addWidget(self.nug_primary)
        l.addLayout(h1)
        
        l.addWidget(QFrame(frameShape=QFrame.Shape.HLine))  # Separator
        
        # Secondary
        l.addWidget(QLabel("<b>Secondary Variogram</b>"))
        h2 = QHBoxLayout()
        self.model_secondary = QComboBox()
        self.model_secondary.addItems(["spherical", "exponential"])
        self.range_secondary = QDoubleSpinBox()
        self.range_secondary.setRange(1, 10000)
        self.range_secondary.setValue(100)
        self.sill_secondary = QDoubleSpinBox()
        self.sill_secondary.setValue(1)
        self.nug_secondary = QDoubleSpinBox()
        self.nug_secondary.setValue(0)
        h2.addWidget(self.model_secondary)
        h2.addWidget(QLabel("R:"))
        h2.addWidget(self.range_secondary)
        h2.addWidget(QLabel("S:"))
        h2.addWidget(self.sill_secondary)
        h2.addWidget(QLabel("N:"))
        h2.addWidget(self.nug_secondary)
        l.addLayout(h2)
        
        l.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        
        # Cross
        self.use_cross = QCheckBox("<b>Use Custom Cross-Variogram</b>")
        l.addWidget(self.use_cross)
        h3 = QHBoxLayout()
        self.range_cross = QDoubleSpinBox()
        self.range_cross.setRange(1, 10000)
        self.range_cross.setValue(100)
        self.range_cross.setEnabled(False)
        self.sill_cross = QDoubleSpinBox()
        self.sill_cross.setValue(0.5)
        self.sill_cross.setEnabled(False)
        self.use_cross.toggled.connect(self.range_cross.setEnabled)
        self.use_cross.toggled.connect(self.sill_cross.setEnabled)
        h3.addWidget(QLabel("Range:"))
        h3.addWidget(self.range_cross)
        h3.addWidget(QLabel("Sill:"))
        h3.addWidget(self.sill_cross)
        l.addLayout(h3)
        
        layout.addWidget(g)

    def _create_search_grid_group(self, layout):
        g = QGroupBox("3. Search & Grid")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #90a4ae; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)
        
        h1 = QHBoxLayout()
        self.neigh_spin = QSpinBox()
        self.neigh_spin.setRange(4, 50)
        self.neigh_spin.setValue(12)
        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(10, 5000)
        self.max_dist_spin.setValue(200)
        h1.addWidget(QLabel("Neighbors:"))
        h1.addWidget(self.neigh_spin)
        h1.addWidget(QLabel("Max Dist:"))
        h1.addWidget(self.max_dist_spin)
        l.addLayout(h1)
        
        # Minimum neighbors (professional audit requirement)
        h_min = QHBoxLayout()
        self.min_neigh_spin = QSpinBox()
        self.min_neigh_spin.setRange(1, 20)
        self.min_neigh_spin.setValue(3)
        self.min_neigh_spin.setToolTip("Minimum neighbors required for valid estimate (recommended: 3)")
        h_min.addWidget(QLabel("Min Neighbors:"))
        h_min.addWidget(self.min_neigh_spin)
        h_min.addStretch()
        l.addLayout(h_min)
        
        # Professional options
        self.use_sk_secondary = QCheckBox("Use SK for secondary interpolation (recommended)")
        self.use_sk_secondary.setChecked(True)
        self.use_sk_secondary.setToolTip("Use Simple Kriging to interpolate secondary variable at targets. "
                                         "This provides block-support correction and is recommended over NN.")
        l.addWidget(self.use_sk_secondary)
        
        # Grid Origin
        origin_label = QLabel("Grid Origin (corner of first block):")
        origin_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(origin_label)
        
        h0 = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setDecimals(1)
        self.xmin_spin.setValue(0)
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(1)
        self.ymin_spin.setValue(0)
        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-1e9, 1e9)
        self.zmin_spin.setDecimals(1)
        self.zmin_spin.setValue(0)
        h0.addWidget(QLabel("X₀:"))
        h0.addWidget(self.xmin_spin)
        h0.addWidget(QLabel("Y₀:"))
        h0.addWidget(self.ymin_spin)
        h0.addWidget(QLabel("Z₀:"))
        h0.addWidget(self.zmin_spin)
        l.addLayout(h0)
        
        # Block Size
        size_label = QLabel("Block Size (meters):")
        size_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(size_label)
        
        h2 = QHBoxLayout()
        self.gx = QDoubleSpinBox()
        self.gx.setValue(10)
        self.gy = QDoubleSpinBox()
        self.gy.setValue(10)
        self.gz = QDoubleSpinBox()
        self.gz.setValue(5)
        h2.addWidget(QLabel("DX:"))
        h2.addWidget(self.gx)
        h2.addWidget(QLabel("DY:"))
        h2.addWidget(self.gy)
        h2.addWidget(QLabel("DZ:"))
        h2.addWidget(self.gz)
        l.addLayout(h2)
        
        # Auto-detect button
        auto_btn = QPushButton("Auto-Detect from Drillholes")
        auto_btn.setToolTip("Calculate grid origin and size to cover all drillhole data with padding")
        auto_btn.clicked.connect(self._auto_detect_grid)
        l.addWidget(auto_btn)
        
        layout.addWidget(g)

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
        if self.drillhole_data is not None:
            # Only update UI if it's been built
            if hasattr(self, 'primary_combo') and hasattr(self, 'secondary_combo'):
                # Populate variable combos using standardized method
                populate_variable_combo(self.primary_combo, self.drillhole_data)
                populate_variable_combo(self.secondary_combo, self.drillhole_data)
            else:
                logger.debug("Co-Kriging panel: UI not ready, data stored for later initialization")

    def _on_vario_loaded(self, res):
        self.variogram_results = res
        # Enable variogram loading buttons if they exist
        if hasattr(self, 'auto_vario_btn'):
            self.auto_vario_btn.setEnabled(res is not None)
        if hasattr(self, 'use_assisted_btn'):
            self.use_assisted_btn.setEnabled(res is not None)

    def load_variogram_parameters(self) -> bool:
        """Populate co-kriging variogram fields from variogram results - loads per variable."""
        if not hasattr(self, 'registry') or not self.registry:
            QMessageBox.warning(self, "No Registry", "Data registry not available.")
            return False

        # Only update UI if it's been built
        required_attrs = ['model_primary', 'range_primary', 'sill_primary', 'nug_primary',
                         'model_secondary', 'range_secondary', 'sill_secondary', 'nug_secondary',
                         'primary_combo', 'secondary_combo']
        if not all(hasattr(self, attr) for attr in required_attrs):
            logger.debug("Co-Kriging panel: UI not ready for variogram parameter loading")
            return False

        try:
            # Get primary and secondary variable names
            primary_var = self.primary_combo.currentText()
            secondary_var = self.secondary_combo.currentText()
            
            if not primary_var or not secondary_var:
                QMessageBox.warning(self, "No Variables", "Please select primary and secondary variables first.")
                return False

            # Load variogram for primary variable
            primary_vario = self.registry.get_variogram_results(variable_name=primary_var)
            if not primary_vario:
                QMessageBox.warning(self, "No Primary Variogram", 
                                  f"No variogram found for primary variable '{primary_var}'. "
                                  f"Please run Variogram 3D Analysis for '{primary_var}' first.")
                return False

            # Load variogram for secondary variable
            secondary_vario = self.registry.get_variogram_results(variable_name=secondary_var)
            if not secondary_vario:
                QMessageBox.warning(self, "No Secondary Variogram", 
                                  f"No variogram found for secondary variable '{secondary_var}'. "
                                  f"Please run Variogram 3D Analysis for '{secondary_var}' first.")
                return False

            # Extract models from variogram results
            def extract_model(vario_results):
                """Extract model from variogram results."""
                model = vario_results.get('combined_3d_model')
                if not model:
                    # Fallback to directional fits
                    fits = vario_results.get('fitted_models', {})
                    omni_dict = fits.get('omni', {})
                    if omni_dict:
                        # Get first model from omni dict
                        model = next(iter(omni_dict.values())) if omni_dict else None
                return model

            primary_model = extract_model(primary_vario)
            secondary_model = extract_model(secondary_vario)

            if not primary_model:
                QMessageBox.warning(self, "No Primary Model", 
                                  f"Could not extract variogram model for '{primary_var}'.")
                return False

            if not secondary_model:
                QMessageBox.warning(self, "No Secondary Model", 
                                  f"Could not extract variogram model for '{secondary_var}'.")
                return False

            # Validate model types
            primary_type = primary_model.get('model_type', '').lower()
            secondary_type = secondary_model.get('model_type', '').lower()
            
            if primary_type not in ['spherical', 'exponential', 'gaussian']:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Primary variogram model type '{primary_type}' not supported.")
                return False

            if secondary_type not in ['spherical', 'exponential', 'gaussian']:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Secondary variogram model type '{secondary_type}' not supported.")
                return False

            # Extract primary variogram parameters
            prim_nugget = primary_model.get('nugget', 0.0)
            prim_total_sill = primary_model.get('sill', 0.0)
            prim_partial_sill = prim_total_sill - prim_nugget
            prim_range = primary_model.get('major_range') or primary_model.get('range', 100.0)

            # Extract secondary variogram parameters
            sec_nugget = secondary_model.get('nugget', 0.0)
            sec_total_sill = secondary_model.get('sill', 0.0)
            sec_partial_sill = sec_total_sill - sec_nugget
            sec_range = secondary_model.get('major_range') or secondary_model.get('range', 100.0)

            # Set primary variogram
            self.model_primary.setCurrentText(primary_type)
            self.range_primary.setValue(prim_range)
            self.sill_primary.setValue(max(0.001, prim_partial_sill))
            self.nug_primary.setValue(prim_nugget)

            # Set secondary variogram
            self.model_secondary.setCurrentText(secondary_type)
            self.range_secondary.setValue(sec_range)
            self.sill_secondary.setValue(max(0.001, sec_partial_sill))
            self.nug_secondary.setValue(sec_nugget)

            # Set cross-variogram sill to geometric mean
            cross_sill = np.sqrt(prim_partial_sill * sec_partial_sill)
            self.sill_cross.setValue(cross_sill)
            # Use primary range for cross-variogram (can be adjusted manually)
            self.range_cross.setValue(prim_range)

            logger.info(f"Co-Kriging: Loaded variogram parameters - "
                       f"Primary ({primary_var}): {primary_type}, range={prim_range:.1f}, sill={prim_partial_sill:.3f}, nugget={prim_nugget:.3f}; "
                       f"Secondary ({secondary_var}): {secondary_type}, range={sec_range:.1f}, sill={sec_partial_sill:.3f}, nugget={sec_nugget:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error loading co-kriging variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load variogram parameters:\n{str(e)}")
            return False

    def _load_assisted_variogram(self):
        """Load variogram parameters from Variogram Assistant."""
        # Check if UI is ready
        required_attrs = ['model_primary', 'range_primary', 'sill_primary', 'nug_primary',
                         'model_secondary', 'range_secondary', 'sill_secondary', 'nug_secondary']
        if not all(hasattr(self, attr) for attr in required_attrs):
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

            # Use the first available assisted model
            assisted_model = assisted_models[0]
            model_type = assisted_model.get('model_type', '').lower()

            if model_type not in ['spherical', 'exponential']:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Assisted model type '{model_type}' not supported for co-kriging.")
                return

            # Apply parameters
            nugget = assisted_model.get('nugget', 0.0)
            sill = assisted_model.get('sill', 1.0)
            range_val = assisted_model.get('range', 100.0)

            # Set both variograms
            self.model_primary.setCurrentText(model_type)
            self.range_primary.setValue(range_val)
            self.sill_primary.setValue(max(0.001, sill))
            self.nug_primary.setValue(nugget)

            self.model_secondary.setCurrentText(model_type)
            self.range_secondary.setValue(range_val)
            self.sill_secondary.setValue(max(0.001, sill))
            self.nug_secondary.setValue(nugget)

            self.sill_cross.setValue(sill * 0.5)  # Half the auto-correlation

            logger.info(f"Co-Kriging: Loaded assisted variogram - {model_type}, "
                       f"range={range_val:.1f}, sill={sill:.3f}, nugget={nugget:.3f}")

        except Exception as e:
            logger.error(f"Error loading assisted co-kriging variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load assisted variogram:\n{str(e)}")

    def gather_parameters(self) -> Dict[str, Any]:
        # Calculate grid counts from origin, spacing, and data bounds
        x0 = self.xmin_spin.value() if hasattr(self, 'xmin_spin') else 0.0
        y0 = self.ymin_spin.value() if hasattr(self, 'ymin_spin') else 0.0
        z0 = self.zmin_spin.value() if hasattr(self, 'zmin_spin') else 0.0
        dx = self.gx.value()
        dy = self.gy.value()
        dz = self.gz.value()
        
        # Validate spacing
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("Grid spacing must be positive")
        
        # Maximum grid dimensions to prevent memory issues (10M points max)
        MAX_GRID_POINTS = 10_000_000
        MAX_DIM = 1000  # Maximum cells per dimension
        
        # Estimate grid counts from data bounds if available
        if self.drillhole_data is not None and not self.drillhole_data.empty:
            df = self.drillhole_data
            if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
                x_min, x_max = df['X'].min(), df['X'].max()
                y_min, y_max = df['Y'].min(), df['Y'].max()
                z_min, z_max = df['Z'].min(), df['Z'].max()
                
                # Add 10% padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min
                x_max += x_range * 0.1
                y_max += y_range * 0.1
                z_max += z_range * 0.1
                x_min -= x_range * 0.1
                y_min -= y_range * 0.1
                z_min -= z_range * 0.1
                
                # Calculate grid counts
                nx = int(np.ceil((x_max - x0) / dx)) if x0 < x_max else int(np.ceil((x_max - x_min) / dx))
                ny = int(np.ceil((y_max - y0) / dy)) if y0 < y_max else int(np.ceil((y_max - y_min) / dy))
                nz = int(np.ceil((z_max - z0) / dz)) if z0 < z_max else int(np.ceil((z_max - z_min) / dz))
                
                # Apply limits
                nx = min(max(1, nx), MAX_DIM)
                ny = min(max(1, ny), MAX_DIM)
                nz = min(max(1, nz), MAX_DIM)
                
                # Check total points
                total_points = nx * ny * nz
                if total_points > MAX_GRID_POINTS:
                    # Scale down proportionally
                    scale = (MAX_GRID_POINTS / total_points) ** (1/3)
                    nx = max(1, int(nx * scale))
                    ny = max(1, int(ny * scale))
                    nz = max(1, int(nz * scale))
                    logger.warning(f"Grid dimensions reduced to {nx}x{ny}x{nz} to prevent memory issues")
            else:
                nx, ny, nz = 50, 50, 20  # Defaults
        else:
            nx, ny, nz = 50, 50, 20  # Defaults
        
        # Final validation
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"Invalid grid dimensions: {nx}x{ny}x{nz}")
        
        total_points = nx * ny * nz
        if total_points > MAX_GRID_POINTS:
            raise ValueError(f"Grid too large: {nx}x{ny}x{nz} = {total_points:,} points (max {MAX_GRID_POINTS:,})")
        
        # Get minimum neighbors (professional audit setting)
        min_neighbors = self.min_neigh_spin.value() if hasattr(self, 'min_neigh_spin') else 3
        
        # Get SK interpolation preference
        use_sk_for_secondary = self.use_sk_secondary.isChecked() if hasattr(self, 'use_sk_secondary') else True
        
        return {
            'data_df': self.drillhole_data,
            'primary_name': self.primary_combo.currentText(),
            'secondary_name': self.secondary_combo.currentText(),
            'method': self.method_combo.currentText(),
            'variogram_primary': {
                'range': self.range_primary.value(),
                'sill': self.sill_primary.value(),
                'nugget': self.nug_primary.value(),
                'model_type': self.model_primary.currentText()
            },
            'variogram_secondary': {
                'range': self.range_secondary.value(),
                'sill': self.sill_secondary.value(),
                'nugget': self.nug_secondary.value(),
                'model_type': self.model_secondary.currentText()
            },
            'cross_variogram': {
                'range': self.range_cross.value(),
                'sill': self.sill_cross.value(),
                'nugget': 0.0,
                'model_type': self.model_primary.currentText()
            } if self.use_cross.isChecked() else None,
            'grid_origin': (x0, y0, z0),
            'grid_spacing': (dx, dy, dz),
            'grid_counts': (nx, ny, nz),
            'n_neighbors': self.neigh_spin.value(),
            'max_distance': self.max_dist_spin.value(),
            # Professional settings
            'min_neighbors': min_neighbors,
            'min_correlation': 0.3,  # Audit threshold
            'fallback_to_ok': True,
            'use_sk_for_secondary': use_sk_for_secondary
        }

    def validate_inputs(self) -> bool:
        if self.drillhole_data is None:
            QMessageBox.warning(self, "Error", "No Data")
            return False
        if self.primary_combo.currentText() == self.secondary_combo.currentText():
            QMessageBox.warning(self, "Error", "Primary and Secondary variables must differ")
            return False
        return True

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Co-Kriging.

        Co-Kriging requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Validated data quality

        Returns:
            True if data is acceptable for Co-Kriging
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
                f"Cannot run Co-Kriging:\n\n{message}\n\n"
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

    def _log_event(self, message: str, level: str = "info"):
        if not hasattr(self, 'log_text') or not self.log_text: return
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"info": f"{ModernColors.TEXT_PRIMARY}", "success": "#81c784", "warning": "#ffb74d", "error": "#e57373", "progress": "#4fc3f7"}
        self.log_text.append(f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {colors.get(level, f"{ModernColors.TEXT_PRIMARY}")};">{message}</span>')
    
    def _update_progress(self, percent: int, message: str = ""):
        """Update progress bar and label with percentage and message."""
        percent = max(0, min(100, int(percent)))
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setValue(percent)
            if message:
                self.progress_bar.setFormat(f"{percent}% - {message}")
            else:
                self.progress_bar.setFormat(f"{percent}%")
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setText(message if message else f"{percent}% complete")
        
        # Log milestones and important progress updates
        should_log = (
            percent % 10 == 0 or 
            percent in [0, 100] or 
            "complete" in message.lower() or
            "done" in message.lower() or
            "error" in message.lower()
        )
        
        if should_log and message:
            self._log_event(f"Progress: {percent}% - {message}", "progress")
    
    def show_progress(self, message: str) -> None:
        self._update_progress(0, message)
        self._log_event(f"Starting: {message}", "progress")
        if hasattr(self, 'run_btn'): self.run_btn.setEnabled(False)
    
    def hide_progress(self) -> None:
        if hasattr(self, 'run_btn'): self.run_btn.setEnabled(True)
    
    def run_analysis(self) -> None:
        """Override to pass progress callback for real-time updates."""
        if not self.controller:
            self.show_warning("Unavailable", "Controller is not connected; cannot run analysis.")
            return

        if not self.validate_inputs():
            return

        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return

        params = self.gather_parameters()
        self.show_progress("Starting Co-Kriging...")
        
        # Log correlation gate check (professional audit requirement)
        try:
            primary_var = params['primary_name']
            secondary_var = params['secondary_name']
            if self.drillhole_data is not None and not self.drillhole_data.empty:
                prim_vals = self.drillhole_data[primary_var].values
                sec_vals = self.drillhole_data[secondary_var].values
                valid_mask = np.isfinite(prim_vals) & np.isfinite(sec_vals)
                if np.sum(valid_mask) >= 10:
                    try:
                        from scipy.stats import pearsonr
                        r, _ = pearsonr(prim_vals[valid_mask], sec_vals[valid_mask])
                    except ImportError:
                        r = np.corrcoef(prim_vals[valid_mask], sec_vals[valid_mask])[0, 1]
                    
                    min_corr = params.get('min_correlation', 0.3)
                    if abs(r) >= min_corr:
                        self._log_event(f"Correlation r = {r:.3f} ({'strong' if abs(r) >= 0.7 else 'moderate' if abs(r) >= 0.5 else 'weak'}). Co-kriging enabled.", "success")
                    else:
                        self._log_event(f"Correlation r = {r:.3f} < {min_corr:.2f}. Falling back to Ordinary Kriging.", "warning")
        except Exception as e:
            logger.debug(f"Could not compute correlation for logging: {e}")

        # ✅ FIX: Create progress callback that updates the progress bar
        self._last_progress_percent = 0
        self._last_progress_message = ""
        
        def progress_callback(percent: int, message: str):
            """Update progress bar from worker thread."""
            # Store latest values - these will be picked up by the timer
            self._last_progress_percent = percent
            self._last_progress_message = message
            # Use QTimer.singleShot to update UI from main thread
            QTimer.singleShot(0, self._apply_progress_update)
        
        def _apply_progress_update(self):
            """Apply stored progress update to UI (called from main thread)."""
            self._update_progress(self._last_progress_percent, self._last_progress_message)
            # NOTE: processEvents() removed to prevent Qt painter reentrancy issues.
            # Progress updates via QTimer.singleShot are sufficient for UI responsiveness.
        
        self._apply_progress_update = _apply_progress_update

        try:
            self.controller.run_cokriging(
                config=params,
                callback=self.handle_results,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Failed to dispatch Co-Kriging: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def handle_results(self, payload):
        """Handle results from controller with professional diagnostics."""
        self._update_progress(100, "Complete!")
        self._log_event("✓ CO-KRIGING COMPLETE", "success")
        
        # Check for fallback to OK warning
        if payload.get('fallback_to_ok'):
            warning_msg = payload.get('warning', 'Correlation too weak for co-kriging')
            correlation_data = payload.get('correlation_analysis', {})
            
            self._log_event(f"⚠ Co-Kriging fallback: {warning_msg}", "warning")
            
            # Build detailed warning message
            pearson_r = correlation_data.get('pearson_r', 'N/A')
            pearson_r_str = f"{pearson_r:.3f}" if isinstance(pearson_r, float) else str(pearson_r)

            detail_msg = (f"Co-Kriging Analysis Result:\n\n"
                         f"Correlation Analysis:\n"
                         f"  Pearson r: {pearson_r_str}\n"
                         f"  Paired samples: {correlation_data.get('n_paired', 'N/A')}\n"
                         f"  Strength: {correlation_data.get('strength', 'N/A')}\n\n"
                         f"Recommendation:\n{correlation_data.get('recommendation', warning_msg)}\n\n"
                         f"Consider using Ordinary Kriging instead.")
            
            QMessageBox.warning(self, "Co-Kriging - Weak Correlation", detail_msg)
            self.results_text.setText(detail_msg)
            self.hide_progress()
            return
        
        if "error" in payload and payload['error']:
            self._log_event(f"Co-Kriging Failed: {payload['error']}", "error")
            QMessageBox.critical(self, "Error", f"Co-Kriging failed:\n{payload['error']}")
            self.hide_progress()
            return
        
        # Extract results from payload
        estimates = payload.get('estimates')
        variances = payload.get('variances')
        grid_x = payload.get('grid_x')
        grid_y = payload.get('grid_y')
        grid_z = payload.get('grid_z')
        metadata = payload.get('metadata', {})
        
        if estimates is None:
            QMessageBox.critical(self, "Error", "Co-Kriging failed to return estimates.")
            self.hide_progress()
            return
        
        # Store results including diagnostics
        self.kriging_results = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'estimates': estimates,
            'variances': variances if variances is not None else np.full_like(estimates, np.nan),
            'variable': metadata.get('primary_name', 'Primary'),
            'model_type': 'cokriging',
            'metadata': metadata,
            # Diagnostic data
            'secondary_influence': payload.get('secondary_influence'),
            'neighbor_counts': payload.get('neighbor_counts'),
        }

        # Register results and block model to DataRegistry
        if self.registry:
            try:
                # Register co-kriging results
                self.registry.register_cokriging_results(self.kriging_results, source_panel="Co-Kriging")
                self._log_event("Results registered to data registry", "info")

                # Also register the block model DataFrame for cross-sections and other panels
                coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                block_df = pd.DataFrame({
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                    f'{self.kriging_results["variable"]}_ck_est': estimates.ravel(),
                    f'{self.kriging_results["variable"]}_ck_var': variances.ravel() if variances is not None else np.full_like(estimates.ravel(), np.nan)
                }).dropna()

                # Register the block model
                self.registry.register_block_model_generated(
                    block_df,
                    source_panel="Co-Kriging",
                    metadata={
                        'primary_variable': self.kriging_results['variable'],
                        'secondary_variable': metadata.get('secondary_name'),
                        'method': 'cokriging',
                        'grid_size': (len(np.unique(grid_x)), len(np.unique(grid_y)), len(np.unique(grid_z))),
                        'n_blocks': len(block_df)
                    }
                )
                self._log_event("Block model registered to data registry", "info")
            except Exception as e:
                logger.warning(f"Failed to register co-kriging results/block model: {e}")

        # Enable visualization and table buttons
        self.visualize_btn.setEnabled(True)
        self.view_table_btn.setEnabled(True)
        self.diagnostics_btn.setEnabled(True)
        
        # Build comprehensive results summary
        est_flat = estimates.flatten()
        n_valid = np.sum(~np.isnan(est_flat))
        
        # Get correlation info
        correlation = metadata.get('correlation', {})
        corr_r_raw = correlation.get('pearson_r_raw')
        corr_r_eff = correlation.get('pearson_r_effective') or correlation.get('pearson_r')
        corr_str = f"r = {corr_r_eff:.3f}" if isinstance(corr_r_eff, float) else "N/A"
        if corr_r_raw and abs(corr_r_raw - corr_r_eff) > 0.01:
            corr_str += f" (clamped from {corr_r_raw:.3f})"
        corr_strength = correlation.get('strength', 'N/A')
        
        # Get enhanced secondary influence diagnostics
        sec_inf = metadata.get('secondary_influence', {})
        sec_inf_mean = sec_inf.get('mean', 0.0)
        sec_inf_p95 = sec_inf.get('p95', 0.0)
        sec_inf_p99 = sec_inf.get('p99', 0.0)
        sec_unused = sec_inf.get('unused_percentage', 0.0)
        
        # Get OK fallback statistics
        ok_fallback = metadata.get('ok_fallback', {})
        n_ok_fallback = ok_fallback.get('n_nodes', 0)
        pct_ok_fallback = ok_fallback.get('percentage', 0.0)
        
        # Get neighbor statistics
        nbr_stats = metadata.get('neighbor_statistics', {})
        n_low_nbr = nbr_stats.get('n_low_neighbors', 0)
        pct_low_nbr = nbr_stats.get('pct_low_neighbors', 0.0)
        mean_nbr = nbr_stats.get('mean_neighbors', 0.0)
        
        # Get Markov-1 cross-sill
        mm1_sill = metadata.get('markov1_cross_sill')
        mm1_str = f"{mm1_sill:.4f}" if mm1_sill else "N/A"
        
        summary = (
            f"=== Co-Kriging Results ===\n\n"
            f"Variables:\n"
            f"  Primary: {metadata.get('primary_name', 'N/A')}\n"
            f"  Secondary: {metadata.get('secondary_name', 'N/A')}\n\n"
            f"Correlation Analysis:\n"
            f"  Pearson r: {corr_str} ({corr_strength})\n"
            f"  Recommendation: {correlation.get('recommendation', 'N/A')[:60]}...\n\n"
            f"Grid Statistics:\n"
            f"  Size: {estimates.shape}\n"
            f"  Valid: {n_valid:,} / {est_flat.size:,}\n\n"
            f"Estimate Statistics:\n"
            f"  Min: {np.nanmin(est_flat):.3f}\n"
            f"  Max: {np.nanmax(est_flat):.3f}\n"
            f"  Mean: {np.nanmean(est_flat):.3f}\n"
            f"  Std: {np.nanstd(est_flat):.3f}\n\n"
            f"Co-Kriging Diagnostics:\n"
            f"  Markov-1 Cross-Sill: {mm1_str}\n"
            f"  Secondary Influence: mean={sec_inf_mean:.1%}, P95={sec_inf_p95:.1%}, P99={sec_inf_p99:.1%}\n"
            f"  Secondary Unused: {sec_unused:.1f}% of nodes\n"
            f"  OK Fallback: {n_ok_fallback:,} nodes ({pct_ok_fallback:.1f}%)\n"
            f"  Low Neighbors: {n_low_nbr:,} nodes ({pct_low_nbr:.1f}%)\n"
            f"  Mean Neighbors: {mean_nbr:.1f}\n"
            f"  SK Interpolation: {'Yes' if metadata.get('used_sk_interpolation') else 'No'}\n"
            f"  Anisotropy: {'Yes' if metadata.get('used_anisotropy') else 'No'}\n"
        )
        
        self.results_text.setText(summary)
        
        # Log comprehensive audit information
        self._log_event(f"Correlation: {corr_str} ({corr_strength})", "info")
        self._log_event(f"Secondary influence: mean={sec_inf_mean:.1%}, unused={sec_unused:.1f}%", "info")
        if n_ok_fallback > 0:
            self._log_event(f"OK fallback: {n_ok_fallback:,} nodes ({pct_ok_fallback:.1f}%)", "info")
        if n_low_nbr > 0:
            self._log_event(f"Low neighbors: {n_low_nbr:,} nodes ({pct_low_nbr:.1f}%)", "warning")
        
        self.hide_progress()
    
    def visualize_results(self):
        """Visualize Co-Kriging results in 3D."""
        if self.kriging_results is None:
            self._log_event("No results to visualize", "warning")
            QMessageBox.warning(self, "No Results", "Please run Co-Kriging first.")
            return
        
        self._log_event("Sending to 3D viewer...", "progress")
        if self.receivers(self.request_visualization) > 0:
            try:
                self.request_visualization.emit(self.kriging_results)
                self._log_event("✓ Sent to 3D viewer", "success")
                return
            except Exception as e:
                logger.error(f"Signal error: {e}")
        
        if self.parent() and hasattr(self.parent(), 'visualize_kriging_results'):
            try:
                self.parent().visualize_kriging_results(self.kriging_results)
                self._log_event("✓ Sent to 3D viewer", "success")
            except Exception as e:
                logger.error(f"Visualization error: {e}", exc_info=True)
                self._log_event(f"ERROR: {e}", "error")
                QMessageBox.critical(self, "Visualization Error", f"Error visualizing results:\n{str(e)}")
        else:
            self._log_event("Cannot access main viewer", "warning")
            QMessageBox.information(self, "Info", "Cannot access main viewer for visualization.")

    def open_results_table(self):
        """Open Co-Kriging results as a table."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run Co-Kriging first.")
            return

        try:
            grid_x = self.kriging_results.get('grid_x')
            grid_y = self.kriging_results.get('grid_y')
            grid_z = self.kriging_results.get('grid_z')
            estimates = self.kriging_results.get('estimates')
            variances = self.kriging_results.get('variances')
            variable = self.kriging_results.get('variable', 'Co-Kriging')

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

            title = f"Co-Kriging Results - {variable}"

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

    def _check_correlation(self):
        """
        Check and display correlation between selected primary and secondary variables.
        Professional audit feature - warns if correlation is too weak for co-kriging.
        """
        if not hasattr(self, 'correlation_label'):
            return
        
        if self.drillhole_data is None or self.drillhole_data.empty:
            self.correlation_label.setText("Load data to check correlation")
            self.correlation_label.setStyleSheet("font-size: 9px; color: #888; padding: 4px;")
            return
        
        primary_var = self.primary_combo.currentText()
        secondary_var = self.secondary_combo.currentText()
        
        if not primary_var or not secondary_var:
            self.correlation_label.setText("Select variables to check correlation")
            self.correlation_label.setStyleSheet("font-size: 9px; color: #888; padding: 4px;")
            return
        
        if primary_var == secondary_var:
            self.correlation_label.setText("⚠ Primary and secondary must differ")
            self.correlation_label.setStyleSheet("font-size: 9px; color: #e57373; padding: 4px;")
            return
        
        if primary_var not in self.drillhole_data.columns or secondary_var not in self.drillhole_data.columns:
            self.correlation_label.setText("Variables not found in data")
            self.correlation_label.setStyleSheet("font-size: 9px; color: #888; padding: 4px;")
            return
        
        try:
            prim_vals = self.drillhole_data[primary_var].values
            sec_vals = self.drillhole_data[secondary_var].values
            
            # Filter paired valid data
            valid_mask = np.isfinite(prim_vals) & np.isfinite(sec_vals)
            n_paired = np.sum(valid_mask)
            
            if n_paired < 10:
                self.correlation_label.setText(f"⚠ Insufficient paired samples ({n_paired})")
                self.correlation_label.setStyleSheet("font-size: 9px; color: #e57373; padding: 4px;")
                return
            
            # Compute Pearson correlation
            prim_valid = prim_vals[valid_mask]
            sec_valid = sec_vals[valid_mask]
            
            try:
                from scipy.stats import pearsonr
                r, p = pearsonr(prim_valid, sec_valid)
            except ImportError:
                r = np.corrcoef(prim_valid, sec_valid)[0, 1]
                p = np.nan
            
            # Determine correlation strength and display
            abs_r = abs(r)
            if abs_r >= 0.7:
                strength = "Strong"
                color = "#4CAF50"  # Green
                icon = "✓"
            elif abs_r >= 0.5:
                strength = "Moderate"
                color = "#81c784"  # Light green
                icon = "✓"
            elif abs_r >= 0.3:
                strength = "Weak"
                color = "#FF9800"  # Orange
                icon = "⚠"
            else:
                strength = "Very weak"
                color = "#e57373"  # Red
                icon = "⚠"
            
            # Build display text
            if abs_r < 0.3:
                msg = f"{icon} r = {r:.3f} ({strength}, n={n_paired})\nCo-Kriging may not improve over OK"
            else:
                msg = f"{icon} r = {r:.3f} ({strength}, n={n_paired})"
            
            self.correlation_label.setText(msg)
            self.correlation_label.setStyleSheet(f"font-size: 9px; color: {color}; padding: 4px;")
            
            # Log for audit trail
            logger.info(f"Co-Kriging correlation check: {primary_var} vs {secondary_var}, r={r:.3f}, n={n_paired}")
            
        except Exception as e:
            self.correlation_label.setText(f"Error computing correlation: {str(e)[:30]}")
            self.correlation_label.setStyleSheet("font-size: 9px; color: #e57373; padding: 4px;")
            logger.warning(f"Correlation check failed: {e}")

    def _auto_detect_grid(self):
        """
        Auto-detect grid parameters from drillhole coordinates.
        Robust grid detection from DataFrame bounds (no renderer hunting).
        """
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "No drillhole data loaded for grid detection.")
            return
        
        df = self.drillhole_data
        if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
            QMessageBox.warning(self, "Missing Coordinates", "Drillhole data must have X, Y, Z columns for grid detection.")
            return
        
        valid_mask = df['X'].notna() & df['Y'].notna() & df['Z'].notna()
        valid_df = df[valid_mask]
        
        if valid_df.empty:
            QMessageBox.warning(self, "No Valid Data", "No valid coordinate data found in drillholes.")
            return
        
        x_min = float(valid_df['X'].min())
        x_max = float(valid_df['X'].max())
        y_min = float(valid_df['Y'].min())
        y_max = float(valid_df['Y'].max())
        z_min = float(valid_df['Z'].min())
        z_max = float(valid_df['Z'].max())
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Ensure non-zero ranges
        if x_range < 1e-6:
            x_range = 100.0
        if y_range < 1e-6:
            y_range = 100.0
        if z_range < 1e-6:
            z_range = 50.0
        
        # Use current block size values as defaults, or sensible mining defaults
        dx = self.gx.value() if self.gx.value() > 0.1 else 10.0
        dy = self.gy.value() if self.gy.value() > 0.1 else 10.0
        dz = self.gz.value() if self.gz.value() > 0.1 else 5.0
        
        # Add padding: 5% of range
        x_pad = x_range * 0.05
        y_pad = y_range * 0.05
        z_pad = z_range * 0.05
        
        # Calculate grid origin (snap to block size for cleaner coordinates)
        xmin = np.floor((x_min - x_pad) / dx) * dx
        ymin = np.floor((y_min - y_pad) / dy) * dy
        zmin = np.floor((z_min - z_pad) / dz) * dz
        
        # Calculate grid counts
        xmax = np.ceil((x_max + x_pad) / dx) * dx
        ymax = np.ceil((y_max + y_pad) / dy) * dy
        zmax = np.ceil((z_max + z_pad) / dz) * dz
        
        nx = int(np.ceil((xmax - xmin) / dx))
        ny = int(np.ceil((ymax - ymin) / dy))
        nz = int(np.ceil((zmax - zmin) / dz))
        
        # Set the UI values
        if hasattr(self, 'xmin_spin'):
            self.xmin_spin.setValue(xmin)
            self.ymin_spin.setValue(ymin)
            self.zmin_spin.setValue(zmin)
        
        logger.info(
            f"Auto-detected grid: origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f}), "
            f"spacing=({dx:.1f}, {dy:.1f}, {dz:.1f})m, counts=({nx}, {ny}, {nz})"
        )
        
        if hasattr(self, 'log_text'):
            self._log_event(
                f"Grid Auto-Detection: origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f}), "
                f"spacing=({dx:.1f}, {dy:.1f}, {dz:.1f})m, grid=({nx}x{ny}x{nz})",
                "success"
            )

    def open_scaling_diagnostics(self):
        """Open comprehensive scaling diagnostics to identify co-kriging issues."""
        if not self.kriging_results or not self.drillhole_data:
            QMessageBox.warning(self, "Scaling Diagnostics",
                              "No co-kriging results or data available for diagnostics.")
            return

        try:
            self._log_event("Running co-kriging scaling diagnostics...", "info")

            # Get variable names
            primary_var = self.primary_var_combo.currentText()
            secondary_var = self.secondary_var_combo.currentText()

            if not primary_var or not secondary_var:
                QMessageBox.warning(self, "Scaling Diagnostics", "Variables not selected.")
                return

            # Extract data
            primary_data = self.drillhole_data[primary_var].values
            secondary_data = self.drillhole_data[secondary_var].values

            # Run diagnostics
            from ..geostats.cokriging3d import _analyze_variable_scaling, _validate_cokriging_scaling

            scaling_analysis = _analyze_variable_scaling(primary_data, secondary_data)

            # Get correlation and sills from results metadata
            metadata = self.kriging_results.get('metadata', {})
            correlation_analysis = metadata.get('correlation_analysis', {})
            correlation = correlation_analysis.get('pearson_r', 0)

            # Get sills from variogram models
            variogram_models = metadata.get('variogram_models', {})
            vp = variogram_models.get('primary', {})
            vs = variogram_models.get('secondary', {})

            sill_p = vp.get('sill', 1.0) + vp.get('nugget', 0.0)
            sill_s = vs.get('sill', 1.0) + vs.get('nugget', 0.0)

            scaling_validation = _validate_cokriging_scaling(
                primary_data, secondary_data, correlation, sill_p, sill_s
            )

            # Display results
            self._show_scaling_diagnostics(scaling_analysis, scaling_validation, metadata)

            self._log_event("Co-kriging scaling diagnostics completed", "success")

        except Exception as e:
            logger.error(f"Scaling diagnostics failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Scaling Diagnostics", f"Diagnostics failed: {str(e)}")

    def _show_scaling_diagnostics(self, scaling_analysis, scaling_validation, metadata):
        """Display scaling diagnostics in a comprehensive dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QGroupBox, QFormLayout, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle("Co-Kriging Scaling Diagnostics")
        dialog.resize(700, 600)

        layout = QVBoxLayout(dialog)

        # Scaling compatibility analysis
        compat_group = QGroupBox("Variable Scaling Compatibility")
        compat_layout = QFormLayout(compat_group)

        scales_compat = scaling_analysis.get('scales_compatible', True)
        status = "✓ Compatible" if scales_compat else "✗ Incompatible"
        compat_layout.addRow("Status:", QLabel(status))

        warning = scaling_analysis.get('warning')
        if warning:
            compat_layout.addRow("Warning:", QLabel(warning))

        scaling_factor = scaling_analysis.get('scaling_factor')
        if scaling_factor:
            compat_layout.addRow("Applied Scaling Factor:", QLabel(f"{scaling_factor:.3f}"))

        # Statistics
        stats = scaling_analysis.get('statistics', {})
        if stats:
            compat_layout.addRow("Primary Mean:", QLabel(f"{stats.get('primary_mean', 0):.3f}"))
            compat_layout.addRow("Secondary Mean:", QLabel(f"{stats.get('secondary_mean', 0):.3f}"))
            compat_layout.addRow("Mean Ratio:", QLabel(f"{stats.get('mean_ratio', 0):.2f}"))

        layout.addWidget(compat_group)

        # Validation results
        validation_group = QGroupBox("Co-Kriging Validation")
        validation_layout = QFormLayout(validation_group)

        issue_detected = scaling_validation.get('issue_detected', False)
        status = "⚠ ISSUE DETECTED" if issue_detected else "✓ No Issues"
        validation_layout.addRow("Validation Status:", QLabel(status))

        message = scaling_validation.get('message', 'No issues found')
        validation_layout.addRow("Message:", QLabel(message))

        layout.addWidget(validation_group)

        # Correlation info
        correlation_group = QGroupBox("Correlation Analysis")
        correlation_layout = QFormLayout(correlation_group)

        corr_analysis = metadata.get('correlation_analysis', {})
        correlation_layout.addRow("Correlation (r):", QLabel(f"{corr_analysis.get('pearson_correlation', 0):.3f}"))
        correlation_layout.addRow("P-value:", QLabel(f"{corr_analysis.get('pearson_pvalue', 0):.3f}" if corr_analysis.get('pearson_pvalue') else "N/A"))
        correlation_layout.addRow("Sample Pairs:", QLabel(str(corr_analysis.get('n_paired', 0))))

        layout.addWidget(correlation_group)

        # Recommendations
        rec_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        recommendations = []

        if issue_detected:
            recommendations.append("• Critical: Fix scaling/unit issues before using results")
            recommendations.append("• Check if variables are in compatible units")
            recommendations.append("• Consider transforming variables to same scale")

        if scaling_analysis.get('scaling_factor'):
            recommendations.append("• Automatic scaling correction was applied")
            recommendations.append("• Verify this correction is geologically reasonable")

        if not recommendations:
            recommendations.append("• Variable scaling appears compatible")
            recommendations.append("• Co-kriging results should be reliable")

        for rec in recommendations:
            rec_layout.addWidget(QLabel(rec))

        layout.addWidget(rec_group)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_btn = QPushButton("Export Diagnostics")
        export_btn.clicked.connect(lambda: self._export_scaling_diagnostics(scaling_analysis, scaling_validation, metadata))
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _export_scaling_diagnostics(self, scaling_analysis, scaling_validation, metadata):
        """Export scaling diagnostics to file."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Scaling Diagnostics", "", "JSON files (*.json);;All files (*)"
            )

            if filename:
                diagnostics = {
                    'scaling_analysis': scaling_analysis,
                    'scaling_validation': scaling_validation,
                    'metadata': metadata,
                    'timestamp': pd.Timestamp.now().isoformat()
                }

                import json
                with open(filename, 'w') as f:
                    json.dump(diagnostics, f, indent=2, default=str)

                QMessageBox.information(self, "Export Complete",
                                      f"Scaling diagnostics saved to {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export diagnostics: {str(e)}")

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
            
            # Variables
            settings['primary_variable'] = get_safe_widget_value(self, 'primary_combo')
            settings['secondary_variable'] = get_safe_widget_value(self, 'secondary_combo')
            settings['cok_method'] = get_safe_widget_value(self, 'method_combo')
            
            # Variogram models
            settings['primary_model'] = get_safe_widget_value(self, 'primary_model_combo')
            settings['primary_nugget'] = get_safe_widget_value(self, 'primary_nugget_spin')
            settings['primary_sill'] = get_safe_widget_value(self, 'primary_sill_spin')
            settings['primary_range'] = get_safe_widget_value(self, 'primary_range_spin')
            
            settings['secondary_model'] = get_safe_widget_value(self, 'secondary_model_combo')
            settings['secondary_nugget'] = get_safe_widget_value(self, 'secondary_nugget_spin')
            settings['secondary_sill'] = get_safe_widget_value(self, 'secondary_sill_spin')
            settings['secondary_range'] = get_safe_widget_value(self, 'secondary_range_spin')
            
            settings['cross_nugget'] = get_safe_widget_value(self, 'cross_nugget_spin')
            settings['cross_sill'] = get_safe_widget_value(self, 'cross_sill_spin')
            settings['cross_range'] = get_safe_widget_value(self, 'cross_range_spin')
            
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
            logger.warning(f"Could not save cokriging panel settings: {e}")
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
            
            # Variables
            set_safe_widget_value(self, 'primary_combo', settings.get('primary_variable'))
            set_safe_widget_value(self, 'secondary_combo', settings.get('secondary_variable'))
            set_safe_widget_value(self, 'method_combo', settings.get('cok_method'))
            
            # Variogram models
            set_safe_widget_value(self, 'primary_model_combo', settings.get('primary_model'))
            set_safe_widget_value(self, 'primary_nugget_spin', settings.get('primary_nugget'))
            set_safe_widget_value(self, 'primary_sill_spin', settings.get('primary_sill'))
            set_safe_widget_value(self, 'primary_range_spin', settings.get('primary_range'))
            
            set_safe_widget_value(self, 'secondary_model_combo', settings.get('secondary_model'))
            set_safe_widget_value(self, 'secondary_nugget_spin', settings.get('secondary_nugget'))
            set_safe_widget_value(self, 'secondary_sill_spin', settings.get('secondary_sill'))
            set_safe_widget_value(self, 'secondary_range_spin', settings.get('secondary_range'))
            
            set_safe_widget_value(self, 'cross_nugget_spin', settings.get('cross_nugget'))
            set_safe_widget_value(self, 'cross_sill_spin', settings.get('cross_sill'))
            set_safe_widget_value(self, 'cross_range_spin', settings.get('cross_range'))
            
            # Grid
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            set_safe_widget_value(self, 'dx_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'dy_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'dz_spin', settings.get('grid_z'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
                
            logger.info("Restored cokriging panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore cokriging panel settings: {e}")
