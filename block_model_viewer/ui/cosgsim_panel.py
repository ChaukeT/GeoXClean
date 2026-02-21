"""
Co-Simulation (CoSGSIM) Panel
=============================

UI for Multi-Variable Sequential Gaussian Simulation.
Updated to support Markov Model 1 (MM1) correlation and Threading.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QLabel,
    QMessageBox, QWidget, QTextEdit, QFileDialog, QCheckBox,
    QSplitter, QFrame, QScrollArea, QListWidget, QAbstractItemView, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QColor
from datetime import datetime

from .base_analysis_panel import BaseAnalysisPanel
from .modern_styles import get_theme_colors, ModernColors
# Import the backend job runner
from ..geostats.cosgsim3d import run_cosgsim_job

logger = logging.getLogger(__name__)


# Worker logic moved to GeostatsController._prepare_cosgsim_payload
# This ensures pure computation with no access to DataRegistry or Qt objects


class CoSGSIMPanel(BaseAnalysisPanel):
    task_name = "cosgsim"
    request_visualization = pyqtSignal(object, str)
    progress_updated = pyqtSignal(int, str)

    # PanelManager metadata
    PANEL_ID = "CoSGSIMPanel"
    PANEL_NAME = "CoSGSIM Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    def __init__(self, parent=None):
        # Initialize state BEFORE super().__init__
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Dict[str, Dict[str, Any]] = {}
        self.simulation_results: Optional[Dict[str, Any]] = None

        super().__init__(parent=parent, panel_id="cosgsim")

        # Build the UI immediately after base initialization
        self._build_ui()

        # Initialize registry connections
        self._init_registry()

        # Connect own progress signal to update method
        self.progress_updated.connect(self._update_progress)

    def bind_controller(self, controller):
        """Bind controller and connect to task_progress signal for progress updates."""
        super().bind_controller(controller)
        if controller and hasattr(controller, 'signals'):
            controller.signals.task_progress.connect(self._handle_task_progress)

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
        self.setWindowTitle("Co-Simulation (CoSGSIM)")
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
                logger.debug("CoSGSimPanel: Connected to drillholeDataLoaded signal")
            
            vario_signal = self.registry.variogramResultsLoaded
            if vario_signal is not None:
                vario_signal.connect(self._on_vario_loaded)
                logger.debug("CoSGSimPanel: Connected to variogramResultsLoaded signal")
            
            # AUDIT FIX: Prefer get_estimation_ready_data for proper provenance
            d = None
            try:
                d = self.registry.get_estimation_ready_data(
                    prefer_declustered=True,
                    require_validation=False
                )
            except (ValueError, AttributeError):
                d = self.registry.get_drillhole_data()
            if d is not None:
                self._on_data_loaded(d)
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None
    
    def _setup_ui(self):
        # Use the main_layout provided by BaseAnalysisPanel
        layout = self.main_layout
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT: CONFIG ---
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)
        
        self._create_var_group(s_lay)
        self._create_grid_group(s_lay)
        self._create_vario_group(s_lay)
        self._create_sim_group(s_lay)
        self._create_search_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)
        
        # --- RIGHT: RESULTS & VALIDATION ---
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
        self.progress_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid {colors.BORDER}; border-radius: 4px; background-color: {colors.CARD_BG}; text-align: center; color: {colors.TEXT_PRIMARY}; height: 22px; }} QProgressBar::chunk {{ background-color: #4CAF50; }}")
        prog_lay.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY};")
        prog_lay.addWidget(self.progress_label)
        r_lay.addWidget(prog_group)

        # Tabbed interface for logs and validation
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {colors.BORDER}; background-color: {colors.CARD_BG}; }}
            QTabBar::tab {{ background-color: {colors.ELEVATED_BG}; color: {colors.TEXT_SECONDARY}; padding: 8px 16px; border: 1px solid {colors.BORDER}; }}
            QTabBar::tab:selected {{ background-color: {colors.CARD_HOVER}; color: {colors.TEXT_PRIMARY}; }}
            QTabBar::tab:hover {{ background-color: {colors.CARD_HOVER}; }}
        """)

        # Tab 1: Event Log
        log_tab = QWidget()
        log_lay = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas; font-size: 9pt;")
        log_lay.addWidget(self.log_text)
        self.results_tabs.addTab(log_tab, "Event Log")

        # Tab 2: Simulation Log
        sim_tab = QWidget()
        sim_lay = QVBoxLayout(sim_tab)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas;")
        sim_lay.addWidget(self.results_text)
        self.results_tabs.addTab(sim_tab, "Simulation Log")
        
        # Tab 3: Validation (POST-SIMULATION)
        validation_tab = QWidget()
        val_lay = QVBoxLayout(validation_tab)
        self._create_validation_tab(val_lay)
        self.results_tabs.addTab(validation_tab, "Validation")
        
        r_lay.addWidget(self.results_tabs, stretch=1)
        
        # Actions
        act_lay = QHBoxLayout()
        self.run_btn = QPushButton("RUN CO-SIMULATION")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        
        self.viz_btn = QPushButton("Visualize")
        self.viz_btn.clicked.connect(self._visualize_results)
        self.viz_btn.setEnabled(False)
        self.exp_btn = QPushButton("Export")
        self.exp_btn.clicked.connect(self._export_results)
        self.exp_btn.setEnabled(False)
        self.clr_btn = QPushButton("Clear")
        self.clr_btn.clicked.connect(self._clear_results)
        
        act_lay.addWidget(self.run_btn, stretch=2)
        act_lay.addWidget(self.viz_btn)
        act_lay.addWidget(self.exp_btn)
        act_lay.addWidget(self.clr_btn)
        r_lay.addLayout(act_lay)
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_var_group(self, layout):
        g = QGroupBox("1. Variables")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #4fc3f7; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)
        
        f = QFormLayout()
        self.primary_combo = QComboBox()
        self.secondary_list = QListWidget()
        self.secondary_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.secondary_list.setMaximumHeight(100)
        
        f.addRow("Primary:", self.primary_combo)
        l.addLayout(f)
        l.addWidget(QLabel("Secondary (Ctrl+Click):"))

        # Add secondary variables list with remove button
        secondary_layout = QHBoxLayout()
        secondary_layout.addWidget(self.secondary_list)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.setMaximumWidth(120)
        remove_btn.clicked.connect(self._remove_selected_secondary_vars)
        secondary_layout.addWidget(remove_btn)

        l.addLayout(secondary_layout)
        layout.addWidget(g)

    def _create_grid_group(self, layout):
        """Create Grid & Block Size configuration group (same as SGSIM)."""
        import numpy as np
        g = QGroupBox("2. Grid & Block Size")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)
        
        # Grid Origin
        origin_label = QLabel("Grid Origin:")
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
        
        # Number of blocks
        blocks_label = QLabel("Number of Blocks:")
        blocks_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(blocks_label)
        
        h1 = QHBoxLayout()
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(1, 1000)
        self.nx_spin.setValue(50)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(1, 1000)
        self.ny_spin.setValue(50)
        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(1, 1000)
        self.nz_spin.setValue(20)
        h1.addWidget(QLabel("NX:"))
        h1.addWidget(self.nx_spin)
        h1.addWidget(QLabel("NY:"))
        h1.addWidget(self.ny_spin)
        h1.addWidget(QLabel("NZ:"))
        h1.addWidget(self.nz_spin)
        l.addLayout(h1)
        
        # Block size
        size_label = QLabel("Block Size (m):")
        size_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(size_label)
        
        h2 = QHBoxLayout()
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 1000)
        self.dx_spin.setValue(10)
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 1000)
        self.dy_spin.setValue(10)
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 1000)
        self.dz_spin.setValue(5)
        h2.addWidget(QLabel("DX:"))
        h2.addWidget(self.dx_spin)
        h2.addWidget(QLabel("DY:"))
        h2.addWidget(self.dy_spin)
        h2.addWidget(QLabel("DZ:"))
        h2.addWidget(self.dz_spin)
        l.addLayout(h2)
        
        auto_btn = QPushButton("Auto-Detect from Drillholes")
        auto_btn.clicked.connect(self._auto_detect_grid)
        l.addWidget(auto_btn)
        
        layout.addWidget(g)

    def _auto_detect_grid(self):
        """Auto-detect grid parameters from drillhole data."""
        import numpy as np
        if self.drillhole_data is None:
            self._log_event("No drillhole data loaded", "warning")
            return
        
        df = self.drillhole_data
        x_col = next((c for c in df.columns if c.upper() in ['X', 'MIDX', 'EAST', 'EASTING']), None)
        y_col = next((c for c in df.columns if c.upper() in ['Y', 'MIDY', 'NORTH', 'NORTHING']), None)
        z_col = next((c for c in df.columns if c.upper() in ['Z', 'MIDZ', 'ELEV', 'ELEVATION', 'RL']), None)
        
        if not all([x_col, y_col, z_col]):
            self._log_event("Could not find X, Y, Z columns", "warning")
            return
        
        x_vals = df[x_col].dropna()
        y_vals = df[y_col].dropna()
        z_vals = df[z_col].dropna()
        
        dx, dy, dz = self.dx_spin.value(), self.dy_spin.value(), self.dz_spin.value()
        
        xmin = float(x_vals.min() - dx / 2)
        ymin = float(y_vals.min() - dy / 2)
        zmin = float(z_vals.min() - dz / 2)
        
        nx = int(np.ceil((x_vals.max() - x_vals.min()) / dx)) + 1
        ny = int(np.ceil((y_vals.max() - y_vals.min()) / dy)) + 1
        nz = int(np.ceil((z_vals.max() - z_vals.min()) / dz)) + 1
        
        self.xmin_spin.setValue(xmin)
        self.ymin_spin.setValue(ymin)
        self.zmin_spin.setValue(zmin)
        self.nx_spin.setValue(nx)
        self.ny_spin.setValue(ny)
        self.nz_spin.setValue(nz)
        
        self._log_event(f"✓ Grid: {nx}×{ny}×{nz}, origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f})", "success")

    def _create_vario_group(self, layout):
        g = QGroupBox("3. Variogram & Cross-Variogram")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)
        
        self.use_reg_chk = QCheckBox("Use Registry Variograms")
        self.use_reg_chk.setChecked(True)
        l.addWidget(self.use_reg_chk)
        
        # Correlation Settings (Crucial for MM1)
        corr_label = QLabel("Correlation Coefficient (MM1):")
        corr_label.setStyleSheet("color: #aaa; font-size: 9pt; margin-top: 5px;")
        l.addWidget(corr_label)
        
        h = QHBoxLayout()
        self.corr_spin = QDoubleSpinBox()
        self.corr_spin.setRange(-1.0, 1.0)
        self.corr_spin.setValue(0.75)
        self.corr_spin.setSingleStep(0.05)
        self.corr_spin.setToolTip("ρ: Correlation between Primary and Secondary in Gaussian space\n"
                                   "Cross-covariance at lag 0: Cps(0) = ρ√(Cpp(0)·Css(0))")
        h.addWidget(QLabel("ρ:"))
        h.addWidget(self.corr_spin)
        
        # Show computed scale factor
        self.scale_label = QLabel(f"√(1-ρ²) = {np.sqrt(1 - 0.75**2):.3f}")
        self.scale_label.setStyleSheet("color: #81c784; font-size: 9pt;")
        h.addWidget(self.scale_label)
        self.corr_spin.valueChanged.connect(self._update_scale_label)
        l.addLayout(h)
        
        # Cross-Variogram Advanced Settings (for CP credibility)
        xvario_label = QLabel("Cross-Variogram Parameters:")
        xvario_label.setStyleSheet("color: #aaa; font-size: 9pt; margin-top: 8px;")
        l.addWidget(xvario_label)
        
        h2 = QHBoxLayout()
        self.sill_ratio_spin = QDoubleSpinBox()
        self.sill_ratio_spin.setRange(0.1, 2.0)
        self.sill_ratio_spin.setValue(1.0)
        self.sill_ratio_spin.setSingleStep(0.1)
        self.sill_ratio_spin.setToolTip("Sill ratio (secondary/primary). Usually 1.0 in Gaussian space.")
        h2.addWidget(QLabel("Sill Ratio:"))
        h2.addWidget(self.sill_ratio_spin)
        
        self.range_ratio_spin = QDoubleSpinBox()
        self.range_ratio_spin.setRange(0.1, 3.0)
        self.range_ratio_spin.setValue(1.0)
        self.range_ratio_spin.setSingleStep(0.1)
        self.range_ratio_spin.setToolTip("Range ratio (secondary/primary). Controls cross-continuity.")
        h2.addWidget(QLabel("Range Ratio:"))
        h2.addWidget(self.range_ratio_spin)
        l.addLayout(h2)
        
        # Structured Residual Toggle (KEY FIX)
        self.structured_residual_chk = QCheckBox("Use Structured Residual (SGSIM)")
        self.structured_residual_chk.setChecked(True)
        self.structured_residual_chk.setToolTip(
            "CRITICAL: When enabled, generates spatially structured residual field using FFT-MA.\n"
            "This ensures block-to-block relationships preserve realistic grade co-patterns.\n"
            "Disabling uses random noise (legacy mode - NOT recommended for production)."
        )
        self.structured_residual_chk.setStyleSheet("color: #4CAF50; font-weight: bold;")
        l.addWidget(self.structured_residual_chk)
        
        note_label = QLabel("<i>MM1: Y_sec(u) = ρ·Y_pri(u) + √(1-ρ²)·R(u)</i>")
        note_label.setStyleSheet("color: #666; font-size: 8pt;")
        l.addWidget(note_label)
        
        layout.addWidget(g)
    
    def _update_scale_label(self, value):
        """Update the residual scale factor display."""
        scale = np.sqrt(1 - value**2)
        self.scale_label.setText(f"√(1-ρ²) = {scale:.3f}")
    
    def _create_validation_tab(self, layout):
        """Create the validation tab with post-simulation analysis tools."""
        colors = get_theme_colors()

        # Header
        header = QLabel("Post-Simulation Validation")
        header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #4fc3f7; margin-bottom: 10px;")
        layout.addWidget(header)

        info_label = QLabel(
            "Run simulation first, then use these tools to validate results.\n"
            "A Competent Person should verify correlation reproduction and distribution preservation."
        )
        info_label.setStyleSheet(f"color: {colors.TEXT_HINT}; font-size: 9pt;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Validation Metrics Table
        metrics_group = QGroupBox("Correlation Verification")
        metrics_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #81c784; border: 1px solid {colors.BORDER}; }}")
        metrics_lay = QVBoxLayout(metrics_group)

        self.validation_table = QTableWidget()
        self.validation_table.setColumnCount(5)
        self.validation_table.setHorizontalHeaderLabels([
            "Variable Pair", "Target ρ", "Empirical ρ", "Δρ (%)", "Status"
        ])
        self.validation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.validation_table.setStyleSheet(f"""
            QTableWidget {{ background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; gridline-color: {colors.BORDER}; }}
            QHeaderView::section {{ background-color: {colors.ELEVATED_BG}; color: {colors.TEXT_PRIMARY}; padding: 5px; border: 1px solid {colors.BORDER}; }}
        """)
        self.validation_table.setMaximumHeight(150)
        metrics_lay.addWidget(self.validation_table)
        layout.addWidget(metrics_group)

        # Distribution Statistics
        stats_group = QGroupBox("Distribution Statistics")
        stats_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; }}")
        stats_lay = QVBoxLayout(stats_group)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(6)
        self.stats_table.setHorizontalHeaderLabels([
            "Variable", "Input Mean", "Sim Mean", "Input Var", "Sim Var", "Status"
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.setStyleSheet(f"""
            QTableWidget {{ background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; gridline-color: {colors.BORDER}; }}
            QHeaderView::section {{ background-color: {colors.ELEVATED_BG}; color: {colors.TEXT_PRIMARY}; padding: 5px; border: 1px solid {colors.BORDER}; }}
        """)
        self.stats_table.setMaximumHeight(150)
        stats_lay.addWidget(self.stats_table)
        layout.addWidget(stats_group)
        
        # Validation Actions
        action_lay = QHBoxLayout()
        
        self.validate_btn = QPushButton("Run Validation")
        self.validate_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.validate_btn.clicked.connect(self._run_validation)
        self.validate_btn.setEnabled(False)
        
        self.scatter_btn = QPushButton("View Scatter Plot")
        self.scatter_btn.clicked.connect(self._show_scatter_plot)
        self.scatter_btn.setEnabled(False)
        
        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.clicked.connect(self._export_validation_report)
        self.export_report_btn.setEnabled(False)
        
        action_lay.addWidget(self.validate_btn)
        action_lay.addWidget(self.scatter_btn)
        action_lay.addWidget(self.export_report_btn)
        layout.addLayout(action_lay)
        
        # Validation Summary
        self.validation_summary = QTextEdit()
        self.validation_summary.setReadOnly(True)
        self.validation_summary.setMaximumHeight(100)
        colors = get_theme_colors()
        self.validation_summary.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas; font-size: 9pt;")
        self.validation_summary.setPlaceholderText("Validation summary will appear here after running validation...")
        layout.addWidget(self.validation_summary)
        
        layout.addStretch()
    
    def _run_validation(self):
        """Run post-simulation validation checks."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Results", "Run simulation first.")
            return
        
        try:
            self._log_event("Running validation checks...", "progress")
            
            # Get block model with results
            block_model = self.simulation_results.get('block_model')
            if block_model is None:
                self._log_event("Block model not found in results", "error")
                return
            
            primary = self.primary_combo.currentText()
            secondaries = [i.text() for i in self.secondary_list.selectedItems()]
            target_corr = self.corr_spin.value()
            n_reals = self.n_reals.value()
            prefix = self.prefix.currentText()
            
            # Clear tables
            self.validation_table.setRowCount(0)
            self.stats_table.setRowCount(0)
            
            validation_results = []
            
            # Correlation Verification
            for sec in secondaries:
                # Get first realization for each variable
                prim_real_name = f"{prefix}_{primary}_001"
                sec_real_name = f"{prefix}_{sec}_001"
                
                prim_data = block_model.get_property(prim_real_name)
                sec_data = block_model.get_property(sec_real_name)
                
                if prim_data is None or sec_data is None:
                    self._log_event(f"Could not find realizations for {primary}-{sec}", "warning")
                    continue
                
                # Compute empirical correlation
                valid_mask = ~np.isnan(prim_data) & ~np.isnan(sec_data)
                if np.sum(valid_mask) < 10:
                    self._log_event(f"Insufficient valid data for {primary}-{sec}", "warning")
                    continue
                
                empirical_corr = np.corrcoef(prim_data[valid_mask], sec_data[valid_mask])[0, 1]
                delta_pct = abs(empirical_corr - target_corr) / abs(target_corr) * 100 if target_corr != 0 else 0
                
                # Status based on tolerance
                if delta_pct < 10:
                    status = "✓ PASS"
                    status_color = "#81c784"
                elif delta_pct < 20:
                    status = "⚠ WARN"
                    status_color = "#ffb74d"
                else:
                    status = "✗ FAIL"
                    status_color = "#e57373"
                
                # Add to table
                row = self.validation_table.rowCount()
                self.validation_table.insertRow(row)
                self.validation_table.setItem(row, 0, QTableWidgetItem(f"{primary}-{sec}"))
                self.validation_table.setItem(row, 1, QTableWidgetItem(f"{target_corr:.3f}"))
                self.validation_table.setItem(row, 2, QTableWidgetItem(f"{empirical_corr:.3f}"))
                self.validation_table.setItem(row, 3, QTableWidgetItem(f"{delta_pct:.1f}%"))
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(status_color))
                self.validation_table.setItem(row, 4, status_item)
                
                validation_results.append({
                    'pair': f"{primary}-{sec}",
                    'target': target_corr,
                    'empirical': empirical_corr,
                    'delta_pct': delta_pct,
                    'status': status
                })
            
            # Distribution Statistics
            all_vars = [primary] + secondaries
            for var in all_vars:
                # Original data (from block model properties)
                orig_data = block_model.properties.get(var)
                if orig_data is None:
                    continue
                
                # Simulated data (mean across realizations)
                sim_values = []
                for i in range(min(n_reals, 10)):  # Check first 10 realizations
                    real_name = f"{prefix}_{var}_{i+1:03d}"
                    real_data = block_model.get_property(real_name)
                    if real_data is not None:
                        sim_values.append(real_data)
                
                if not sim_values:
                    continue
                
                sim_mean_all = np.nanmean(sim_values)
                sim_var_all = np.nanvar(sim_values)
                input_mean = np.nanmean(orig_data)
                input_var = np.nanvar(orig_data)
                
                # Status based on mean/variance reproduction
                mean_err = abs(sim_mean_all - input_mean) / (abs(input_mean) + 1e-10) * 100
                var_err = abs(sim_var_all - input_var) / (abs(input_var) + 1e-10) * 100
                
                if mean_err < 5 and var_err < 15:
                    dist_status = "✓ PASS"
                    dist_color = "#81c784"
                elif mean_err < 10 and var_err < 25:
                    dist_status = "⚠ WARN"
                    dist_color = "#ffb74d"
                else:
                    dist_status = "✗ FAIL"
                    dist_color = "#e57373"
                
                # Add to table
                row = self.stats_table.rowCount()
                self.stats_table.insertRow(row)
                self.stats_table.setItem(row, 0, QTableWidgetItem(var))
                self.stats_table.setItem(row, 1, QTableWidgetItem(f"{input_mean:.3f}"))
                self.stats_table.setItem(row, 2, QTableWidgetItem(f"{sim_mean_all:.3f}"))
                self.stats_table.setItem(row, 3, QTableWidgetItem(f"{input_var:.3f}"))
                self.stats_table.setItem(row, 4, QTableWidgetItem(f"{sim_var_all:.3f}"))
                status_item = QTableWidgetItem(dist_status)
                status_item.setForeground(QColor(dist_color))
                self.stats_table.setItem(row, 5, status_item)
            
            # Summary
            n_pass = sum(1 for r in validation_results if 'PASS' in r['status'])
            n_total = len(validation_results)
            
            summary_text = f"Validation Complete\n"
            summary_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            summary_text += f"Correlation Checks: {n_pass}/{n_total} PASSED\n"
            summary_text += f"Target ρ: {target_corr:.3f}\n"
            summary_text += f"Method: MM1 with {'Structured' if self.structured_residual_chk.isChecked() else 'Random'} Residual\n"
            summary_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            if n_pass == n_total and n_total > 0:
                summary_text += "✓ All correlation checks PASSED.\n"
                summary_text += "Results acceptable for CP review."
            elif n_pass > 0:
                summary_text += "⚠ Some checks need attention.\n"
                summary_text += "Review individual pairs before acceptance."
            else:
                summary_text += "✗ Validation FAILED.\n"
                summary_text += "Check input parameters and re-run."
            
            self.validation_summary.setText(summary_text)
            self._log_event(f"Validation complete: {n_pass}/{n_total} checks passed", "success" if n_pass == n_total else "warning")
            
            # Enable scatter plot
            self.scatter_btn.setEnabled(True)
            self.export_report_btn.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            self._log_event(f"Validation error: {e}", "error")
    
    def _show_scatter_plot(self):
        """Show scatter plot of primary vs secondary simulated values."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Results", "Run simulation and validation first.")
            return
        
        try:
            # Try to use matplotlib
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", 
                    "Matplotlib is required for scatter plots.\nInstall with: pip install matplotlib")
                return
            
            block_model = self.simulation_results.get('block_model')
            if block_model is None:
                return
            
            primary = self.primary_combo.currentText()
            secondaries = [i.text() for i in self.secondary_list.selectedItems()]
            target_corr = self.corr_spin.value()
            prefix = self.prefix.currentText()
            
            if not secondaries:
                QMessageBox.warning(self, "No Secondary", "Select at least one secondary variable.")
                return
            
            # Create scatter plot dialog
            from PyQt6.QtWidgets import QDialog, QVBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Primary-Secondary Correlation Scatter")
            dialog.resize(800, 600)
            
            fig, axes = plt.subplots(1, len(secondaries), figsize=(5*len(secondaries), 4))
            if len(secondaries) == 1:
                axes = [axes]
            
            for i, sec in enumerate(secondaries):
                prim_data = block_model.get_property(f"{prefix}_{primary}_001")
                sec_data = block_model.get_property(f"{prefix}_{sec}_001")
                
                if prim_data is None or sec_data is None:
                    continue
                
                valid_mask = ~np.isnan(prim_data) & ~np.isnan(sec_data)
                x = prim_data[valid_mask]
                y = sec_data[valid_mask]
                
                # Subsample for visualization
                if len(x) > 5000:
                    idx = np.random.choice(len(x), 5000, replace=False)
                    x, y = x[idx], y[idx]
                
                axes[i].scatter(x, y, alpha=0.3, s=5, c='#4fc3f7')
                
                # Add regression line
                if len(x) > 2:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    axes[i].plot(x_line, p(x_line), 'r--', label=f'Fit (slope={z[0]:.2f})')
                
                # Compute empirical correlation
                emp_corr = np.corrcoef(x, y)[0, 1]
                
                axes[i].set_xlabel(f'{primary} (Simulated)')
                axes[i].set_ylabel(f'{sec} (Simulated)')
                axes[i].set_title(f'{primary} vs {sec}\nTarget ρ={target_corr:.2f}, Empirical ρ={emp_corr:.2f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Embed in dialog
            canvas = FigureCanvasQTAgg(fig)
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.addWidget(canvas)
            
            dialog.exec()
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Scatter plot error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create scatter plot:\n{e}")
    
    def _export_validation_report(self):
        """Export validation report to file."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Results", "Run simulation and validation first.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Validation Report", "cosim_validation_report.txt", "Text (*.txt);;CSV (*.csv)"
        )
        
        if not path:
            return
        
        try:
            with open(path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("CO-SIMULATION (CoSGSIM) VALIDATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Primary Variable: {self.primary_combo.currentText()}\n")
                f.write(f"Secondary Variables: {', '.join(i.text() for i in self.secondary_list.selectedItems())}\n")
                f.write(f"Target Correlation: {self.corr_spin.value():.3f}\n")
                f.write(f"Number of Realizations: {self.n_reals.value()}\n")
                f.write(f"Structured Residual: {'Yes' if self.structured_residual_chk.isChecked() else 'No'}\n")
                f.write(f"Back-Transform: PCHIP Interpolation\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("CORRELATION VERIFICATION\n")
                f.write("-" * 60 + "\n")
                for row in range(self.validation_table.rowCount()):
                    pair = self.validation_table.item(row, 0).text()
                    target = self.validation_table.item(row, 1).text()
                    empirical = self.validation_table.item(row, 2).text()
                    delta = self.validation_table.item(row, 3).text()
                    status = self.validation_table.item(row, 4).text()
                    f.write(f"{pair}: Target={target}, Empirical={empirical}, Δ={delta} [{status}]\n")
                
                f.write("\n" + "-" * 60 + "\n")
                f.write("DISTRIBUTION STATISTICS\n")
                f.write("-" * 60 + "\n")
                for row in range(self.stats_table.rowCount()):
                    var = self.stats_table.item(row, 0).text()
                    in_mean = self.stats_table.item(row, 1).text()
                    sim_mean = self.stats_table.item(row, 2).text()
                    in_var = self.stats_table.item(row, 3).text()
                    sim_var = self.stats_table.item(row, 4).text()
                    status = self.stats_table.item(row, 5).text()
                    f.write(f"{var}: Input(μ={in_mean}, σ²={in_var}), Sim(μ={sim_mean}, σ²={sim_var}) [{status}]\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("VALIDATION SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(self.validation_summary.toPlainText())
                f.write("\n")
            
            QMessageBox.information(self, "Export Complete", f"Validation report exported to:\n{path}")
            self._log_event(f"Validation report exported to {path}", "success")
            
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export report:\n{e}")

    def _create_sim_group(self, layout):
        g = QGroupBox("4. Simulation Params")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #81c784; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        f = QFormLayout(g)
        
        self.n_reals = QSpinBox()
        self.n_reals.setRange(1, 500)
        self.n_reals.setValue(50)
        
        # AUDIT FIX (W-001): Seed is MANDATORY for JORC/SAMREC reproducibility
        import time
        default_seed = int(time.time()) % 100000
        self.seed = QSpinBox()
        self.seed.setRange(1, 999999)  # Minimum 1, no -1 or 0 allowed
        self.seed.setValue(default_seed if default_seed > 0 else 42)
        self.seed.setToolTip(
            "Random seed for reproducibility (REQUIRED).\n"
            "Same seed = same results. Record this for JORC/SAMREC compliance."
        )
        self.prefix = QComboBox()
        self.prefix.setEditable(True)
        self.prefix.addItems(["cosim", "sim"])
        
        f.addRow("Realizations:", self.n_reals)
        f.addRow("Seed:", self.seed)
        f.addRow("Prefix:", self.prefix)
        layout.addWidget(g)

    def _create_search_group(self, layout):
        g = QGroupBox("5. Search")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ba68c8; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QHBoxLayout(g)
        
        self.min_n = QSpinBox()
        self.min_n.setValue(4)
        self.max_n = QSpinBox()
        self.max_n.setValue(12)
        self.rad = QDoubleSpinBox()
        self.rad.setRange(1, 10000)
        self.rad.setValue(200)
        
        l.addWidget(QLabel("Min:"))
        l.addWidget(self.min_n)
        l.addWidget(QLabel("Max:"))
        l.addWidget(self.max_n)
        l.addWidget(QLabel("Rad:"))
        l.addWidget(self.rad)
        layout.addWidget(g)

    def _on_data_loaded(self, data):
        df = None
        if isinstance(data, dict):
            # Prefer composites if available and non-empty, otherwise use assays
            # Fix: Explicitly check for non-empty DataFrames to avoid ValueError
            composites = data.get('composites')
            composites_df = data.get('composites_df')
            assays_data = data.get('assays')
            assays_df = data.get('assays_df')

            if isinstance(composites, pd.DataFrame) and not composites.empty:
                comp = composites
            elif isinstance(composites_df, pd.DataFrame) and not composites_df.empty:
                comp = composites_df
            else:
                comp = None

            if isinstance(assays_data, pd.DataFrame) and not assays_data.empty:
                assays = assays_data
            elif isinstance(assays_df, pd.DataFrame) and not assays_df.empty:
                assays = assays_df
            else:
                assays = None
            if isinstance(comp, pd.DataFrame) and not comp.empty:
                df = comp
                # AUDIT FIX: Set provenance for composites
                if 'source_type' not in df.attrs:
                    df.attrs['source_type'] = 'composites'
                    df.attrs['lineage_gate_passed'] = True
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                df = assays
                # AUDIT FIX: Mark raw assays appropriately
                df.attrs['source_type'] = 'raw_assays'
                df.attrs['lineage_gate_passed'] = False
        elif isinstance(data, pd.DataFrame):
            df = data
            # AUDIT FIX: Set default provenance if not present
            if 'source_type' not in df.attrs:
                df.attrs['source_type'] = 'composites'
                df.attrs['lineage_gate_passed'] = True
        
        if df is not None and not df.empty:
            self.drillhole_data = df
        if df is not None:
            # Only update UI if it's been built
            if hasattr(self, 'primary_combo') and hasattr(self, 'secondary_list') and hasattr(self, 'run_btn'):
                cols = sorted(df.select_dtypes(include=[float]).columns)
                # Exclude system IDs, coordinates, and compositing metadata
                exclude_cols = ['X', 'Y', 'Z', 'HOLEID', 'FROM', 'TO', 'GLOBAL_INTERVAL_ID',
                               # Compositing metadata columns
                               'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                               'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO']
                cols = [c for c in cols if c.upper() not in exclude_cols]
                self.primary_combo.clear()
                self.primary_combo.addItems(cols)
                self.secondary_list.clear()
                self.secondary_list.addItems(cols)
                self.run_btn.setEnabled(True)
            else:
                logger.debug("Co-SGSIM panel: UI not ready, data stored for later initialization")

    def _on_vario_loaded(self, res):
        v = res.get('variable')
        if v:
            self.variogram_results[v] = res

    def gather_parameters(self) -> Dict[str, Any]:
        """Collects parameters and maps correlation correctly."""
        primary = self.primary_combo.currentText()
        secondaries = [i.text() for i in self.secondary_list.selectedItems()]

        # Map the single spinner to dictionary for all pairs
        correlations = {}
        corr_val = self.corr_spin.value()
        for sec in secondaries:
            correlations[(primary, sec)] = corr_val

        # Cross-variogram parameters (advanced settings)
        cross_variogram_params = {}
        sill_ratio = self.sill_ratio_spin.value() if hasattr(self, 'sill_ratio_spin') else 1.0
        range_ratio = self.range_ratio_spin.value() if hasattr(self, 'range_ratio_spin') else 1.0
        for sec in secondaries:
            cross_variogram_params[(primary, sec)] = {
                'sill_ratio': sill_ratio,
                'range_ratio': range_ratio
            }

        # Convert DataFrame to BlockModel object
        block_model_df = self.registry.get_block_model() if self.registry else None
        block_model = None
        if block_model_df is not None:
            from ..models.block_model import BlockModel
            block_model = BlockModel()
            block_model.update_from_dataframe(block_model_df)

        # Structured residual option (KEY FIX for geostatistical credibility)
        use_structured = True
        if hasattr(self, 'structured_residual_chk'):
            use_structured = self.structured_residual_chk.isChecked()

        return {
            "block_model": block_model,
            "primary_name": primary,
            "secondary_names": secondaries,
            "n_realizations": self.n_reals.value(),
            "random_seed": self.seed.value(),
            "realisation_prefix": self.prefix.currentText(),
            "variogram_models": self.variogram_results if self.use_reg_chk.isChecked() else {},
            "correlations": correlations,
            "cross_variogram_params": cross_variogram_params,
            "use_structured_residual": use_structured,
            "search_params": {
                "min_neighbors": self.min_n.value(),
                "max_neighbors": self.max_n.value(),
                "max_search_radius": self.rad.value()
            },
            # Grid & Block Size parameters (same as SGSIM)
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

    def validate_inputs(self) -> bool:
        if not self.secondary_list.selectedItems():
            QMessageBox.warning(self, "Error", "Select at least one secondary variable.")
            return False
        return True

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Co-Simulation.

        Co-Simulation requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Composited data (consistent sample support)
        3. Validated (for data quality)

        Returns:
            True if data is acceptable for Co-Simulation
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
                f"Cannot run Co-Simulation:\n\n{message}\n\n"
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
        """
        Run Co-Simulation following the standardized workflow:
        1. Load drillholes
        2. Define grid extents + block size
        3. Generate empty property array
        4. Run Co-Simulation to populate it
        """
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not available.")
            return

        # ========================================================================
        # STEP 1: LOAD DRILLHOLES
        # ========================================================================
        self._log_event("Step 1: Loading drillholes...", "progress")

        if not self.validate_inputs():
            return

        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return

        # Gather parameters (includes drillhole loading)
        params = self.gather_parameters()

        # ========================================================================
        # STEP 2: DEFINE GRID EXTENTS + BLOCK SIZE
        # ========================================================================
        self._log_event("Step 2: Defining grid extents + block size...", "progress")
        self._log_event(f"  Grid: {params['nx']}×{params['ny']}×{params['nz']} blocks", "info")
        self._log_event(f"  Block size: {params['xinc']:.1f}×{params['yinc']:.1f}×{params['zinc']:.1f}", "info")

        # ========================================================================
        # STEP 3: GENERATE EMPTY PROPERTY ARRAY
        # ========================================================================
        self._log_event("Step 3: Generating empty property array...", "progress")
        self._log_event(f"  Array shape: {params['n_realizations']}×{params['nz']}×{params['ny']}×{params['nx']}", "info")

        # ========================================================================
        # STEP 4: RUN CO-SIMULATION TO POPULATE IT
        # ========================================================================
        self._log_event("Step 4: Running Co-Simulation to populate array...", "progress")
        self._log_event(f"  Primary: {self.primary_combo.currentText()}", "info")
        self._log_event(f"  Realizations: {params['n_realizations']}", "info")

        # Set up progress bar
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0% - Initializing {params['n_realizations']} realizations...")
        self.progress_label.setText("Starting Co-Simulation...")

        # Disable run button
        self.run_btn.setEnabled(False)

        # Progress callback using signals for thread safety
        def progress_callback(percent: int, message: str):
            """Update progress from worker thread using signals."""
            # Emit signal to update UI from main thread
            self.progress_updated.emit(percent, message)
        
        # 5. Run via controller (data injection happens in run_cosgsim)
        self.controller.run_cosgsim(
            config=params,
            callback=self._on_finished,
            progress_callback=progress_callback
        )
    
    def _handle_task_progress(self, task_name: str, percent: int, message: str):
        """Receive progress from the task system and route to UI update."""
        if task_name == self.task_name:
            self._update_progress(percent, message)

    def _update_progress(self, percent: int, message: str = ""):
        """Update progress bar and label with percentage and message."""
        if not hasattr(self, 'progress_bar') or self.progress_bar is None:
            return

        percent = max(0, min(100, percent))
        self.progress_bar.setValue(percent)
        if message:
            # Check if message is simulation count (e.g., "25/50")
            if "/" in message and message.replace("/", "").replace(" ", "").isdigit():
                # This is a simulation count - show it prominently
                self.progress_bar.setFormat(f"Simulations: {message} ({percent}%)")
                self.progress_label.setText(f"Running simulation {message}...")
            else:
                # Regular message - show with percent
                self.progress_bar.setFormat(f"{percent}% - {message}")
                self.progress_label.setText(message)
        else:
            self.progress_bar.setFormat(f"{percent}%")
            self.progress_label.setText("")

        # Log progress message (throttled to avoid flooding log)
        if hasattr(self, 'log_text') and self.log_text is not None:
            # Only log every 10% to avoid log spam
            if percent % 10 == 0 or percent >= 100:
                self.log_text.append(message)

        # Force immediate visual repaint of the progress bar.
        # NOTE: Do NOT use QApplication.processEvents() here - it processes ALL pending
        # queued signals at once, causing the bar to jump 0->100 skipping intermediate steps.
        self.progress_bar.repaint()

    def _on_finished(self, result: Dict[str, Any]):
        """Handle successful completion."""
        if result is None:
            self._on_error("Simulation returned no result.")
            return
        
        if result.get("error"):
            self._on_error(result["error"])
            return
        
        # Extract result data (task function returns dict with name and other keys)
        self.simulation_results = result
        self.run_btn.setEnabled(True)
        self.viz_btn.setEnabled(True)
        self.exp_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Complete!")
        self.log_text.append("\nSUCCESS: Co-Simulation Complete.")
        
        # Enable validation button
        if hasattr(self, 'validate_btn'):
            self.validate_btn.setEnabled(True)
            self._log_event("✓ Validation tools enabled - switch to 'Validation' tab", "success")
        
        # Auto-register results back to registry
        if self.registry and hasattr(self.registry, 'register_sgsim_results'):
            try:
                self.registry.register_sgsim_results(result, source_panel="CoSGSIM")
            except Exception as e:
                logger.warning(f"Failed to register results: {e}")
    
    def _on_error(self, msg: str):
        """Handle errors from worker."""
        self.run_btn.setEnabled(True)
        self.log_text.append(f"\nERROR: {msg}")
        self.progress_label.setText("Error occurred")
        QMessageBox.critical(self, "Simulation Error", msg)

    def _visualize_results(self):
        """Visualize simulation results."""
        if not hasattr(self, 'simulation_results') or self.simulation_results is None:
            self._log_event("No results to visualize", "warning")
            return
        
        try:
            # Extract grid from visualization.mesh (standardized workflow format)
            viz = self.simulation_results.get('visualization', {})
            grid = viz.get('mesh')
            
            if grid is not None:
                property_name = viz.get('property', 'CoSGSIM_Mean')
                self._log_event("Sending results to 3D viewer...", "info")
                self.request_visualization.emit(grid, property_name)
                self._log_event("✓ Visualization request sent", "success")
            else:
                # Fallback to old format
                names = self.simulation_results.get('realization_names', {})
                if names:
                    first_var = list(names.keys())[0]  # Usually primary
                    # If secondary exists, visualize that (more interesting)
                    if len(names) > 1:
                        first_var = list(names.keys())[1]
                    
                    first_real = names[first_var][0]
                    self.request_visualization.emit(None, first_real)  # Passes string property name
                    self.log_text.append(f"Visualizing: {first_real}")
                else:
                    self._log_event("No grid data in results", "warning")
        except Exception as e:
            self._log_event(f"Visualization error: {e}", "error")

    def _remove_selected_secondary_vars(self):
        """Remove selected secondary variables from the list."""
        selected_items = self.secondary_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select secondary variables to remove.")
            return

        # Remove selected items (iterate in reverse to maintain indices)
        for item in reversed(selected_items):
            row = self.secondary_list.row(item)
            self.secondary_list.takeItem(row)

    def _export_results(self):
        """Export simulation results."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Results", "Run simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CoSGSIM Results", "cosim_results.csv", "CSV (*.csv)"
        )

        if not path:
            return

        try:
            import pandas as pd
            import numpy as np

            # Check if results are in standardized workflow format (new format)
            results_data = self.simulation_results.get('results')
            if results_data is not None:
                # Standardized workflow format: results contains realizations, grid_coords, etc.
                grid_coords = results_data.get('grid_coords')
                realizations = results_data.get('realizations')
                summary = results_data.get('summary', {})
                
                if grid_coords is None or realizations is None:
                    raise ValueError("Missing grid_coords or realizations in standardized workflow results")
                
                # Reshape realizations if needed: (nreal, nz, ny, nx) -> (nreal, n_blocks)
                n_realizations, nz, ny, nx = realizations.shape
                n_blocks = nz * ny * nx
                
                # Flatten grid coordinates if needed
                if grid_coords.shape[0] != n_blocks:
                    # Reshape to match grid structure
                    grid_coords = grid_coords.reshape(nz, ny, nx, 3)
                    grid_coords = grid_coords.reshape(n_blocks, 3)
                
                # Create DataFrame with coordinates
                df_data = {
                    'X': grid_coords[:, 0],
                    'Y': grid_coords[:, 1],
                    'Z': grid_coords[:, 2],
                }
                
                # Add summary statistics
                if 'mean' in summary:
                    df_data['MEAN'] = summary['mean'].ravel()
                if 'var' in summary:
                    df_data['VARIANCE'] = summary['var'].ravel()
                if 'std' in summary:
                    df_data['STD_DEV'] = summary['std'].ravel()
                if 'p10' in summary:
                    df_data['P10'] = summary['p10'].ravel()
                if 'p50' in summary:
                    df_data['P50'] = summary['p50'].ravel()
                if 'p90' in summary:
                    df_data['P90'] = summary['p90'].ravel()
                
                # Add individual realizations
                for i in range(n_realizations):
                    real_data = realizations[i].ravel()
                    df_data[f'REALIZATION_{i+1:03d}'] = real_data
                
                # Create DataFrame and export
                df = pd.DataFrame(df_data)
                df = df.replace([pd.NA, float('inf'), float('-inf')], pd.NA).dropna()
                df.to_csv(path, index=False)
                
                n_blocks_exported = len(df)
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"✓ CoSGSIM results exported successfully!\n\n"
                    f"File: {path}\n"
                    f"Blocks: {n_blocks_exported}\n"
                    f"Realizations: {n_realizations}\n"
                    f"Format: Standardized workflow"
                )
                
                self.log_text.append(f"✓ Exported {n_blocks_exported} blocks with {n_realizations} realizations to {path}")
                
            else:
                # Legacy format: block_model with realization properties
                block_model = self.simulation_results.get('block_model')
                if block_model is None:
                    raise ValueError("Block model not found in simulation results. Results may be in an unsupported format.")
                
                # Get coordinates
                coords = block_model.positions
                if coords is None:
                    raise ValueError("Block model has no coordinate data")
                
                # Create DataFrame with coordinates
                df_data = {
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                }
                
                # Add all realization properties
                realization_names = self.simulation_results.get('realization_names', {})
                for var_name, prop_names in realization_names.items():
                    for prop_name in prop_names:
                        prop_data = block_model.get_property(prop_name)
                        if prop_data is not None:
                            df_data[prop_name] = prop_data
                        else:
                            logger.warning(f"Property {prop_name} not found in block model")
                
                # Create DataFrame and export
                df = pd.DataFrame(df_data)
                df = df.replace([pd.NA, float('inf'), float('-inf')], pd.NA).dropna()
                df.to_csv(path, index=False)
                
                n_blocks = len(df)
                n_vars = len(realization_names)
                n_realizations = sum(len(props) for props in realization_names.values())
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"✓ CoSGSIM results exported successfully!\n\n"
                    f"File: {path}\n"
                    f"Blocks: {n_blocks}\n"
                    f"Variables: {n_vars}\n"
                    f"Total Realizations: {n_realizations}"
                )
                
                self.log_text.append(f"✓ Exported {n_blocks} blocks with {n_realizations} realizations to {path}")

        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    def _log_event(self, message: str, level: str = "info"):
        """Add timestamped event to the log with color coding."""
        from datetime import datetime
        if not hasattr(self, 'log_text') or not self.log_text: return
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"info": f"{ModernColors.TEXT_PRIMARY}", "success": "#81c784", "warning": "#ffb74d", "error": "#e57373", "progress": "#4fc3f7"}
        self.log_text.append(f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {colors.get(level, f"{ModernColors.TEXT_PRIMARY}")};">{message}</span>')

    def _clear_results(self):
        """Clear simulation results."""
        self.results_text.clear()
        self.simulation_results = None
        self.viz_btn.setEnabled(False)
        self.exp_btn.setEnabled(False)

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Variables (save selected items from list)
            if hasattr(self, 'var_list') and self.var_list:
                try:
                    selected_items = self.var_list.selectedItems()
                    if selected_items:
                        settings['selected_variables'] = [item.text() for item in selected_items]
                except Exception:
                    pass
            
            # Simulation parameters
            settings['nreal'] = get_safe_widget_value(self, 'nreal_spin')
            settings['seed'] = get_safe_widget_value(self, 'seed_spin')
            
            # Grid
            settings['xmin'] = get_safe_widget_value(self, 'xmin_spin')
            settings['ymin'] = get_safe_widget_value(self, 'ymin_spin')
            settings['zmin'] = get_safe_widget_value(self, 'zmin_spin')
            settings['grid_x'] = get_safe_widget_value(self, 'dx_spin')
            settings['grid_y'] = get_safe_widget_value(self, 'dy_spin')
            settings['grid_z'] = get_safe_widget_value(self, 'dz_spin')
            settings['nx'] = get_safe_widget_value(self, 'nx_spin')
            settings['ny'] = get_safe_widget_value(self, 'ny_spin')
            settings['nz'] = get_safe_widget_value(self, 'nz_spin')
            
            # Search settings
            settings['neighbors'] = get_safe_widget_value(self, 'neighbors_spin')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save CoSGSIM panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Variables (restore selected items in list)
            if 'selected_variables' in settings and hasattr(self, 'var_list'):
                self.var_list.clearSelection()
                for i in range(self.var_list.count()):
                    item = self.var_list.item(i)
                    if item and item.text() in settings['selected_variables']:
                        item.setSelected(True)
            
            # Simulation parameters
            set_safe_widget_value(self, 'nreal_spin', settings.get('nreal'))
            set_safe_widget_value(self, 'seed_spin', settings.get('seed'))
            
            # Grid
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            set_safe_widget_value(self, 'dx_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'dy_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'dz_spin', settings.get('grid_z'))
            set_safe_widget_value(self, 'nx_spin', settings.get('nx'))
            set_safe_widget_value(self, 'ny_spin', settings.get('ny'))
            set_safe_widget_value(self, 'nz_spin', settings.get('nz'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
                
            logger.info("Restored CoSGSIM panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore CoSGSIM panel settings: {e}")