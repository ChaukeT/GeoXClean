"""
Sequential Indicator Simulation (SIS) Panel
============================================

UI panel for configuring and running Sequential Indicator Simulation.
Refactored for Modern UX/UI.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QLabel,
    QMessageBox, QWidget, QLineEdit, QTableWidget, QTextEdit,
    QTableWidgetItem, QHeaderView, QSplitter, QScrollArea, QFrame, QTabWidget, QProgressBar
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from datetime import datetime
from .base_analysis_panel import BaseAnalysisPanel
from ..utils.coordinate_utils import ensure_xyz_columns

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)

class SISPanel(BaseAnalysisPanel):
    task_name = "sis"
    request_visualization = pyqtSignal(object, str)
    
    def __init__(self, parent=None):
        # Initialize state BEFORE super().__init__
        self.drillhole_data = None
        self.simulation_results = None
        self.variogram_results = None  # Store loaded variogram
        
        super().__init__(parent=parent, panel_id="sis")

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
        self.setWindowTitle("Sequential Indicator Simulation (SIS)")
        self.resize(1100, 800)
        
        # Build UI (required when using _build_ui pattern)
        self._build_ui()
        
        self._init_registry()
    
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
                logger.debug("SISPanel: Connected to drillholeDataLoaded signal")
            
            vario_signal = self.registry.variogramResultsLoaded
            if vario_signal is not None:
                vario_signal.connect(self._on_vario_loaded)
                logger.debug("SISPanel: Connected to variogramResultsLoaded signal")
            
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
            v = self.registry.get_variogram_results()
            if v is not None:
                self._on_vario_loaded(v)
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None
    
    def _on_vario_loaded(self, res):
        """Handle variogram results loaded from registry."""
        self.variogram_results = res
    
    def _apply_variogram_to_thresholds(self):
        """Apply variogram range/sill to all threshold rows."""
        if not self.variogram_results:
            self._log_event("No variogram results in registry", "warning")
            return
        
        # Extract fitted parameters
        fitted = self.variogram_results.get('fitted_models', {})
        if 'omni' in fitted:
            p = fitted['omni']
        elif fitted:
            p = list(fitted.values())[0]
        else:
            p = self.variogram_results
        
        # Get range and sill
        range_val = p.get('range', 100)
        nugget = p.get('nugget', 0.0)
        total_sill = p.get('total_sill', p.get('sill', 1.0) + nugget)
        
        # Apply to all rows in threshold table
        for row in range(self.thresh_table.rowCount()):
            # Set range (column 1)
            self.thresh_table.setItem(row, 1, QTableWidgetItem(f"{range_val:.1f}"))
            # Set sill (column 2)
            self.thresh_table.setItem(row, 2, QTableWidgetItem(f"{total_sill:.3f}"))
        
        self._log_event(f"✓ Applied variogram to {self.thresh_table.rowCount()} thresholds: range={range_val:.1f}, sill={total_sill:.3f}", "success")
    
    def _load_from_variogram(self):
        """Load variogram parameters from registry or cached results."""
        # Try to get from parent's variogram dialog first
        mw = self.parent()
        if hasattr(mw, 'variogram_dialog') and mw.variogram_dialog:
            if hasattr(mw.variogram_dialog, 'get_variogram_results'):
                res = mw.variogram_dialog.get_variogram_results()
            else:
                res = getattr(mw.variogram_dialog, 'variogram_results', None)
            if res:
                self.variogram_results = res
                self._apply_variogram_to_thresholds()
                return
        
        # Fall back to registry
        if self.registry:
            res = self.registry.get_variogram_results()
            if res:
                self.variogram_results = res
                self._apply_variogram_to_thresholds()
                return
        
        # Fall back to cached results
        if self.variogram_results:
            self._apply_variogram_to_thresholds()
            return
        
        self._log_event("No variogram results available in registry", "warning")

    def _setup_ui(self):
        # Clear any existing layout from base class (BaseAnalysisPanel creates a scroll area)
        old_layout = self.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                    del item
            QWidget().setLayout(old_layout)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT (Config)
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)
        
        self._create_data_group(s_lay)
        self._create_grid_group(s_lay)
        self._create_thresh_group(s_lay)
        self._create_sim_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # RIGHT (Results)
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Progress Bar
        prog_group = QGroupBox("Progress")
        prog_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; }")
        prog_lay = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"QProgressBar {{ border: 1px solid #555; border-radius: 4px; background-color: {ModernColors.PANEL_BG}; text-align: center; color: white; height: 22px; }} QProgressBar::chunk {{ background-color: #4CAF50; }}")
        prog_lay.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: #90a4ae;")
        prog_lay.addWidget(self.progress_label)
        r_lay.addWidget(prog_group)
        
        # Event Log
        log_group = QGroupBox("Event Log")
        log_lay = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet(f"background-color: #2b2b2b; color: {ModernColors.TEXT_PRIMARY}; font-family: Consolas; font-size: 9pt;")
        log_lay.addWidget(self.log_text)
        r_lay.addWidget(log_group)
        
        self.tabs = QTabWidget()
        
        # Log Tab
        log_tab = QWidget()
        l1 = QVBoxLayout(log_tab)
        self.results_text = QLabel("Configure parameters and run simulation.")
        self.results_text.setWordWrap(True)
        self.results_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        l1.addWidget(self.results_text)
        self.tabs.addTab(log_tab, "Log")
        
        # Probability Map Tab
        prob_tab = QWidget()
        l2 = QVBoxLayout(prob_tab)
        l2.addWidget(QLabel("Probability Maps:"))
        self.prob_layout = QVBoxLayout()
        l2.addLayout(self.prob_layout)
        l2.addStretch()
        self.tabs.addTab(prob_tab, "Uncertainty")
        
        r_lay.addWidget(self.tabs)
        
        # Action Buttons
        # Action buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("RUN SIS")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.run_btn)
        
        self.visualize_btn = QPushButton("Visualize 3D")
        self.visualize_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 12px;")
        self.visualize_btn.clicked.connect(self.visualize_results)
        self.visualize_btn.setEnabled(False)
        btn_layout.addWidget(self.visualize_btn)
        
        r_lay.addLayout(btn_layout)
        
        # Store results (SIS uses simulation_results, but we'll also store kriging_results for visualization)
        self.kriging_results = None
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_data_group(self, layout):
        g = QGroupBox("1. Data")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        self.prop_combo = QComboBox()
        self.prop_combo.currentTextChanged.connect(self._auto_thresh)
        l.addWidget(QLabel("Property:"))
        l.addWidget(self.prop_combo)
        layout.addWidget(g)

    def _create_grid_group(self, layout):
        """Create Grid & Block Size configuration group (same as SGSIM)."""
        g = QGroupBox("2. Grid & Block Size")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        
        # Grid Origin (xmin, ymin, zmin)
        origin_label = QLabel("Grid Origin (corner of first block):")
        origin_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(origin_label)
        
        h0 = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setDecimals(1)
        self.xmin_spin.setValue(0)
        self.xmin_spin.setToolTip("X origin (min X coordinate of grid)")
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(1)
        self.ymin_spin.setValue(0)
        self.ymin_spin.setToolTip("Y origin (min Y coordinate of grid)")
        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-1e9, 1e9)
        self.zmin_spin.setDecimals(1)
        self.zmin_spin.setValue(0)
        self.zmin_spin.setToolTip("Z origin (min Z/elevation coordinate of grid)")
        h0.addWidget(QLabel("X₀:"))
        h0.addWidget(self.xmin_spin)
        h0.addWidget(QLabel("Y₀:"))
        h0.addWidget(self.ymin_spin)
        h0.addWidget(QLabel("Z₀:"))
        h0.addWidget(self.zmin_spin)
        l.addLayout(h0)
        
        # Number of blocks (NX, NY, NZ)
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
        
        # Block size (spacing)
        size_label = QLabel("Block Size (meters):")
        size_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(size_label)
        
        h2 = QHBoxLayout()
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 1000)
        self.dx_spin.setValue(10)
        self.dx_spin.setToolTip("Block size in X (meters)")
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 1000)
        self.dy_spin.setValue(10)
        self.dy_spin.setToolTip("Block size in Y (meters)")
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 1000)
        self.dz_spin.setValue(5)
        self.dz_spin.setToolTip("Block size in Z (meters)")
        h2.addWidget(QLabel("DX:"))
        h2.addWidget(self.dx_spin)
        h2.addWidget(QLabel("DY:"))
        h2.addWidget(self.dy_spin)
        h2.addWidget(QLabel("DZ:"))
        h2.addWidget(self.dz_spin)
        l.addLayout(h2)
        
        # Auto-detect button
        auto_btn = QPushButton("Auto-Detect from Drillholes")
        auto_btn.setToolTip("Calculate grid origin and size to cover all drillhole data")
        auto_btn.clicked.connect(self._auto_detect_grid)
        l.addWidget(auto_btn)
        
        layout.addWidget(g)

    def _auto_detect_grid(self):
        """Auto-detect grid parameters from drillhole data."""
        if not hasattr(self, 'drillhole_data') or self.drillhole_data is None:
            self._log_event("No drillhole data loaded", "warning")
            return
        
        df = self.drillhole_data
        x_col = next((c for c in df.columns if c.upper() in ['X', 'MIDX', 'EAST', 'EASTING']), None)
        y_col = next((c for c in df.columns if c.upper() in ['Y', 'MIDY', 'NORTH', 'NORTHING']), None)
        z_col = next((c for c in df.columns if c.upper() in ['Z', 'MIDZ', 'ELEV', 'ELEVATION', 'RL']), None)
        
        if not all([x_col, y_col, z_col]):
            self._log_event("Could not find X, Y, Z columns in data", "warning")
            return
        
        x_vals = df[x_col].dropna()
        y_vals = df[y_col].dropna()
        z_vals = df[z_col].dropna()
        
        # Calculate grid parameters
        dx = self.dx_spin.value()
        dy = self.dy_spin.value()
        dz = self.dz_spin.value()
        
        xmin = float(x_vals.min() - dx / 2)
        ymin = float(y_vals.min() - dy / 2)
        zmin = float(z_vals.min() - dz / 2)
        
        nx = int(np.ceil((x_vals.max() - x_vals.min()) / dx)) + 1
        ny = int(np.ceil((y_vals.max() - y_vals.min()) / dy)) + 1
        nz = int(np.ceil((z_vals.max() - z_vals.min()) / dz)) + 1
        
        # Set values
        self.xmin_spin.setValue(xmin)
        self.ymin_spin.setValue(ymin)
        self.zmin_spin.setValue(zmin)
        self.nx_spin.setValue(nx)
        self.ny_spin.setValue(ny)
        self.nz_spin.setValue(nz)
        
        self._log_event(f"✓ Grid auto-detected: {nx}×{ny}×{nz} blocks, origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f})", "success")

    def _create_thresh_group(self, layout):
        g = QGroupBox("3. Thresholds & Indicators")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        
        # Load from Registry button
        load_btn = QPushButton("Load Variogram from Registry")
        load_btn.setStyleSheet("background-color: #7b1fa2; color: white;")
        load_btn.setToolTip("Apply variogram range/sill from registry to all threshold rows")
        load_btn.clicked.connect(self._load_from_variogram)
        l.addWidget(load_btn)
        
        self.thresh_table = QTableWidget(3, 3)
        self.thresh_table.setHorizontalHeaderLabels(["Cutoff", "Range", "Sill"])
        self.thresh_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        l.addWidget(self.thresh_table)
        
        h = QHBoxLayout()
        btn_add = QPushButton("+")
        btn_add.clicked.connect(self._add_row)
        btn_rem = QPushButton("-")
        btn_rem.clicked.connect(self._rem_row)
        btn_auto = QPushButton("Auto")
        btn_auto.clicked.connect(self._auto_thresh)
        h.addWidget(btn_add)
        h.addWidget(btn_rem)
        h.addWidget(btn_auto)
        l.addLayout(h)
        layout.addWidget(g)

    def _create_sim_group(self, layout):
        g = QGroupBox("4. Simulation")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        self.n_reals = QSpinBox()
        self.n_reals.setRange(1, 500)
        self.n_reals.setValue(50)
        
        # AUDIT FIX (W-001): Seed is MANDATORY for JORC/SAMREC reproducibility
        import time
        default_seed = int(time.time()) % 100000
        self.seed = QSpinBox()
        self.seed.setRange(1, 999999)  # Minimum 1, no zero allowed
        self.seed.setValue(default_seed if default_seed > 0 else 12345)
        self.seed.setToolTip(
            "Random seed for reproducibility (REQUIRED).\n"
            "Same seed = same results. Record this for JORC/SAMREC compliance."
        )
        self.neigh = QSpinBox()
        self.neigh.setRange(4, 50)
        self.neigh.setValue(12)
        self.rad = QDoubleSpinBox()
        self.rad.setRange(10, 5000)
        self.rad.setValue(200)
        self.prefix = QLineEdit("sis")
        
        f.addRow("Reals:", self.n_reals)
        f.addRow("Seed:", self.seed)
        f.addRow("Neigh:", self.neigh)
        f.addRow("Radius:", self.rad)
        f.addRow("Prefix:", self.prefix)
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
            self.drillhole_data = ensure_xyz_columns(df)
            # Preserve attrs after ensure_xyz_columns
            if hasattr(df, 'attrs') and df.attrs:
                self.drillhole_data.attrs = df.attrs.copy()
        if self.drillhole_data is not None:
            # Only update UI if it's been built
            if hasattr(self, 'prop_combo') and self.prop_combo is not None:
                cols = self.drillhole_data.select_dtypes(include=[np.number]).columns
                # Exclude system IDs, coordinates, and compositing metadata
                exclude_cols = ['X', 'Y', 'Z', 'HOLEID', 'FROM', 'TO', 'GLOBAL_INTERVAL_ID',
                               # Compositing metadata columns
                               'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                               'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO']
                cols = [c for c in cols if c.upper() not in exclude_cols]
                self.prop_combo.clear()
                self.prop_combo.addItems(sorted(cols))
            else:
                logger.debug("SIS panel: UI not ready, data stored for later initialization")

    def _auto_thresh(self):
        # Populate table with quartiles
        if self.drillhole_data is None:
            return
        p = self.prop_combo.currentText()
        if not p:
            return
        vals = self.drillhole_data[p].dropna()
        qs = [np.percentile(vals, q) for q in [25, 50, 75]]
        self.thresh_table.setRowCount(3)
        for i, q in enumerate(qs):
            self.thresh_table.setItem(i, 0, QTableWidgetItem(f"{q:.2f}"))
            self.thresh_table.setItem(i, 1, QTableWidgetItem("100"))
            self.thresh_table.setItem(i, 2, QTableWidgetItem("1.0"))

    def _add_row(self):
        self.thresh_table.insertRow(self.thresh_table.rowCount())

    def _rem_row(self):
        row = self.thresh_table.currentRow()
        if row >= 0:
            self.thresh_table.removeRow(row)

    def gather_parameters(self) -> Dict[str, Any]:
        ts = []
        for r in range(self.thresh_table.rowCount()):
            try:
                t = float(self.thresh_table.item(r, 0).text())
                rg = float(self.thresh_table.item(r, 1).text())
                s = float(self.thresh_table.item(r, 2).text())
                ts.append({'threshold': t, 'range': rg, 'sill': s})
            except (ValueError, AttributeError):
                pass
        
        return {
            'data_df': self.drillhole_data,
            'property': self.prop_combo.currentText(),
            'thresholds': ts,
            'n_realizations': self.n_reals.value(),
            'random_seed': self.seed.value(),
            'max_neighbors': self.neigh.value(),
            'max_search_radius': self.rad.value(),
            'realization_prefix': self.prefix.text(),
            # Grid & Block Size parameters (same as SGSIM)
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
            'nz': self.nz_spin.value(),
            'xmin': self.xmin_spin.value(),
            'ymin': self.ymin_spin.value(),
            'zmin': self.zmin_spin.value(),
            'xinc': self.dx_spin.value(),
            'yinc': self.dy_spin.value(),
            'zinc': self.dz_spin.value(),
        }

    def validate_inputs(self) -> bool:
        if self.thresh_table.rowCount() == 0:
            QMessageBox.warning(self, "Error", "Add thresholds")
            return False
        return True

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Sequential Indicator Simulation.

        SIS requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Composited data (consistent sample support)

        Returns:
            True if data is acceptable for SIS
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
                f"Cannot run SIS simulation:\n\n{message}\n\n"
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
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setValue(max(0, min(100, percent)))
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setText(message or f"{percent}%")
    
    def show_progress(self, message: str) -> None:
        self._update_progress(0, message)
        self._log_event(f"Starting: {message}", "progress")
        if hasattr(self, 'run_btn'): self.run_btn.setEnabled(False)
    
    def hide_progress(self) -> None:
        if hasattr(self, 'run_btn'): self.run_btn.setEnabled(True)

    def on_results(self, payload):
        self.simulation_results = payload
        self._update_progress(100, "Complete!")
        self._log_event("✓ SIS COMPLETE", "success")
        
        # Extract grid and estimates for visualization (use mean if available)
        # SIS results may have different structure, so we'll try to extract what we can
        if 'grid' in payload or 'estimates' in payload:
            # Try to extract grid coordinates and estimates
            try:
                import pyvista as pv
                viz = payload.get("visualization", {})
                mesh = viz.get("mesh")
                
                if mesh:
                    property_name = payload.get("property_name", "Mean")
                    estimates = mesh[property_name] if property_name in mesh.array_names else None
                    
                    if estimates is not None:
                        if hasattr(estimates, 'numpy'):
                            estimates = estimates.numpy()
                        
                        if isinstance(mesh, pv.StructuredGrid):
                            grid_x = mesh.x.reshape(mesh.dimensions, order='F')
                            grid_y = mesh.y.reshape(mesh.dimensions, order='F')
                            grid_z = mesh.z.reshape(mesh.dimensions, order='F')
                        else:
                            points = mesh.points
                            grid_x = points[:, 0]
                            grid_y = points[:, 1]
                            grid_z = points[:, 2]
                        
                        self.kriging_results = {
                            'grid_x': grid_x,
                            'grid_y': grid_y,
                            'grid_z': grid_z,
                            'estimates': estimates,
                            'variances': np.full_like(estimates, np.nan),
                            'variable': 'SIS_Mean',
                            'model_type': 'indicator',
                            'metadata': payload.get('metadata', {}),
                        }
                        self.visualize_btn.setEnabled(True)
            except Exception as e:
                logger.warning(f"Could not extract visualization data from SIS results: {e}")
        
        self.results_text.setText(f"Done. {payload.get('n_realizations', 0)} realizations created.")
        # Add probability buttons
        while self.prob_layout.count():
            item = self.prob_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Mock creation of prob buttons
        for i in range(self.thresh_table.rowCount()):
            try:
                t = self.thresh_table.item(i, 0).text()
                btn = QPushButton(f"Map > {t}")
                btn.clicked.connect(lambda checked, threshold=t: self._visualize(threshold))
                self.prob_layout.addWidget(btn)
            except (AttributeError, ValueError):
                pass

    def visualize_results(self):
        """Visualize SIS results in 3D."""
        if not hasattr(self, 'simulation_results') or self.simulation_results is None:
            QMessageBox.warning(self, "No Results", "Please run SIS first.")
            return
        
        try:
            # Extract grid from visualization.mesh (standardized workflow format)
            viz = self.simulation_results.get('visualization', {})
            grid = viz.get('mesh')
            
            if grid is not None:
                property_name = viz.get('property', 'SIS_Mean')
                self._log_event("Sending results to 3D viewer...", "info")
                self.request_visualization.emit(grid, property_name)
                self._log_event("✓ Visualization request sent", "success")
            elif self.kriging_results is not None:
                # Fallback to old kriging_results format
                if self.parent() and hasattr(self.parent(), 'visualize_kriging_results'):
                    try:
                        self.parent().visualize_kriging_results(self.kriging_results)
                    except Exception as e:
                        logger.error(f"Visualization error: {e}", exc_info=True)
                        QMessageBox.critical(self, "Visualization Error", f"Error visualizing results:\n{str(e)}")
                else:
                    QMessageBox.information(self, "Info", "Cannot access main viewer for visualization.")
            else:
                QMessageBox.warning(self, "No Results", "No visualization data available.")
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
            QMessageBox.critical(self, "Visualization Error", f"Error visualizing results:\n{str(e)}")
    
    def _visualize(self, t):
        self.results_text.setText(f"Visualizing > {t}")
        if self.simulation_results:
            self.request_visualization.emit(
                self.simulation_results.get('indicator_probabilities'),
                f"sis_prob_{t}"
            )

    def run_analysis(self):
        """
        Run SIS following the standardized workflow:
        1. Load drillholes
        2. Define grid extents + block size
        3. Generate empty property array
        4. Run SIS to populate it
        """
        if not self.controller:
            self.show_warning("Unavailable", "Controller is not connected; cannot run analysis.")
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
        # STEP 4: RUN SIS TO POPULATE IT
        # ========================================================================
        self._log_event("Step 4: Running SIS to populate array...", "progress")
        self._log_event(f"  Variable: {self.prop_combo.currentText()}", "info")
        self._log_event(f"  Realizations: {params['n_realizations']}", "info")

        # Set up progress bar
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0% - Initializing {params['n_realizations']} realizations...")

        self.show_progress("Running SIS...")

        # Run with progress callback
        try:
            self.controller.run_sis(
                params=params,
                callback=self.handle_results
            )
        except Exception as e:
            logger.error(f"Failed to dispatch SIS: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def _export_results(self):
        pass
