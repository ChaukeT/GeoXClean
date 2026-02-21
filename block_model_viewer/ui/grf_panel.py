"""
Gaussian Random Fields (GRF) Panel
==================================

Refactored for Modern UX/UI.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import pandas as pd
from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QLabel, QMessageBox, QWidget, QSplitter, QScrollArea, QFrame, QCheckBox,
    QProgressBar, QTextEdit
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QDateTime
from .base_analysis_panel import BaseAnalysisPanel
from ..utils.coordinate_utils import ensure_xyz_columns

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class GRFPanel(BaseAnalysisPanel):
    task_name = "grf"
    request_visualization = pyqtSignal(object, str)

    # PanelManager metadata
    PANEL_ID = "GRFPanel"
    PANEL_NAME = "GRF Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    def __init__(self, parent=None):
        self.drillhole_data = None
        self.variogram_results = None  # Store loaded variogram
        super().__init__(parent=parent, panel_id="grf")
        self.setWindowTitle("Gaussian Random Fields")
        self.resize(900, 700)
        
        # Build UI (required when using _build_ui pattern)
        self._build_ui()

        self._init_registry()

    def bind_controller(self, controller):
        """Bind controller and connect to task_progress signal for progress updates."""
        super().bind_controller(controller)
        if controller and hasattr(controller, 'signals'):
            controller.signals.task_progress.connect(self._handle_task_progress)

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _build_ui(self):
        """Build custom UI. Called by base class."""
        self._setup_ui()

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                # FIX: Check if signals are available before connecting
                dh_signal = self.registry.drillholeDataLoaded
                if dh_signal is not None:
                    dh_signal.connect(self._on_data_loaded)
                    logger.debug("GRFPanel: Connected to drillholeDataLoaded signal")
                
                vario_signal = self.registry.variogramResultsLoaded
                if vario_signal is not None:
                    vario_signal.connect(self._on_vario_loaded)
                    logger.debug("GRFPanel: Connected to variogramResultsLoaded signal")
                
                d = self.registry.get_drillhole_data()
                if d is not None:
                    self._on_data_loaded(d)
                v = self.registry.get_variogram_results()
                if v is not None:
                    self._on_vario_loaded(v)
        except Exception:
            pass
    
    def _on_vario_loaded(self, res):
        """Handle variogram results loaded from registry."""
        self.variogram_results = res
        self._apply_variogram_results(res)
    
    def _apply_variogram_results(self, res):
        """Apply variogram results to UI fields (covariance parameters)."""
        if not res:
            return
        
        # Check if UI is ready
        if not (hasattr(self, 'cov_type') and hasattr(self, 'rx') and 
                hasattr(self, 'ry') and hasattr(self, 'rz')):
            return
        
        # Extract fitted parameters
        fitted = res.get('fitted_models', {})
        if 'omni' in fitted:
            p = fitted['omni']
        elif fitted:
            p = list(fitted.values())[0]
        else:
            p = res
        
        # Set covariance type (model type maps to covariance)
        model_type = p.get('model_type', 'spherical').capitalize()
        idx = self.cov_type.findText(model_type)
        if idx >= 0:
            self.cov_type.setCurrentIndex(idx)
        
        # Set ranges
        omni_range = p.get('range', 100)
        
        # Try directional ranges first
        if 'horizontal' in fitted:
            h_range = fitted['horizontal'].get('range', omni_range)
        else:
            h_range = p.get('range_major', omni_range)
        
        if 'vertical' in fitted:
            v_range = fitted['vertical'].get('range', omni_range * 0.5)
        else:
            v_range = p.get('range_vert', omni_range * 0.5)
        
        self.rx.setValue(h_range)
        self.ry.setValue(h_range)
        self.rz.setValue(v_range)
        
        self._log_event(f"✓ Loaded covariance: {model_type}, ranges=({h_range:.1f}, {h_range:.1f}, {v_range:.1f})", "success")
    
    def _load_from_variogram(self):
        """Load variogram/covariance parameters from registry or cached results."""
        # Try to get from parent's variogram dialog first
        mw = self.parent()
        if hasattr(mw, 'variogram_dialog') and mw.variogram_dialog:
            if hasattr(mw.variogram_dialog, 'get_variogram_results'):
                res = mw.variogram_dialog.get_variogram_results()
            else:
                res = getattr(mw.variogram_dialog, 'variogram_results', None)
            if res:
                self._apply_variogram_results(res)
                return
        
        # Fall back to registry
        if self.registry:
            res = self.registry.get_variogram_results()
            if res:
                self._apply_variogram_results(res)
                return
        
        # Fall back to cached results
        if self.variogram_results:
            self._apply_variogram_results(self.variogram_results)
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

        # LEFT
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)
        
        self._create_cov_group(s_lay)
        self._create_grid_group(s_lay)
        self._create_sim_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # RIGHT
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Progress section
        progress_box = QGroupBox("Progress")
        progress_box.setStyleSheet("QGroupBox { font-weight: bold; color: #26c6da; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        progress_layout = QVBoxLayout(progress_box)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid #555; border-radius: 3px; background: {ModernColors.CARD_BG}; height: 20px; text-align: center; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #26c6da, stop:1 #4CAF50); border-radius: 3px; }}
        """)
        progress_layout.addWidget(self.progress_bar)
        r_lay.addWidget(progress_box)
        
        # Results box
        r_box = QGroupBox("Results")
        r_box.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        r_l = QVBoxLayout(r_box)
        self.res_lbl = QLabel("Configured for FFT/Cholesky simulation.")
        self.res_lbl.setWordWrap(True)
        self.res_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        r_l.addWidget(self.res_lbl)
        r_lay.addWidget(r_box)
        
        # Event log
        log_box = QGroupBox("Event Log")
        log_box.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        log_layout = QVBoxLayout(log_box)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setStyleSheet(f"QTextEdit {{ background: {ModernColors.PANEL_BG}; border: 1px solid #444; font-family: monospace; font-size: 11px; }}")
        log_layout.addWidget(self.results_text)
        r_lay.addWidget(log_box)
        
        self.run_btn = QPushButton("RUN GRF")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)
        r_lay.addWidget(self.run_btn)
        
        # Visualization button
        self.vis_btn = QPushButton("VISUALIZE MEAN")
        self.vis_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.vis_btn.setEnabled(False)
        self.vis_btn.clicked.connect(self._visualize_results)
        r_lay.addWidget(self.vis_btn)
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_cov_group(self, layout):
        g = QGroupBox("1. Covariance")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        
        # Load from Registry button
        load_btn = QPushButton("Load from Registry")
        load_btn.setStyleSheet("background-color: #7b1fa2; color: white;")
        load_btn.clicked.connect(self._load_from_variogram)
        l.addWidget(load_btn)
        
        self.cov_type = QComboBox()
        self.cov_type.addItems(["Spherical", "Exponential", "Matern"])
        l.addWidget(QLabel("Type:"))
        l.addWidget(self.cov_type)
        
        h = QHBoxLayout()
        self.rx = QDoubleSpinBox()
        self.rx.setRange(1, 10000)
        self.rx.setValue(100)
        self.ry = QDoubleSpinBox()
        self.ry.setRange(1, 10000)
        self.ry.setValue(100)
        self.rz = QDoubleSpinBox()
        self.rz.setRange(1, 10000)
        self.rz.setValue(50)
        h.addWidget(QLabel("Rx:"))
        h.addWidget(self.rx)
        h.addWidget(QLabel("Ry:"))
        h.addWidget(self.ry)
        h.addWidget(QLabel("Rz:"))
        h.addWidget(self.rz)
        l.addLayout(h)
        layout.addWidget(g)

    def _create_grid_group(self, layout):
        """Create Grid & Block Size configuration group (same as SGSIM)."""
        import numpy as np
        g = QGroupBox("2. Grid & Block Size")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
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
        self.nx = QSpinBox()
        self.nx.setRange(1, 1000)
        self.nx.setValue(100)
        self.ny = QSpinBox()
        self.ny.setRange(1, 1000)
        self.ny.setValue(100)
        self.nz = QSpinBox()
        self.nz.setRange(1, 1000)
        self.nz.setValue(20)
        h1.addWidget(QLabel("NX:"))
        h1.addWidget(self.nx)
        h1.addWidget(QLabel("NY:"))
        h1.addWidget(self.ny)
        h1.addWidget(QLabel("NZ:"))
        h1.addWidget(self.nz)
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
        self.nx.setValue(nx)
        self.ny.setValue(ny)
        self.nz.setValue(nz)
        
        self._log_event(f"✓ Grid: {nx}×{ny}×{nz}, origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f})", "success")

    def _create_sim_group(self, layout):
        g = QGroupBox("3. Simulation")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        self.method = QComboBox()
        self.method.addItems(["FFT", "Cholesky"])
        self.cond = QCheckBox("Condition")
        self.cond.setChecked(True)
        self.reals = QSpinBox()
        self.reals.setValue(100)
        f.addRow("Method:", self.method)
        f.addRow("Realizations:", self.reals)
        f.addRow("", self.cond)
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
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                df = assays
        elif isinstance(data, pd.DataFrame):
            df = data
        
        if df is not None and not df.empty:
            self.drillhole_data = ensure_xyz_columns(df)

    def gather_parameters(self) -> Dict[str, Any]:
        return {
            'drillhole_data': self.drillhole_data,
            'covariance_type': self.cov_type.currentText().lower(),
            'range_x': self.rx.value(),
            'range_y': self.ry.value(),
            'range_z': self.rz.value(),
            'grid_shape': (self.nz.value(), self.ny.value(), self.nx.value()),
            'method': self.method.currentText().lower(),
            'condition': self.cond.isChecked(),
            'n_realizations': self.reals.value(),
            # Grid & Block Size parameters (same as SGSIM)
            'nx': self.nx.value(),
            'ny': self.ny.value(),
            'nz': self.nz.value(),
            'xmin': self.xmin_spin.value(),
            'ymin': self.ymin_spin.value(),
            'zmin': self.zmin_spin.value(),
            'dx': self.dx_spin.value(),
            'dy': self.dy_spin.value(),
            'dz': self.dz_spin.value(),
        }

    def validate_inputs(self) -> bool:
        return True
    
    def _log_event(self, message: str, level: str = "info"):
        """Add timestamped message to event log."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        color_map = {"info": "#b0bec5", "success": "#81c784", "warning": "#ffb74d", "error": "#ef5350"}
        color = color_map.get(level, "#b0bec5")
        self.results_text.append(f'<span style="color:{color}">[{timestamp}] {message}</span>')
    
    def _handle_task_progress(self, task_name: str, percent: int, message: str):
        """Receive progress from the task system and route to UI update."""
        if task_name == self.task_name:
            self._update_progress(percent, message)

    def _update_progress(self, percent: int, message: str = ""):
        """Update progress bar and optionally log message."""
        self.progress_bar.setValue(percent)
        if message:
            # Check if message is simulation count (e.g., "25/50")
            if "/" in message and message.replace("/", "").replace(" ", "").isdigit():
                # This is a simulation count - show it prominently
                self.progress_bar.setFormat(f"Simulations: {message} ({percent}%)")
                self._log_event(f"Running simulation {message}...")
            else:
                # Regular message
                self.progress_bar.setFormat(f"{percent}% - {message}")
                self._log_event(message)
        else:
            self.progress_bar.setFormat(f"{percent}%")
        self.progress_bar.repaint()
    
    def run_analysis(self):
        """
        Run GRF following the standardized workflow:
        1. Load drillholes
        2. Define grid extents + block size
        3. Generate empty property array
        4. Run GRF to populate it
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

        # Gather parameters (includes drillhole loading)
        params = self.gather_parameters()

        # ========================================================================
        # STEP 2: DEFINE GRID EXTENTS + BLOCK SIZE
        # ========================================================================
        self._log_event("Step 2: Defining grid extents + block size...", "progress")
        self._log_event(f"  Grid: {params['nx']}×{params['ny']}×{params['nz']} blocks", "info")
        self._log_event(f"  Block size: {params['dx']:.1f}×{params['dy']:.1f}×{params['dz']:.1f}", "info")

        # ========================================================================
        # STEP 3: GENERATE EMPTY PROPERTY ARRAY
        # ========================================================================
        self._log_event("Step 3: Generating empty property array...", "progress")
        self._log_event(f"  Array shape: {params['n_realizations']}×{params['nz']}×{params['ny']}×{params['nx']}", "info")

        # ========================================================================
        # STEP 4: RUN GRF TO POPULATE IT
        # ========================================================================
        self._log_event("Step 4: Running GRF to populate array...", "progress")
        self._log_event(f"  Covariance: {self.cov_type.currentText()}", "info")
        self._log_event(f"  Realizations: {params['n_realizations']}", "info")

        # Set up progress bar
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0% - Initializing {params['n_realizations']} realizations...")

        self.show_progress("Running GRF...")

        # Run with progress callback
        try:
            self.controller.run_grf(
                params=params,
                callback=self.on_results
            )
        except Exception as e:
            logger.error(f"Failed to dispatch GRF: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def on_results(self, payload):
        self._update_progress(100, "GRF simulation complete!")
        self._log_event("✓ Simulation finished successfully", "success")
        self.res_lbl.setText("GRF Simulation Complete.")
        self.vis_btn.setEnabled(True)
        self._sim_results = payload
    
    def _visualize_results(self):
        """Visualize simulation results in main 3D viewer."""
        if not hasattr(self, '_sim_results') or self._sim_results is None:
            self._log_event("No results to visualize", "warning")
            return
        
        try:
            # Extract grid from visualization.mesh (standardized workflow format)
            viz = self._sim_results.get('visualization', {})
            grid = viz.get('mesh') or self._sim_results.get('grid')
            
            if grid is not None:
                property_name = viz.get('property', 'GRF_Mean')
                self._log_event("Sending results to 3D viewer...", "info")
                self.request_visualization.emit(grid, property_name)
                self._log_event("✓ Visualization request sent", "success")
            else:
                self._log_event("No grid data in results", "warning")
        except Exception as e:
            self._log_event(f"Visualization error: {e}", "error")
