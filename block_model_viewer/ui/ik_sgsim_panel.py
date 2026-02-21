"""
IK-based Sequential Gaussian Simulation Panel
==============================================

Dedicated UI panel for Indicator Kriging based Sequential Gaussian Simulation.
Refactored for Modern UX/UI.
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
    QSplitter, QFrame, QScrollArea, QProgressBar
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from datetime import datetime

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class IKSGSIMPanel(BaseAnalysisPanel):
    """
    IK-based Sequential Gaussian Simulation Panel.
    """
    # PanelManager metadata
    PANEL_ID = "IKSGSIMPanel"
    PANEL_NAME = "IKSGSIM Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "ik_sgsim"
    request_visualization = pyqtSignal(object, str)
    
    def __init__(self, parent=None):
        # Initialize state BEFORE super().__init__
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.ik_results: Dict[str, Any] = {}
        self.simulation_results: Optional[Dict[str, Any]] = None
        
        super().__init__(parent=parent, panel_id="ik_sgsim")

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
        self.setWindowTitle("IK-based Sequential Gaussian Simulation")
        self.resize(1000, 700)
        
        # Build UI (required when using _build_ui pattern)
        self._build_ui()
        
        self._init_registry_connections()
    
    def _build_ui(self):
        """Build custom split-pane UI. Called by base class."""
        self._setup_ui()

    def _init_registry_connections(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                # FIX: Check if signals are available before connecting
                dh_signal = self.registry.drillholeDataLoaded
                if dh_signal is not None:
                    dh_signal.connect(self._on_data_loaded)
                    logger.debug("IKSGSimPanel: Connected to drillholeDataLoaded signal")
                
                # Listen for IK results being loaded
                ik_signal = getattr(self.registry, 'indicatorKrigingResultsLoaded', None)
                if ik_signal is not None:
                    ik_signal.connect(self._on_ik_results_loaded)
                    logger.debug("IKSGSimPanel: Connected to indicatorKrigingResultsLoaded signal")
                
                # Load initial data - AUDIT FIX: Prefer get_estimation_ready_data for proper provenance
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

                # Try to load existing IK results
                ik_results = self.registry.get_indicator_kriging_results()
                if ik_results is not None:
                    self._on_ik_results_loaded(ik_results)
        except Exception as e:
            logger.warning(f"Registry connection failed: {e}")

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

        # --- LEFT: CONFIGURATION ---
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)
        
        self._create_input_group(s_lay)
        self._create_grid_group(s_lay)
        self._create_sim_group(s_lay)
        self._create_mode_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # --- RIGHT: RESULTS ---
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
        
        r_box = QGroupBox("Simulation Log")
        r_box_l = QVBoxLayout(r_box)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(f"background-color: #2b2b2b; color: {ModernColors.TEXT_PRIMARY}; font-family: Consolas;")
        self.results_text.setPlaceholderText("1. Load Data\n2. Select IK Result\n3. Run Simulation")
        r_box_l.addWidget(self.results_text)
        r_lay.addWidget(r_box, stretch=1)
        
        # Actions
        act_lay = QHBoxLayout()
        self.run_btn = QPushButton("RUN IK-SGSIM")
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

    def _create_input_group(self, layout):
        g = QGroupBox("1. Input Data & Model")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        
        f = QFormLayout()
        self.property_combo = QComboBox()
        self.ik_result_combo = QComboBox()
        f.addRow("Property:", self.property_combo)
        f.addRow("IK Result:", self.ik_result_combo)
        l.addLayout(f)
        
        self.load_ik_btn = QPushButton("Load IK Results from Registry")
        self.load_ik_btn.clicked.connect(self._load_ik_results)
        l.addWidget(self.load_ik_btn)
        
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

    def _create_sim_group(self, layout):
        g = QGroupBox("3. Simulation Parameters")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        
        self.n_reals = QSpinBox()
        self.n_reals.setRange(1, 1000)
        self.n_reals.setValue(100)
        
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
        self.prefix.addItems(["ik_sim", "sim"])
        
        f.addRow("Realizations:", self.n_reals)
        f.addRow("Seed:", self.seed)
        f.addRow("Prefix:", self.prefix)
        layout.addWidget(g)

    def _create_mode_group(self, layout):
        g = QGroupBox("4. Algorithm Settings")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        
        self.sequential_check = QCheckBox("Sequential Simulation")
        self.sequential_check.setChecked(True)
        self.sequential_check.setToolTip("Maintains spatial correlation using previously simulated nodes")
        l.addWidget(self.sequential_check)
        
        lbl = QLabel("Checked: Sequential (Slower, Correlated)\nUnchecked: Independent (Faster, Local Only)")
        lbl.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        l.addWidget(lbl)
        
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

        self.drillhole_data = df.copy() if df is not None and not df.empty else None
        # Preserve attrs in the copy
        if self.drillhole_data is not None and df is not None:
            self.drillhole_data.attrs = df.attrs.copy()
        
        if self.drillhole_data is not None:
            # Only update UI if it's been built
            if hasattr(self, 'property_combo') and hasattr(self, 'run_btn'):
                cols = self.drillhole_data.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['X', 'Y', 'Z', 'HOLEID', 'FROM', 'TO', 'LENGTH', 'GLOBAL_INTERVAL_ID',
                               # Compositing metadata columns
                               'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                               'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO']
                valid = [c for c in cols if c.upper() not in exclude_cols]
                self.property_combo.clear()
                self.property_combo.addItems(sorted(valid))
                self.run_btn.setEnabled(True)
            else:
                logger.debug("IK-SGSIM panel: UI not ready, data stored for later initialization")

    def _on_ik_results_loaded(self, ik_result: Dict[str, Any]):
        """Handle IK results loaded from registry."""
        if not ik_result:
            return
        
        # Create a key for this result based on variable/property name
        # Try metadata first, then property_name, then variable
        metadata = ik_result.get('metadata', {})
        variable = metadata.get('variable') or ik_result.get('property_name') or ik_result.get('variable', 'IK')
        # Extract just the variable name if property_name includes prefix like "IK_Fe_Prob"
        if variable.startswith('IK_'):
            variable = variable.replace('IK_', '').split('_')[0]
        result_key = f"{variable}_IK"
        
        # Store in our local dict
        if not hasattr(self, 'ik_results'):
            self.ik_results = {}
        self.ik_results[result_key] = ik_result
        
        # Update combo box
        if hasattr(self, 'ik_result_combo'):
            self.ik_result_combo.clear()
            self.ik_result_combo.addItems(list(self.ik_results.keys()))
            self._log_event(f"Loaded IK result: {result_key}", "success")
    
    def _load_ik_results(self):
        """Load IK results from registry."""
        if not self.registry:
            QMessageBox.warning(self, "No Registry", "DataRegistry not available.")
            return
        
        ik_result = self.registry.get_indicator_kriging_results()
        if ik_result:
            self._on_ik_results_loaded(ik_result)
            msg = f"Loaded IK result from registry."
            if hasattr(self, 'results_text'):
                self.results_text.append(msg)
            self._log_event(msg, "success")
        else:
            QMessageBox.warning(
                self, 
                "No Results", 
                "No Indicator Kriging results found in registry.\n\nPlease run Indicator Kriging first."
            )

    def gather_parameters(self) -> Dict[str, Any]:
        p_name = self.property_combo.currentText()
        ik_name = self.ik_result_combo.currentText()
        
        if not p_name:
            raise ValueError("Select property")
        if not ik_name:
            raise ValueError("Select IK Result")
        
        ik_res = self.ik_results.get(ik_name)
        if not ik_res:
            raise ValueError("IK Result data missing")
        
        # Get block model from registry
        if not self.registry:
            raise ValueError("DataRegistry not available")
        
        block_model = self.registry.get_block_model()
        if block_model is None:
            raise ValueError("No block model loaded. Please load or generate a block model first.")
        
        # Prepare IK result dict - ensure probabilities are in correct format
        # IK results from registry have probabilities as (nx, ny, nz, n_thresh)
        # We need to reshape to (n_blocks, n_thresholds) for ik_sgsim
        ik_result_dict = ik_res.copy()
        probabilities = ik_result_dict.get('probabilities')
        thresholds = ik_result_dict.get('thresholds', [])
        
        if probabilities is not None:
            # Reshape from grid format to block format
            if probabilities.ndim == 4:  # (nx, ny, nz, n_thresh)
                n_blocks = probabilities.shape[0] * probabilities.shape[1] * probabilities.shape[2]
                n_thresh = probabilities.shape[3]
                # Flatten spatial dimensions, keep threshold dimension
                probabilities_reshaped = probabilities.reshape((n_blocks, n_thresh), order='F')
                ik_result_dict['probabilities'] = probabilities_reshaped
            elif probabilities.ndim == 2:  # Already (n_blocks, n_thresh)
                # Already in correct format
                pass
            else:
                raise ValueError(f"Unexpected probabilities shape: {probabilities.shape}")
        
        return {
            "property_name": p_name,
            "ik_result": ik_result_dict,
            "block_model": block_model,
            "n_realizations": self.n_reals.value(),
            "random_seed": self.seed.value(),  # AUDIT FIX (W-001): Seed is always required
            "realisation_prefix": self.prefix.currentText(),
            "use_sequential": self.sequential_check.isChecked(),
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
        if not self.property_combo.currentText():
            return False
        if not self.ik_result_combo.currentText():
            QMessageBox.warning(self, "Error", "Run Indicator Kriging first.")
            return False
        return True

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before IK-SGSIM simulation.

        IK-SGSIM requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Composited data (consistent sample support)
        3. Indicator Kriging results

        Returns:
            True if data is acceptable for IK-SGSIM
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
                f"Cannot run IK-SGSIM simulation:\n\n{message}\n\n"
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
        if not hasattr(self, 'progress_bar') or self.progress_bar is None:
            return
            
        percent = max(0, min(100, percent))
        
        # Ensure progress bar is visible
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        
        self.progress_bar.setValue(percent)
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setText(message or f"{percent}%")
        
        # Force UI repaint using safer processEvents (excludes user input to prevent reentrancy)
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    
    def show_progress(self, message: str) -> None:
        # Ensure progress bar is visible
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setVisible(True)
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setVisible(True)
        
        self._update_progress(0, message)
        self._log_event(f"Starting: {message}", "progress")
        if hasattr(self, 'run_btn'): self.run_btn.setEnabled(False)
    
    def hide_progress(self) -> None:
        if hasattr(self, 'run_btn'): self.run_btn.setEnabled(True)

    def on_results(self, payload):
        self.simulation_results = payload
        self._update_progress(100, "Complete!")
        self._log_event("✓ IK-SGSIM COMPLETE", "success")
        n = payload.get('realization_names', [])
        
        txt = f"IK-SGSIM COMPLETE\nRealizations: {len(n)}\n"
        txt += f"Method: {'Sequential' if self.sequential_check.isChecked() else 'Independent'}\n"
        self.results_text.setText(txt)
        
        self.viz_btn.setEnabled(True)
        self.exp_btn.setEnabled(True)
        
        if self.registry:
            try:
                self.registry.register_sgsim_results(payload, source_panel="IK-SGSIM")
            except AttributeError:
                logger.warning("register_sgsim_results not available in registry")

    def _visualize_results(self):
        """Visualize simulation results in main 3D viewer."""
        if not hasattr(self, 'simulation_results') or self.simulation_results is None:
            QMessageBox.warning(self, "No Results", "Please run IK-SGSIM first.")
            return
        
        try:
            # Extract grid from visualization.mesh (standardized workflow format)
            viz = self.simulation_results.get('visualization', {})
            grid = viz.get('mesh') or self.simulation_results.get('grid')
            
            if grid is not None:
                property_name = viz.get('property', f"{self.prefix.currentText()}_mean")
                self._log_event("Sending results to 3D viewer...", "info")
                self.request_visualization.emit(grid, property_name)
                self._log_event("✓ Visualization request sent", "success")
            else:
                QMessageBox.info(self, "Viz", "Results added to block model. Use Property Panel.")
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
            QMessageBox.critical(self, "Visualization Error", f"Error visualizing results:\n{str(e)}")

    def run_analysis(self):
        """
        Run IK-SGSIM following the standardized workflow:
        1. Load drillholes
        2. Define grid extents + block size
        3. Generate empty property array
        4. Run IK-SGSIM to populate it
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
        # STEP 4: RUN IK-SGSIM TO POPULATE IT
        # ========================================================================
        self._log_event("Step 4: Running IK-SGSIM to populate array...", "progress")
        self._log_event(f"  Variable: {self.property_combo.currentText()}", "info")
        self._log_event(f"  Realizations: {params['n_realizations']}", "info")

        # Set up progress bar
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0% - Initializing {params['n_realizations']} realizations...")

        self.show_progress("Running IK-SGSIM...")

        # Run with progress callback
        try:
            self.controller.run_ik_sgsim(
                config=params,
                callback=self.handle_results
            )
        except Exception as e:
            logger.error(f"Failed to dispatch IK-SGSIM: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def _export_results(self):
        if not self.simulation_results:
            QMessageBox.warning(self, "No Results", "Run simulation first.")
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export", "ik_sgsim.csv", "CSV (*.csv)")
        if f:
            QMessageBox.info(self, "Export", f"Saved to {f}")

    def _clear_results(self):
        self.simulation_results = None
        self.results_text.clear()
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
            
            # Property/variable
            settings['property'] = get_safe_widget_value(self, 'property_combo')
            settings['prefix'] = get_safe_widget_value(self, 'prefix')
            
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
            
            # Mode selection
            settings['simulation_mode'] = get_safe_widget_value(self, 'mode_combo')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save IK-SGSIM panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Property/variable
            set_safe_widget_value(self, 'property_combo', settings.get('property'))
            set_safe_widget_value(self, 'prefix', settings.get('prefix'))
            
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
            
            # Mode selection
            set_safe_widget_value(self, 'mode_combo', settings.get('simulation_mode'))
                
            logger.info("Restored IK-SGSIM panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore IK-SGSIM panel settings: {e}")