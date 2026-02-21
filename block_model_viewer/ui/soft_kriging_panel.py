"""
Soft / Bayesian Kriging Panel (Threaded).

Features:
1. Threaded execution (Worker pattern) to prevent UI freezing.
2. Robust parameter gathering.
3. Clean modern Qt layout.
4. Validation Mode for testing data integrity before heavy computation.
"""

from __future__ import annotations

import logging
import traceback
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QTextEdit,
    QFileDialog, QLineEdit, QWidget, QSplitter, QScrollArea, 
    QFrame, QMessageBox, QProgressBar, QTabWidget, QDialog
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal

from ..utils.variable_utils import populate_variable_combo
from .base_analysis_panel import BaseAnalysisPanel
from ..geostats.soft_data import soft_from_csv, soft_from_ik_result  # Import optimized functions

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)

# Worker logic moved to GeostatsController._prepare_bayesian_kriging_payload
# VALIDATE mode kept as lightweight local operation

# --- MAIN PANEL ---
class SoftKrigingPanel(BaseAnalysisPanel):
    task_name = "soft_kriging"
    
    def __init__(self, parent=None):
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.kriging_results: Optional[Dict[str, Any]] = None

        super().__init__(parent=parent, panel_id="soft_kriging")

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
        self.setWindowTitle("Soft / Bayesian Kriging")
        self.resize(1100, 750)
        
        self._build_ui()
        self._init_registry()
    
    def _build_ui(self):
        """Constructs the UI cleanly."""
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tabs for Config vs Results
        self.tabs = QTabWidget()
        
        # --- TAB 1: CONFIGURATION ---
        config_tab = QWidget()
        c_layout = QVBoxLayout(config_tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        config_container = QWidget()
        config_layout = QVBoxLayout(config_container)
        
        self._create_method_group(config_layout)
        self._create_soft_data_group(config_layout)
        self._create_bayesian_group(config_layout)
        self._create_vario_group(config_layout)
        self._create_grid_group(config_layout)
        
        config_layout.addStretch()
        scroll.setWidget(config_container)
        c_layout.addWidget(scroll)
        
        # Validation Buttons
        h_btns = QHBoxLayout()
        
        self.btn_validate = QPushButton("Test / Validate Data")
        self.btn_validate.setStyleSheet("background-color: #FFA726; color: white;")
        self.btn_validate.clicked.connect(lambda: self._start_worker('VALIDATE'))
        
        self.run_btn = QPushButton("RUN KRIGING")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(lambda: self._start_worker('RUN'))

        self.view_table_btn = QPushButton("View Table")
        self.view_table_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.view_table_btn.clicked.connect(self.open_results_table)
        self.view_table_btn.setEnabled(False)

        h_btns.addWidget(self.btn_validate)
        h_btns.addWidget(self.run_btn)
        h_btns.addWidget(self.view_table_btn)
        c_layout.addLayout(h_btns)
        
        self.tabs.addTab(config_tab, "Configuration")
        
        # --- TAB 2: LOGS ---
        log_tab = QWidget()
        l_log = QVBoxLayout(log_tab)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("background-color: #222; color: #0F0; font-family: Monospace;")
        self.progress_bar = QProgressBar()
        l_log.addWidget(self.results_text)
        l_log.addWidget(self.progress_bar)
        self.tabs.addTab(log_tab, "Logs / Results")
        
        main_layout.addWidget(self.tabs)

    def _init_registry(self):
        """Connect to DataRegistry safely."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
                self.registry.variogramResultsLoaded.connect(self._on_variogram_results_loaded)

                # Initial load - AUDIT FIX: Prefer get_estimation_ready_data for proper provenance
                data = None
                try:
                    data = self.registry.get_estimation_ready_data(
                        prefer_declustered=True,
                        require_validation=False
                    )
                except (ValueError, AttributeError):
                    data = self.registry.get_drillhole_data()

                if data is not None:
                    self._on_data_loaded(data)

                # Check for existing variogram results
                vario = self.registry.get_variogram_results()
                if vario is not None:
                    self._on_variogram_results_loaded(vario)
        except ImportError:
            logger.warning("DataRegistry not found. Running in standalone mode.")
            self.registry = None

    def _create_method_group(self, layout):
        g = QGroupBox("1. Method & Variable")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        self.base_method_combo = QComboBox()
        self.base_method_combo.addItems(["Ordinary Kriging (OK)", "Universal Kriging (UK)"])
        self.variable_combo = QComboBox()
        f.addRow("Base Method:", self.base_method_combo)
        f.addRow("Primary Var:", self.variable_combo)
        layout.addWidget(g)

    def _create_soft_data_group(self, layout):
        g = QGroupBox("2. Soft Data Source")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        self.soft_source_combo = QComboBox()
        self.soft_source_combo.addItems(["From Block Model", "From External CSV"])
        self.soft_source_combo.currentTextChanged.connect(self._update_soft_ui)
        l.addWidget(self.soft_source_combo)
        
        self.csv_frame = QWidget()
        h = QHBoxLayout(self.csv_frame)
        h.setContentsMargins(0, 0, 0, 0)
        self.csv_path = QLineEdit()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)
        h.addWidget(self.csv_path)
        h.addWidget(self.browse_btn)
        l.addWidget(self.csv_frame)
        self.csv_frame.setVisible(False)
        layout.addWidget(g)
    
    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open Soft Data", "", "CSV (*.csv)")
        if f: 
            self.csv_path.setText(f)

    def _create_bayesian_group(self, layout):
        g = QGroupBox("3. Bayesian Config")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #90a4ae; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        self.prior_combo = QComboBox()
        self.prior_combo.addItems(["Mean & Variance", "Mean Only"])
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 1)
        self.weight_spin.setValue(0.5)
        f.addRow("Prior Mode:", self.prior_combo)
        f.addRow("Soft Confidence:", self.weight_spin)
        layout.addWidget(g)

    def _create_vario_group(self, layout):
        g = QGroupBox("4. Variogram Model")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        v = QVBoxLayout(g)

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
        v.addLayout(btn_layout)

        # Parameters
        h = QHBoxLayout()
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1, 10000)
        self.range_spin.setValue(100)
        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setValue(1.0)
        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setValue(0.1)

        for lbl, spin in [("Range:", self.range_spin), ("Sill:", self.sill_spin), ("Nugget:", self.nugget_spin)]:
            h.addWidget(QLabel(lbl))
            h.addWidget(spin)
        v.addLayout(h)

        layout.addWidget(g)

    def _create_grid_group(self, layout):
        g = QGroupBox("5. Block Grid")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ba68c8; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        h = QHBoxLayout(g)
        self.grid_x = QDoubleSpinBox()
        self.grid_x.setValue(10)
        self.grid_y = QDoubleSpinBox()
        self.grid_y.setValue(10)
        self.grid_z = QDoubleSpinBox()
        self.grid_z.setValue(5)
        
        h.addWidget(QLabel("DX:"))
        h.addWidget(self.grid_x)
        h.addWidget(QLabel("DY:"))
        h.addWidget(self.grid_y)
        h.addWidget(QLabel("DZ:"))
        h.addWidget(self.grid_z)
        layout.addWidget(g)

    def _update_soft_ui(self, txt):
        self.csv_frame.setVisible("CSV" in txt)

    def _on_data_loaded(self, data):
        # Handle dictionary wrapper or direct dataframe
        df = None
        if isinstance(data, dict):
            # Check for composites/assays (correct keys used by data registry)
            composites = data.get('composites')
            assays = data.get('assays')

            # Also check legacy keys for backwards compatibility
            if composites is None:
                composites = data.get('composites_df')
            if assays is None:
                assays = data.get('assays_df')

            # Prefer composites if available
            if isinstance(composites, pd.DataFrame) and not composites.empty:
                df = composites
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
            populate_variable_combo(self.variable_combo, df)
            logger.info(f"Soft Kriging: Loaded {len(df)} records")

    def _on_variogram_results_loaded(self, results):
        """Store variogram results and enable loading buttons."""
        self.variogram_results = results
        # Enable variogram loading buttons if they exist
        if hasattr(self, 'auto_vario_btn'):
            self.auto_vario_btn.setEnabled(results is not None)
        if hasattr(self, 'use_assisted_btn'):
            self.use_assisted_btn.setEnabled(results is not None)

    def load_variogram_parameters(self) -> bool:
        """Populate variogram fields from variogram results."""
        if not self.variogram_results:
            return False

        # Only update UI if it's been built
        required_attrs = ['range_spin', 'sill_spin', 'nugget_spin']
        if not all(hasattr(self, attr) for attr in required_attrs):
            logger.debug("Soft Kriging panel: UI not ready for variogram parameter loading")
            return False

        try:
            # Check for combined_3d_model first (preferred), then fallback to fitted_models
            model = self.variogram_results.get('combined_3d_model')
            if not model:
                # Fallback to directional fits
                fits = self.variogram_results.get('fitted_models', {})
                omni_dict = fits.get('omni', {})
                if omni_dict:
                    # Get first model from omni dict
                    model = next(iter(omni_dict.values())) if omni_dict else None
            
            if not model:
                QMessageBox.warning(self, "No Variogram", "No variogram model found in results. Please run Variogram 3D Analysis first.")
                return False

            model_type = model.get('model_type', '').lower()
            if model_type not in ['spherical', 'exponential', 'gaussian']:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Model type '{model_type}' not supported for soft kriging.")
                return False

            # Apply parameters
            nugget = model.get('nugget', 0.0)
            total_sill = model.get('sill', 0.0)
            partial_sill = total_sill - nugget
            range_val = model.get('major_range') or model.get('range', 100.0)

            self.range_spin.setValue(range_val)
            self.sill_spin.setValue(max(0.001, partial_sill))
            self.nugget_spin.setValue(nugget)

            logger.info(f"Soft Kriging: Loaded variogram parameters - {model_type}, "
                       f"range={range_val:.1f}, sill={partial_sill:.3f}, nugget={nugget:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error loading soft kriging variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load variogram parameters:\n{str(e)}")
            return False

    def _load_assisted_variogram(self):
        """Load variogram parameters from Variogram Assistant."""
        # Check if UI is ready
        required_attrs = ['range_spin', 'sill_spin', 'nugget_spin']
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

            if model_type not in ['spherical', 'exponential', 'gaussian']:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Assisted model type '{model_type}' not supported for soft kriging.")
                return

            # Apply parameters
            nugget = assisted_model.get('nugget', 0.0)
            sill = assisted_model.get('sill', 1.0)
            range_val = assisted_model.get('range', 100.0)

            self.range_spin.setValue(range_val)
            self.sill_spin.setValue(max(0.001, sill))
            self.nugget_spin.setValue(nugget)

            logger.info(f"Soft Kriging: Loaded assisted variogram - {model_type}, "
                       f"range={range_val:.1f}, sill={sill:.3f}, nugget={nugget:.3f}")

        except Exception as e:
            logger.error(f"Error loading assisted soft kriging variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load assisted variogram:\n{str(e)}")

    def gather_parameters(self) -> Dict[str, Any]:
        """Collects all UI state into a dictionary."""
        return {
            "data_df": self.drillhole_data,
            "variable": self.variable_combo.currentText(),
            "soft_source": self.soft_source_combo.currentText(),
            "soft_path": self.csv_path.text(),
            "bayesian": {
                "mode": self.prior_combo.currentText(),
                "weight": self.weight_spin.value()
            },
            "variogram": {
                "range": self.range_spin.value(),
                "sill": self.sill_spin.value(),
                "nugget": self.nugget_spin.value()
            },
            "grid": (self.grid_x.value(), self.grid_y.value(), self.grid_z.value())
        }

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Soft/Bayesian Kriging.

        Soft Kriging requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Validated data quality

        Returns:
            True if data is acceptable for Soft Kriging
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
                f"Cannot run Soft Kriging:\n\n{message}\n\n"
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

    def _start_worker(self, mode):
        """Starts analysis in VALIDATE or RUN mode."""
        # Gather Params
        params = self.gather_parameters()
        
        if params['soft_source'] == "From External CSV" and not params['soft_path']:
            QMessageBox.warning(self, "Error", "Please select a CSV file.")
            return

        # UI State
        self.tabs.setCurrentIndex(1)  # Switch to log tab
        self.results_text.clear()
        self.run_btn.setEnabled(False)
        self.btn_validate.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        if mode == 'VALIDATE':
            # VALIDATE is lightweight - run locally
            self._run_validation_local(params)
        else:
            # RUN uses controller.run_task()
            if not self.controller:
                QMessageBox.warning(self, "Error", "Controller not available.")
                self.run_btn.setEnabled(True)
                self.btn_validate.setEnabled(True)
                return

            # HARD GATE: Check data lineage before proceeding with RUN
            if not self._check_data_lineage():
                self.run_btn.setEnabled(True)
                self.btn_validate.setEnabled(True)
                return

            # Progress callback
            def progress_callback(percent: int, message: str):
                pct = max(0, min(100, int(percent)))
                self.progress_bar.setValue(pct)
                if message:
                    self.progress_bar.setFormat(f"{pct}% - {message}")
                else:
                    self.progress_bar.setFormat(f"{pct}%")
                self.results_text.append(message)
            
            # Run via controller
            self.controller.run_task(
                'soft_kriging',
                params,
                callback=self._on_kriging_complete,
                progress_callback=progress_callback
            )
    
    def _run_validation_local(self, params):
        """Run validation locally (lightweight operation)."""
        try:
            self.results_text.append("Starting VALIDATE process...")
            self.progress_bar.setValue(20)
            
            # Load soft data
            soft_data = None
            if params['soft_source'] == "From External CSV":
                self.results_text.append(f"Loading CSV: {params['soft_path']}...")
                t0 = time.time()
                soft_data = soft_from_csv(params['soft_path'])
                dt = time.time() - t0
                self.results_text.append(f"Loaded {soft_data.n_points:,} points in {dt:.2f}s")
            elif params['soft_source'] == "From Block Model":
                self.results_text.append("Block Model source not yet implemented.")
                return
            
            if not soft_data:
                raise ValueError("No Soft Data Loaded.")
            
            # Check Stats
            mean_val = float(np.mean(soft_data.means))
            var_val = float(np.mean(soft_data.variances))
            min_v, max_v = float(np.min(soft_data.means)), float(np.max(soft_data.means))
            
            report = (
                f"\n--- DATA VALIDATION REPORT ---\n"
                f"Count:      {soft_data.n_points:,}\n"
                f"Mean Grade: {mean_val:.4f}\n"
                f"Avg Var:    {var_val:.4f}\n"
                f"Range:      [{min_v:.4f}, {max_v:.4f}]\n"
                f"Memory:     {soft_data.coords.nbytes / 1e6:.1f} MB (Coords)\n"
            )
            
            # Check for negative variances
            neg_vars = np.sum(soft_data.variances < 0)
            if neg_vars > 0:
                report += f"\nWARNING: Found {neg_vars} negative variances! These will be clipped to 0."
            else:
                report += "\nVariance Integrity: OK"
            
            self.progress_bar.setValue(100)
            self.results_text.append(report)
            self.results_text.append("\nValidation Complete.")
            
        except Exception as e:
            self.results_text.append(f"\nERROR: {e}")
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.run_btn.setEnabled(True)
            self.btn_validate.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_kriging_complete(self, result: Dict[str, Any]):
        """Handle completion of kriging task."""
        self.run_btn.setEnabled(True)
        self.btn_validate.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result is None:
            self.results_text.append("\nERROR: Kriging returned no result.")
            QMessageBox.critical(self, "Error", "Kriging returned no result.")
            return

        if result.get("error"):
            error_msg = result["error"]
            self.results_text.append(f"\nERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
            return

        # Store results for visualization and table viewing
        self.kriging_results = result

        # Register results and block model to DataRegistry
        if self.registry:
            try:
                # Register soft kriging results
                self.registry.register_soft_kriging_results(result, source_panel="Soft Kriging")
                self.results_text.append("Results registered to data registry")

                # Also register the block model DataFrame for cross-sections and other panels
                grid_x = result.get('grid_x')
                grid_y = result.get('grid_y')
                grid_z = result.get('grid_z')
                estimates = result.get('estimates')
                variances = result.get('variances')

                if grid_x is not None and estimates is not None:
                    coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                    block_df = pd.DataFrame({
                        'X': coords[:, 0],
                        'Y': coords[:, 1],
                        'Z': coords[:, 2],
                        f'{result.get("variable", "soft")}_est': estimates.ravel(),
                        f'{result.get("variable", "soft")}_var': variances.ravel() if variances is not None else np.full_like(estimates.ravel(), np.nan)
                    }).dropna()

                    # Register the block model
                    self.registry.register_block_model_generated(
                        block_df,
                        source_panel="Soft Kriging",
                        metadata={
                            'variable': result.get('variable', 'unknown'),
                            'method': 'soft_kriging',
                            'grid_size': (len(np.unique(grid_x)), len(np.unique(grid_y)), len(np.unique(grid_z))),
                            'n_blocks': len(block_df)
                        }
                    )
                    self.results_text.append("Block model registered to data registry")
            except Exception as e:
                logger.warning(f"Failed to register soft kriging results/block model: {e}")
                self.results_text.append(f"Warning: Failed to register results: {e}")

        # Enable view table button if it exists and we have grid results
        if hasattr(self, 'view_table_btn') and 'grid_x' in result:
            self.view_table_btn.setEnabled(True)

        self.results_text.append("\nDone.")
        QMessageBox.information(self, "Success", "Operation completed successfully.")
    
    def start_analysis(self):
        """Legacy method for compatibility - redirects to RUN mode."""
        self._start_worker('RUN')

    def on_worker_finished(self, result):
        """Legacy method for compatibility."""
        self._on_finished(result)

    def on_worker_error(self, msg):
        """Legacy method for compatibility."""
        self._on_error(msg)
    
    def open_results_table(self):
        """Open Soft/Bayesian Kriging results as a table."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run kriging first.")
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

            title = f"Soft/Bayesian Kriging Results - {variable}"

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

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Method and variable
            settings['base_method'] = get_safe_widget_value(self, 'base_method_combo')
            settings['variable'] = get_safe_widget_value(self, 'variable_combo')
            
            # Soft data source
            settings['soft_source'] = get_safe_widget_value(self, 'soft_source_combo')
            settings['soft_file_path'] = get_safe_widget_value(self, 'soft_file_edit')
            
            # Bayesian prior settings
            settings['prior_type'] = get_safe_widget_value(self, 'prior_combo')
            settings['prior_mean'] = get_safe_widget_value(self, 'prior_mean_spin')
            settings['prior_variance'] = get_safe_widget_value(self, 'prior_var_spin')
            
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
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save soft kriging panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Method and variable
            set_safe_widget_value(self, 'base_method_combo', settings.get('base_method'))
            set_safe_widget_value(self, 'variable_combo', settings.get('variable'))
            
            # Soft data source
            set_safe_widget_value(self, 'soft_source_combo', settings.get('soft_source'))
            set_safe_widget_value(self, 'soft_file_edit', settings.get('soft_file_path'))
            
            # Bayesian prior settings
            set_safe_widget_value(self, 'prior_combo', settings.get('prior_type'))
            set_safe_widget_value(self, 'prior_mean_spin', settings.get('prior_mean'))
            set_safe_widget_value(self, 'prior_var_spin', settings.get('prior_variance'))
            
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
                
            logger.info("Restored soft kriging panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore soft kriging panel settings: {e}")