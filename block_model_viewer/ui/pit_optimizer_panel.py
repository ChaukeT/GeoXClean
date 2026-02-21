"""
Pit Optimizer Panel

===================

UI for running Lerchs-Grossmann optimization.
Uses Threading to prevent GUI freeze during MaxFlow calculation.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QComboBox, QPushButton, QLabel,
    QMessageBox, QWidget, QProgressBar, QTextEdit, QSpinBox
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal

from .base_analysis_panel import BaseAnalysisPanel
from ..models.pit_optimizer import is_fast_solver_available

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)

# Worker logic moved to MiningController._prepare_pit_optimisation_payload
# This ensures pure computation with no access to DataRegistry or Qt objects

class PitOptimizerPanel(BaseAnalysisPanel):
    task_name = "pit_opt"
    request_visualization = pyqtSignal(object, str) # Signals to Main Window
    
    # Columns that indicate block model has been estimated (kriged/simulated)
    ESTIMATION_INDICATORS = [
        'ok_', 'uk_', 'cok_', 'ik_', 'sgsim_', 'sis_',
        'KRIGING_', 'EST_', 'ESTIMATED', 'est_'
    ]
    
    # Recognized classification column names
    CLASSIFICATION_COLUMNS = ['CLASSIFICATION', 'CLASS', 'CLASS_FINAL', 'Category', 'RESOURCE_CLASS']
    
    def __init__(self, parent=None):
        self._block_model = None  # Use _block_model instead of block_model property
        self.result_df = None
        self._is_estimated = False
        self._is_classified = False
        self._classification_column = None
        
        super().__init__(parent=parent, panel_id="pit_opt")

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
        self.setWindowTitle("Pit Optimization (Lerchs-Grossmann)")
        self.resize(1000, 700)
    
    @property
    def block_model(self) -> Optional[pd.DataFrame]:
        """Read-only property to access the block model (MP-019 fix)."""
        return self._block_model
        
    def _init_registry(self):
        """Connect to DataRegistry to get block model updates."""
        try:
            registry = self.get_registry()
            if registry:
                registry.blockModelLoaded.connect(self._on_block_model_loaded)
                registry.blockModelGenerated.connect(self._on_block_model_loaded)
                registry.blockModelClassified.connect(self._on_block_model_loaded)
                
                # Prefer classified block model when available.
                existing = registry.get_classified_block_model()
                if existing is None:
                    existing = registry.get_block_model()
                if existing is not None:
                    self._on_block_model_loaded(existing)
        except ImportError:
            logger.warning("DataRegistry not found. Running in standalone mode.")

    def _on_block_model_loaded(self, block_model):
        """Handle block model loaded from registry."""
        try:
            if hasattr(block_model, 'to_dataframe'):
                self._block_model = block_model.to_dataframe()  # Use _block_model
            elif isinstance(block_model, pd.DataFrame):
                self._block_model = block_model  # Use _block_model
            else:
                logger.warning(f"Unknown block model type: {type(block_model)}")
                return
            
            # Invalidate any previous results when block model changes
            self.result_df = None
            if hasattr(self, 'btn_viz'):
                self.btn_viz.setEnabled(False)
            
            # Check for estimation indicators (MP-003 fix)
            self._is_estimated = self._check_estimation_status()
            
            # Check for classification (MP-001 fix)
            self._is_classified, self._classification_column = self._check_classification_status()
            
            # Update grade column combo - use _block_model to avoid property issues (MP-019 fix)
            if hasattr(self, 'combo_grade'):
                self.combo_grade.clear()
                # Find potential grade columns - prefer estimated grades
                estimated_cols = [col for col in self._block_model.columns 
                                  if any(col.lower().startswith(prefix.lower()) for prefix in self.ESTIMATION_INDICATORS)]
                raw_grade_cols = [col for col in self._block_model.columns 
                                  if any(term in col.upper() for term in ['GRADE', 'G_', 'FE', 'CU', 'AU', 'AG', 'ZN', 'PB'])]
                
                if estimated_cols:
                    # Prioritize estimated grades
                    self.combo_grade.addItems(estimated_cols)
                    self.combo_grade.setCurrentIndex(0)
                    logger.info(f"Found estimated grade columns: {estimated_cols}")
                elif raw_grade_cols:
                    self.combo_grade.addItems(raw_grade_cols)
                    self.combo_grade.setCurrentIndex(0)
                    logger.warning("No estimated grades found - using raw grades (not recommended)")
                else:
                    # Add all numeric columns as candidates
                    numeric_cols = self._block_model.select_dtypes(include=[np.number]).columns.tolist()
                    self.combo_grade.addItems(numeric_cols[:10])  # Limit to first 10
            
            # Update UI state based on readiness
            self._update_readiness_status()
            
            logger.info(f"Block model loaded: {len(self._block_model)} blocks, "
                       f"estimated={self._is_estimated}, classified={self._is_classified}")
        except Exception as e:
            logger.error(f"Error loading block model: {e}", exc_info=True)
    
    def _check_estimation_status(self) -> bool:
        """Check if block model has been estimated (kriged/simulated)."""
        if self._block_model is None:
            return False
        
        for col in self._block_model.columns:
            col_lower = col.lower()
            for indicator in self.ESTIMATION_INDICATORS:
                if col_lower.startswith(indicator.lower()):
                    logger.info(f"Estimation detected: column '{col}' matches indicator '{indicator}'")
                    return True
        return False
    
    def _check_classification_status(self) -> tuple:
        """Check if block model has been classified.
        
        Returns:
            Tuple of (is_classified: bool, classification_column: str or None)
        """
        if self._block_model is None:
            return False, None
        
        for col_name in self.CLASSIFICATION_COLUMNS:
            if col_name in self._block_model.columns:
                logger.info(f"Classification detected: column '{col_name}'")
                return True, col_name
            # Case-insensitive check
            for col in self._block_model.columns:
                if col.upper() == col_name.upper():
                    logger.info(f"Classification detected: column '{col}'")
                    return True, col
        return False, None
    
    def _update_readiness_status(self):
        """Update UI to reflect block model readiness for optimization."""
        if not hasattr(self, 'log_text'):
            return
        
        status_msgs = []
        if self._block_model is not None:
            status_msgs.append(f"Block model: {len(self._block_model)} blocks")
            
            if self._is_estimated:
                status_msgs.append("✓ Grades are estimated (kriged/simulated)")
            else:
                status_msgs.append("⚠ WARNING: No estimated grades detected - using raw grades violates support assumptions")
            
            if self._is_classified:
                status_msgs.append(f"✓ Classification found: {self._classification_column}")
            else:
                status_msgs.append("⚠ WARNING: Block model not classified - results may include Inferred resources")
        
        # Log status
        for msg in status_msgs:
            logger.info(msg)

    def _setup_ui(self):
        layout = self.main_layout
        
        # 1. Economics Group
        g_econ = QGroupBox("Economic Parameters")
        f_econ = QFormLayout(g_econ)
        
        self.spin_price = QDoubleSpinBox()
        self.spin_price.setRange(0, 10000)
        self.spin_price.setValue(1500)
        self.spin_price.setSuffix(" $/unit")
        
        self.spin_cost_m = QDoubleSpinBox()
        self.spin_cost_m.setRange(0, 1000)
        self.spin_cost_m.setValue(2.5)
        self.spin_cost_m.setSuffix(" $/t")
        
        self.spin_cost_p = QDoubleSpinBox()
        self.spin_cost_p.setRange(0, 1000)
        self.spin_cost_p.setValue(15.0)
        self.spin_cost_p.setSuffix(" $/t")
        
        self.spin_rec = QDoubleSpinBox()
        self.spin_rec.setRange(0, 1)
        self.spin_rec.setDecimals(3)
        self.spin_rec.setSingleStep(0.01)
        self.spin_rec.setValue(0.9)
        
        self.combo_grade = QComboBox()
        self.combo_grade.setEditable(True)
        self.combo_grade.addItem("GRADE")  # Default placeholder
        
        f_econ.addRow("Metal Price ($/unit):", self.spin_price)
        f_econ.addRow("Mining Cost ($/t):", self.spin_cost_m)
        f_econ.addRow("Processing Cost ($/t):", self.spin_cost_p)
        f_econ.addRow("Recovery (0-1):", self.spin_rec)
        f_econ.addRow("Grade Column:", self.combo_grade)
        
        layout.addWidget(g_econ)
        
        # 2. Geotech Group
        g_geo = QGroupBox("Geotechnical")
        f_geo = QFormLayout(g_geo)
        self.spin_slope = QSpinBox()
        self.spin_slope.setRange(1, 89)
        self.spin_slope.setValue(45)
        self.spin_slope.setEnabled(False)
        self.spin_slope.setSuffix("°")
        f_geo.addRow("Global Slope Angle:", self.spin_slope)
        f_geo.addRow(QLabel("<i>Current version supports 45° (1:1 block) slopes only.</i>"))
        
        layout.addWidget(g_geo)
        
        # 3. Actions
        h_btn = QHBoxLayout()
        self.btn_run = QPushButton("Run Optimizer")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self.run_optimization)
        
        self.btn_viz = QPushButton("Visualize Pit")
        self.btn_viz.setEnabled(False)
        self.btn_viz.clicked.connect(self._visualize)
        
        h_btn.addWidget(self.btn_run)
        h_btn.addWidget(self.btn_viz)
        h_btn.addStretch()
        layout.addLayout(h_btn)
        
        # 4. Progress Bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # 5. Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("background-color: #f5f5f5; font-family: monospace; font-size: 9pt;")
        layout.addWidget(self.log_text)

    def run_optimization(self):
        """Run pit optimization using controller.run_task() pipeline."""
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not available.")
            return
        
        if self._block_model is None:
            QMessageBox.warning(self, "Error", "No Block Model Loaded")
            return
        
        grade_col = self.combo_grade.currentText().strip()
        if not grade_col:
            QMessageBox.warning(self, "Error", "Please select a grade column")
            return
        
        # Validate economic parameters (MP-002 fix)
        price = self.spin_price.value()
        mining_cost = self.spin_cost_m.value()
        proc_cost = self.spin_cost_p.value()
        recovery = self.spin_rec.value()
        
        if price <= 0:
            QMessageBox.critical(
                self, "Invalid Parameters",
                "Metal price must be positive. Please enter a valid price."
            )
            return
        
        if recovery <= 0 or recovery > 1:
            QMessageBox.critical(
                self, "Invalid Parameters",
                "Recovery must be between 0 and 1 (e.g., 0.9 for 90%)."
            )
            return
        
        # MP-003: Warn if using raw (non-estimated) grades
        if not self._is_estimated:
            response = QMessageBox.warning(
                self, "Estimation Recommended",
                "Block model does not appear to have estimated grades (kriging/simulation).\n\n"
                "Using raw assay grades for pit optimization violates geostatistical support assumptions "
                "and may produce unreliable results.\n\n"
                "It is strongly recommended to run grade estimation before pit optimization.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        
        # MP-001: Warn if block model is not classified
        if not self._is_classified:
            response = QMessageBox.warning(
                self, "Classification Recommended",
                "Block model has not been classified (Measured/Indicated/Inferred).\n\n"
                "Pit optimization results may include Inferred resources which should not be "
                "used for mine planning per JORC/SAMREC/NI 43-101 guidelines.\n\n"
                "It is strongly recommended to classify resources before pit optimization.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log_text.clear()
        
        # Log readiness status
        if not self._is_estimated:
            self.log_text.append("<b style='color: orange;'>⚠ WARNING: Using non-estimated grades</b>")
        if not self._is_classified:
            self.log_text.append("<b style='color: orange;'>⚠ WARNING: Block model not classified</b>")
        
        self.log_text.append("Starting Lerchs-Grossmann Optimization...")
        self.log_text.append(f"  Price: ${price:.2f}/unit")
        self.log_text.append(f"  Mining Cost: ${mining_cost:.2f}/t")
        self.log_text.append(f"  Processing Cost: ${proc_cost:.2f}/t")
        self.log_text.append(f"  Recovery: {recovery*100:.1f}%")
        self.log_text.append(f"  Grade Column: {grade_col}")
        
        # Prepare params - copy block model to avoid mutation
        params = {
            'block_model': self._block_model.copy(),
            'price': price,
            'mining_cost': mining_cost,
            'proc_cost': proc_cost,
            'recovery': recovery,
            'grade_col': grade_col,
            'use_fast_solver': self.use_fast_solver_checkbox.isChecked() if hasattr(self, 'use_fast_solver_checkbox') else True,
            # Include metadata for provenance
            'is_estimated': self._is_estimated,
            'is_classified': self._is_classified,
            'classification_column': self._classification_column
        }
        
        # Progress callback to update UI
        def progress_callback(percent: int, message: str):
            self.progress.setValue(percent)
            self.log_text.append(message)
        
        # Run task via controller
        self.controller.run_task(
            'pit_opt',
            params,
            callback=self._on_optimization_complete,
            progress_callback=progress_callback
        )

    def _on_optimization_complete(self, result: Dict[str, Any]):
        """Handle completion of pit optimization task."""
        self.btn_run.setEnabled(True)
        # MP-013 fix: Set progress to 100 before hiding (gives visual feedback)
        self.progress.setValue(100)
        
        if result is None:
            self.progress.setVisible(False)
            self.log_text.append("<b style='color: red;'>ERROR: Optimization returned no result.</b>")
            QMessageBox.critical(self, "Optimization Failed", "Optimization returned no result.")
            return
        
        if result.get("error"):
            self.progress.setVisible(False)
            error_msg = result["error"]
            self.log_text.append(f"<b style='color: red;'>ERROR: {error_msg}</b>")
            QMessageBox.critical(self, "Optimization Failed", error_msg)
            return
        
        # Extract result DataFrame
        result_df = result.get("result_df")
        if result_df is None:
            self.progress.setVisible(False)
            self.log_text.append("<b style='color: red;'>ERROR: No result DataFrame in result.</b>")
            QMessageBox.critical(self, "Optimization Failed", "No result DataFrame in result.")
            return
        
        self.result_df = result_df
        self.btn_viz.setEnabled(True)
        
        total_val = result.get("total_value", result_df[result_df['IN_PIT'] == 1]['LG_VALUE'].sum())
        
        # Get tonnage - warn if missing (MP-010 related)
        tonnage_col = None
        for col in ['TONNAGE', 'TONNES', 'T']:
            if col in result_df.columns:
                tonnage_col = col
                break
        
        self.log_text.append("<b style='color: green;'>✓ Optimization Complete</b>")
        
        if tonnage_col:
            tonnes = result_df[result_df['IN_PIT'] == 1][tonnage_col].sum()
            self.log_text.append(f"Total Pit Value: ${total_val:,.0f}")
            self.log_text.append(f"Total Tonnage: {tonnes:,.0f} t")
        else:
            pit_blocks = result.get("blocks_in_pit", (result_df['IN_PIT'] == 1).sum())
            self.log_text.append(f"Total Pit Value: ${total_val:,.0f}")
            self.log_text.append(f"Blocks in Pit: {pit_blocks:,}")
            self.log_text.append("<b style='color: orange;'>⚠ Note: TONNAGE field missing - values may use default 1.0 t/block</b>")
        
        # Log warnings if input data was not ideal
        if not self._is_estimated:
            self.log_text.append("<b style='color: orange;'>⚠ Results based on non-estimated grades</b>")
        if not self._is_classified:
            self.log_text.append("<b style='color: orange;'>⚠ Results may include Inferred resources</b>")
        
        # Hide progress bar after all messages logged
        self.progress.setVisible(False)
        
    def _visualize(self):
        if self.result_df is not None:
            # Send DataFrame directly - the main window handler will convert it
            # The handler expects either a mesh/grid or a DataFrame with IN_PIT column
            self.request_visualization.emit(self.result_df, "IN_PIT")
        else:
            QMessageBox.warning(self, "No Results", "Please run optimization first.")
    
    def set_block_model(self, block_model: pd.DataFrame, grid_spec: Optional[Dict] = None):
        """Set block model data (called from main window).
        
        Args:
            block_model: Block model DataFrame
            grid_spec: Optional grid specification (for compatibility with existing interface)
        """
        self._block_model = block_model  # Use _block_model instead of block_model property
        self._on_block_model_loaded(block_model)

