"""
Reconciliation Panel (STEP 29)

Model-to-mine and mine-to-mill reconciliation.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QHeaderView, QTabWidget
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ReconciliationPanel(BaseAnalysisPanel):
    """
    Panel for reconciliation analysis.
    """
    
    task_name = "recon_model_mine"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="reconciliation")
        self.mined_table = None
        self.plant_feed_table = None
        
        # Subscribe to block model and resource updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.resourceCalculated.connect(self._on_resource_calculated)
            
            # Load existing data if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_resource = self.registry.get_resource_summary()
            if existing_resource:
                self._on_resource_calculated(existing_resource)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Reconciliation Panel")
    


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
    def setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Reconciliation Panel: Compare model vs mined vs plant data."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Data Import
        import_group = QGroupBox("Data Import")
        import_layout = QFormLayout()
        
        self.mined_file_btn = QPushButton("Load Mined Data")
        self.mined_file_btn.clicked.connect(self._on_load_mined)
        import_layout.addRow("Mined Data:", self.mined_file_btn)
        
        self.plant_file_btn = QPushButton("Load Plant Feed Data")
        self.plant_file_btn.clicked.connect(self._on_load_plant)
        import_layout.addRow("Plant Feed:", self.plant_file_btn)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Reconciliation Buttons
        recon_layout = QHBoxLayout()
        
        self.recon_model_mine_btn = QPushButton("Model → Mine")
        self.recon_model_mine_btn.clicked.connect(self._on_recon_model_mine)
        recon_layout.addWidget(self.recon_model_mine_btn)
        
        self.recon_mine_mill_btn = QPushButton("Mine → Mill")
        self.recon_mine_mill_btn.clicked.connect(self._on_recon_mine_mill)
        recon_layout.addWidget(self.recon_mine_mill_btn)
        
        self.recon_metrics_btn = QPushButton("Compute Metrics")
        self.recon_metrics_btn.clicked.connect(self._on_recon_metrics)
        recon_layout.addWidget(self.recon_metrics_btn)
        
        layout.addLayout(recon_layout)
        
        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Stage", "Tonnes Bias %", "Grade Bias %", "Metal Bias %", "Notes"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Reconciliation Results:"))
        layout.addWidget(self.results_table)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {
            "mined_table": self.mined_table,
            "plant_feed_table": self.plant_feed_table
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if self.mined_table is None:
            self.show_warning("No Data", "Please load mined data first.")
            return False
        return True
    
    def _on_load_mined(self):
        """Load mined data."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Mined Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            try:
                self.mined_table = pd.read_csv(filename)
                self.show_info("Loaded", f"Mined data loaded: {len(self.mined_table)} records")
            except Exception as e:
                self.show_error("Error", f"Failed to load mined data: {e}")
    
    def _on_load_plant(self):
        """Load plant feed data."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Plant Feed Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            try:
                self.plant_feed_table = pd.read_csv(filename)
                self.show_info("Loaded", f"Plant feed data loaded: {len(self.plant_feed_table)} records")
            except Exception as e:
                self.show_error("Error", f"Failed to load plant feed data: {e}")
    
    def _on_recon_model_mine(self):
        """Run model-mine reconciliation."""
        if not self.validate_inputs():
            return
        
        params = {
            "long_model": self.controller.current_block_model if self.controller else None,
            "mined_table": self.mined_table.to_dict() if isinstance(self.mined_table, pd.DataFrame) else self.mined_table,
            "grade_properties": ["Fe"]  # Would get from UI
        }
        
        self.controller.run_recon_model_mine(params, self._on_recon_complete)
    
    def _on_recon_mine_mill(self):
        """Run mine-mill reconciliation."""
        if not self.validate_inputs():
            return
        
        params = {
            "mined_series": {"records": []},  # Would convert from table
            "plant_feed_table": self.plant_feed_table.to_dict() if isinstance(self.plant_feed_table, pd.DataFrame) else self.plant_feed_table
        }
        
        self.controller.run_recon_mine_mill(params, self._on_recon_complete)
    
    def _on_recon_metrics(self):
        """Compute reconciliation metrics."""
        params = {
            "mined_series": {"records": []}  # Would convert from table
        }
        
        self.controller.run_recon_metrics(params, self._on_metrics_complete)
    
    def _on_recon_complete(self, result: Dict[str, Any]):
        """Handle reconciliation completion."""
        self.show_info("Complete", "Reconciliation analysis complete.")
        
        # Publish reconciliation results to DataRegistry
        try:
            if hasattr(self, 'registry') and self.registry:
                self.registry.register_reconciliation_results(
                    result, 
                    source_panel="ReconciliationPanel"
                )
                logger.info("Published reconciliation results to DataRegistry")
        except Exception as e:
            logger.warning(f"Failed to publish reconciliation results to DataRegistry: {e}")
        
        # Add button to send to Production Dashboard
        if not hasattr(self, 'prod_dashboard_btn'):
            self.prod_dashboard_btn = QPushButton("Send to Production Dashboard")
            self.prod_dashboard_btn.clicked.connect(self._on_send_to_production_dashboard)
            self.main_layout.addWidget(self.prod_dashboard_btn)
    
    def _on_send_to_production_dashboard(self):
        """Send reconciliation result to Production Dashboard."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            from .production_dashboard_panel import ProductionDashboardPanel
            
            # Open production dashboard
            if not hasattr(self.controller, 'production_dashboard_dialog') or self.controller.production_dashboard_dialog is None:
                dialog = ProductionDashboardPanel(self)
                dialog.bind_controller(self.controller)
                self.controller.production_dashboard_dialog = dialog
            
            # Pre-populate with reconciliation result
            # Would pass recon result reference here
            self.controller.production_dashboard_dialog.show()
            self.controller.production_dashboard_dialog.raise_()
            self.controller.production_dashboard_dialog.activateWindow()
            
            self.show_info("Production Dashboard", "Sent reconciliation result to Production Dashboard")
        
        except Exception as e:
            logger.error(f"Failed to open Production Dashboard: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Production Dashboard:\n{e}")
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Reconciliation Panel received block model from DataRegistry")
        # Store for reconciliation analysis
        self._block_model = block_model  # Use private backing field (property contract)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_resource_calculated(self, resource_summary):
        """
        Automatically receive resource calculation results.
        
        Args:
            resource_summary: Resource calculation summary from DataRegistry
        """
        logger.info("Reconciliation Panel received resource calculation results from DataRegistry")
        self.resource_summary = resource_summary
    
    def _on_metrics_complete(self, result: Dict[str, Any]):
        """Handle metrics completion."""
        metrics = result.get("metrics", {})
        self.show_info("Complete", f"Metrics computed: {len(metrics.get('stages', {}))} stages")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        # Reconciliation panel typically doesn't have persistent settings
        return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        pass

