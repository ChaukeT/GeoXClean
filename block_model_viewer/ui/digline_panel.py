"""
Digline Panel (STEP 29)

Manage diglines and ore/waste polygons.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QHeaderView, QTextEdit
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class DiglinePanel(BaseAnalysisPanel):
    """
    Panel for managing diglines and dig polygons.
    """
    
    task_name = "gc_summarise_digpolys"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="digline")
        self.diglines = None
        self._block_model = None  # Use _block_model instead of block_model property
        self.schedule = None
        
        # Subscribe to block model and schedule from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.scheduleGenerated.connect(self._on_schedule_generated)
            
            # Load existing data if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_schedule = self.registry.get_schedule()
            if existing_schedule:
                self._on_schedule_generated(existing_schedule)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Digline Panel")
    


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
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Digline Panel received block model from DataRegistry")
        self._block_model = block_model  # Use _block_model instead of block_model property
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("Digline Panel received schedule from DataRegistry")
        self.schedule = schedule
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Digline Panel: Manage dig polygons, compute tonnes/grade per polygon."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Import/Export
        io_layout = QHBoxLayout()
        
        self.import_btn = QPushButton("Import Diglines")
        self.import_btn.clicked.connect(self._on_import)
        io_layout.addWidget(self.import_btn)
        
        self.export_btn = QPushButton("Export Diglines")
        self.export_btn.clicked.connect(self._on_export)
        io_layout.addWidget(self.export_btn)
        
        layout.addLayout(io_layout)
        
        # Polygon Table
        self.polygon_table = QTableWidget()
        self.polygon_table.setColumnCount(6)
        self.polygon_table.setHorizontalHeaderLabels([
            "ID", "Bench", "Elevation", "Ore/Waste", "Material Type", "Destination"
        ])
        self.polygon_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("Dig Polygons:"))
        layout.addWidget(self.polygon_table)
        
        # Summarize Button
        self.summarize_btn = QPushButton("Summarize by Polygon")
        self.summarize_btn.clicked.connect(self._on_summarize)
        layout.addWidget(self.summarize_btn)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_text)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def _on_import(self):
        """Import diglines."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Diglines", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self.show_info("Import", f"Importing from {filename}")
            # Would implement actual import logic
    
    def _on_export(self):
        """Export diglines."""
        if not self.diglines:
            self.show_warning("No Diglines", "No diglines to export.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Diglines", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self.show_info("Export", f"Exporting to {filename}")
            # Would implement actual export logic
    
    def _on_summarize(self):
        """Summarize by polygon."""
        if not self.diglines:
            self.show_warning("No Diglines", "Please import diglines first.")
            return
        
        self.show_info("Not Implemented", "Summarization requires GC model.")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

