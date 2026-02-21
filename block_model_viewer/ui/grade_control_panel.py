"""
Grade Control Panel (STEP 29)

Define GC grid, run GC kriging, classify ore/waste.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QDoubleSpinBox,
    QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView, QMessageBox
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class GradeControlPanel(BaseAnalysisPanel):
    """
    Panel for Grade Control operations.
    """
    # PanelManager metadata
    PANEL_ID = "GradeControlPanel"
    PANEL_NAME = "GradeControl Panel"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "gc_build_support_model"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="grade_control")
        self.gc_model = None
        self.gc_grid = None
        self._block_model = None  # Use _block_model instead of block_model property
        self.drillhole_data = None
        
        # Subscribe to block model and drillhole data from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
            
            # Load existing data if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_drillhole = self.registry.get_drillhole_data()
            if existing_drillhole:
                self._on_drillhole_data_loaded(existing_drillhole)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Grade Control Panel")
    


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
            "Grade Control Panel: Define SMU support, build GC model, run GC kriging, classify ore/waste."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # SMU Grid Definition
        grid_group = QGroupBox("SMU Grid Definition")
        grid_layout = QFormLayout()
        
        self.smu_dx = QDoubleSpinBox()
        self.smu_dx.setRange(0.1, 100.0)
        self.smu_dx.setValue(2.5)
        self.smu_dx.setSuffix(" m")
        grid_layout.addRow("SMU Size X:", self.smu_dx)
        
        self.smu_dy = QDoubleSpinBox()
        self.smu_dy.setRange(0.1, 100.0)
        self.smu_dy.setValue(2.5)
        self.smu_dy.setSuffix(" m")
        grid_layout.addRow("SMU Size Y:", self.smu_dy)
        
        self.smu_dz = QDoubleSpinBox()
        self.smu_dz.setRange(0.1, 100.0)
        self.smu_dz.setValue(2.5)
        self.smu_dz.setSuffix(" m")
        grid_layout.addRow("SMU Size Z:", self.smu_dz)
        
        self.resample_method = QComboBox()
        self.resample_method.addItems(["volume_weighted", "nearest", "average"])
        grid_layout.addRow("Resample Method:", self.resample_method)
        
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)
        
        # GC Kriging
        kriging_group = QGroupBox("GC Kriging")
        kriging_layout = QFormLayout()
        
        self.property_combo = QComboBox()
        kriging_layout.addRow("Property:", self.property_combo)
        
        self.use_dh = QComboBox()
        self.use_dh.addItems(["Yes", "No"])
        kriging_layout.addRow("Use Drillholes:", self.use_dh)
        
        self.use_bh = QComboBox()
        self.use_bh.addItems(["Yes", "No"])
        kriging_layout.addRow("Use Blast-holes:", self.use_bh)
        
        kriging_group.setLayout(kriging_layout)
        layout.addWidget(kriging_group)
        
        # Ore/Waste Classification
        cutoff_group = QGroupBox("Ore/Waste Classification")
        cutoff_layout = QFormLayout()
        
        self.cutoff_property = QComboBox()
        cutoff_layout.addRow("Property:", self.cutoff_property)

        # Cutoff value with auto-suggest button
        cutoff_row = QHBoxLayout()
        self.cutoff_value = QDoubleSpinBox()
        self.cutoff_value.setRange(0.0, 100000.0)
        self.cutoff_value.setValue(0.0)
        self.cutoff_value.setDecimals(3)
        cutoff_row.addWidget(self.cutoff_value, stretch=3)

        self.cutoff_suggest_btn = QPushButton("Auto")
        self.cutoff_suggest_btn.setToolTip("Suggest cutoff based on data percentiles")
        self.cutoff_suggest_btn.setStyleSheet("background-color: #388e3c; color: white;")
        self.cutoff_suggest_btn.clicked.connect(self._auto_suggest_cutoff)
        cutoff_row.addWidget(self.cutoff_suggest_btn, stretch=1)

        cutoff_layout.addRow("Cutoff:", cutoff_row)
        
        self.cutoff_direction = QComboBox()
        self.cutoff_direction.addItems([">=", "<=", ">", "<"])
        cutoff_layout.addRow("Direction:", self.cutoff_direction)
        
        cutoff_group.setLayout(cutoff_layout)
        layout.addWidget(cutoff_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.build_btn = QPushButton("Build GC Model")
        self.build_btn.clicked.connect(self._on_build_gc_model)
        button_layout.addWidget(self.build_btn)
        
        self.krige_btn = QPushButton("Run GC Kriging")
        self.krige_btn.clicked.connect(self._on_run_gc_kriging)
        button_layout.addWidget(self.krige_btn)
        
        self.classify_btn = QPushButton("Classify Ore/Waste")
        self.classify_btn.clicked.connect(self._on_classify)
        button_layout.addWidget(self.classify_btn)
        
        layout.addLayout(button_layout)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_text)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {
            "smu_size": (self.smu_dx.value(), self.smu_dy.value(), self.smu_dz.value()),
            "method": self.resample_method.currentText()
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if not self.controller or not self.controller.current_block_model:
            self.show_warning("No Model", "Please load a block model first.")
            return False
        return True
    
    def _on_build_gc_model(self):
        """Build GC support model."""
        if not self.validate_inputs():
            return
        
        params = {
            "long_model": self.controller.current_block_model,
            **self.gather_parameters()
        }
        
        self.controller.build_gc_support_model(params, self._on_build_complete)
    
    def _on_build_complete(self, result: Dict[str, Any]):
        """Handle build completion."""
        self.gc_grid = result.get("gc_grid")
        self.gc_model = result.get("gc_model")
        
        if self.gc_model:
            self.results_text.append(f"GC Model built: {self.gc_model.grid.get_block_count()} blocks")
            # Update property combos
            props = self.gc_model.get_property_names()
            self.property_combo.clear()
            self.property_combo.addItems(props)
            self.cutoff_property.clear()
            self.cutoff_property.addItems(props)
    
    def _on_run_gc_kriging(self):
        """Run GC kriging."""
        if not self.gc_model:
            self.show_warning("No GC Model", "Please build GC model first.")
            return
        
        # Simplified - would need sample data selection
        self.show_info("Not Implemented", "GC kriging requires sample data selection.")
    
    def _on_classify(self):
        """Classify ore/waste."""
        if not self.gc_model:
            self.show_warning("No GC Model", "Please build GC model first.")
            return
        
        from ..grade_control.ore_waste_marking import OreWasteCutoffRule
        
        cutoff_rules = [OreWasteCutoffRule(
            property_name=self.cutoff_property.currentText(),
            cutoff=self.cutoff_value.value(),
            direction=self.cutoff_direction.currentText()
        )]
        
        params = {
            "gc_model": self.gc_model,
            "cutoff_rules": [{"property_name": r.property_name, "cutoff": r.cutoff, "direction": r.direction} for r in cutoff_rules]
        }
        
        self.controller.classify_gc_ore_waste(params, self._on_classify_complete)
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Grade Control Panel received block model from DataRegistry")
        self._block_model = block_model  # Use _block_model instead of block_model property
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_drillhole_data_loaded(self, drillhole_data):
        """
        Automatically receive drillhole data when it's loaded.

        Args:
            drillhole_data: Dict[str, pd.DataFrame] from DataRegistry
        """
        logger.info("Grade Control Panel received drillhole data from DataRegistry")
        self.drillhole_data = drillhole_data

    def _auto_suggest_cutoff(self):
        """Auto-suggest cutoff value based on data percentiles.

        Uses P50 (median) as default cutoff for ore/waste classification.
        Works for any commodity.
        """
        import pandas as pd

        # Try block model first, then drillhole data
        df = None
        if self._block_model is not None:
            if hasattr(self._block_model, 'df'):
                df = self._block_model.df
            elif isinstance(self._block_model, pd.DataFrame):
                df = self._block_model

        if df is None or df.empty:
            QMessageBox.warning(self, "No Data", "Load a block model first to auto-suggest cutoff.")
            return

        grade_col = self.cutoff_property.currentText()
        if not grade_col or grade_col not in df.columns:
            QMessageBox.warning(self, "No Property", "Select a cutoff property first.")
            return

        try:
            values = df[grade_col].dropna()
            if len(values) == 0:
                QMessageBox.warning(self, "No Data", f"No valid data for '{grade_col}'.")
                return

            # Use P50 (median) as suggested cutoff for ore/waste
            p50 = values.quantile(0.50)

            # Determine decimal places based on data magnitude
            grade_max = values.max()
            if grade_max > 50:
                decimals = 1
            elif grade_max > 10:
                decimals = 1
            elif grade_max > 1:
                decimals = 2
            else:
                decimals = 3

            suggested = round(p50, decimals)
            self.cutoff_value.setValue(suggested)

            logger.info(f"GradeControl: Auto-suggested cutoff for '{grade_col}': {suggested}")

        except Exception as e:
            logger.warning(f"GradeControl: Could not auto-suggest cutoff: {e}")
            QMessageBox.warning(self, "Error", f"Could not calculate cutoff: {e}")

    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        # Publish grade control results to DataRegistry
        if hasattr(self, 'registry') and self.registry:
            try:
                grade_control_results = {
                    'classification': payload.get('classification'),
                    'dig_polygons': payload.get('dig_polygons'),
                    'source': 'grade_control'
                }
                # Store as block model classification or separate storage
                # For now, update block model if classification is available
                if 'classification' in payload:
                    existing_block_model = self.registry.get_block_model()
                    if existing_block_model is not None:
                        if hasattr(existing_block_model, 'df') and isinstance(payload['classification'], pd.DataFrame):
                            # Merge classification into block model
                            for col in payload['classification'].columns:
                                if col not in existing_block_model.df.columns:
                                    existing_block_model.df[col] = payload['classification'][col]
                            self.registry.register_block_model(existing_block_model, source_panel="GradeControlPanel")
                            logger.info("Grade Control Panel published classification to block model in DataRegistry")
            except Exception as e:
                logger.warning(f"Failed to register grade control results: {e}")
    
    def _on_classify_complete(self, result: Dict[str, Any]):
        """Handle classification completion."""
        classification_result = result.get("result")
        if classification_result:
            self.results_text.append(
                f"Classification complete:\n"
                f"Ore: {classification_result.tonnage_by_class.get(1, 0):.1f} t\n"
                f"Waste: {classification_result.tonnage_by_class.get(0, 0):.1f} t"
            )

