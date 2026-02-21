"""
Geomet Overview / Block Model Panel (STEP 28)

Run full geometallurgical pipeline at block scale and show summary KPIs.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QSplitter, QHeaderView
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class GeometPanel(BaseAnalysisPanel):
    """
    Main panel for running geometallurgical analysis at block scale.
    """
    
    task_name = "geomet_compute_block_attrs"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="geomet")
        
        self._block_model = None  # Use _block_model instead of block_model property
        
        # Subscribe to block model updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_classified)
            
            # Load existing block model if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_classified = self.registry.get_classified_block_model()
            if existing_classified:
                self._on_block_model_classified(existing_classified)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Geomet Panel")
    


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
            "Run full geometallurgical pipeline at block scale:\n"
            "• Assign ore types to blocks\n"
            "• Compute geomet attributes (recovery, concentrate grades, energy)\n"
            "• Generate geomet-adjusted block values\n"
            "• Push to IRR/pit/schedule analysis"
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Configuration section
        config_group = self._create_config_group()
        layout.addWidget(config_group)
        
        # Create splitter for results
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Summary KPIs
        kpi_group = self._create_kpi_group()
        splitter.addWidget(kpi_group)
        
        # Results table
        results_group = self._create_results_group()
        splitter.addWidget(results_group)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        run_btn = QPushButton("▶️ Compute Geomet Attributes")
        run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        run_btn.clicked.connect(self._compute_attributes)
        button_layout.addWidget(run_btn)
        
        button_layout.addStretch()
        
        push_irr_btn = QPushButton("📊 Push to IRR/NPV")
        push_irr_btn.clicked.connect(self._push_to_irr)
        button_layout.addWidget(push_irr_btn)
        
        push_pit_btn = QPushButton("⛏️ Push to Pit Optimizer")
        push_pit_btn.clicked.connect(self._push_to_pit)
        button_layout.addWidget(push_pit_btn)
        
        layout.addLayout(button_layout)
    
    def _create_config_group(self) -> QGroupBox:
        """Create configuration group."""
        group = QGroupBox("Configuration")
        layout = QFormLayout()
        
        self.plant_combo = QComboBox()
        self.plant_combo.setToolTip("Select plant configuration to apply")
        layout.addRow("Plant Configuration:", self.plant_combo)
        
        self.domain_map_combo = QComboBox()
        self.domain_map_combo.setToolTip("Select domain mapping (ore types)")
        layout.addRow("Domain Mapping:", self.domain_map_combo)
        
        self.value_field_combo = QComboBox()
        self.value_field_combo.addItems(["Base Value", "Geomet-Adjusted Value"])
        self.value_field_combo.setToolTip("Select which value field to use for economic analysis")
        layout.addRow("Value Field:", self.value_field_combo)
        
        group.setLayout(layout)
        return group
    
    def _create_kpi_group(self) -> QGroupBox:
        """Create KPI summary group."""
        group = QGroupBox("Summary KPIs")
        layout = QVBoxLayout()
        
        self.kpi_text = QTextEdit()
        self.kpi_text.setReadOnly(True)
        self.kpi_text.setMaximumHeight(150)
        self.kpi_text.setPlaceholderText("Run analysis to see summary KPIs...")
        layout.addWidget(self.kpi_text)
        
        group.setLayout(layout)
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Create results table group."""
        group = QGroupBox("Results by Ore Type")
        layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Ore Type", "Avg Recovery", "Conc Grade", "Mass Pull", "Throughput", "Energy"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        
        group.setLayout(layout)
        return group
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        try:
            # Convert to BlockModel if needed
            if hasattr(block_model, 'to_dataframe'):
                # Already a BlockModel
                self._block_model = block_model  # Use _block_model
            elif isinstance(block_model, pd.DataFrame):
                # Convert DataFrame to BlockModel if possible
                from ..models.block_model import BlockModel
                # Try to construct BlockModel from DataFrame
                # For now, just store the DataFrame and use validate_block_model_loaded() which gets from controller
                # The block model will be set properly when controller is available
                logger.info(f"Geomet Panel received block model DataFrame: {len(block_model)} blocks")
            else:
                logger.warning(f"Unexpected block model type: {type(block_model)}")
                return
            
            logger.info(f"Geomet Panel auto-received block model: {len(block_model) if hasattr(block_model, '__len__') else 'N/A'}")
        except Exception as e:
            logger.error(f"Error processing block model in Geomet Panel: {e}", exc_info=True)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        # Use same handler as generated
        self._on_block_model_generated(block_model)
    
    def _on_block_model_classified(self, block_model):
        """
        Automatically receive classified block model when it's classified.
        Prefer classified model as it has resource categories.
        
        Args:
            block_model: Classified BlockModel from DataRegistry
        """
        # Use same handler as generated (classified is just an enhanced version)
        self._on_block_model_generated(block_model)
    
    def _compute_attributes(self):
        """Compute geomet attributes for blocks."""
        if not self.validate_block_model_loaded():
            return
        
        plant_name = self.plant_combo.currentText()
        if not plant_name:
            self.show_error("Configuration Error", "Please select a plant configuration.")
            return
        
        # Gather parameters
        params = {
            "block_model": self.block_model,
            "plant_config_name": plant_name,
            "domain_map_name": self.domain_map_combo.currentText()
        }
        
        # Run via controller
        if self.controller:
            self.controller.compute_geomet_block_attributes(params, self._on_compute_complete)
        else:
            self.show_error("No Controller", "Controller not available.")
    
    def _on_compute_complete(self, result: Dict[str, Any]):
        """Handle computation completion."""
        if result.get("error"):
            self.show_error("Computation Error", result["error"])
            return
        
        geomet_attrs = result.get("geomet_attrs")
        if not geomet_attrs:
            self.show_error("No Results", "Computation completed but no results returned.")
            return
        
        # Update KPI summary
        self._update_kpi_summary(geomet_attrs)
        
        # Update results table
        self._update_results_table(geomet_attrs)
        
        # Attach to block model
        if self.block_model and self.controller:
            from ..geomet.geomet_block_model import attach_geomet_to_block_model
            attach_geomet_to_block_model(self.block_model, geomet_attrs)
            self.emit_status("Geomet attributes attached to block model.")
        
        # Publish results to DataRegistry
        try:
            if hasattr(self, 'registry') and self.registry:
                self.registry.register_geomet_results(geomet_attrs, source_panel="GeometPanel")
                logger.info("Published geomet results to DataRegistry")
        except Exception as e:
            logger.warning(f"Failed to publish geomet results to DataRegistry: {e}")
    
    def _update_kpi_summary(self, geomet_attrs: Any):
        """Update KPI summary text."""
        text = "Geometallurgical Summary:\n\n"
        
        # Overall statistics
        n_blocks = len(geomet_attrs.ore_type_code)
        unique_ore_types = np.unique(geomet_attrs.ore_type_code)
        
        text += f"Total Blocks: {n_blocks:,}\n"
        text += f"Unique Ore Types: {len(unique_ore_types)}\n\n"
        
        # Average recovery
        if geomet_attrs.recovery_by_element:
            text += "Average Recoveries:\n"
            for element, recovery_array in geomet_attrs.recovery_by_element.items():
                avg_recovery = np.nanmean(recovery_array)
                text += f"  {element}: {avg_recovery:.2%}\n"
        
        # Average energy
        if len(geomet_attrs.plant_specific_energy) > 0:
            avg_energy = np.nanmean(geomet_attrs.plant_specific_energy)
            text += f"\nAverage Specific Energy: {avg_energy:.2f} kWh/t\n"
        
        self.kpi_text.setText(text)
    
    def _update_results_table(self, geomet_attrs: Any):
        """Update results table."""
        unique_ore_types = np.unique(geomet_attrs.ore_type_code)
        
        self.results_table.setRowCount(len(unique_ore_types))
        
        for row, ore_type in enumerate(unique_ore_types):
            mask = geomet_attrs.ore_type_code == ore_type
            
            self.results_table.setItem(row, 0, QTableWidgetItem(str(ore_type)))
            
            # Average recovery
            if geomet_attrs.recovery_by_element:
                element = list(geomet_attrs.recovery_by_element.keys())[0]
                recovery_array = geomet_attrs.recovery_by_element[element]
                avg_recovery = np.nanmean(recovery_array[mask])
                self.results_table.setItem(row, 1, QTableWidgetItem(f"{avg_recovery:.2%}"))
            
            # Average concentrate grade
            if geomet_attrs.concentrate_grade_by_element:
                element = list(geomet_attrs.concentrate_grade_by_element.keys())[0]
                grade_array = geomet_attrs.concentrate_grade_by_element[element]
                avg_grade = np.nanmean(grade_array[mask])
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{avg_grade:.2f}%"))
            
            # Mass pull (average)
            if len(geomet_attrs.plant_tonnage_factor) > 0:
                avg_factor = np.nanmean(geomet_attrs.plant_tonnage_factor[mask])
                self.results_table.setItem(row, 3, QTableWidgetItem(f"{avg_factor:.3f}"))
            
            # Throughput (placeholder)
            self.results_table.setItem(row, 4, QTableWidgetItem("N/A"))
            
            # Energy
            if len(geomet_attrs.plant_specific_energy) > 0:
                avg_energy = np.nanmean(geomet_attrs.plant_specific_energy[mask])
                self.results_table.setItem(row, 5, QTableWidgetItem(f"{avg_energy:.2f} kWh/t"))
    
    def _push_to_irr(self):
        """Push geomet values to IRR/NPV analysis."""
        if not self.block_model:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        # Check if geomet value field exists
        plant_name = self.plant_combo.currentText()
        if not plant_name:
            self.show_error("No Plant", "Please select a plant configuration.")
            return
        
        value_field = f"gvalue_{plant_name}"
        if value_field not in self.block_model.get_property_names():
            self.show_error("No Geomet Values", 
                          f"Geomet values not found. Please run 'Compute Geomet Attributes' first.")
            return
        
        self.show_info("Success", 
                      f"Geomet-adjusted values ({value_field}) are available for IRR/NPV analysis.\n"
                      f"Select '{value_field}' as the value field in the IRR panel.")
        self.emit_status(f"Geomet values ready: {value_field}")
    
    def _push_to_pit(self):
        """Push geomet values to pit optimizer."""
        if not self.block_model:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        plant_name = self.plant_combo.currentText()
        if not plant_name:
            self.show_error("No Plant", "Please select a plant configuration.")
            return
        
        value_field = f"gvalue_{plant_name}"
        if value_field not in self.block_model.get_property_names():
            self.show_error("No Geomet Values", 
                          f"Geomet values not found. Please run 'Compute Geomet Attributes' first.")
            return
        
        self.show_info("Success", 
                      f"Geomet-adjusted values ({value_field}) are available for pit optimization.\n"
                      f"Select '{value_field}' as the value field in the Pit Optimizer panel.")
        self.emit_status(f"Geomet values ready: {value_field}")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters."""
        return {
            "plant_config_name": self.plant_combo.currentText(),
            "domain_map_name": self.domain_map_combo.currentText()
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if not self.validate_block_model_loaded():
            return False
        
        if not self.plant_combo.currentText():
            self.show_error("Configuration Error", "Please select a plant configuration.")
            return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        if payload.get("error"):
            self.show_error("Error", payload["error"])
        else:
            self._on_compute_complete(payload)

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Configuration
            settings['plant'] = get_safe_widget_value(self, 'plant_combo')
            settings['domain_map'] = get_safe_widget_value(self, 'domain_map_combo')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save geomet panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Configuration
            set_safe_widget_value(self, 'plant_combo', settings.get('plant'))
            set_safe_widget_value(self, 'domain_map_combo', settings.get('domain_map'))
                
            logger.info("Restored geomet panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore geomet panel settings: {e}")
