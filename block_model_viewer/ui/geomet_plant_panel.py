"""
Geomet Plant Panel (STEP 28)

Define plant configuration and test plant response.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QDoubleSpinBox,
    QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class GeometPlantPanel(BaseAnalysisPanel):
    """
    Panel for defining plant configurations and testing plant response.
    """
    
    task_name = "geomet_plant_response"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="geomet_plant")
        self.plant_configs: Dict[str, Any] = {}
        self.comminution_props: Dict[str, Any] = {}
        self.liberation_models: Dict[tuple, Any] = {}
        self._block_model = None  # Use _block_model instead of block_model property
        self.geomet_results = None
        
        # Subscribe to data from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.geometResultsLoaded.connect(self._on_geomet_results_loaded)
            
            # Load existing data if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_geomet = self.registry.get_geomet_results()
            if existing_geomet:
                self._on_geomet_results_loaded(existing_geomet)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Geomet Plant Panel")
    


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
        logger.info("Geomet Plant Panel received block model from DataRegistry")
        self._block_model = block_model  # Use _block_model instead of block_model property
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_geomet_results_loaded(self, geomet_results):
        """
        Automatically receive geomet results when they're loaded.
        
        Args:
            geomet_results: Geomet results from DataRegistry
        """
        logger.info("Geomet Plant Panel received geomet results from DataRegistry")
        self.geomet_results = geomet_results
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Define plant configurations (comminution + separation) and ore-specific response parameters.\n"
            "Test plant response for representative ore types."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Plant configuration tab
        plant_tab = self._create_plant_config_tab()
        tabs.addTab(plant_tab, "Plant Configuration")
        
        # Comminution properties tab
        comminution_tab = self._create_comminution_tab()
        tabs.addTab(comminution_tab, "Comminution Properties")
        
        # Liberation models tab
        liberation_tab = self._create_liberation_tab()
        tabs.addTab(liberation_tab, "Liberation Models")
        
        # Test response tab
        test_tab = self._create_test_tab()
        tabs.addTab(test_tab, "Test Response")
        
        layout.addWidget(tabs)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        test_btn = QPushButton("▶️ Test Plant Response")
        test_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        test_btn.clicked.connect(self._test_response)
        button_layout.addWidget(test_btn)
        
        layout.addLayout(button_layout)
    
    def _create_plant_config_tab(self) -> QWidget:
        """Create plant configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        form = QFormLayout()
        
        self.plant_name = QLineEdit()
        self.plant_name.setPlaceholderText("e.g., Plant_A, Main_Plant")
        form.addRow("Plant Name:", self.plant_name)
        
        self.circuit_type = QComboBox()
        self.circuit_type.addItems(["SAG-Ball", "HPGR-Ball", "Ball only", "SAG only"])
        form.addRow("Comminution Circuit:", self.circuit_type)
        
        self.target_p80 = QDoubleSpinBox()
        self.target_p80.setRange(10, 1000)
        self.target_p80.setValue(150)
        self.target_p80.setSuffix(" μm")
        form.addRow("Target P80:", self.target_p80)
        
        self.f80 = QDoubleSpinBox()
        self.f80.setRange(1, 500)
        self.f80.setValue(100)
        self.f80.setSuffix(" mm")
        form.addRow("Feed F80:", self.f80)
        
        self.max_throughput = QDoubleSpinBox()
        self.max_throughput.setRange(1, 10000)
        self.max_throughput.setValue(1000)
        self.max_throughput.setSuffix(" t/h")
        form.addRow("Max Throughput:", self.max_throughput)
        
        layout.addLayout(form)
        
        # Separation stages
        sep_group = QGroupBox("Separation Stages")
        sep_layout = QVBoxLayout()
        
        self.separation_table = QTableWidget()
        self.separation_table.setColumnCount(3)
        self.separation_table.setHorizontalHeaderLabels(["Method", "Ore Type", "Target Grade"])
        self.separation_table.horizontalHeader().setStretchLastSection(True)
        sep_layout.addWidget(self.separation_table)
        
        sep_btn_layout = QHBoxLayout()
        add_sep_btn = QPushButton("➕ Add Stage")
        add_sep_btn.clicked.connect(self._add_separation_stage)
        sep_btn_layout.addWidget(add_sep_btn)
        
        remove_sep_btn = QPushButton("➖ Remove Selected")
        remove_sep_btn.clicked.connect(self._remove_separation_stage)
        sep_btn_layout.addWidget(remove_sep_btn)
        sep_layout.addLayout(sep_btn_layout)
        
        sep_group.setLayout(sep_layout)
        layout.addWidget(sep_group)
        
        # Save plant config button
        save_btn = QPushButton("💾 Save Plant Configuration")
        save_btn.clicked.connect(self._save_plant_config)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        return widget
    
    def _create_comminution_tab(self) -> QWidget:
        """Create comminution properties tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        form = QFormLayout()
        
        self.comm_ore_type = QComboBox()
        form.addRow("Ore Type:", self.comm_ore_type)
        
        self.bond_wi = QDoubleSpinBox()
        self.bond_wi.setRange(1, 50)
        self.bond_wi.setValue(12.0)
        self.bond_wi.setSuffix(" kWh/t")
        form.addRow("Bond Work Index:", self.bond_wi)
        
        self.jk_a = QDoubleSpinBox()
        self.jk_a.setRange(0.1, 100)
        self.jk_a.setValue(50.0)
        self.jk_a.setSuffix(" kWh/t")
        form.addRow("JK A Parameter:", self.jk_a)
        
        self.jk_b = QDoubleSpinBox()
        self.jk_b.setRange(0.1, 2.0)
        self.jk_b.setValue(0.5)
        form.addRow("JK b Parameter:", self.jk_b)
        
        layout.addLayout(form)
        
        save_btn = QPushButton("💾 Save Comminution Properties")
        save_btn.clicked.connect(self._save_comminution_props)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        return widget
    
    def _create_liberation_tab(self) -> QWidget:
        """Create liberation models tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info = QLabel("Liberation models define liberation fraction vs. particle size for each ore type and mineral.")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        form = QFormLayout()
        
        self.lib_ore_type = QComboBox()
        form.addRow("Ore Type:", self.lib_ore_type)
        
        self.lib_mineral = QLineEdit()
        self.lib_mineral.setPlaceholderText("e.g., Fe, Cu")
        form.addRow("Mineral:", self.lib_mineral)
        
        self.lib_note = QTextEdit()
        self.lib_note.setMaximumHeight(60)
        self.lib_note.setPlaceholderText("Note: Liberation curves are typically loaded from testwork data.")
        form.addRow("Notes:", self.lib_note)
        
        layout.addLayout(form)
        
        save_btn = QPushButton("💾 Save Liberation Model")
        save_btn.clicked.connect(self._save_liberation_model)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        return widget
    
    def _create_test_tab(self) -> QWidget:
        """Create test response tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        form = QFormLayout()
        
        self.test_plant = QComboBox()
        form.addRow("Plant Configuration:", self.test_plant)
        
        self.test_ore_type = QComboBox()
        form.addRow("Ore Type:", self.test_ore_type)
        
        self.test_fe_grade = QDoubleSpinBox()
        self.test_fe_grade.setRange(0, 100)
        self.test_fe_grade.setValue(50.0)
        self.test_fe_grade.setSuffix(" %")
        form.addRow("Fe Grade:", self.test_fe_grade)
        
        self.test_sio2_grade = QDoubleSpinBox()
        self.test_sio2_grade.setRange(0, 100)
        self.test_sio2_grade.setValue(5.0)
        self.test_sio2_grade.setSuffix(" %")
        form.addRow("SiO2 Grade:", self.test_sio2_grade)
        
        layout.addLayout(form)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        layout.addWidget(self.results_text)
        
        return widget
    
    def _add_separation_stage(self):
        """Add separation stage."""
        row = self.separation_table.rowCount()
        self.separation_table.insertRow(row)
        
        method_combo = QComboBox()
        method_combo.addItems(["magnetic", "gravity", "flotation"])
        self.separation_table.setCellWidget(row, 0, method_combo)
        
        ore_type_combo = QComboBox()
        self.separation_table.setCellWidget(row, 1, ore_type_combo)
        
        target_grade = QDoubleSpinBox()
        target_grade.setRange(0, 100)
        target_grade.setValue(65.0)
        self.separation_table.setCellWidget(row, 2, target_grade)
    
    def _remove_separation_stage(self):
        """Remove selected separation stage."""
        row = self.separation_table.currentRow()
        if row >= 0:
            self.separation_table.removeRow(row)
    
    def _save_plant_config(self):
        """Save plant configuration."""
        plant_name = self.plant_name.text().strip()
        if not plant_name:
            self.show_error("Invalid Input", "Plant name is required.")
            return
        
        from ..geomet.comminution_model import ComminutionCircuitConfig
        from ..geomet.separation_model import SeparationConfig
        from ..geomet.plant_response import PlantConfig
        
        circuit_config = ComminutionCircuitConfig(
            circuit_type=self.circuit_type.currentText(),
            target_p80=self.target_p80.value(),
            f80=self.f80.value(),
            plant_throughput_limit=self.max_throughput.value()
        )
        
        separation_configs = []
        for row in range(self.separation_table.rowCount()):
            method_widget = self.separation_table.cellWidget(row, 0)
            ore_type_widget = self.separation_table.cellWidget(row, 1)
            target_widget = self.separation_table.cellWidget(row, 2)
            
            if method_widget and target_widget:
                sep_config = SeparationConfig(
                    method=method_widget.currentText(),
                    ore_type_code=ore_type_widget.currentText() if ore_type_widget else "",
                    target_grade=target_widget.value() if target_widget else None
                )
                separation_configs.append(sep_config)
        
        plant_config = PlantConfig(
            name=plant_name,
            comminution_config=circuit_config,
            separation_configs=separation_configs,
            constraints={"max_throughput": self.max_throughput.value()}
        )
        
        self.plant_configs[plant_name] = plant_config
        self.test_plant.addItem(plant_name)
        self.show_info("Success", f"Saved plant configuration: {plant_name}")
    
    def _save_comminution_props(self):
        """Save comminution properties."""
        ore_type = self.comm_ore_type.currentText()
        if not ore_type:
            self.show_error("Invalid Input", "Ore type is required.")
            return
        
        from ..geomet.comminution_model import ComminutionOreProperties
        
        props = ComminutionOreProperties(
            ore_type_code=ore_type,
            work_index_bond=self.bond_wi.value(),
            A=self.jk_a.value(),
            b=self.jk_b.value()
        )
        
        self.comminution_props[ore_type] = props
        self.show_info("Success", f"Saved comminution properties for {ore_type}")
    
    def _save_liberation_model(self):
        """Save liberation model."""
        ore_type = self.lib_ore_type.currentText()
        mineral = self.lib_mineral.text().strip()
        
        if not ore_type or not mineral:
            self.show_error("Invalid Input", "Ore type and mineral are required.")
            return
        
        # Create default liberation curve (would normally come from testwork)
        import numpy as np
        from ..geomet.liberation_model import LiberationCurve, LiberationModelConfig
        
        size_classes = np.array([150, 106, 75, 53, 38]) * 1000  # microns
        liberation_fractions = np.array([0.3, 0.5, 0.7, 0.85, 0.95])  # Default curve
        
        curve = LiberationCurve(
            size_classes=size_classes,
            liberation_fraction=liberation_fractions,
            mineral_name=mineral,
            ore_type_code=ore_type
        )
        
        config = LiberationModelConfig(
            ore_type_code=ore_type,
            mineral_name=mineral,
            base_curve=curve
        )
        
        self.liberation_models[(ore_type, mineral)] = config
        self.show_info("Success", f"Saved liberation model for {ore_type} - {mineral}")
    
    def _test_response(self):
        """Test plant response."""
        plant_name = self.test_plant.currentText()
        ore_type = self.test_ore_type.currentText()
        
        if not plant_name or plant_name not in self.plant_configs:
            self.show_error("Invalid Input", "Please select a valid plant configuration.")
            return
        
        plant_config = self.plant_configs[plant_name]
        
        chemistry = {
            "Fe": self.test_fe_grade.value(),
            "SiO2": self.test_sio2_grade.value()
        }
        
        # Run via controller
        params = {
            "ore_type_code": ore_type,
            "chemistry": chemistry,
            "plant_config": plant_config,
            "liberation_models": self.liberation_models,
            "comminution_props": self.comminution_props
        }
        
        if self.controller:
            self.controller.evaluate_geomet_plant_response(params, self._on_test_complete)
        else:
            self.show_error("No Controller", "Controller not available.")
    
    def _on_test_complete(self, result: Dict[str, Any]):
        """Handle test response completion."""
        if result.get("error"):
            self.results_text.setText(f"Error: {result['error']}")
        else:
            response = result.get("response", {})
            text = f"Plant Response Results:\n\n"
            text += f"Ore Type: {response.get('ore_type_code', 'N/A')}\n"
            text += f"Throughput: {response.get('throughput', 0):.1f} t/h\n"
            text += f"Specific Energy: {response.get('specific_energy', 0):.2f} kWh/t\n"
            text += f"Mass Pull: {response.get('mass_pull', 0):.2%}\n\n"
            
            text += "Recoveries:\n"
            for element, recovery in response.get("recovery_by_element", {}).items():
                text += f"  {element}: {recovery:.2%}\n"
            
            text += "\nConcentrate Grades:\n"
            for element, grade in response.get("concentrate_grade_by_element", {}).items():
                text += f"  {element}: {grade:.2f}%\n"
            
            self.results_text.setText(text)
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Plant configuration
            settings['plant_name'] = get_safe_widget_value(self, 'plant_name')
            settings['circuit_type'] = get_safe_widget_value(self, 'circuit_type')
            settings['target_p80'] = get_safe_widget_value(self, 'target_p80')
            settings['f80'] = get_safe_widget_value(self, 'f80')
            
            # Test parameters
            settings['test_plant'] = get_safe_widget_value(self, 'test_plant')
            settings['test_ore_type'] = get_safe_widget_value(self, 'test_ore_type')
            settings['test_fe_grade'] = get_safe_widget_value(self, 'test_fe_grade')
            settings['test_sio2_grade'] = get_safe_widget_value(self, 'test_sio2_grade')
            
            # Save plant configs if any
            if self.plant_configs:
                settings['plant_configs'] = {k: v.__dict__ if hasattr(v, '__dict__') else v 
                                             for k, v in self.plant_configs.items()}
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save geomet plant panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Plant configuration
            set_safe_widget_value(self, 'plant_name', settings.get('plant_name'))
            set_safe_widget_value(self, 'circuit_type', settings.get('circuit_type'))
            set_safe_widget_value(self, 'target_p80', settings.get('target_p80'))
            set_safe_widget_value(self, 'f80', settings.get('f80'))
            
            # Test parameters
            set_safe_widget_value(self, 'test_plant', settings.get('test_plant'))
            set_safe_widget_value(self, 'test_ore_type', settings.get('test_ore_type'))
            set_safe_widget_value(self, 'test_fe_grade', settings.get('test_fe_grade'))
            set_safe_widget_value(self, 'test_sio2_grade', settings.get('test_sio2_grade'))
                
            logger.info("Restored geomet plant panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore geomet plant panel settings: {e}")

