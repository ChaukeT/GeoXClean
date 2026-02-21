"""
IRR Input Configuration Module

Manages numeric input limits, unit conversions, and user preferences
for the IRR Analysis panel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget,
    QWidget, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .modern_styles import get_complete_panel_stylesheet, ModernColors

logger = logging.getLogger(__name__)


# Default maximum limits for numeric inputs
DEFAULT_MAX_LIMITS = {
    # Scenario Generation
    "num_scenarios": 1000,
    "num_periods": 50,
    "random_seed": 99999,
    
    # Price Parameters
    "metal_price": 10000.0,
    "price_volatility": 1.0,
    "price_drift": 1.0,
    "price_mean_reversion": 1.0,
    
    # Cost Parameters
    "mining_cost": 500.0,
    "processing_cost": 500.0,
    "selling_cost": 100.0,
    "cost_inflation": 0.5,
    "cost_uncertainty": 0.5,
    
    # Recovery
    "recovery": 1.0,
    "recovery_uncertainty": 0.5,
    
    # Grade
    "grade_uncertainty": 1.0,
    "grade_spatial_correlation": 1.0,
    
    # IRR Search
    "r_low": 1.0,
    "r_high": 2.0,
    "alpha": 1.0,
    "tolerance": 0.01,
    "max_iterations": 100,
    
    # Optimization
    "production_capacity": 10000000.0,
    "annual_rom": 50000000.0,
    "min_bottom_width": 1000.0,
    "num_phases": 10,
    "phase_gap": 50,
    
    # Slope Angles
    "slope_angle": 90.0,
    
    # By-products
    "byproduct_price": 10000.0,
    "byproduct_recovery": 1.0
}


# Unit conversion factors (to base units)
UNIT_CONVERSION = {
    "$/tonne": 1.0,
    "$/gram": 1000000.0,
    "$/kg": 1000.0,
    "$/oz": 31103.5,  # Troy ounce to gram
    "$/unit": 1.0
}


# Available units for different parameter types
PRICE_UNITS = ["$/tonne", "$/gram", "$/kg", "$/oz", "$/unit"]
COST_UNITS = ["$/tonne", "$/kg", "$/unit"]


class IRRInputLimits:
    """Manages maximum limits for IRR input fields."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".block_model_viewer" / "irr_input_limits.json"
        self.limits = DEFAULT_MAX_LIMITS.copy()
        self.load()
    
    def load(self):
        """Load limits from configuration file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_limits = json.load(f)
                    self.limits.update(user_limits)
                logger.info(f"Loaded input limits from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load input limits: {e}")
    
    def save(self):
        """Save limits to configuration file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.limits, f, indent=2)
            logger.info(f"Saved input limits to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save input limits: {e}")
    
    def get_limit(self, key: str) -> float:
        """Get maximum limit for a parameter."""
        return self.limits.get(key, 1000.0)
    
    def set_limit(self, key: str, value: float):
        """Set maximum limit for a parameter."""
        self.limits[key] = value
        self.save()
    
    def reset_to_defaults(self):
        """Reset all limits to default values."""
        self.limits = DEFAULT_MAX_LIMITS.copy()
        self.save()


class IRRUnitPreferences:
    """Manages unit preferences for IRR economic parameters."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".block_model_viewer" / "irr_unit_prefs.json"
        self.units = {
            "metal_price": "$/oz",
            "mining_cost": "$/tonne",
            "processing_cost": "$/tonne",
            "selling_cost": "$/oz",
            "byproduct_price": "$/oz"
        }
        self.load()
    
    def load(self):
        """Load unit preferences from configuration file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_units = json.load(f)
                    self.units.update(user_units)
                logger.info(f"Loaded unit preferences from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load unit preferences: {e}")
    
    def save(self):
        """Save unit preferences to configuration file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.units, f, indent=2)
            logger.info(f"Saved unit preferences to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save unit preferences: {e}")
    
    def get_unit(self, param: str) -> str:
        """Get unit for a parameter."""
        return self.units.get(param, "$/tonne")
    
    def set_unit(self, param: str, unit: str):
        """Set unit for a parameter."""
        self.units[param] = unit
        self.save()
    
    def get_conversion_factor(self, param: str) -> float:
        """Get conversion factor for a parameter to base units."""
        unit = self.get_unit(param)
        return UNIT_CONVERSION.get(unit, 1.0)
    
    def convert_to_base(self, param: str, value: float) -> float:
        """Convert value from user units to base units ($/tonne)."""
        factor = self.get_conversion_factor(param)
        return value * factor
    
    def convert_from_base(self, param: str, value: float) -> float:
        """Convert value from base units ($/tonne) to user units."""
        factor = self.get_conversion_factor(param)
        return value / factor if factor != 0 else value


class InputLimitsDialog(QDialog):
    """Dialog for configuring maximum input limits with modern styling."""
    
    limits_updated = pyqtSignal()
    
    def __init__(self, limits_manager: IRRInputLimits, parent=None):
        super().__init__(parent)
        self.limits_manager = limits_manager
        self.setWindowTitle("⚙️ Configure Input Limits")
        self.resize(650, 550)
        self._setup_ui()
        self._load_values()
    
    def _setup_ui(self):
        """Setup the dialog UI with modern styling."""
        # Apply modern stylesheet
        self.setStyleSheet(get_complete_panel_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Modern info banner
        info_label = QLabel(
            "ℹ️ Set maximum values for numeric input fields.\n"
            "These limits prevent accidental entry of unreasonable values."
        )
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Segoe UI", 10))
        info_label.setStyleSheet(f"""
            padding: 14px;
            background-color: rgba(14, 122, 202, 0.15);
            color: {ModernColors.INFO};
            border: 1px solid {ModernColors.INFO};
            border-radius: 8px;
        """)
        layout.addWidget(info_label)
        
        # Modern tabs for different categories
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)
        layout.addWidget(tab_widget)
        
        # Scenario Generation tab
        scenario_tab = self._create_scenario_limits_tab()
        tab_widget.addTab(scenario_tab, "🎲 Scenario Generation")
        
        # Economic Parameters tab
        economic_tab = self._create_economic_limits_tab()
        tab_widget.addTab(economic_tab, "💵 Economic Parameters")
        
        # IRR & Optimization tab
        irr_tab = self._create_irr_limits_tab()
        tab_widget.addTab(irr_tab, "🔍 IRR & Optimization")
        
        # Modern action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("✖ Cancel")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("✔ Apply")
        apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {ModernColors.ACCENT_HOVER};
            }}
        """)
        apply_btn.clicked.connect(self._apply_limits)
        apply_btn.setDefault(True)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
        
        # Store spin boxes for later access
        self.spin_boxes = {}
    
    def _create_scenario_limits_tab(self) -> QWidget:
        """Create scenario generation limits tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.spin_boxes['num_scenarios'] = QSpinBox()
        self.spin_boxes['num_scenarios'].setRange(10, 100000)
        layout.addRow("Max Scenarios:", self.spin_boxes['num_scenarios'])
        
        self.spin_boxes['num_periods'] = QSpinBox()
        self.spin_boxes['num_periods'].setRange(5, 1000)
        layout.addRow("Max Periods:", self.spin_boxes['num_periods'])
        
        self.spin_boxes['price_volatility'] = QDoubleSpinBox()
        self.spin_boxes['price_volatility'].setRange(0.1, 10.0)
        self.spin_boxes['price_volatility'].setDecimals(2)
        layout.addRow("Max Price Volatility:", self.spin_boxes['price_volatility'])
        
        self.spin_boxes['grade_uncertainty'] = QDoubleSpinBox()
        self.spin_boxes['grade_uncertainty'].setRange(0.1, 10.0)
        self.spin_boxes['grade_uncertainty'].setDecimals(2)
        layout.addRow("Max Grade Uncertainty:", self.spin_boxes['grade_uncertainty'])
        
        self.spin_boxes['cost_uncertainty'] = QDoubleSpinBox()
        self.spin_boxes['cost_uncertainty'].setRange(0.1, 10.0)
        self.spin_boxes['cost_uncertainty'].setDecimals(2)
        layout.addRow("Max Cost Uncertainty:", self.spin_boxes['cost_uncertainty'])
        
        layout.addRow(QLabel(""))  # Spacer
        return widget
    
    def _create_economic_limits_tab(self) -> QWidget:
        """Create economic parameters limits tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.spin_boxes['metal_price'] = QDoubleSpinBox()
        self.spin_boxes['metal_price'].setRange(1.0, 1000000.0)
        self.spin_boxes['metal_price'].setDecimals(2)
        layout.addRow("Max Metal Price:", self.spin_boxes['metal_price'])
        
        self.spin_boxes['mining_cost'] = QDoubleSpinBox()
        self.spin_boxes['mining_cost'].setRange(1.0, 10000.0)
        self.spin_boxes['mining_cost'].setDecimals(2)
        layout.addRow("Max Mining Cost:", self.spin_boxes['mining_cost'])
        
        self.spin_boxes['processing_cost'] = QDoubleSpinBox()
        self.spin_boxes['processing_cost'].setRange(1.0, 10000.0)
        self.spin_boxes['processing_cost'].setDecimals(2)
        layout.addRow("Max Processing Cost:", self.spin_boxes['processing_cost'])
        
        self.spin_boxes['selling_cost'] = QDoubleSpinBox()
        self.spin_boxes['selling_cost'].setRange(1.0, 10000.0)
        self.spin_boxes['selling_cost'].setDecimals(2)
        layout.addRow("Max Selling Cost:", self.spin_boxes['selling_cost'])
        
        self.spin_boxes['recovery'] = QDoubleSpinBox()
        self.spin_boxes['recovery'].setRange(0.01, 1.0)
        self.spin_boxes['recovery'].setDecimals(3)
        layout.addRow("Max Recovery:", self.spin_boxes['recovery'])
        
        layout.addRow(QLabel(""))  # Spacer
        return widget
    
    def _create_irr_limits_tab(self) -> QWidget:
        """Create IRR and optimization limits tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        self.spin_boxes['production_capacity'] = QDoubleSpinBox()
        self.spin_boxes['production_capacity'].setRange(1000.0, 1e10)
        self.spin_boxes['production_capacity'].setDecimals(0)
        layout.addRow("Max Production Capacity:", self.spin_boxes['production_capacity'])
        
        self.spin_boxes['annual_rom'] = QDoubleSpinBox()
        self.spin_boxes['annual_rom'].setRange(1000.0, 1e10)
        self.spin_boxes['annual_rom'].setDecimals(0)
        layout.addRow("Max Annual ROM:", self.spin_boxes['annual_rom'])
        
        self.spin_boxes['num_phases'] = QSpinBox()
        self.spin_boxes['num_phases'].setRange(1, 100)
        layout.addRow("Max Pit Phases:", self.spin_boxes['num_phases'])
        
        self.spin_boxes['max_iterations'] = QSpinBox()
        self.spin_boxes['max_iterations'].setRange(5, 1000)
        layout.addRow("Max IRR Iterations:", self.spin_boxes['max_iterations'])
        
        self.spin_boxes['slope_angle'] = QDoubleSpinBox()
        self.spin_boxes['slope_angle'].setRange(10.0, 90.0)
        self.spin_boxes['slope_angle'].setDecimals(1)
        layout.addRow("Max Slope Angle (°):", self.spin_boxes['slope_angle'])
        
        layout.addRow(QLabel(""))  # Spacer
        return widget
    
    def _load_values(self):
        """Load current limit values into spin boxes."""
        for key, spin_box in self.spin_boxes.items():
            value = self.limits_manager.get_limit(key)
            spin_box.setValue(value)
    
    def _apply_limits(self):
        """Apply the configured limits."""
        for key, spin_box in self.spin_boxes.items():
            self.limits_manager.set_limit(key, spin_box.value())
        
        self.limits_updated.emit()
        QMessageBox.information(
            self,
            "Limits Updated",
            "Input limits have been updated successfully."
        )
        self.accept()
    
    def _reset_defaults(self):
        """Reset all limits to default values."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all limits to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.limits_manager.reset_to_defaults()
            self._load_values()
            QMessageBox.information(
                self,
                "Reset Complete",
                "All limits have been reset to default values."
            )




