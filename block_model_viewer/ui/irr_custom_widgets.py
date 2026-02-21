"""
Custom Widgets for IRR Input Panel

Provides auto-adjusting spin boxes and unit selectors for IRR analysis inputs.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QSpinBox, QWidget, QHBoxLayout, QComboBox,
    QLabel, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from .modern_styles import ModernColors

logger = logging.getLogger(__name__)


class AutoAdjustSpinBox(QSpinBox):
    """
    Spin box that automatically adjusts its maximum when user exceeds limit.
    """
    
    limit_exceeded = pyqtSignal(int)  # Emits proposed new maximum
    
    def __init__(self, param_name: str, limits_manager=None, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.limits_manager = limits_manager
        self.auto_adjust_enabled = True
        
        # Set initial range
        if limits_manager:
            max_value = int(limits_manager.get_limit(param_name))
            self.setRange(0, max_value)
        
        # Connect signals
        self.valueChanged.connect(self._check_limit)
    
    def _check_limit(self, value: int):
        """Check if value approaches maximum and offer to increase."""
        if not self.auto_adjust_enabled:
            return
        
        if value >= self.maximum() * 0.95:  # Within 5% of max
            self._offer_limit_increase(value)
    
    def _offer_limit_increase(self, proposed_value: int):
        """Offer to increase the maximum limit."""
        new_max = max(proposed_value * 2, self.maximum() * 2)
        
        reply = QMessageBox.question(
            self,
            "Increase Limit?",
            f"The entered value ({proposed_value}) is close to the current maximum ({self.maximum()}).\n\n"
            f"Would you like to increase the maximum to {new_max}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.setMaximum(new_max)
            if self.limits_manager:
                self.limits_manager.set_limit(self.param_name, new_max)
            self.limit_exceeded.emit(new_max)
            logger.info(f"Increased {self.param_name} limit to {new_max}")
    
    def update_limit(self, new_max: int):
        """Update the maximum value from external source."""
        self.setMaximum(new_max)


class AutoAdjustDoubleSpinBox(QDoubleSpinBox):
    """
    Double spin box that automatically adjusts its maximum when user exceeds limit.
    """
    
    limit_exceeded = pyqtSignal(float)  # Emits proposed new maximum
    
    def __init__(self, param_name: str, limits_manager=None, decimals: int = 2, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.limits_manager = limits_manager
        self.auto_adjust_enabled = True
        
        self.setDecimals(decimals)
        
        # Set initial range
        if limits_manager:
            max_value = limits_manager.get_limit(param_name)
            self.setRange(0.0, max_value)
        
        # Connect signals
        self.valueChanged.connect(self._check_limit)
    
    def _check_limit(self, value: float):
        """Check if value approaches maximum and offer to increase."""
        if not self.auto_adjust_enabled:
            return
        
        if value >= self.maximum() * 0.95:  # Within 5% of max
            self._offer_limit_increase(value)
    
    def _offer_limit_increase(self, proposed_value: float):
        """Offer to increase the maximum limit."""
        new_max = max(proposed_value * 2, self.maximum() * 2)
        
        reply = QMessageBox.question(
            self,
            "Increase Limit?",
            f"The entered value ({proposed_value:.2f}) is close to the current maximum ({self.maximum():.2f}).\n\n"
            f"Would you like to increase the maximum to {new_max:.2f}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.setMaximum(new_max)
            if self.limits_manager:
                self.limits_manager.set_limit(self.param_name, new_max)
            self.limit_exceeded.emit(new_max)
            logger.info(f"Increased {self.param_name} limit to {new_max:.2f}")
    
    def update_limit(self, new_max: float):
        """Update the maximum value from external source."""
        self.setMaximum(new_max)


class UnitValueSpinBox(QWidget):
    """
    Combined widget with spin box and unit selector.
    """
    
    valueChanged = pyqtSignal(float)  # Emits value in base units
    unitChanged = pyqtSignal(str)  # Emits selected unit
    
    def __init__(self, param_name: str, available_units: list, 
                 limits_manager=None, unit_manager=None, 
                 decimals: int = 2, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.limits_manager = limits_manager
        self.unit_manager = unit_manager
        self.available_units = available_units
        self._updating = False  # Prevent recursive updates
        
        self._setup_ui(decimals)
        
        # Load saved unit preference
        if unit_manager:
            saved_unit = unit_manager.get_unit(param_name)
            idx = self.unit_combo.findText(saved_unit)
            if idx >= 0:
                self.unit_combo.setCurrentIndex(idx)
    
    def _setup_ui(self, decimals: int):
        """Setup the combined widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Spin box
        self.spin_box = AutoAdjustDoubleSpinBox(
            self.param_name, 
            self.limits_manager, 
            decimals, 
            self
        )
        self.spin_box.setSingleStep(0.1 if decimals > 0 else 1.0)
        self.spin_box.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.spin_box, stretch=2)
        
        # Unit selector
        self.unit_combo = QComboBox(self)
        self.unit_combo.addItems(self.available_units)
        self.unit_combo.currentTextChanged.connect(self._on_unit_changed)
        self.unit_combo.setMinimumWidth(100)
        layout.addWidget(self.unit_combo, stretch=1)
    
    def _on_value_changed(self, value: float):
        """Handle spin box value change."""
        if not self._updating:
            base_value = self._convert_to_base(value)
            self.valueChanged.emit(base_value)
    
    def _on_unit_changed(self, unit: str):
        """Handle unit selection change."""
        if self._updating:
            return
        
        # Save unit preference
        if self.unit_manager:
            self.unit_manager.set_unit(self.param_name, unit)
        
        # Update spin box range for new unit
        self._update_range_for_unit()
        
        self.unitChanged.emit(unit)
        logger.info(f"Changed {self.param_name} unit to {unit}")
    
    def _update_range_for_unit(self):
        """Update spin box range based on current unit."""
        if not self.limits_manager:
            return
        
        base_max = self.limits_manager.get_limit(self.param_name)
        display_max = self._convert_from_base(base_max)
        
        self._updating = True
        current_base_value = self.get_base_value()
        self.spin_box.setRange(0.0, display_max)
        self.set_base_value(current_base_value)
        self._updating = False
    
    def _convert_to_base(self, value: float) -> float:
        """Convert displayed value to base units."""
        if self.unit_manager:
            return self.unit_manager.convert_to_base(self.param_name, value)
        return value
    
    def _convert_from_base(self, value: float) -> float:
        """Convert base units to displayed value."""
        if self.unit_manager:
            return self.unit_manager.convert_from_base(self.param_name, value)
        return value
    
    def get_base_value(self) -> float:
        """Get value in base units."""
        return self._convert_to_base(self.spin_box.value())
    
    def set_base_value(self, base_value: float):
        """Set value from base units."""
        display_value = self._convert_from_base(base_value)
        self._updating = True
        self.spin_box.setValue(display_value)
        self._updating = False
    
    def value(self) -> float:
        """Get displayed value (not base units)."""
        return self.spin_box.value()
    
    def setValue(self, value: float):
        """Set displayed value (not base units)."""
        self.spin_box.setValue(value)
    
    def get_unit(self) -> str:
        """Get current unit."""
        return self.unit_combo.currentText()
    
    def set_unit(self, unit: str):
        """Set current unit."""
        idx = self.unit_combo.findText(unit)
        if idx >= 0:
            self.unit_combo.setCurrentIndex(idx)
    
    def setToolTip(self, tooltip: str):
        """Set tooltip for both widgets."""
        self.spin_box.setToolTip(tooltip)
        self.unit_combo.setToolTip(tooltip)


class ParameterRow(QWidget):
    """
    Complete parameter row with label, tooltip, and input widget.
    Modern styled version with better visual hierarchy.
    """
    
    def __init__(self, label_text: str, widget: QWidget, 
                 tooltip: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._setup_ui(label_text, widget, tooltip)
    
    def _setup_ui(self, label_text: str, widget: QWidget, tooltip: Optional[str]):
        """Setup the parameter row UI with modern styling."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)
        
        # Modern styled label with tooltip
        label = QLabel(label_text)
        label.setMinimumWidth(200)
        label.setFont(QFont("Segoe UI", 10))
        label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        
        if tooltip:
            label.setToolTip(tooltip)
            label.setStyleSheet(f"""
                color: {ModernColors.TEXT_PRIMARY};
                padding: 2px 4px;
                border-radius: 4px;
            """)
            label.setCursor(Qt.CursorShape.WhatsThisCursor)
            widget.setToolTip(tooltip)
        
        layout.addWidget(label)
        layout.addWidget(widget, stretch=1)


def create_tooltip(param_name: str, unit: str = "") -> str:
    """Create informative tooltip for a parameter."""
    tooltips = {
        "num_scenarios": "Number of stochastic scenarios to generate for uncertainty analysis",
        "num_periods": "Number of time periods for mine scheduling and cash flow analysis",
        "metal_price": f"Initial metal price ({unit}) for revenue calculations",
        "mining_cost": f"Cost per tonne ({unit}) for mining operations",
        "processing_cost": f"Cost per tonne ({unit}) for ore processing and beneficiation",
        "selling_cost": f"Marketing and selling cost ({unit}) per unit sold",
        "recovery": "Metallurgical recovery rate (fraction of metal recovered from ore)",
        "price_volatility": "Standard deviation of price fluctuations (higher = more volatile)",
        "grade_uncertainty": "Coefficient of variation for grade estimation uncertainty",
        "cost_uncertainty": "Coefficient of variation for cost estimation uncertainty",
        "production_capacity": "Maximum annual production capacity in tonnes",
        "annual_rom": "Annual run-of-mine (ROM) tonnage target",
        "num_phases": "Number of mining phases for pit development",
        "alpha": "Confidence level for risk-adjusted IRR (0.75 = 75% confidence)",
        "slope_angle": "Overall pit slope angle in degrees for wall stability"
    }
    
    return tooltips.get(param_name, f"Parameter: {param_name}")




