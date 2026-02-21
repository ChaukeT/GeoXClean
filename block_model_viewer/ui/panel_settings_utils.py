"""
Panel Settings Utilities for GeoX.

Helper functions for saving and restoring Qt widget states in analysis panels.
These utilities provide a consistent way to serialize and deserialize widget
values for project save/load operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


def save_widget_state(widget: 'QWidget') -> Optional[Any]:
    """
    Save the state of a Qt widget to a JSON-serializable value.
    
    Supports:
    - QComboBox: currentText()
    - QSpinBox: value()
    - QDoubleSpinBox: value()
    - QCheckBox: isChecked()
    - QRadioButton: isChecked()
    - QLineEdit: text()
    - QTextEdit/QPlainTextEdit: toPlainText()
    - QSlider: value()
    
    Args:
        widget: Qt widget to save state from
        
    Returns:
        JSON-serializable value, or None if widget type not supported
    """
    if widget is None:
        return None
    
    try:
        from PyQt6.QtWidgets import (
            QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
            QLineEdit, QTextEdit, QPlainTextEdit, QSlider
        )
        
        if isinstance(widget, QComboBox):
            return widget.currentText()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        elif isinstance(widget, (QCheckBox, QRadioButton)):
            return widget.isChecked()
        elif isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
            return widget.toPlainText()
        elif isinstance(widget, QSlider):
            return widget.value()
        else:
            logger.debug(f"Unsupported widget type for saving: {type(widget).__name__}")
            return None
    except Exception as e:
        logger.warning(f"Error saving widget state: {e}")
        return None


def restore_widget_state(widget: 'QWidget', value: Any) -> bool:
    """
    Restore the state of a Qt widget from a saved value.
    
    Args:
        widget: Qt widget to restore state to
        value: Previously saved value
        
    Returns:
        True if restoration succeeded, False otherwise
    """
    if widget is None or value is None:
        return False
    
    try:
        from PyQt6.QtWidgets import (
            QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
            QLineEdit, QTextEdit, QPlainTextEdit, QSlider
        )
        
        if isinstance(widget, QComboBox):
            if isinstance(value, str):
                idx = widget.findText(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                    return True
                # If exact text not found, try case-insensitive search
                for i in range(widget.count()):
                    if widget.itemText(i).lower() == value.lower():
                        widget.setCurrentIndex(i)
                        return True
            elif isinstance(value, int):
                if 0 <= value < widget.count():
                    widget.setCurrentIndex(value)
                    return True
            return False
            
        elif isinstance(widget, QSpinBox):
            if isinstance(value, (int, float)):
                # Clamp to widget's range
                clamped = max(widget.minimum(), min(widget.maximum(), int(value)))
                widget.setValue(clamped)
                return True
            return False
            
        elif isinstance(widget, QDoubleSpinBox):
            if isinstance(value, (int, float)):
                # Clamp to widget's range
                clamped = max(widget.minimum(), min(widget.maximum(), float(value)))
                widget.setValue(clamped)
                return True
            return False
            
        elif isinstance(widget, (QCheckBox, QRadioButton)):
            if isinstance(value, bool):
                widget.setChecked(value)
                return True
            return False
            
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if value is not None else "")
            return True
            
        elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
            widget.setPlainText(str(value) if value is not None else "")
            return True
            
        elif isinstance(widget, QSlider):
            if isinstance(value, (int, float)):
                clamped = max(widget.minimum(), min(widget.maximum(), int(value)))
                widget.setValue(clamped)
                return True
            return False
        else:
            logger.debug(f"Unsupported widget type for restoring: {type(widget).__name__}")
            return False
            
    except Exception as e:
        logger.warning(f"Error restoring widget state: {e}")
        return False


def save_widgets_dict(widgets: Dict[str, 'QWidget']) -> Dict[str, Any]:
    """
    Save multiple widget states to a dictionary.
    
    Args:
        widgets: Dictionary mapping names to widget instances
        
    Returns:
        Dictionary of saved states (only includes successful saves)
    """
    result = {}
    for name, widget in widgets.items():
        try:
            value = save_widget_state(widget)
            if value is not None:
                result[name] = value
        except Exception as e:
            logger.debug(f"Could not save widget '{name}': {e}")
    return result


def restore_widgets_dict(widgets: Dict[str, 'QWidget'], settings: Dict[str, Any]) -> int:
    """
    Restore multiple widget states from a dictionary.
    
    Args:
        widgets: Dictionary mapping names to widget instances
        settings: Dictionary of saved states
        
    Returns:
        Number of widgets successfully restored
    """
    restored_count = 0
    for name, value in settings.items():
        if name in widgets:
            try:
                if restore_widget_state(widgets[name], value):
                    restored_count += 1
            except Exception as e:
                logger.debug(f"Could not restore widget '{name}': {e}")
    return restored_count


def save_radio_group(checked_widget_name: str) -> str:
    """
    Save the state of a radio button group by returning the name of the checked button.
    
    Args:
        checked_widget_name: Name/identifier of the checked radio button
        
    Returns:
        The name to save
    """
    return checked_widget_name


def save_table_widget(table_widget: 'QWidget') -> Optional[list]:
    """
    Save the contents of a QTableWidget to a list of rows.
    
    Args:
        table_widget: QTableWidget instance
        
    Returns:
        List of row data, or None if not a table widget
    """
    try:
        from PyQt6.QtWidgets import QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox
        
        if not isinstance(table_widget, QTableWidget):
            return None
        
        rows = []
        for row in range(table_widget.rowCount()):
            row_data = []
            for col in range(table_widget.columnCount()):
                item = table_widget.item(row, col)
                cell_widget = table_widget.cellWidget(row, col)
                
                if cell_widget:
                    # Handle cell widgets
                    value = save_widget_state(cell_widget)
                    row_data.append({'type': 'widget', 'value': value, 'widget_type': type(cell_widget).__name__})
                elif item:
                    row_data.append({'type': 'item', 'value': item.text()})
                else:
                    row_data.append({'type': 'empty', 'value': None})
            rows.append(row_data)
        return rows
    except Exception as e:
        logger.warning(f"Error saving table widget: {e}")
        return None


def get_safe_widget_value(panel, attr_name: str, default: Any = None) -> Any:
    """
    Safely get a widget value from a panel attribute.
    
    Args:
        panel: Panel instance
        attr_name: Attribute name of the widget
        default: Default value if widget doesn't exist or fails
        
    Returns:
        Widget value or default
    """
    try:
        if not hasattr(panel, attr_name):
            return default
        widget = getattr(panel, attr_name)
        if widget is None:
            return default
        value = save_widget_state(widget)
        return value if value is not None else default
    except Exception:
        return default


def set_safe_widget_value(panel, attr_name: str, value: Any) -> bool:
    """
    Safely set a widget value on a panel attribute.
    
    Args:
        panel: Panel instance
        attr_name: Attribute name of the widget
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not hasattr(panel, attr_name):
            return False
        widget = getattr(panel, attr_name)
        if widget is None:
            return False
        return restore_widget_state(widget, value)
    except Exception:
        return False


def collect_panel_settings(panel, widget_attrs: Dict[str, str]) -> Dict[str, Any]:
    """
    Collect settings from multiple widget attributes on a panel.
    
    Args:
        panel: Panel instance
        widget_attrs: Dictionary mapping setting names to widget attribute names
                     e.g., {'variable': 'variable_combo', 'nreal': 'nreal_spin'}
        
    Returns:
        Dictionary of collected settings
    """
    settings = {}
    for setting_name, attr_name in widget_attrs.items():
        value = get_safe_widget_value(panel, attr_name)
        if value is not None:
            settings[setting_name] = value
    return settings


def apply_panel_settings(panel, settings: Dict[str, Any], widget_attrs: Dict[str, str]) -> int:
    """
    Apply settings to multiple widget attributes on a panel.
    
    Args:
        panel: Panel instance
        settings: Dictionary of settings to apply
        widget_attrs: Dictionary mapping setting names to widget attribute names
        
    Returns:
        Number of settings successfully applied
    """
    applied_count = 0
    for setting_name, attr_name in widget_attrs.items():
        if setting_name in settings:
            if set_safe_widget_value(panel, attr_name, settings[setting_name]):
                applied_count += 1
    return applied_count

