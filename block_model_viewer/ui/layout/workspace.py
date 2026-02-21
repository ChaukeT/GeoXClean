"""Workspace layout management for MainWindow."""

import json
import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QByteArray

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


def load_workspace_layout(main_window: 'MainWindow', layout_name: str) -> None:
    """Load a predefined workspace layout.
    
    Args:
        main_window: MainWindow instance.
        layout_name: Layout name ("geology", "resource", "planning", "analytics").
    """
    try:
        layout_config = main_window.panel_registry.get_workspace_layout(layout_name)
        main_window.current_workspace_layout = layout_name
        
        # Apply layout configuration
        # This is a simplified version - full implementation would restore dock positions
        logger.info(f"Loading workspace layout: {layout_name}")
        main_window.status_bar.showMessage(f"Workspace layout: {layout_name.capitalize()}", 2000)
    except Exception as e:
        logger.error(f"Error loading workspace layout: {e}", exc_info=True)


def reset_workspace_layout(main_window: 'MainWindow') -> None:
    """Reset workspace to default layout.
    
    Args:
        main_window: MainWindow instance.
    """
    try:
        layout_config = main_window.panel_registry.get_default_layout()
        main_window.current_workspace_layout = "default"
        main_window.reset_layout()
        logger.info("Reset workspace layout to default")
        main_window.status_bar.showMessage("Workspace layout reset", 2000)
    except Exception as e:
        logger.error(f"Error resetting workspace layout: {e}", exc_info=True)


def save_workspace_layout(main_window: 'MainWindow') -> None:
    """Save current workspace layout to file.
    
    Args:
        main_window: MainWindow instance.
    """
    try:
        filename, _ = QFileDialog.getSaveFileName(
            main_window,
            "Save Workspace Layout",
            "",
            "JSON Files (*.json)"
        )
        if filename:
            layout_data = {
                "layout_name": main_window.current_workspace_layout,
                "dock_states": main_window.saveState().toHex().data().decode() if hasattr(main_window.saveState(), 'toHex') else None,
                "geometry": main_window.saveGeometry().toHex().data().decode() if hasattr(main_window.saveGeometry(), 'toHex') else None
            }
            with open(filename, 'w') as f:
                json.dump(layout_data, f, indent=2)
            logger.info(f"Saved workspace layout to {filename}")
            main_window.status_bar.showMessage(f"Workspace layout saved", 2000)
    except Exception as e:
        logger.error(f"Error saving workspace layout: {e}", exc_info=True)


def load_workspace_layout_file(main_window: 'MainWindow') -> None:
    """Load workspace layout from file.
    
    Args:
        main_window: MainWindow instance.
    """
    try:
        filename, _ = QFileDialog.getOpenFileName(
            main_window,
            "Load Workspace Layout",
            "",
            "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'r') as f:
                layout_data = json.load(f)
            
            # Restore dock states
            if 'dock_states' in layout_data and layout_data['dock_states']:
                state = QByteArray.fromHex(layout_data['dock_states'].encode())
                main_window.restoreState(state)
            
            # Ensure drillhole control dock is visible after layout restoration
            if main_window.drillhole_control_dock:
                main_window.drillhole_control_dock.setVisible(True)
                main_window.drillhole_control_dock.raise_()
            
            logger.info(f"Loaded workspace layout from {filename}")
            main_window.status_bar.showMessage(f"Workspace layout loaded", 2000)
    except Exception as e:
        logger.error(f"Error loading workspace layout: {e}", exc_info=True)

