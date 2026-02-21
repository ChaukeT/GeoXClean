"""
Resources menu construction for GeoX.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
except ImportError:
    def get_menu_icon(category, name):
        return None


def build_resources_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Resources menu."""
    resources_menu = menubar.addMenu("&Resources")
    
    # Block Model Resources
    block_resource_action = QAction(get_menu_icon("resources", "block_resources"), "Block Model Resources", main_window)
    block_resource_action.setStatusTip("Calculate block model resources with cut-off logic")
    block_resource_action.triggered.connect(main_window.open_block_resource_panel)
    resources_menu.addAction(block_resource_action)
    
    resources_menu.addSeparator()
    
    # Sensitivity Analysis
    sensitivity_action = QAction(get_menu_icon("resources", "sensitivity"), "Cut-off Sensitivity Analysis", main_window)
    sensitivity_action.setStatusTip("Perform cut-off sensitivity analysis")
    sensitivity_action.triggered.connect(main_window.open_sensitivity_panel)
    resources_menu.addAction(sensitivity_action)
    
    resources_menu.addSeparator()
    
    return resources_menu

