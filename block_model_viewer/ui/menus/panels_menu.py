"""
Dynamic Panels menu for GeoX.

Lists all registered panels grouped by category with toggle actions.
Includes utility actions for clearing panels and resetting layout.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMenuBar, QMenu, QMessageBox
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

from ..panel_manager import PanelCategory

logger = logging.getLogger(__name__)

# User-friendly category names and display order
CATEGORY_DISPLAY = [
    (PanelCategory.DRILLHOLE, "Drillhole"),
    (PanelCategory.GEOSTATS, "Geostatistics"),
    (PanelCategory.RESOURCE, "Resource Estimation"),
    (PanelCategory.PLANNING, "Mine Planning"),
    (PanelCategory.OPTIMIZATION, "Optimization"),
    (PanelCategory.GEOTECH, "Geotechnical"),
    (PanelCategory.ESG, "ESG & Sustainability"),
    (PanelCategory.ANALYSIS, "Analysis"),
    (PanelCategory.CHART, "Charts & Visualization"),
    (PanelCategory.PROPERTY, "Property Controls"),
    (PanelCategory.SCENE, "Scene"),
    (PanelCategory.INFO, "Information"),
    (PanelCategory.SELECTION, "Selection"),
    (PanelCategory.DISPLAY, "Display Settings"),
    (PanelCategory.CROSS_SECTION, "Cross-Section"),
    (PanelCategory.CONFIG, "Configuration"),
    (PanelCategory.REPORT, "Reports"),
    (PanelCategory.OTHER, "Other"),
]


def build_panels_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build dynamic Panels menu grouped by category."""
    panels_menu = menubar.addMenu("&Panels")

    # Get panel manager
    panel_manager = getattr(main_window, 'panel_manager', None)

    if panel_manager:
        _build_category_submenus(main_window, panels_menu, panel_manager)
    else:
        # Fallback: basic menu if panel_manager not available
        _build_basic_menu(main_window, panels_menu)

    # Add separator before utility actions
    panels_menu.addSeparator()

    # Clear All Visible Panels action
    clear_action = QAction("Clear All &Visible Panels", main_window)
    clear_action.setStatusTip("Clear UI state of all visible panels (reset to defaults)")
    clear_action.triggered.connect(lambda: _clear_all_visible_panels(main_window))
    panels_menu.addAction(clear_action)

    panels_menu.addSeparator()

    # Reset Layout action
    reset_action = QAction("&Reset Layout", main_window)
    reset_action.setStatusTip("Reset all panels to default positions")
    reset_action.triggered.connect(main_window.reset_layout)
    panels_menu.addAction(reset_action)

    return panels_menu


def _build_category_submenus(main_window: 'MainWindow', panels_menu: QMenu, panel_manager):
    """Build submenus for each category with panel toggle actions."""
    for category, display_name in CATEGORY_DISPLAY:
        try:
            panels = panel_manager.get_panels_by_category(category)
            if not panels:
                continue

            # Create category submenu
            submenu = panels_menu.addMenu(display_name)
            submenu.setToolTipsVisible(True)

            # Sort panels alphabetically by name
            for panel_info in sorted(panels, key=lambda p: p.name):
                action = QAction(panel_info.name, main_window)
                action.setCheckable(True)

                # Check current visibility
                try:
                    is_visible = panel_manager.is_panel_visible(panel_info.panel_id)
                    action.setChecked(is_visible)
                except Exception:
                    action.setChecked(False)

                # Set tooltip if available
                if panel_info.tooltip:
                    action.setToolTip(panel_info.tooltip)

                # Connect toggle action
                panel_id = panel_info.panel_id
                action.triggered.connect(
                    lambda checked, pid=panel_id: _toggle_panel(panel_manager, pid)
                )

                # Store action reference for state updates
                panel_info.menu_action = action
                submenu.addAction(action)

        except Exception as e:
            logger.warning(f"Error building submenu for category {category}: {e}")
            continue


def _build_basic_menu(main_window: 'MainWindow', panels_menu: QMenu):
    """Fallback basic menu if panel_manager not available."""
    # Add toggle for left dock if it exists
    if hasattr(main_window, 'left_dock') and main_window.left_dock:
        panels_menu.addAction(main_window.left_dock.toggleViewAction())
        main_window.left_dock.toggleViewAction().setText("&Controls && Scene (Left)")


def _toggle_panel(panel_manager, panel_id: str):
    """Toggle panel visibility."""
    try:
        panel_manager.toggle_panel(panel_id)
    except Exception as e:
        logger.error(f"Error toggling panel {panel_id}: {e}")


def _clear_all_visible_panels(main_window: 'MainWindow'):
    """Clear UI state of all visible panels."""
    panel_manager = getattr(main_window, 'panel_manager', None)
    if not panel_manager:
        QMessageBox.warning(
            main_window,
            "Not Available",
            "Panel manager not initialized."
        )
        return

    try:
        cleared = panel_manager.clear_all_visible_panels()
        main_window.statusBar().showMessage(f"Cleared {cleared} panels", 3000)
        logger.info(f"Cleared {cleared} visible panels")
    except Exception as e:
        logger.error(f"Error clearing panels: {e}")
        QMessageBox.warning(
            main_window,
            "Error",
            f"Failed to clear panels: {e}"
        )

