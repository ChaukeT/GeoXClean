"""
Mouse menu construction for GeoX.

Extracted from MainWindow to improve maintainability.
Handles mouse interaction modes and zoom controls.

Architecture:
- Actions are created here and connected to MainWindow methods (for compatibility)
- Action references are bound to InteractionController for state management
- The controller owns all interaction logic; menu just wires UI elements
"""

import logging
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
except ImportError:
    def get_menu_icon(category, name):
        return None

logger = logging.getLogger(__name__)


def build_mouse_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """
    Build and return the Mouse menu.
    
    This function creates the Mouse menu and wires actions to MainWindow methods.
    The InteractionController (if present) is given references to the actions
    so it can manage their checked states.
    """
    mouse_menu = menubar.addMenu("&Mouse")
    
    # --- Interaction Mode Actions ---
    mouse_select_action = QAction(get_menu_icon("mouse", "select"), "&Select (Click)", main_window)
    mouse_select_action.setShortcut(QKeySequence("S"))
    mouse_select_action.setStatusTip("Enable selection/click mode (S)")
    mouse_select_action.triggered.connect(main_window.set_mouse_mode_select)
    mouse_select_action.setCheckable(True)
    mouse_menu.addAction(mouse_select_action)

    mouse_pan_action = QAction(get_menu_icon("mouse", "pan"), "&Pan", main_window)
    mouse_pan_action.setShortcut(QKeySequence("P"))
    mouse_pan_action.setStatusTip("Pan the view (P)")
    mouse_pan_action.triggered.connect(main_window.set_mouse_mode_pan)
    mouse_pan_action.setCheckable(True)
    mouse_menu.addAction(mouse_pan_action)

    mouse_zoom_box_action = QAction(get_menu_icon("mouse", "zoom"), "&Zoom (Box)", main_window)
    mouse_zoom_box_action.setShortcut(QKeySequence("Z"))
    mouse_zoom_box_action.setStatusTip("Drag a box to zoom (Z)")
    mouse_zoom_box_action.triggered.connect(main_window.set_mouse_mode_zoom_box)
    mouse_zoom_box_action.setCheckable(True)
    mouse_menu.addAction(mouse_zoom_box_action)

    # --- Action Group (exclusive selection) ---
    mouse_action_group = QActionGroup(main_window)
    mouse_action_group.setExclusive(True)
    mouse_action_group.addAction(mouse_select_action)
    mouse_action_group.addAction(mouse_pan_action)
    mouse_action_group.addAction(mouse_zoom_box_action)

    # Keep references on MainWindow for backward compatibility
    main_window.mouse_action_group = mouse_action_group
    main_window.mouse_select_action = mouse_select_action
    main_window.mouse_pan_action = mouse_pan_action
    main_window.mouse_zoom_box_action = mouse_zoom_box_action

    # --- Bind actions to InteractionController ---
    # The controller owns state management; giving it action references
    # allows it to update checked states without MainWindow involvement.
    interaction = getattr(main_window, 'interaction', None)
    if interaction is not None:
        try:
            interaction.bind_actions(
                select_action=mouse_select_action,
                pan_action=mouse_pan_action,
                zoom_box_action=mouse_zoom_box_action,
                action_group=mouse_action_group
            )
            # Sync controller state from viewer (if viewer already has a mode)
            interaction.sync_from_viewer()
            logger.debug("Bound mouse actions to InteractionController")
        except Exception as e:
            logger.debug(f"Could not bind actions to InteractionController: {e}")
    else:
        # Fallback: sync via MainWindow method (legacy path)
        try:
            current_mode = None
            if main_window.viewer_widget and getattr(main_window.viewer_widget, 'renderer', None) is not None:
                current_mode = getattr(main_window.viewer_widget.renderer, '_current_mouse_mode', None)
            if not current_mode:
                current_mode = 'original'
            main_window.update_mouse_action_checks(current_mode, show_message=False)
        except Exception:
            pass

    mouse_menu.addSeparator()

    # --- Zoom Actions ---
    zoom_in_action = QAction(get_menu_icon("mouse", "zoom_in"), "Zoom &In", main_window)
    zoom_in_action.setShortcut(QKeySequence("+"))
    zoom_in_action.setStatusTip("Zoom in")
    zoom_in_action.triggered.connect(main_window.zoom_in)
    mouse_menu.addAction(zoom_in_action)

    zoom_out_action = QAction(get_menu_icon("mouse", "zoom_out"), "Zoom &Out", main_window)
    zoom_out_action.setShortcut(QKeySequence("-"))
    zoom_out_action.setStatusTip("Zoom out")
    zoom_out_action.triggered.connect(main_window.zoom_out)
    mouse_menu.addAction(zoom_out_action)

    zoom_extents_action = QAction(get_menu_icon("mouse", "zoom_extents"), "Zoom to &Extents", main_window)
    zoom_extents_action.setShortcut(QKeySequence("E"))
    zoom_extents_action.setStatusTip("Zoom to full model extents")
    zoom_extents_action.triggered.connect(main_window.fit_to_view)
    mouse_menu.addAction(zoom_extents_action)

    # --- Reset Mode Action ---
    mouse_menu.addSeparator()
    reset_mouse_action = QAction("Reset Mouse (Original)", main_window)
    reset_mouse_action.setStatusTip("Restore original mouse interaction mode")
    reset_mouse_action.triggered.connect(main_window.set_mouse_mode_reset)
    mouse_menu.addAction(reset_mouse_action)
    
    return mouse_menu

