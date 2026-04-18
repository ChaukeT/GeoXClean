"""
Tools menu construction for GeoX.
"""

import logging
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
except ImportError:
    def get_menu_icon(category, name):
        return None

logger = logging.getLogger(__name__)


def _safe_connect(action, main_window, method_name):
    handler = getattr(main_window, method_name, None)
    if handler is not None:
        action.triggered.connect(handler)
    else:
        action.setEnabled(False)
        logger.debug("tools_menu: MainWindow.%s not found — action disabled", method_name)


def build_tools_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Tools menu."""
    tools_menu = menubar.addMenu("&Tools")

    slice_tool = QAction(get_menu_icon("tools", "slice"), "&Slice Tool", main_window)
    slice_tool.setStatusTip("Open spatial slicing controls")
    _safe_connect(slice_tool, main_window, 'open_slice_tool')
    tools_menu.addAction(slice_tool)

    filter_tool = QAction(get_menu_icon("tools", "filter"), "&Filter Tool", main_window)
    filter_tool.setStatusTip("Open property filtering controls")
    _safe_connect(filter_tool, main_window, 'open_filter_tool')
    tools_menu.addAction(filter_tool)

    tools_menu.addSeparator()

    pick_action = QAction(get_menu_icon("tools", "selection"), "Selection &Mode", main_window)
    pick_action.setCheckable(True)
    pick_action.setChecked(True)
    pick_action.setStatusTip("Enable/disable block selection")
    _safe_connect(pick_action, main_window, 'toggle_pick_mode')
    main_window.pick_mode_action = pick_action
    tools_menu.addAction(pick_action)

    tools_menu.addSeparator()

    stats_tool = QAction(get_menu_icon("tools", "statistics"), "Property &Statistics", main_window)
    stats_tool.setStatusTip("Show property statistics")
    _safe_connect(stats_tool, main_window, 'show_statistics')
    tools_menu.addAction(stats_tool)

    tools_menu.addSeparator()

    selection_tool = QAction(get_menu_icon("tools", "block_selection"), "Block Selection Manager...", main_window)
    selection_tool.setStatusTip("Multi-block selection, named sets, and export")
    _safe_connect(selection_tool, main_window, 'open_selection_manager')
    tools_menu.addAction(selection_tool)

    cross_section_tool = QAction(get_menu_icon("view", "cross_section"), "Cross-Section Manager...", main_window)
    cross_section_tool.setStatusTip("Manage named cross-sections and quick rendering")
    _safe_connect(cross_section_tool, main_window, 'open_cross_section_manager')
    tools_menu.addAction(cross_section_tool)

    interactive_slicer_tool = QAction(get_menu_icon("tools", "slice"), "Interactive Slicer...", main_window)
    interactive_slicer_tool.setStatusTip("Interactive slicing with draggable plane, box, sphere widgets")
    _safe_connect(interactive_slicer_tool, main_window, 'open_interactive_slicer')
    tools_menu.addAction(interactive_slicer_tool)

    clip_plane_tool = QAction(get_menu_icon("tools", "clip"), "&Clip Plane...", main_window)
    clip_plane_tool.setShortcut(QKeySequence("Ctrl+Shift+C"))
    clip_plane_tool.setStatusTip("ParaView-style clip plane with draggable handles")
    _safe_connect(clip_plane_tool, main_window, 'toggle_clip_plane')
    tools_menu.addAction(clip_plane_tool)

    return tools_menu
