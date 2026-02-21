"""
Tools menu construction for GeoX.
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


def build_tools_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Tools menu."""
    tools_menu = menubar.addMenu("&Tools")
    
    slice_tool = QAction(get_menu_icon("tools", "slice"), "&Slice Tool", main_window)
    slice_tool.setStatusTip("Open spatial slicing controls")
    slice_tool.triggered.connect(main_window.open_slice_tool)
    tools_menu.addAction(slice_tool)
    
    filter_tool = QAction(get_menu_icon("tools", "filter"), "&Filter Tool", main_window)
    filter_tool.setStatusTip("Open property filtering controls")
    filter_tool.triggered.connect(main_window.open_filter_tool)
    tools_menu.addAction(filter_tool)
    
    tools_menu.addSeparator()
    
    main_window.pick_mode_action = QAction(get_menu_icon("tools", "selection"), "Selection &Mode", main_window)
    main_window.pick_mode_action.setCheckable(True)
    main_window.pick_mode_action.setChecked(True)
    main_window.pick_mode_action.setStatusTip("Enable/disable block selection")
    main_window.pick_mode_action.triggered.connect(main_window.toggle_pick_mode)
    tools_menu.addAction(main_window.pick_mode_action)
    
    tools_menu.addSeparator()
    
    stats_tool = QAction(get_menu_icon("tools", "statistics"), "Property &Statistics", main_window)
    stats_tool.setStatusTip("Show property statistics")
    stats_tool.triggered.connect(main_window.show_statistics)
    tools_menu.addAction(stats_tool)
    
    tools_menu.addSeparator()
    
    # Selection Manager
    selection_tool = QAction(get_menu_icon("tools", "block_selection"), "Block Selection Manager...", main_window)
    selection_tool.setStatusTip("Multi-block selection, named sets, and export")
    selection_tool.triggered.connect(main_window.open_selection_manager)
    tools_menu.addAction(selection_tool)
    
    # Cross-Section Manager
    cross_section_tool = QAction(get_menu_icon("view", "cross_section"), "Cross-Section Manager...", main_window)
    cross_section_tool.setStatusTip("Manage named cross-sections and quick rendering")
    cross_section_tool.triggered.connect(main_window.open_cross_section_manager)
    tools_menu.addAction(cross_section_tool)

    # Interactive Slicer
    interactive_slicer_tool = QAction(get_menu_icon("tools", "slice"), "Interactive Slicer...", main_window)
    interactive_slicer_tool.setStatusTip("Interactive slicing with draggable plane, box, sphere widgets")
    interactive_slicer_tool.triggered.connect(main_window.open_interactive_slicer)
    tools_menu.addAction(interactive_slicer_tool)

    return tools_menu

