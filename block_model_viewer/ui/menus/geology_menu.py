"""
Geological Modeling menu construction for GeoX.
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


def build_geology_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Geological Modeling menu."""
    geology_menu = menubar.addMenu("&Geological Modeling")

    # LoopStructural Modeling
    loopstructural_action = QAction(get_menu_icon("geology", "loopstructural"), "LoopStructural Modeler...", main_window)
    loopstructural_action.setStatusTip("Industry-grade geological modeling with JORC/SAMREC compliance")
    loopstructural_action.triggered.connect(main_window.open_loopstructural_panel)
    geology_menu.addAction(loopstructural_action)

    geology_menu.addSeparator()

    # Note about LoopStructural capabilities
    geology_info_action = QAction("About LoopStructural...", main_window)
    geology_info_action.setStatusTip("Information about LoopStructural geological modeling")
    geology_info_action.triggered.connect(main_window._show_loopstructural_info)
    geology_menu.addAction(geology_info_action)

    return geology_menu

