"""
Search menu construction for GeoX.

Provides a modern command palette for quickly finding and executing
modules, actions, and commands throughout the application.
"""

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


def build_search_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Search menu.

    Creates a Search menu with a command palette action that allows
    users to quickly search and execute any action in the application.
    """
    search_menu = menubar.addMenu("&Search")
    search_menu.setObjectName("searchMenu")

    # Command Palette action
    search_action = QAction(
        get_menu_icon("search", "search"),
        "Command Palette...",
        main_window
    )
    search_action.setShortcut(QKeySequence.StandardKey.Find)
    search_action.setStatusTip("Search for modules, actions, and commands (Ctrl+F)")
    search_action.triggered.connect(main_window._search_modules)
    search_menu.addAction(search_action)

    return search_menu

