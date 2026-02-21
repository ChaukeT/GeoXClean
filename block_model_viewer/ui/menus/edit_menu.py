"""
Edit menu construction for GeoX.
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


def build_edit_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Edit menu."""
    edit_menu = menubar.addMenu("&Edit")
    
    # Undo/Redo
    main_window.undo_action = QAction(get_menu_icon("edit", "undo"), "&Undo", main_window)
    main_window.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
    main_window.undo_action.setStatusTip("Undo last visual change")
    main_window.undo_action.triggered.connect(main_window._undo)
    edit_menu.addAction(main_window.undo_action)

    main_window.redo_action = QAction(get_menu_icon("edit", "redo"), "&Redo", main_window)
    main_window.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
    main_window.redo_action.setStatusTip("Redo last undone change")
    main_window.redo_action.triggered.connect(main_window._redo)
    edit_menu.addAction(main_window.redo_action)

    # Preferences
    preferences_action = QAction(get_menu_icon("edit", "preferences"), "&Preferences...", main_window)
    preferences_action.setShortcut(QKeySequence("Ctrl+,"))
    preferences_action.setStatusTip("Open application preferences")
    preferences_action.triggered.connect(main_window.open_preferences)
    edit_menu.addSeparator()
    edit_menu.addAction(preferences_action)

    return edit_menu

