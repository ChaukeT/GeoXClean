"""
Help menu construction for GeoX.
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


def build_help_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Help menu."""
    help_menu = menubar.addMenu("&Help")
    
    docs_action = QAction(get_menu_icon("help", "documentation"), "&Documentation", main_window)
    docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
    docs_action.setStatusTip("Open documentation")
    docs_action.triggered.connect(main_window.show_documentation)
    help_menu.addAction(docs_action)
    
    shortcuts_action = QAction(get_menu_icon("help", "keyboard"), "&Keyboard Shortcuts", main_window)
    shortcuts_action.setShortcut(QKeySequence.StandardKey.HelpContents)
    shortcuts_action.setStatusTip("Show keyboard shortcuts (F1)")
    shortcuts_action.triggered.connect(main_window.show_shortcuts)
    help_menu.addAction(shortcuts_action)
    
    help_menu.addSeparator()
    
    about_action = QAction(get_menu_icon("help", "about"), "&About", main_window)
    about_action.setStatusTip("About this application")
    about_action.triggered.connect(main_window.show_about)
    help_menu.addAction(about_action)
    
    return help_menu

