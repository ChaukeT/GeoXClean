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


def _safe_connect(action, main_window, method_name):
    handler = getattr(main_window, method_name, None)
    if handler is not None:
        action.triggered.connect(handler)
    else:
        action.setEnabled(False)


def build_help_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Help menu."""
    help_menu = menubar.addMenu("&Help")

    docs_action = QAction(get_menu_icon("help", "documentation"), "&Documentation", main_window)
    docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
    docs_action.setStatusTip("Open documentation")
    _safe_connect(docs_action, main_window, 'show_documentation')
    help_menu.addAction(docs_action)

    shortcuts_action = QAction(get_menu_icon("help", "keyboard"), "&Keyboard Shortcuts", main_window)
    shortcuts_action.setStatusTip("Show keyboard shortcuts")
    _safe_connect(shortcuts_action, main_window, 'show_shortcuts')
    help_menu.addAction(shortcuts_action)

    whats_new_action = QAction(get_menu_icon("help", "whats_new"), "&What's New", main_window)
    whats_new_action.setStatusTip("See the latest release notes")
    _safe_connect(whats_new_action, main_window, 'show_whats_new')
    help_menu.addAction(whats_new_action)

    help_menu.addSeparator()

    updates_action = QAction(get_menu_icon("help", "updates"), "Check for &Updates...", main_window)
    updates_action.setStatusTip("Check for software updates")
    _safe_connect(updates_action, main_window, 'check_for_updates')
    help_menu.addAction(updates_action)

    bug_action = QAction(get_menu_icon("help", "bug"), "&Report a Bug...", main_window)
    bug_action.setStatusTip("Open the bug report page in your browser")
    _safe_connect(bug_action, main_window, 'report_bug')
    help_menu.addAction(bug_action)

    help_menu.addSeparator()

    license_action = QAction(get_menu_icon("help", "license"), "&License Information", main_window)
    license_action.setStatusTip("Show the software license information")
    _safe_connect(license_action, main_window, 'show_license_info')
    help_menu.addAction(license_action)

    about_action = QAction(get_menu_icon("help", "about"), "&About", main_window)
    about_action.setStatusTip("About this application")
    _safe_connect(about_action, main_window, 'show_about')
    help_menu.addAction(about_action)

    return help_menu

