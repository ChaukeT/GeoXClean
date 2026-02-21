"""
Layout/Reports menu construction for GeoX.

Provides menu items for the Layout Composer and quick export functionality.
"""

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
    ICONS_AVAILABLE = True
except ImportError:
    ICONS_AVAILABLE = False
    def get_menu_icon(category, name):
        return None


def build_layout_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """
    Build and return the Reports/Layout menu.

    Args:
        main_window: The main application window
        menubar: The menu bar to add the menu to

    Returns:
        The created QMenu
    """
    menu = menubar.addMenu("&Reports")

    # Layout Composer
    layout_action = QAction(
        get_menu_icon("file", "advanced_screenshot") if ICONS_AVAILABLE else None,
        "&Layout Composer...",
        main_window
    )
    layout_action.setShortcut(QKeySequence("Ctrl+L"))
    layout_action.setStatusTip("Open the print layout composer for creating publication-ready exports")
    layout_action.triggered.connect(main_window.open_layout_composer)
    menu.addAction(layout_action)

    menu.addSeparator()

    # Quick exports submenu
    quick_menu = menu.addMenu("Quick &Export")

    quick_pdf = QAction("Export PDF (300 DPI)...", main_window)
    quick_pdf.setStatusTip("Quick export current view to PDF with standard layout")
    quick_pdf.triggered.connect(lambda: main_window.quick_layout_export("pdf"))
    quick_menu.addAction(quick_pdf)

    quick_png = QAction("Export PNG (300 DPI)...", main_window)
    quick_png.setStatusTip("Quick export current view to PNG with standard layout")
    quick_png.triggered.connect(lambda: main_window.quick_layout_export("png"))
    quick_menu.addAction(quick_png)

    quick_png_hd = QAction("Export PNG (600 DPI - Poster)...", main_window)
    quick_png_hd.setStatusTip("Quick export current view to high-resolution PNG")
    quick_png_hd.triggered.connect(lambda: main_window.quick_layout_export("png_hd"))
    quick_menu.addAction(quick_png_hd)

    return menu
