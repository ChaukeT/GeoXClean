"""
Scan menu construction for GeoX.
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


def build_scan_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Scan menu."""
    scan_menu = menubar.addMenu("&Scan")

    # Load Scan action
    load_scan_action = QAction(get_menu_icon("scan", "load_scan"), "&Load Scan...", main_window)
    load_scan_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
    load_scan_action.setStatusTip("Load and analyze drone scan data (point clouds and meshes)")
    load_scan_action.triggered.connect(main_window.load_scan_file)
    scan_menu.addAction(load_scan_action)

    scan_menu.addSeparator()

    # Open Scan Panel action
    open_scan_panel_action = QAction(get_menu_icon("scan", "scan_panel"), "&Scan Analysis Panel", main_window)
    open_scan_panel_action.setStatusTip("Open the scan analysis and fragmentation panel")
    open_scan_panel_action.triggered.connect(main_window.open_scan_panel)
    scan_menu.addAction(open_scan_panel_action)

    scan_menu.addSeparator()

    # Export submenu
    export_menu = scan_menu.addMenu("&Export")

    export_psd_action = QAction(get_menu_icon("scan", "export_psd"), "Fragment Metrics &PSD (CSV)...", main_window)
    export_psd_action.setStatusTip("Export particle size distribution and fragment metrics to CSV")
    export_psd_action.triggered.connect(main_window.export_scan_psd)
    export_menu.addAction(export_psd_action)

    export_fragments_action = QAction(get_menu_icon("scan", "export_fragments"), "&Fragments (OBJ/STL)...", main_window)
    export_fragments_action.setStatusTip("Export individual fragments as mesh files")
    export_fragments_action.triggered.connect(main_window.export_scan_fragments)
    export_menu.addAction(export_fragments_action)

    # Clear Scan Data action
    scan_menu.addSeparator()
    clear_scan_action = QAction(get_menu_icon("scan", "clear_scan"), "&Clear Scan Data", main_window)
    clear_scan_action.setStatusTip("Remove all loaded scan data and results")
    clear_scan_action.triggered.connect(main_window.clear_scan_data)
    scan_menu.addAction(clear_scan_action)

    return scan_menu

