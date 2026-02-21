"""
Remote Sensing menu construction for GeoX.

Adds access to external InSAR deformation workflows.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
    ICONS_AVAILABLE = True
except ImportError:
    ICONS_AVAILABLE = False
    def get_menu_icon(category, name):
        return None


def build_remote_sensing_menu(main_window: "MainWindow", menubar: QMenuBar) -> QMenu:
    remote_menu = menubar.addMenu("&Remote Sensing")

    open_panel_action = QAction(get_menu_icon("scan", "scan_panel"), "InSAR Deformation Panel", main_window)
    open_panel_action.setStatusTip("Open InSAR Deformation panel")
    open_panel_action.triggered.connect(main_window.open_insar_panel)
    remote_menu.addAction(open_panel_action)

    remote_menu.addSeparator()

    run_action = QAction(get_menu_icon("scan", "run"), "Run InSAR Job", main_window)
    run_action.setStatusTip("Launch ISCE-2 InSAR job via WSL2")
    run_action.triggered.connect(main_window.run_insar_job)
    remote_menu.addAction(run_action)

    ingest_action = QAction(get_menu_icon("scan", "import"), "Ingest InSAR Results", main_window)
    ingest_action.setStatusTip("Register InSAR outputs into GeoX registry")
    ingest_action.triggered.connect(main_window.ingest_insar_results)
    remote_menu.addAction(ingest_action)

    return remote_menu

