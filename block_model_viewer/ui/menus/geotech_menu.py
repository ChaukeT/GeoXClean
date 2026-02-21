"""
Geotechnical Modelling menu construction for GeoX.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow


def build_geotech_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Geotechnical Modelling menu."""
    geotech_top_menu = menubar.addMenu("&Geotechnical Modelling")

    geotech_action = QAction("Geotechnical Dashboard", main_window)
    geotech_action.setStatusTip("Rock-mass property interpolation and geotechnical analysis")
    geotech_action.triggered.connect(main_window.open_geotech_panel)
    geotech_top_menu.addAction(geotech_action)
    
    return geotech_top_menu

