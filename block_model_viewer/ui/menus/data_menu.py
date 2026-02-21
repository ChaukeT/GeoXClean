"""
Data & Analysis menu construction for GeoX.
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


def build_data_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Data & Analysis menu."""
    data_menu = menubar.addMenu("&Data && Analysis")
    
    # Statistics Window
    statistics_action = QAction(get_menu_icon("data_analysis", "statistics"), "Statistics", main_window)
    statistics_action.setStatusTip("Open statistics and data summary window")
    statistics_action.triggered.connect(main_window.open_statistics_window)
    data_menu.addAction(statistics_action)
    
    # Charts & Visualization Window
    charts_action = QAction(get_menu_icon("data_analysis", "charts"), "Charts && Visualization", main_window)
    charts_action.setStatusTip("Open charts and data visualization window")
    charts_action.triggered.connect(main_window.open_charts_window)
    data_menu.addAction(charts_action)
    
    # 3D Swath Analysis
    swath_3d_action = QAction(get_menu_icon("data_analysis", "swath_3d"), "3D Swath Analysis", main_window)
    swath_3d_action.setStatusTip("Geostatistical estimation reliability assessment using 3D swath analysis")
    swath_3d_action.triggered.connect(main_window.open_swath_analysis_3d_panel)
    data_menu.addAction(swath_3d_action)
    
    data_menu.addSeparator()
    
    # Structural Analysis
    structural_action = QAction(get_menu_icon("data_analysis", "structural"), "Structural Analysis...", main_window)
    structural_action.setStatusTip("Stereonet, rose diagrams, and kinematic feasibility analysis")
    structural_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
    structural_action.triggered.connect(main_window.open_structural_panel)
    data_menu.addAction(structural_action)

    # Swath Plot Analysis Window
    swath_action = QAction("Swath Plot Analysis", main_window)
    swath_action.setStatusTip("Open swath plot analysis with 3D linking")
    swath_action.triggered.connect(main_window.open_swath_window)
    data_menu.addAction(swath_action)

    data_menu.addSeparator()
    
    return data_menu

