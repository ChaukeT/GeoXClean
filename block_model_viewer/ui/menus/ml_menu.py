"""
Machine Learning menu construction for GeoX.
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


def build_ml_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Machine Learning menu."""
    ml_menu = menubar.addMenu("&Machine Learning")

    kmeans_action = QAction(get_menu_icon("machine_learning", "kmeans"), "K-Means Clustering Analysis", main_window)
    kmeans_action.setStatusTip("Unsupervised K-Means clustering for geological domain classification")
    kmeans_action.triggered.connect(main_window.open_kmeans_panel)
    ml_menu.addAction(kmeans_action)
    
    return ml_menu

