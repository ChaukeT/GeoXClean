"""
Dashboards & Views menu construction for GeoX.
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


def build_dashboards_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Dashboards & Views menu."""
    dashboards_menu = menubar.addMenu("&Dashboards && Views")

    dashboards_menu.addSeparator()

    # Planning / production dashboards
    planning_action = QAction(get_menu_icon("dashboards", "planning"), "Planning Dashboard", main_window)
    planning_action.setStatusTip("Define, run, compare, and export planning scenarios")
    planning_action.triggered.connect(main_window.open_planning_dashboard_panel)
    dashboards_menu.addAction(planning_action)

    production_action = QAction(get_menu_icon("dashboards", "production"), "Production Dashboard", main_window)
    production_action.setStatusTip("Joint dashboard for NPVS, haulage, and reconciliation")
    production_action.triggered.connect(main_window.open_production_dashboard_panel)
    dashboards_menu.addAction(production_action)

    geotech_summary_action2 = QAction(get_menu_icon("geotech", "geotech_summary"), "Geotech Summary", main_window)
    geotech_summary_action2.setStatusTip("Geotechnical summary dashboard")
    geotech_summary_action2.triggered.connect(main_window.open_geotech_summary_panel)
    dashboards_menu.addAction(geotech_summary_action2)

    research_dashboard_action = QAction(get_menu_icon("dashboards", "research"), "Research Mode Dashboard", main_window)
    research_dashboard_action.setStatusTip("Research Mode: Experiment Runner, Parameter Sweeps, Batch Reports")
    research_dashboard_action.triggered.connect(main_window.open_research_dashboard_panel)
    dashboards_menu.addAction(research_dashboard_action)
    
    return dashboards_menu

