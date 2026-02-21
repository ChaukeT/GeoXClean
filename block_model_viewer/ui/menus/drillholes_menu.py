"""
Drillholes menu construction for GeoX.
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


def build_drillholes_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Drillholes menu."""
    drillholes_menu = menubar.addMenu("&Drillholes")
    
    # Step 1: Drillhole Loading
    domain_comp_action = QAction(get_menu_icon("drillholes", "loading"), "Drillhole Loading", main_window)
    domain_comp_action.setStatusTip("Step 1: Load drillhole data (collar, assay, survey, lithology)")
    domain_comp_action.triggered.connect(main_window.open_domain_compositing_panel)
    drillholes_menu.addAction(domain_comp_action)

    # Step 2: Data Validation & QC
    drillholes_menu.addSeparator()
    
    qc_action = QAction(get_menu_icon("drillholes", "validation"), "Data Validation & QC", main_window)
    qc_action.setStatusTip("Run comprehensive drillhole data validation and quality control")
    qc_action.triggered.connect(main_window.open_drillhole_qc_window)
    drillholes_menu.addAction(qc_action)

    # Step 3: Compositing
    drillholes_menu.addSeparator()
    compositing_action = QAction(get_menu_icon("drillholes", "compositing"), "Compositing", main_window)
    compositing_action.setStatusTip("Composite drillhole data using various methods")
    compositing_action.triggered.connect(main_window.open_compositing_window)
    drillholes_menu.addAction(compositing_action)

    # Step 4: Declustering (after compositing)
    declustering_action = QAction(get_menu_icon("drillholes", "declustering"), "Declustering Analysis", main_window)
    declustering_action.setStatusTip("Cell-based declustering for statistical defensibility (JORC/SAMREC compliant)")
    declustering_action.triggered.connect(main_window.open_declustering_panel)
    drillholes_menu.addAction(declustering_action)

    drillholes_menu.addSeparator()

    # Step 5: Statistics & Reporting submenu
    stats_reporting_menu = drillholes_menu.addMenu("Statistics & Reporting")
    
    # Reporting
    reporting_action = QAction(get_menu_icon("drillholes", "reporting"), "Reporting", main_window)
    reporting_action.setStatusTip("Generate statistics and reports with summary statistics, grade-tonnage curves, and export")
    reporting_action.triggered.connect(main_window.open_drillhole_reporting_panel)
    stats_reporting_menu.addAction(reporting_action)
    
    # Plotting
    plotting_action = QAction(get_menu_icon("drillholes", "plotting"), "Plotting", main_window)
    plotting_action.setStatusTip("Generate downhole plots, strip logs, and fence diagrams")
    plotting_action.triggered.connect(main_window.open_drillhole_plotting_panel)
    stats_reporting_menu.addAction(plotting_action)
    
    drillholes_menu.addSeparator()
    
    # Grade Transformation (Step 7)
    transform_action = QAction(get_menu_icon("drillholes", "transform"), "Transform Grade Data", main_window)
    transform_action.setStatusTip("Step 7: Transform drillhole grade data (log, sqrt, Box-Cox) for estimation")
    transform_action.triggered.connect(main_window.open_grade_transformation_panel)
    drillholes_menu.addAction(transform_action)
    
    drillholes_menu.addSeparator()
    
    return drillholes_menu

