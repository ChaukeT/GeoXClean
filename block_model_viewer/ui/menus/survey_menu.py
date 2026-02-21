"""
Survey menu construction for GeoX.

Extracted from MainWindow to improve maintainability.
This menu handles survey deformation and subsidence analysis.
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


def build_survey_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """
    Build and return the Survey menu.
    
    Args:
        main_window: Reference to MainWindow instance
        menubar: Menu bar to add menu to
        
    Returns:
        Created Survey menu
    """
    survey_menu = menubar.addMenu("&Survey")
    
    # Open Panel Action
    open_survey_panel_action = QAction(get_menu_icon("drillholes", "loading"), "Open Deformation Panel", main_window)
    open_survey_panel_action.setStatusTip("Open Survey Deformation & Subsidence panel")
    open_survey_panel_action.triggered.connect(
        lambda: main_window.panel_manager.show_panel("SurveyDeformationPanel")
    )
    survey_menu.addAction(open_survey_panel_action)
    
    survey_menu.addSeparator()
    
    # Import Actions
    import_subsidence_action = QAction(
        get_menu_icon("drillholes", "loading"), 
        "Import Subsidence Survey Data...", 
        main_window
    )
    import_subsidence_action.setStatusTip(
        "Load subsidence survey CSV/Excel with Point ID, Easting, Northing, Elevation, Survey Date"
    )
    import_subsidence_action.triggered.connect(main_window.import_subsidence_survey_data)
    survey_menu.addAction(import_subsidence_action)
    
    import_groundwater_action = QAction(
        get_menu_icon("drillholes", "validation"), 
        "Import Groundwater Data...", 
        main_window
    )
    import_groundwater_action.setStatusTip(
        "Load groundwater well dipping CSV/Excel with Well ID, Easting, Northing, Date, Water Level"
    )
    import_groundwater_action.triggered.connect(main_window.import_groundwater_data)
    survey_menu.addAction(import_groundwater_action)
    
    survey_menu.addSeparator()
    
    # Analysis Actions
    run_subsidence_action = QAction(
        get_menu_icon("drillholes", "plotting"), 
        "Run Subsidence Analysis", 
        main_window
    )
    run_subsidence_action.setStatusTip("Compute displacement, velocity, and acceleration per survey point")
    run_subsidence_action.triggered.connect(main_window.run_subsidence_analysis)
    survey_menu.addAction(run_subsidence_action)
    
    control_stability_action = QAction(
        get_menu_icon("drillholes", "validation"), 
        "Run Control Stability", 
        main_window
    )
    control_stability_action.setStatusTip(
        "Classify benchmarks as Stable / Suspect / Failed using survey precision thresholds"
    )
    control_stability_action.triggered.connect(main_window.run_control_stability)
    survey_menu.addAction(control_stability_action)
    
    groundwater_ts_action = QAction(
        get_menu_icon("drillholes", "transform"), 
        "Run Groundwater Analysis", 
        main_window
    )
    groundwater_ts_action.setStatusTip("Standardise groundwater wells and compute drawdown rates")
    groundwater_ts_action.triggered.connect(main_window.run_groundwater_analysis)
    survey_menu.addAction(groundwater_ts_action)
    
    coupling_action = QAction(
        get_menu_icon("drillholes", "reporting"), 
        "Run Coupled Interpretation", 
        main_window
    )
    coupling_action.setStatusTip(
        "Relate subsidence behaviour to nearest groundwater wells (spatial/temporal correlation)"
    )
    coupling_action.triggered.connect(main_window.run_coupled_interpretation)
    survey_menu.addAction(coupling_action)
    
    deformation_index_action = QAction(
        get_menu_icon("drillholes", "reporting"), 
        "Compute Deformation Index", 
        main_window
    )
    deformation_index_action.setStatusTip(
        "Combine rate, acceleration, stability, and coupling into a single index"
    )
    deformation_index_action.triggered.connect(main_window.compute_deformation_index)
    survey_menu.addAction(deformation_index_action)
    
    return survey_menu

