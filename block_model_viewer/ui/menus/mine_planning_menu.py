"""
Mine Planning menu construction for GeoX.
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


def build_mine_planning_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Mine Planning menu."""
    mine_planning_menu = menubar.addMenu("&Mine Planning")
    
    pit_optimisation_action = QAction(get_menu_icon("mine_planning", "pit_optimisation"), "Pit &Optimisation", main_window)
    pit_optimisation_action.setStatusTip("Open-pit optimization using Lerchs-Grossmann algorithm")
    pit_optimisation_action.triggered.connect(main_window.open_pit_optimisation_panel)
    mine_planning_menu.addAction(pit_optimisation_action)
    
    underground_action = QAction(get_menu_icon("mine_planning", "underground"), "&Underground Mining", main_window)
    underground_action.setStatusTip("Stope optimization, scheduling, ground control & equipment")
    underground_action.triggered.connect(main_window.open_underground_panel)
    mine_planning_menu.addAction(underground_action)

    # Advanced Underground
    ug_advanced_action = QAction(get_menu_icon("mine_planning", "ug_advanced"), "Advanced Underground", main_window)
    ug_advanced_action.setStatusTip("SLOS, caving, dilution, and void tracking")
    ug_advanced_action.triggered.connect(main_window.open_ug_advanced_panel)
    mine_planning_menu.addAction(ug_advanced_action)

    mine_planning_menu.addSeparator()

    # Scheduling submenu
    scheduling_menu = mine_planning_menu.addMenu("Scheduling")

    strategic_action = QAction(get_menu_icon("mine_planning", "strategic"), "Strategic Schedule", main_window)
    strategic_action.setStatusTip("Annual LOM scheduling with MILP, nested shells, cutoff")
    strategic_action.triggered.connect(main_window.open_strategic_schedule_panel)
    scheduling_menu.addAction(strategic_action)

    tactical_action = QAction(get_menu_icon("mine_planning", "tactical"), "Tactical Schedule", main_window)
    tactical_action.setStatusTip("Monthly/quarterly pushback, bench, development scheduling")
    tactical_action.triggered.connect(main_window.open_tactical_schedule_panel)
    scheduling_menu.addAction(tactical_action)

    short_term_action = QAction(get_menu_icon("mine_planning", "short_term"), "Short-Term Schedule", main_window)
    short_term_action.setStatusTip("Weekly/daily digline scheduling and shift planning")
    short_term_action.triggered.connect(main_window.open_short_term_schedule_panel)
    scheduling_menu.addAction(short_term_action)

    fleet_action = QAction(get_menu_icon("mine_planning", "fleet"), "Fleet & Haulage", main_window)
    fleet_action.setStatusTip("Fleet configuration, cycle time, and dispatch")
    fleet_action.triggered.connect(main_window.open_fleet_panel)
    scheduling_menu.addAction(fleet_action)

    mine_planning_menu.addSeparator()

    # NPVS Optimisation
    npvs_action = QAction(get_menu_icon("mine_planning", "npv"), "NPVS Optimisation", main_window)
    npvs_action.setStatusTip("Net Present Value Scheduling optimization")
    npvs_action.triggered.connect(main_window.open_npvs_panel)
    mine_planning_menu.addAction(npvs_action)

    # Pushback Visual Designer
    pushback_action = QAction(get_menu_icon("mine_planning", "pushback"), "Pushback Visual Designer", main_window)
    pushback_action.setStatusTip("Design pushbacks and integrate with NPVS")
    pushback_action.triggered.connect(main_window.open_pushback_designer_panel)
    mine_planning_menu.addAction(pushback_action)

    # IRR Optimization
    irr_action = QAction(get_menu_icon("mine_planning", "irr"), "IRR Optimization", main_window)
    irr_action.setStatusTip("Risk-Adjusted Internal Rate of Return analysis with stochastic scenarios")
    irr_action.triggered.connect(main_window.open_irr_panel)
    mine_planning_menu.addAction(irr_action)

    mine_planning_menu.addSeparator()

    esg_action = QAction(get_menu_icon("mine_planning", "esg"), "&ESG Dashboard", main_window)
    esg_action.setStatusTip("Environmental, Social & Governance metrics and reporting")
    esg_action.triggered.connect(main_window.open_esg_panel)
    mine_planning_menu.addAction(esg_action)
    
    mine_planning_menu.addSeparator()
    
    uncertainty_action = QAction(get_menu_icon("mine_planning", "uncertainty"), "&Uncertainty Analysis", main_window)
    uncertainty_action.setStatusTip("Monte Carlo, Bootstrap, and probabilistic risk analysis")
    uncertainty_action.triggered.connect(main_window.open_uncertainty_panel)
    mine_planning_menu.addAction(uncertainty_action)
    
    return mine_planning_menu

