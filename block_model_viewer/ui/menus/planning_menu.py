"""
Planning menu — GeoX consolidated menu system.

Position: 7th.
Handles: pit optimisation, underground mining, scheduling, fleet,
         economics, ESG, uncertainty, dashboards.
Workflow: Design → Schedule → Evaluate → Assess Risk.
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


def build_planning_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    menu = menubar.addMenu("&Planning")

    # ── Open Pit ─────────────────────────────────────────────────
    pit_menu = menu.addMenu("&Open Pit")

    act = QAction(get_menu_icon("mine_planning", "pit_optimisation"), "&Pit Optimisation", main_window)
    act.setStatusTip("Open-pit optimization using Lerchs-Grossmann algorithm")
    act.triggered.connect(main_window.open_pit_optimisation_panel)
    pit_menu.addAction(act)

    pit_menu.addSeparator()

    act = QAction(get_menu_icon("mine_planning", "pushback"), "Pushback &Visual Designer", main_window)
    act.setStatusTip("Design pushbacks and integrate with NPVS")
    act.triggered.connect(main_window.open_pushback_designer_panel)
    pit_menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "bench"), "&Bench Design", main_window)
    act.setStatusTip("Design bench geometry and ramp layout")
    act.triggered.connect(main_window.open_bench_design_panel)
    pit_menu.addAction(act)

    menu.addSeparator()

    # ── Underground ──────────────────────────────────────────────
    ug_menu = menu.addMenu("&Underground Mining")

    act = QAction(get_menu_icon("mine_planning", "underground"), "&Underground Mining", main_window)
    act.setStatusTip("Stope optimization, scheduling, ground control, SLOS, caving & void management")
    act.triggered.connect(main_window.open_underground_panel)
    ug_menu.addAction(act)

    ug_menu.addSeparator()

    act = QAction(get_menu_icon("mine_planning", "stope"), "Stope &Stability", main_window)
    act.setStatusTip("Stope stability analysis using Mathews stability graph method")
    act.triggered.connect(main_window.open_stope_stability_panel)
    ug_menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "underground"), "&Rockburst Assessment", main_window)
    act.setStatusTip("Seismic hazard and rockburst risk assessment")
    act.triggered.connect(main_window.open_rockburst_panel)
    ug_menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "underground"), "UG &Advanced (SLOS, Caving, Void)", main_window)
    act.setStatusTip("Advanced underground operations: SLOS, caving, void management")
    if hasattr(main_window, 'open_ug_advanced_panel'):
        act.triggered.connect(main_window.open_ug_advanced_panel)
    else:
        act.setEnabled(False)
    ug_menu.addAction(act)

    menu.addSeparator()

    # ── Scheduling ───────────────────────────────────────────────
    sched_menu = menu.addMenu("&Scheduling")

    act = QAction(get_menu_icon("mine_planning", "strategic"), "&Strategic (Annual LOM)", main_window)
    act.setStatusTip("Annual LOM scheduling with MILP, nested shells, cutoff")
    act.triggered.connect(main_window.open_strategic_schedule_panel)
    sched_menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "tactical"), "&Tactical (Monthly/Quarterly)", main_window)
    act.setStatusTip("Monthly/quarterly pushback, bench, development scheduling")
    act.triggered.connect(main_window.open_tactical_schedule_panel)
    sched_menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "short_term"), "S&hort-Term (Weekly/Daily)", main_window)
    act.setStatusTip("Weekly/daily digline scheduling and shift planning")
    act.triggered.connect(main_window.open_short_term_schedule_panel)
    sched_menu.addAction(act)

    sched_menu.addSeparator()

    act = QAction(get_menu_icon("mine_planning", "fleet"), "&Fleet && Haulage", main_window)
    act.setStatusTip("Fleet configuration, cycle time, and dispatch")
    act.triggered.connect(main_window.open_fleet_panel)
    sched_menu.addAction(act)

    menu.addSeparator()

    # ── Economics ─────────────────────────────────────────────────
    econ_menu = menu.addMenu("&Economics")

    act = QAction(get_menu_icon("mine_planning", "npv"), "&NPVS Optimisation", main_window)
    act.setStatusTip("Net Present Value Scheduling optimization")
    act.triggered.connect(main_window.open_npvs_panel)
    econ_menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "irr"), "&IRR Optimization", main_window)
    act.setStatusTip("Risk-Adjusted Internal Rate of Return analysis")
    act.triggered.connect(main_window.open_irr_panel)
    econ_menu.addAction(act)

    menu.addSeparator()

    # ── Risk & Sustainability ────────────────────────────────────
    act = QAction(get_menu_icon("mine_planning", "uncertainty"), "&Uncertainty Analysis", main_window)
    act.setStatusTip("Monte Carlo, Bootstrap, and probabilistic risk analysis")
    act.triggered.connect(main_window.open_uncertainty_panel)
    menu.addAction(act)

    act = QAction(get_menu_icon("mine_planning", "esg"), "ES&G Dashboard", main_window)
    act.setStatusTip("Environmental, Social & Governance metrics and reporting")
    act.triggered.connect(main_window.open_esg_panel)
    menu.addAction(act)

    menu.addSeparator()

    # ── Dashboards ───────────────────────────────────────────────
    dash_menu = menu.addMenu("&Dashboards")

    act = QAction(get_menu_icon("dashboards", "planning"), "&Planning Dashboard", main_window)
    act.setStatusTip("Define, run, compare, and export planning scenarios")
    act.triggered.connect(main_window.open_planning_dashboard_panel)
    dash_menu.addAction(act)

    act = QAction(get_menu_icon("dashboards", "production"), "P&roduction Dashboard", main_window)
    act.setStatusTip("Joint dashboard for NPVS, haulage, and reconciliation")
    act.triggered.connect(main_window.open_production_dashboard_panel)
    dash_menu.addAction(act)

    act = QAction(get_menu_icon("geotech", "geotech_summary"), "&Geotech Dashboard", main_window)
    act.setStatusTip("Geotechnical dashboard (rock mass, interpolation, stability summary)")
    act.triggered.connect(main_window.open_geotech_panel)
    dash_menu.addAction(act)

    act = QAction(get_menu_icon("dashboards", "research"), "&Research Dashboard", main_window)
    act.setStatusTip("Experiment configuration and results comparison dashboard")
    if hasattr(main_window, 'open_research_dashboard_panel'):
        act.triggered.connect(main_window.open_research_dashboard_panel)
    else:
        act.setEnabled(False)
    dash_menu.addAction(act)

    return menu
