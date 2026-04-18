"""
Resources menu construction for GeoX.
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


def _safe_connect(action, main_window, method_name):
    """Connect an action's triggered signal to a main-window method, disabling the action if missing."""
    handler = getattr(main_window, method_name, None)
    if handler is not None:
        action.triggered.connect(handler)
    else:
        action.setEnabled(False)


def build_resources_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Resources menu."""
    resources_menu = menubar.addMenu("&Resources")

    # ── Classification ──────────────────────────────────────────
    jorc_action = QAction(get_menu_icon("resources", "classification"), "Resource &Classification (JORC)...", main_window)
    jorc_action.setStatusTip("Classify blocks as Measured / Indicated / Inferred (JORC/SAMREC/NI 43-101)")
    _safe_connect(jorc_action, main_window, 'open_resource_classification_panel')
    resources_menu.addAction(jorc_action)

    reporting_action = QAction(get_menu_icon("resources", "reporting"), "Resource &Reporting...", main_window)
    reporting_action.setStatusTip("Generate resource reports by classification and grade bin")
    _safe_connect(reporting_action, main_window, 'open_resource_reporting_panel')
    resources_menu.addAction(reporting_action)

    resources_menu.addSeparator()

    # ── Grade-Tonnage Analysis ──────────────────────────────────
    gt_action = QAction(get_menu_icon("resources", "grade_tonnage"), "&Grade-Tonnage Analysis...", main_window)
    gt_action.setStatusTip("Grade-tonnage curves with cutoff sweeps")
    _safe_connect(gt_action, main_window, 'open_grade_tonnage_panel')
    resources_menu.addAction(gt_action)

    gt_basic_action = QAction(get_menu_icon("resources", "grade_tonnage"), "Grade-Tonnage (&Basic)...", main_window)
    gt_basic_action.setStatusTip("Simple grade-tonnage analysis with single cutoff")
    _safe_connect(gt_basic_action, main_window, 'open_grade_tonnage_basic_panel')
    resources_menu.addAction(gt_basic_action)

    cutoff_action = QAction(get_menu_icon("resources", "cutoff"), "Cut&off Optimization...", main_window)
    cutoff_action.setStatusTip("Optimize economic cutoff grade with NPV-aware search")
    _safe_connect(cutoff_action, main_window, 'open_cutoff_optimization_panel')
    resources_menu.addAction(cutoff_action)

    resources_menu.addSeparator()

    # ── Block Model Resources ───────────────────────────────────
    block_resource_action = QAction(get_menu_icon("resources", "block_resources"), "&Block Model Resources", main_window)
    block_resource_action.setStatusTip("Calculate block model resources with cut-off logic")
    _safe_connect(block_resource_action, main_window, 'open_block_resource_panel')
    resources_menu.addAction(block_resource_action)

    sensitivity_action = QAction(get_menu_icon("resources", "sensitivity"), "Cut-off &Sensitivity Analysis", main_window)
    sensitivity_action.setStatusTip("Perform cut-off sensitivity analysis")
    _safe_connect(sensitivity_action, main_window, 'open_sensitivity_panel')
    resources_menu.addAction(sensitivity_action)

    return resources_menu

