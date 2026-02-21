"""
Workbench menu construction for GeoX.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow


def build_workbench_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Workbench menu."""
    workbench_menu = menubar.addMenu("&Workbench")
    
    # Role-based layouts
    geologist_layout_action = QAction("Geologist Layout", main_window)
    geologist_layout_action.setStatusTip("Switch to geologist workbench layout")
    geologist_layout_action.triggered.connect(lambda: main_window.apply_workbench_profile("geologist"))
    workbench_menu.addAction(geologist_layout_action)
    
    planner_layout_action = QAction("Mine Planner Layout", main_window)
    planner_layout_action.setStatusTip("Switch to mine planner workbench layout")
    planner_layout_action.triggered.connect(lambda: main_window.apply_workbench_profile("planner"))
    workbench_menu.addAction(planner_layout_action)
    
    metallurgist_layout_action = QAction("Metallurgist Layout", main_window)
    metallurgist_layout_action.setStatusTip("Switch to metallurgist workbench layout")
    metallurgist_layout_action.triggered.connect(lambda: main_window.apply_workbench_profile("metallurgist"))
    workbench_menu.addAction(metallurgist_layout_action)
    
    esg_layout_action = QAction("ESG / Reporting Layout", main_window)
    esg_layout_action.setStatusTip("Switch to ESG/reporting workbench layout")
    esg_layout_action.triggered.connect(lambda: main_window.apply_workbench_profile("esg"))
    workbench_menu.addAction(esg_layout_action)
    
    workbench_menu.addSeparator()
    
    # Save layout
    save_layout_action = QAction("Save Current Layout...", main_window)
    save_layout_action.setStatusTip("Save current layout as a profile")
    save_layout_action.triggered.connect(main_window.save_current_layout)
    workbench_menu.addAction(save_layout_action)
    
    return workbench_menu

