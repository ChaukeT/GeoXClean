"""
Workflows menu construction for GeoX.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow


def build_workflows_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Workflows menu."""
    workflows_menu = menubar.addMenu("&Workflows")
    
    # Predefined workflows
    open_pit_workflow_action = QAction("Open Pit Planning Wizard", main_window)
    open_pit_workflow_action.setStatusTip("Guided workflow for open pit planning")
    open_pit_workflow_action.triggered.connect(lambda: main_window.start_workflow("open_pit_planning"))
    workflows_menu.addAction(open_pit_workflow_action)
    
    geomet_workflow_action = QAction("Geomet Planning Wizard", main_window)
    geomet_workflow_action.setStatusTip("Guided workflow for geometallurgical planning")
    geomet_workflow_action.triggered.connect(lambda: main_window.start_workflow("geomet_planning"))
    workflows_menu.addAction(geomet_workflow_action)
    
    gc_workflow_action = QAction("Grade Control & Short-Term Wizard", main_window)
    gc_workflow_action.setStatusTip("Guided workflow for GC and short-term planning")
    gc_workflow_action.triggered.connect(lambda: main_window.start_workflow("gc_short_term"))
    workflows_menu.addAction(gc_workflow_action)
    
    workflows_menu.addSeparator()
    
    # Template management
    save_template_action = QAction("Save Current Session as Template...", main_window)
    save_template_action.setStatusTip("Save current session as a workflow template")
    save_template_action.triggered.connect(main_window.save_session_template)
    workflows_menu.addAction(save_template_action)
    
    load_template_action = QAction("Load Template...", main_window)
    load_template_action.setStatusTip("Load a saved workflow template")
    load_template_action.triggered.connect(main_window.load_session_template)
    workflows_menu.addAction(load_template_action)
    
    delete_template_action = QAction("Delete Template...", main_window)
    delete_template_action.setStatusTip("Delete a saved workflow template")
    delete_template_action.triggered.connect(main_window.delete_session_template)
    workflows_menu.addAction(delete_template_action)
    
    return workflows_menu

