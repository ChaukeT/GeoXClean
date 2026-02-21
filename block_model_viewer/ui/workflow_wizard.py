"""
Workflow Wizard Framework (STEP 39)

Multi-step workflow guidance for non-expert users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWizard, QWizardPage, QTextEdit, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Step status enumeration."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """
    Single step in a workflow.
    
    Attributes:
        id: Step identifier
        title: Step title
        description: Step description
        panel_id: Panel ID to focus on
        action_id: Optional action to trigger
        status: Current status
    """
    id: str
    title: str
    description: str = ""
    panel_id: Optional[str] = None
    action_id: Optional[str] = None
    status: StepStatus = StepStatus.PENDING


@dataclass
class WorkflowDefinition:
    """
    Workflow definition.
    
    Attributes:
        id: Workflow identifier
        name: Display name
        description: Workflow description
        steps: List of WorkflowStep objects
    """
    id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)


class WorkflowWizard(QWizard):
    """
    Workflow wizard widget for guided multi-step workflows.
    """
    
    step_completed = pyqtSignal(str)  # Emitted when a step is completed
    
    def __init__(self, main_window: Any, workflow: WorkflowDefinition, parent: Optional[QWidget] = None):
        """
        Initialize workflow wizard.
        
        Args:
            main_window: MainWindow instance
            workflow: WorkflowDefinition
            parent: Parent widget
        """
        super().__init__(parent)
        self.main_window = main_window
        self.workflow = workflow
        self.current_step_index = 0
        
        self.setWindowTitle(f"Workflow: {workflow.name}")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        
        # Create pages for each step
        self._create_pages()
        
        logger.info(f"Initialized workflow wizard: {workflow.name}")
    
    def _create_pages(self) -> None:
        """Create wizard pages for each step."""
        for step in self.workflow.steps:
            page = self._create_step_page(step)
            self.addPage(page)
    
    def _create_step_page(self, step: WorkflowStep) -> QWizardPage:
        """Create a wizard page for a step."""
        page = QWizardPage()
        page.setTitle(step.title)
        page.setSubTitle(step.description)
        
        layout = QVBoxLayout(page)
        
        # Instructions
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setMaximumHeight(150)
        instructions.setText(self._get_step_instructions(step))
        layout.addWidget(instructions)
        
        # Action button (if action_id specified)
        if step.action_id:
            action_btn = QPushButton(f"Run: {step.action_id}")
            action_btn.clicked.connect(lambda: self._trigger_action(step.action_id))
            layout.addWidget(action_btn)
        
        # Status indicator
        status_label = QLabel(f"Status: {step.status.value}")
        status_label.setObjectName("status_label")
        layout.addWidget(status_label)
        
        layout.addStretch()
        
        return page
    
    def _get_step_instructions(self, step: WorkflowStep) -> str:
        """Get instructions text for a step."""
        instructions = f"<b>{step.title}</b><br><br>"
        instructions += step.description + "<br><br>"
        
        if step.panel_id:
            instructions += f"<i>Focus: {step.panel_id}</i><br>"
        
        if step.action_id:
            instructions += f"<i>Action: {step.action_id}</i>"
        
        return instructions
    
    def _trigger_action(self, action_id: str) -> None:
        """Trigger an action for the current step."""
        logger.info(f"Triggering action: {action_id}")
        
        # Map action IDs to panel opening methods
        action_map = {
            "open_block_model": lambda: self.main_window.open_block_model(),
            "run_pit_optimization": lambda: self._open_panel("pit_optimizer"),
            "design_pushbacks": lambda: self._open_panel("pushback_designer"),
            "build_strategic_schedule": lambda: self._open_panel("strategic_schedule"),
            "run_npvs": lambda: self._open_panel("npvs"),
            "run_irr": lambda: self._open_panel("irr"),
            "configure_ore_types": lambda: self._open_panel("geomet_chain"),
            "configure_plant_routes": lambda: self._open_panel("geomet_chain"),
            "compute_geomet_values": lambda: self._open_panel("geomet_chain"),
            "run_gc_estimation": lambda: self._open_panel("grade_control"),
            "design_diglines": lambda: self._open_panel("digline"),
            "build_short_term_schedule": lambda: self._open_panel("short_term_schedule"),
            "evaluate_haulage": lambda: self._open_panel("fleet"),
        }
        
        action_func = action_map.get(action_id)
        if action_func:
            try:
                action_func()
            except Exception as e:
                logger.error(f"Error triggering action {action_id}: {e}", exc_info=True)
        else:
            logger.warning(f"Unknown action ID: {action_id}")
    
    def _open_panel(self, panel_id: str) -> None:
        """Open a panel by ID."""
        panel_map = {
            "pit_optimizer": "open_pit_optimizer_panel",
            "pushback_designer": "open_pushback_designer_panel",
            "strategic_schedule": "open_strategic_schedule_panel",
            "npvs": "open_npvs_panel",
            "irr": "open_irr_panel",
            "geomet_chain": "open_geomet_chain_panel",
            "grade_control": "open_grade_control_panel",
            "digline": "open_digline_panel",
            "short_term_schedule": "open_short_term_schedule_panel",
            "fleet": "open_fleet_panel",
        }
        
        method_name = panel_map.get(panel_id)
        if method_name and hasattr(self.main_window, method_name):
            getattr(self.main_window, method_name)()
        else:
            logger.warning(f"Panel method not found: {panel_id}")
    
    def start(self) -> None:
        """Start the workflow wizard."""
        self.current_step_index = 0
        self.show()
        logger.info(f"Started workflow: {self.workflow.name}")
    
    def next(self) -> None:
        """Move to next step."""
        if self.current_step_index < len(self.workflow.steps) - 1:
            # Mark current step as completed
            if self.current_step_index < len(self.workflow.steps):
                step = self.workflow.steps[self.current_step_index]
                step.status = StepStatus.COMPLETED
                self.step_completed.emit(step.id)
            
            self.current_step_index += 1
            super().next()
    
    def prev(self) -> None:
        """Move to previous step."""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            super().back()
    
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get current step."""
        if 0 <= self.current_step_index < len(self.workflow.steps):
            return self.workflow.steps[self.current_step_index]
        return None


# Predefined Workflow Definitions

def create_open_pit_workflow() -> WorkflowDefinition:
    """Create Open Pit Planning Workflow."""
    return WorkflowDefinition(
        id="open_pit_planning",
        name="Open Pit Planning",
        description="Complete workflow from pit optimization to NPV analysis",
        steps=[
            WorkflowStep(
                id="load_model",
                title="Load Block Model",
                description="Load or create a block model for pit optimization",
                panel_id="block_model",
                action_id="open_block_model"
            ),
            WorkflowStep(
                id="pit_optimization",
                title="Run Pit Optimization",
                description="Optimize pit shells using Lerchs-Grossmann or discounted LG",
                panel_id="pit_optimizer",
                action_id="run_pit_optimization"
            ),
            WorkflowStep(
                id="design_pushbacks",
                title="Design Pushbacks",
                description="Design pushback phases from optimized pit shells",
                panel_id="pushback_designer",
                action_id="design_pushbacks"
            ),
            WorkflowStep(
                id="strategic_schedule",
                title="Build Strategic Schedule",
                description="Create annual LOM schedule using MILP or nested shells",
                panel_id="strategic_schedule",
                action_id="build_strategic_schedule"
            ),
            WorkflowStep(
                id="npvs",
                title="Run NPVS",
                description="Optimize schedule for maximum Net Present Value",
                panel_id="npvs",
                action_id="run_npvs"
            ),
            WorkflowStep(
                id="irr",
                title="Run IRR Analysis",
                description="Calculate Internal Rate of Return and cashflow analysis",
                panel_id="irr",
                action_id="run_irr"
            ),
            WorkflowStep(
                id="scenario",
                title="Send to Scenario Manager",
                description="Save results to scenario manager for comparison",
                panel_id="scenario_manager"
            ),
        ]
    )


def create_geomet_workflow() -> WorkflowDefinition:
    """Create Geomet Planning Workflow."""
    return WorkflowDefinition(
        id="geomet_planning",
        name="Geomet Planning",
        description="Workflow for geometallurgical planning and NPVS",
        steps=[
            WorkflowStep(
                id="configure_ore_types",
                title="Configure Ore Types",
                description="Define ore domains and link to geological model",
                panel_id="geomet_chain",
                action_id="configure_ore_types"
            ),
            WorkflowStep(
                id="configure_routes",
                title="Configure Plant Routes",
                description="Define plant routes with recovery and throughput surfaces",
                panel_id="geomet_chain",
                action_id="configure_plant_routes"
            ),
            WorkflowStep(
                id="compute_values",
                title="Compute Geomet Values",
                description="Compute geomet-adjusted block values for all blocks",
                panel_id="geomet_chain",
                action_id="compute_geomet_values"
            ),
            WorkflowStep(
                id="run_npvs_geomet",
                title="Run NPVS (Geomet Mode)",
                description="Run NPVS optimization using geomet-adjusted values",
                panel_id="npvs",
                action_id="run_npvs"
            ),
            WorkflowStep(
                id="compare_scenarios",
                title="Compare Scenarios",
                description="Compare base vs geomet scenarios in Scenario Manager",
                panel_id="scenario_manager"
            ),
        ]
    )


def create_gc_short_term_workflow() -> WorkflowDefinition:
    """Create Grade Control & Short-Term Workflow."""
    return WorkflowDefinition(
        id="gc_short_term",
        name="Grade Control & Short-Term Planning",
        description="Workflow for GC estimation, diglines, and short-term scheduling",
        steps=[
            WorkflowStep(
                id="load_gc_data",
                title="Load GC Data",
                description="Load composites, blast-hole data, or GC samples",
                panel_id="grade_control"
            ),
            WorkflowStep(
                id="gc_estimation",
                title="Run GC Estimation",
                description="Estimate grades at SMU support using kriging or simulation",
                panel_id="grade_control",
                action_id="run_gc_estimation"
            ),
            WorkflowStep(
                id="design_diglines",
                title="Design Diglines",
                description="Create ore/waste polygons based on GC model and cutoffs",
                panel_id="digline",
                action_id="design_diglines"
            ),
            WorkflowStep(
                id="short_term_schedule",
                title="Build Short-Term Schedule",
                description="Create short-term schedule from diglines",
                panel_id="short_term_schedule",
                action_id="build_short_term_schedule"
            ),
            WorkflowStep(
                id="evaluate_haulage",
                title="Evaluate Haulage",
                description="Evaluate schedule feasibility against fleet capacity",
                panel_id="fleet",
                action_id="evaluate_haulage"
            ),
            WorkflowStep(
                id="production_dashboard",
                title="Send to Production Dashboard",
                description="View aligned production metrics (plan vs actual)",
                panel_id="production_dashboard"
            ),
        ]
    )


# Registry of predefined workflows
WORKFLOW_REGISTRY: Dict[str, Callable[[], WorkflowDefinition]] = {
    "open_pit_planning": create_open_pit_workflow,
    "geomet_planning": create_geomet_workflow,
    "gc_short_term": create_gc_short_term_workflow,
}


def get_workflow_definition(workflow_id: str) -> Optional[WorkflowDefinition]:
    """
    Get a workflow definition by ID.
    
    Args:
        workflow_id: Workflow identifier
    
    Returns:
        WorkflowDefinition or None
    """
    factory = WORKFLOW_REGISTRY.get(workflow_id)
    if factory:
        return factory()
    return None


def list_available_workflows() -> List[str]:
    """
    List all available workflow IDs.
    
    Returns:
        List of workflow IDs
    """
    return list(WORKFLOW_REGISTRY.keys())

