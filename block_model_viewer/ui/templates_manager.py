"""
Templates Manager (STEP 39)

Save and reload workflow templates (scenarios, layouts, configs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TemplateMetadata:
    """
    Template metadata.
    
    Attributes:
        id: Template identifier
        name: Display name
        description: Description
        workflow_id: Associated workflow ID
        created_at: Creation timestamp
    """
    id: str
    name: str
    description: str = ""
    workflow_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WorkflowTemplate:
    """
    Complete workflow template.
    
    Attributes:
        meta: TemplateMetadata
        scenario_snapshot: Scenario configuration snapshot
        layout_snapshot: Qt layout state (as bytes or base64 string)
        extra_state: Additional state (panel configs, etc.)
    """
    meta: TemplateMetadata
    scenario_snapshot: Dict[str, Any] = field(default_factory=dict)
    layout_snapshot: Optional[str] = None  # Base64 encoded or JSON string
    extra_state: Dict[str, Any] = field(default_factory=dict)


class TemplatesManager:
    """
    Manages workflow templates.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize templates manager.
        
        Args:
            base_path: Base path for template storage (defaults to user config dir)
        """
        if base_path is None:
            # Use user config directory
            from PyQt6.QtCore import QStandardPaths
            config_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
            base_path = Path(config_dir) / "BlockModelViewer" / "templates"
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TemplatesManager at {self.base_path}")
    
    def list_templates(self) -> List[TemplateMetadata]:
        """
        List all available templates.
        
        Returns:
            List of TemplateMetadata objects
        """
        templates = []
        
        if not self.base_path.exists():
            return templates
        
        for template_file in self.base_path.glob("*.json"):
            try:
                template = self.load_template(template_file.stem)
                if template:
                    templates.append(template.meta)
            except Exception as e:
                logger.warning(f"Failed to load template {template_file.stem}: {e}")
        
        return templates
    
    def save_template(self, template: WorkflowTemplate) -> None:
        """
        Save a workflow template.
        
        Args:
            template: WorkflowTemplate instance
        """
        template_file = self.base_path / f"{template.meta.id}.json"
        
        # Convert to dict for JSON serialization
        template_dict = {
            "meta": asdict(template.meta),
            "scenario_snapshot": template.scenario_snapshot,
            "layout_snapshot": template.layout_snapshot,
            "extra_state": template.extra_state
        }
        
        try:
            with open(template_file, 'w') as f:
                json.dump(template_dict, f, indent=2)
            
            logger.info(f"Saved template: {template.meta.id}")
        except Exception as e:
            logger.error(f"Failed to save template {template.meta.id}: {e}", exc_info=True)
            raise
    
    def load_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """
        Load a workflow template.
        
        Args:
            template_id: Template identifier
        
        Returns:
            WorkflowTemplate or None
        """
        template_file = self.base_path / f"{template_id}.json"
        
        if not template_file.exists():
            logger.warning(f"Template file not found: {template_file}")
            return None
        
        try:
            with open(template_file, 'r') as f:
                template_dict = json.load(f)
            
            meta = TemplateMetadata(**template_dict["meta"])
            template = WorkflowTemplate(
                meta=meta,
                scenario_snapshot=template_dict.get("scenario_snapshot", {}),
                layout_snapshot=template_dict.get("layout_snapshot"),
                extra_state=template_dict.get("extra_state", {})
            )
            
            logger.info(f"Loaded template: {template_id}")
            return template
        except Exception as e:
            logger.error(f"Failed to load template {template_id}: {e}", exc_info=True)
            return None
    
    def delete_template(self, template_id: str) -> None:
        """
        Delete a workflow template.
        
        Args:
            template_id: Template identifier
        """
        template_file = self.base_path / f"{template_id}.json"
        
        if template_file.exists():
            try:
                template_file.unlink()
                logger.info(f"Deleted template: {template_id}")
            except Exception as e:
                logger.error(f"Failed to delete template {template_id}: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"Template file not found: {template_file}")
    
    def create_template_from_current_session(
        self,
        template_id: str,
        name: str,
        description: str = "",
        workflow_id: Optional[str] = None,
        main_window: Optional[Any] = None,
        controller: Optional[Any] = None
    ) -> WorkflowTemplate:
        """
        Create a template from the current session state.
        
        Args:
            template_id: Template identifier
            name: Display name
            description: Description
            workflow_id: Associated workflow ID
            main_window: MainWindow instance (for layout snapshot)
            controller: AppController instance (for scenario snapshot)
        
        Returns:
            WorkflowTemplate instance
        """
        meta = TemplateMetadata(
            id=template_id,
            name=name,
            description=description,
            workflow_id=workflow_id,
            created_at=datetime.now().isoformat()
        )
        
        # Capture layout state
        layout_snapshot = None
        if main_window:
            try:
                layout_state = main_window.saveState()
                # Convert QByteArray to base64 string
                import base64
                layout_snapshot = base64.b64encode(layout_state.data()).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to capture layout state: {e}")
        
        # Capture scenario state
        scenario_snapshot = {}
        if controller and hasattr(controller, 'get_current_scenario'):
            try:
                scenario = controller.get_current_scenario()
                if scenario:
                    # Convert scenario to dict (simplified)
                    scenario_snapshot = {"scenario_id": str(scenario.id) if hasattr(scenario, 'id') else None}
            except Exception as e:
                logger.warning(f"Failed to capture scenario state: {e}")
        
        template = WorkflowTemplate(
            meta=meta,
            scenario_snapshot=scenario_snapshot,
            layout_snapshot=layout_snapshot,
            extra_state={}
        )
        
        return template

