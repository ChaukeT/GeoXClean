"""
Workbench Manager (STEP 39)

Manages role-based layouts and panel presets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from PyQt6.QtCore import QByteArray, QSettings

logger = logging.getLogger(__name__)


@dataclass
class WorkbenchProfile:
    """
    Workbench profile definition for a role.
    
    Attributes:
        id: Profile identifier (e.g., "geologist", "planner")
        name: Display name
        description: Description
        default_panels: List of panel IDs to show/dock
        layout_preset_id: Reference to saved dock layout
    """
    id: str
    name: str
    description: str = ""
    default_panels: List[str] = field(default_factory=list)
    layout_preset_id: Optional[str] = None


class WorkbenchManager:
    """
    Manages workbench profiles and layouts.
    """
    
    def __init__(self, main_window: Any):
        """
        Initialize workbench manager.
        
        Args:
            main_window: MainWindow instance
        """
        self.main_window = main_window
        self.profiles: Dict[str, WorkbenchProfile] = {}
        self.settings = QSettings("GeoX", "Workbench")
        
        # Register default profiles
        self._register_default_profiles()
        
        logger.info("Initialized WorkbenchManager")
    
    def _register_default_profiles(self) -> None:
        """Register default workbench profiles."""
        # Geologist Profile
        self.register_profile(WorkbenchProfile(
            id="geologist",
            name="Geologist",
            description="Layout optimized for geological modeling and drillhole analysis",
            default_panels=[
                "drillhole_viewer",
                "domain_compositing",
                "geology_panel",
                "structural_panel",
                "grade_stats"
            ]
        ))
        
        # Mine Planner Profile
        self.register_profile(WorkbenchProfile(
            id="planner",
            name="Mine Planner",
            description="Layout optimized for pit optimization, scheduling, and NPV analysis",
            default_panels=[
                "pit_optimizer",
                "strategic_schedule",
                "npvs",
                "irr",
                "pushback_designer"
            ]
        ))
        
        # Metallurgist Profile
        self.register_profile(WorkbenchProfile(
            id="metallurgist",
            name="Metallurgist",
            description="Layout optimized for geometallurgy and plant response analysis",
            default_panels=[
                "geomet_chain",
                "geomet_panel",
                "geomet_domain_panel",
                "geomet_plant_panel",
                "npvs"
            ]
        ))
        
        # ESG Profile
        self.register_profile(WorkbenchProfile(
            id="esg",
            name="ESG / Reporting",
            description="Layout optimized for ESG metrics and reporting",
            default_panels=[
                "esg_dashboard",
                "production_dashboard",
                "reconciliation_panel",
                "grade_stats"
            ]
        ))
    
    def register_profile(self, profile: WorkbenchProfile) -> None:
        """
        Register a workbench profile.
        
        Args:
            profile: WorkbenchProfile instance
        """
        self.profiles[profile.id] = profile
        logger.info(f"Registered workbench profile: {profile.id}")
    
    def apply_profile(self, profile_id: str) -> None:
        """
        Apply a workbench profile to the main window.
        
        Args:
            profile_id: Profile identifier
        """
        if profile_id not in self.profiles:
            logger.warning(f"Profile '{profile_id}' not found")
            return
        
        profile = self.profiles[profile_id]
        logger.info(f"Applying workbench profile: {profile.name}")
        
        # Restore layout if preset exists
        if profile.layout_preset_id:
            self._restore_layout(profile.layout_preset_id)
        
        # Show/hide panels based on default_panels
        self._apply_panel_visibility(profile.default_panels)
        
        logger.info(f"Applied profile '{profile.name}'")
    
    def _apply_panel_visibility(self, panel_ids: List[str]) -> None:
        """
        Show/hide panels based on panel IDs.
        
        Args:
            panel_ids: List of panel IDs to show
        """
        # Get all available panels from main window
        # This is a simplified implementation - would need to track all panels
        available_panels = {
            "drillhole_viewer": getattr(self.main_window, 'drillhole_viewer_dialog', None),
            "domain_compositing": getattr(self.main_window, 'domain_compositing_dialog', None),
            "geology_panel": getattr(self.main_window, 'geology_dialog', None),
            "structural_panel": getattr(self.main_window, 'structural_dialog', None),
            "pit_optimizer": getattr(self.main_window, 'pit_optimizer_dialog', None),
            "strategic_schedule": getattr(self.main_window, 'strategic_schedule_dialog', None),
            "npvs": getattr(self.main_window, 'npvs_dialog', None),
            "irr": getattr(self.main_window, 'irr_dialog', None),
            "pushback_designer": getattr(self.main_window, 'pushback_designer_dialog', None),
            "geomet_chain": getattr(self.main_window, 'geomet_chain_dialog', None),
            "geomet_panel": getattr(self.main_window, 'geomet_dialog', None),
            "geomet_domain_panel": getattr(self.main_window, 'geomet_domain_dialog', None),
            "geomet_plant_panel": getattr(self.main_window, 'geomet_plant_dialog', None),
            "esg_dashboard": getattr(self.main_window, 'esg_dashboard_dialog', None),
            "production_dashboard": getattr(self.main_window, 'production_dashboard_dialog', None),
            "reconciliation_panel": getattr(self.main_window, 'reconciliation_dialog', None),
        }
        
        # Show panels in the list
        for panel_id in panel_ids:
            panel = available_panels.get(panel_id)
            if panel:
                panel.show()
                panel.raise_()
                panel.activateWindow()
        
        logger.debug(f"Applied visibility for {len(panel_ids)} panels")
    
    def save_current_layout_as_profile(self, profile_id: str, name: str) -> None:
        """
        Save current window layout as a profile preset.
        
        Args:
            profile_id: Profile identifier
            name: Display name
        """
        # Save Qt window state
        layout_state = self.main_window.saveState()
        
        # Store in settings
        key = f"layout_preset_{profile_id}"
        self.settings.setValue(key, layout_state)
        
        # Update profile
        if profile_id in self.profiles:
            profile = self.profiles[profile_id]
            profile.layout_preset_id = profile_id
            profile.name = name
        
        logger.info(f"Saved layout as profile preset: {profile_id}")
    
    def _restore_layout(self, preset_id: str) -> None:
        """
        Restore a saved layout preset.
        
        Args:
            preset_id: Preset identifier
        """
        key = f"layout_preset_{preset_id}"
        layout_state = self.settings.value(key)
        
        if layout_state:
            self.main_window.restoreState(layout_state)
            logger.info(f"Restored layout preset: {preset_id}")
        else:
            logger.warning(f"Layout preset '{preset_id}' not found")
    
    def get_profile(self, profile_id: str) -> Optional[WorkbenchProfile]:
        """
        Get a profile by ID.
        
        Args:
            profile_id: Profile identifier
        
        Returns:
            WorkbenchProfile or None
        """
        return self.profiles.get(profile_id)
    
    def list_profiles(self) -> List[WorkbenchProfile]:
        """
        List all registered profiles.
        
        Returns:
            List of WorkbenchProfile objects
        """
        return list(self.profiles.values())

