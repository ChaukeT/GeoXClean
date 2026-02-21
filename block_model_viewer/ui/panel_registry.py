"""
Panel Registry for BlockModelViewer.

Central registry for all UI panels with metadata, docking rules, and layout management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QWidget
from PyQt6.QtGui import QIcon

logger = logging.getLogger(__name__)


class PanelCategory(Enum):
    """Panel categories for organization."""
    PROPERTY = "property"
    SCENE = "scene"
    LAYER = "layer"
    INFO = "info"
    ANALYSIS = "analysis"
    SELECTION = "selection"
    CROSS_SECTION = "cross_section"
    DISPLAY = "display"
    RESOURCE = "resource"
    PLANNING = "planning"


class DockArea(Enum):
    """Dock widget areas."""
    LEFT = Qt.DockWidgetArea.LeftDockWidgetArea
    RIGHT = Qt.DockWidgetArea.RightDockWidgetArea
    TOP = Qt.DockWidgetArea.TopDockWidgetArea
    BOTTOM = Qt.DockWidgetArea.BottomDockWidgetArea
    FLOATING = None  # Floating window


@dataclass
class PanelMetadata:
    """Metadata for a UI panel."""
    name: str
    category: PanelCategory
    icon_name: Optional[str] = None
    shortcut: Optional[str] = None
    default_dock_area: DockArea = DockArea.LEFT
    default_visible: bool = True
    minimum_width: int = 250
    minimum_height: int = 200
    factory: Optional[Callable[[], QWidget]] = None
    tooltip: Optional[str] = None
    
    def get_icon(self) -> QIcon:
        """Get icon for this panel."""
        if not self.icon_name:
            return QIcon()
        
        from pathlib import Path
        icons_dir = Path(__file__).parent.parent / "assets" / "icons"
        icon_path = icons_dir / f"{self.icon_name}.svg"
        
        if icon_path.exists():
            return QIcon(str(icon_path))
        
        # Fallback to ui/icons
        ui_icons_dir = Path(__file__).parent / "icons"
        icon_path = ui_icons_dir / f"{self.icon_name}.svg"
        if icon_path.exists():
            return QIcon(str(icon_path))
        
        return QIcon()


class PanelRegistry:
    """
    Central registry for all UI panels.
    
    Provides:
    - Panel metadata and factory functions
    - Default layout configurations
    - Docking rules
    """
    
    _instance: Optional['PanelRegistry'] = None
    _panels: Dict[str, PanelMetadata] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize panel registry with default panels."""
        # Property panels
        self.register_panel(
            PanelMetadata(
                name="Property Controls",
                category=PanelCategory.PROPERTY,
                icon_name="block",
                shortcut="Ctrl+1",
                default_dock_area=DockArea.LEFT,
                tooltip="Property selection and filtering"
            )
        )
        
        self.register_panel(
            PanelMetadata(
                name="Scene Inspector",
                category=PanelCategory.SCENE,
                icon_name="pit",
                shortcut="Ctrl+3",
                default_dock_area=DockArea.LEFT,
                tooltip="Scene settings and overlays"
            )
        )
        
        # Info panels
        self.register_panel(
            PanelMetadata(
                name="Block Information",
                category=PanelCategory.INFO,
                icon_name="block",
                shortcut="Ctrl+I",
                default_dock_area=DockArea.RIGHT,
                tooltip="Selected block information"
            )
        )
        
        self.register_panel(
            PanelMetadata(
                name="Pick Info",
                category=PanelCategory.INFO,
                icon_name="block",
                shortcut="Ctrl+P",
                default_dock_area=DockArea.RIGHT,
                tooltip="Picked element information"
            )
        )
        
        # Selection panels
        self.register_panel(
            PanelMetadata(
                name="Selection Manager",
                category=PanelCategory.SELECTION,
                icon_name="block",
                shortcut="Ctrl+S",
                default_dock_area=DockArea.RIGHT,
                default_visible=False,
                tooltip="Multi-block selection management"
            )
        )
        
        # Cross-section panels
        self.register_panel(
            PanelMetadata(
                name="Cross-Section Manager",
                category=PanelCategory.CROSS_SECTION,
                icon_name="block",
                shortcut="Ctrl+X",
                default_dock_area=DockArea.RIGHT,
                default_visible=False,
                tooltip="Cross-section creation and management"
            )
        )
        
        # Display panels
        self.register_panel(
            PanelMetadata(
                name="Display Settings",
                category=PanelCategory.DISPLAY,
                icon_name="layers",
                shortcut="Ctrl+D",
                default_dock_area=DockArea.RIGHT,
                default_visible=False,
                tooltip="Display and rendering settings"
            )
        )
        
        logger.info(f"Initialized PanelRegistry with {len(self._panels)} panels")
    
    def register_panel(self, metadata: PanelMetadata):
        """
        Register a panel with the registry.
        
        Args:
            metadata: Panel metadata
        """
        self._panels[metadata.name] = metadata
        logger.debug(f"Registered panel: {metadata.name}")
    
    def get_panel(self, name: str) -> Optional[PanelMetadata]:
        """
        Get panel metadata by name.
        
        Args:
            name: Panel name
            
        Returns:
            Panel metadata or None if not found
        """
        return self._panels.get(name)
    
    def get_panels_by_category(self, category: PanelCategory) -> List[PanelMetadata]:
        """
        Get all panels in a category.
        
        Args:
            category: Panel category
            
        Returns:
            List of panel metadata
        """
        return [p for p in self._panels.values() if p.category == category]
    
    def get_default_layout(self) -> Dict[str, Any]:
        """
        Get default workspace layout configuration.
        
        Returns:
            Dictionary with layout configuration
        """
        return {
            "left_dock": {
                "panels": ["Property Controls", "Scene Inspector"],
                "area": DockArea.LEFT,
                "tabs": True
            },
            "right_dock": {
                "panels": ["Block Information", "Pick Info"],
                "area": DockArea.RIGHT,
                "tabs": True
            },
            "floating": {
                "panels": ["Selection Manager", "Cross-Section Manager", "Display Settings"],
                "visible": False
            }
        }
    
    def get_workspace_layout(self, layout_name: str) -> Dict[str, Any]:
        """
        Get predefined workspace layout.
        
        Args:
            layout_name: Layout name ("geology", "resource", "planning", "analytics")
            
        Returns:
            Layout configuration
        """
        layouts = {
            "geology": {
                "left_dock": {
                    "panels": ["Property Controls", "Scene Inspector"],
                    "area": DockArea.LEFT
                },
                "right_dock": {
                    "panels": ["Block Information", "Pick Info"],
                    "area": DockArea.RIGHT
                }
            },
            "resource": {
                "left_dock": {
                    "panels": ["Property Controls"],
                    "area": DockArea.LEFT
                },
                "right_dock": {
                    "panels": ["Block Information", "Selection Manager"],
                    "area": DockArea.RIGHT
                }
            },
            "planning": {
                "left_dock": {
                    "panels": ["Property Controls", "Scene Inspector"],
                    "area": DockArea.LEFT
                },
                "right_dock": {
                    "panels": ["Cross-Section Manager", "Block Information"],
                    "area": DockArea.RIGHT
                }
            },
            "analytics": {
                "left_dock": {
                    "panels": ["Property Controls"],
                    "area": DockArea.LEFT
                },
                "right_dock": {
                    "panels": ["Block Information", "Display Settings"],
                    "area": DockArea.RIGHT
                }
            }
        }
        
        return layouts.get(layout_name, self.get_default_layout())
    
    def create_dock_widget(self, panel_name: str, parent: QWidget) -> Optional[QDockWidget]:
        """
        Create a dock widget for a panel.
        
        Args:
            panel_name: Panel name
            parent: Parent widget
            
        Returns:
            QDockWidget or None if panel not found
        """
        metadata = self.get_panel(panel_name)
        if not metadata:
            logger.warning(f"Panel not found: {panel_name}")
            return None
        
        if not metadata.factory:
            logger.warning(f"Panel {panel_name} has no factory function")
            return None
        
        # Create panel widget
        panel_widget = metadata.factory()
        
        # Create dock widget
        dock = QDockWidget(metadata.name, parent)
        dock.setObjectName(f"Dock_{panel_name.replace(' ', '_')}")
        dock.setWidget(panel_widget)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        
        # Set minimum size
        dock.setMinimumWidth(metadata.minimum_width)
        dock.setMinimumHeight(metadata.minimum_height)
        
        # Set icon
        icon = metadata.get_icon()
        if not icon.isNull():
            dock.setWindowIcon(icon)
        
        # Set tooltip
        if metadata.tooltip:
            dock.setToolTip(metadata.tooltip)
        
        return dock
    
    def get_all_panels(self) -> List[PanelMetadata]:
        """Get all registered panels."""
        return list(self._panels.values())


# Global registry instance
def get_panel_registry() -> PanelRegistry:
    """Get the global panel registry instance."""
    return PanelRegistry()

