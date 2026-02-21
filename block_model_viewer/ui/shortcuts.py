"""
Unified Shortcut System for BlockModelViewer.

Centralized keyboard shortcut definitions and management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from enum import Enum

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class ShortcutCategory(Enum):
    """Shortcut categories."""
    MOVEMENT = "movement"
    VIEW = "view"
    SELECTION = "selection"
    PROPERTY = "property"
    LEGEND = "legend"
    CROSS_SECTION = "cross_section"
    EXPORT = "export"
    PANEL = "panel"


@dataclass
class ShortcutDefinition:
    """Definition for a keyboard shortcut."""
    name: str
    key_sequence: str
    category: ShortcutCategory
    description: str
    handler: Optional[Callable] = None
    
    def get_qkeysequence(self) -> QKeySequence:
        """Get QKeySequence for this shortcut."""
        return QKeySequence(self.key_sequence)


class Shortcuts:
    """
    Unified shortcut system.
    
    Provides centralized keyboard shortcut definitions.
    """
    
    _shortcuts: Dict[str, ShortcutDefinition] = {}
    
    @classmethod
    def initialize(cls):
        """Initialize default shortcuts."""
        # Movement shortcuts
        cls.register(
            ShortcutDefinition(
                name="move_forward",
                key_sequence="W",
                category=ShortcutCategory.MOVEMENT,
                description="Move camera forward"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="move_backward",
                key_sequence="S",
                category=ShortcutCategory.MOVEMENT,
                description="Move camera backward"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="move_left",
                key_sequence="A",
                category=ShortcutCategory.MOVEMENT,
                description="Move camera left"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="move_right",
                key_sequence="D",
                category=ShortcutCategory.MOVEMENT,
                description="Move camera right"
            )
        )
        
        # View shortcuts
        cls.register(
            ShortcutDefinition(
                name="reset_view",
                key_sequence="R",
                category=ShortcutCategory.VIEW,
                description="Reset camera to default view"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="fit_view",
                key_sequence="F",
                category=ShortcutCategory.VIEW,
                description="Fit model to viewport"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="rotate_left",
                key_sequence="Left",
                category=ShortcutCategory.VIEW,
                description="Rotate view left"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="rotate_right",
                key_sequence="Right",
                category=ShortcutCategory.VIEW,
                description="Rotate view right"
            )
        )
        
        # Selection shortcuts
        cls.register(
            ShortcutDefinition(
                name="select_mode",
                key_sequence="S",
                category=ShortcutCategory.SELECTION,
                description="Enable selection mode"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="pick_block",
                key_sequence="P",
                category=ShortcutCategory.SELECTION,
                description="Pick block at cursor"
            )
        )
        
        # Property shortcuts
        cls.register(
            ShortcutDefinition(
                name="property_1",
                key_sequence="Ctrl+1",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 1"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_2",
                key_sequence="Ctrl+2",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 2"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_3",
                key_sequence="Ctrl+3",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 3"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_4",
                key_sequence="Ctrl+4",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 4"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_5",
                key_sequence="Ctrl+5",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 5"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_6",
                key_sequence="Ctrl+6",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 6"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_7",
                key_sequence="Ctrl+7",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 7"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_8",
                key_sequence="Ctrl+8",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 8"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="property_9",
                key_sequence="Ctrl+9",
                category=ShortcutCategory.PROPERTY,
                description="Switch to property 9"
            )
        )
        
        # Legend shortcuts
        cls.register(
            ShortcutDefinition(
                name="toggle_legend",
                key_sequence="L",
                category=ShortcutCategory.LEGEND,
                description="Toggle legend visibility"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="reverse_colormap",
                key_sequence="Ctrl+R",
                category=ShortcutCategory.LEGEND,
                description="Reverse colormap"
            )
        )
        
        # Cross-section shortcuts
        cls.register(
            ShortcutDefinition(
                name="toggle_cross_section",
                key_sequence="X",
                category=ShortcutCategory.CROSS_SECTION,
                description="Toggle cross-section tool"
            )
        )
        
        # Export shortcuts
        cls.register(
            ShortcutDefinition(
                name="screenshot",
                key_sequence="F12",
                category=ShortcutCategory.EXPORT,
                description="Export screenshot"
            )
        )
        
        # Panel shortcuts
        cls.register(
            ShortcutDefinition(
                name="toggle_property_panel",
                key_sequence="Ctrl+1",
                category=ShortcutCategory.PANEL,
                description="Toggle Property Controls panel"
            )
        )
        cls.register(
            ShortcutDefinition(
                name="toggle_scene_panel",
                key_sequence="Ctrl+3",
                category=ShortcutCategory.PANEL,
                description="Toggle Scene Inspector panel"
            )
        )
        
        logger.info(f"Initialized Shortcuts with {len(cls._shortcuts)} shortcuts")
    
    @classmethod
    def register(cls, shortcut: ShortcutDefinition):
        """
        Register a shortcut.
        
        Args:
            shortcut: Shortcut definition
        """
        cls._shortcuts[shortcut.name] = shortcut
        logger.debug(f"Registered shortcut: {shortcut.name} -> {shortcut.key_sequence}")
    
    @classmethod
    def get(cls, name: str) -> Optional[ShortcutDefinition]:
        """
        Get shortcut by name.
        
        Args:
            name: Shortcut name
            
        Returns:
            Shortcut definition or None
        """
        return cls._shortcuts.get(name)
    
    @classmethod
    def get_by_category(cls, category: ShortcutCategory) -> list[ShortcutDefinition]:
        """
        Get shortcuts by category.
        
        Args:
            category: Shortcut category
            
        Returns:
            List of shortcut definitions
        """
        return [s for s in cls._shortcuts.values() if s.category == category]
    
    @classmethod
    def get_all(cls) -> Dict[str, ShortcutDefinition]:
        """Get all shortcuts."""
        return cls._shortcuts.copy()
    
    @classmethod
    def create_shortcut(cls, name: str, parent, handler: Callable) -> Optional[QShortcut]:
        """
        Create a QShortcut for a registered shortcut.
        
        Args:
            name: Shortcut name
            parent: Parent widget
            handler: Handler function
            
        Returns:
            QShortcut or None if shortcut not found
        """
        shortcut_def = cls.get(name)
        if not shortcut_def:
            logger.warning(f"Shortcut not found: {name}")
            return None
        
        shortcut = QShortcut(shortcut_def.get_qkeysequence(), parent)
        shortcut.activated.connect(handler)
        shortcut_def.handler = handler
        
        return shortcut


# Initialize shortcuts on import
Shortcuts.initialize()

