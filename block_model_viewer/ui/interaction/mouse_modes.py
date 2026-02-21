"""
Mouse mode definitions and constants.

Centralizes all mouse/interaction mode identifiers to avoid magic strings.
"""

from enum import Enum, auto
from typing import Dict


class MouseMode(Enum):
    """Enumeration of supported mouse interaction modes."""
    SELECT = auto()
    PAN = auto()
    ZOOM_BOX = auto()
    ROTATE = auto()
    ORIGINAL = auto()
    
    @classmethod
    def from_string(cls, mode_str: str) -> 'MouseMode':
        """Convert a string to MouseMode, with normalization."""
        m = (mode_str or "").lower().strip()
        mapping = {
            'select': cls.SELECT,
            'click': cls.SELECT,
            'pan': cls.PAN,
            'zoom': cls.ZOOM_BOX,
            'zoom_box': cls.ZOOM_BOX,
            'zoombox': cls.ZOOM_BOX,
            'rotate': cls.ROTATE,
            'trackball': cls.ROTATE,
            'original': cls.ORIGINAL,
            'default': cls.ORIGINAL,
            'reset': cls.ORIGINAL,
        }
        return mapping.get(m, cls.ORIGINAL)
    
    def to_string(self) -> str:
        """Convert MouseMode to a normalized string."""
        return {
            MouseMode.SELECT: 'select',
            MouseMode.PAN: 'pan',
            MouseMode.ZOOM_BOX: 'zoom_box',
            MouseMode.ROTATE: 'rotate',
            MouseMode.ORIGINAL: 'original',
        }.get(self, 'original')


# Human-readable descriptions for each mode
MOUSE_MODE_DESCRIPTIONS: Dict[MouseMode, str] = {
    MouseMode.SELECT: "Mouse Mode: Select/Click (Left-click to pick blocks)",
    MouseMode.PAN: "Mouse Mode: Pan (Drag to pan view)",
    MouseMode.ZOOM_BOX: "Mouse Mode: Zoom Box (Drag a rectangle to zoom)",
    MouseMode.ROTATE: "Mouse Mode: Rotate (Drag to rotate view)",
    MouseMode.ORIGINAL: "Mouse Mode: Original/Default",
}


# Short descriptions for status bar
MOUSE_MODE_SHORT_DESCRIPTIONS: Dict[MouseMode, str] = {
    MouseMode.SELECT: "Mouse Mode: Select/Click",
    MouseMode.PAN: "Mouse Mode: Pan",
    MouseMode.ZOOM_BOX: "Mouse Mode: Zoom Box",
    MouseMode.ROTATE: "Mouse Mode: Rotate",
    MouseMode.ORIGINAL: "Mouse Mode: Original/Default",
}

