"""
Interaction module - Handles mouse modes, VTK interactor styles, and camera controls.

This module extracts interaction logic from MainWindow for cleaner architecture.
"""

from .mouse_modes import MouseMode, MOUSE_MODE_DESCRIPTIONS
from .interaction_controller import InteractionController

__all__ = [
    'MouseMode',
    'MOUSE_MODE_DESCRIPTIONS',
    'InteractionController',
]

