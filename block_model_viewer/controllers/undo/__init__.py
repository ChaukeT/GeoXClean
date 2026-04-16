"""
Undo/Redo System — Command Pattern for GeoX.

Provides undoable actions throughout the application. Every state change
that a user might want to reverse is wrapped in an UndoableCommand and
executed through the UndoManager.

Package structure:
    base_command.py   — UndoableCommand ABC, MacroCommand
    undo_manager.py   — Stack-based manager with merge + signals
    commands.py       — All concrete command classes

Integration:
    The UndoManager lives on AppController and is wired to Edit > Undo/Redo.
    Panels create commands and submit them via controller.undo_manager.execute(cmd).

Quick start for panel developers:
    from core.undo.commands import SetColormapCommand

    # In a panel method:
    cmd = SetColormapCommand(self.controller, "plasma")
    self.controller.undo_manager.execute(cmd)

    # For generic parameter changes:
    from core.undo.commands import ParameterChangeCommand
    cmd = ParameterChangeCommand(
        target=self, attr_name="cutoff_grade",
        new_value=0.5, description="Change Cut-off Grade"
    )
    self.controller.undo_manager.execute(cmd)
"""

from .base_command import UndoableCommand, MacroCommand
from .undo_manager import UndoManager
from .commands import (
    # Visualization
    SetActivePropertyCommand,
    SetColormapCommand,
    SetTransparencyCommand,
    SetGlobalOpacityCommand,
    SetBackgroundColorCommand,
    SetEdgeVisibilityCommand,
    SetEdgeColorCommand,
    SetLightingCommand,
    # Filters
    ApplyFiltersCommand,
    ApplySliceCommand,
    # Layers
    SetLayerVisibilityCommand,
    SetLayerOpacityCommand,
    SetActiveLayerCommand,
    # Legend
    SetLegendVisibilityCommand,
    SetLegendOrientationCommand,
    # Overlays
    SetAxesVisibleCommand,
    SetGroundGridVisibleCommand,
    SetGroundGridSpacingCommand,
    # Generic
    ParameterChangeCommand,
    CallableCommand,
)

__all__ = [
    # Core
    'UndoableCommand',
    'MacroCommand',
    'UndoManager',
    # Visualization
    'SetActivePropertyCommand',
    'SetColormapCommand',
    'SetTransparencyCommand',
    'SetGlobalOpacityCommand',
    'SetBackgroundColorCommand',
    'SetEdgeVisibilityCommand',
    'SetEdgeColorCommand',
    'SetLightingCommand',
    # Filters
    'ApplyFiltersCommand',
    'ApplySliceCommand',
    # Layers
    'SetLayerVisibilityCommand',
    'SetLayerOpacityCommand',
    'SetActiveLayerCommand',
    # Legend
    'SetLegendVisibilityCommand',
    'SetLegendOrientationCommand',
    # Overlays
    'SetAxesVisibleCommand',
    'SetGroundGridVisibleCommand',
    'SetGroundGridSpacingCommand',
    # Generic
    'ParameterChangeCommand',
    'CallableCommand',
]
