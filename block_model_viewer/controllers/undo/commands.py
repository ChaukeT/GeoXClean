"""
Concrete Undoable Commands for GeoX.

Each command captures the old value before execution and restores it on undo.
Commands that represent incremental slider-like changes support merging.

Command categories:
    - Visualization: colormap, transparency, background, edges, lighting
    - Property: active property selection
    - Filters: property filters, spatial slices
    - Layers: visibility, opacity, removal
    - Selection: block selection changes
    - Legend: visibility, orientation, font size
    - Parameters: generic panel parameter changes (for analysis panels)
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .base_command import UndoableCommand

if TYPE_CHECKING:
    from ..app_controller import AppController

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION COMMANDS
# ═══════════════════════════════════════════════════════════════════

class SetActivePropertyCommand(UndoableCommand):
    """Change the active display property."""

    def __init__(self, controller: 'AppController', new_property: Optional[str]):
        old = controller.s.color_property if hasattr(controller.s, 'color_property') else None
        super().__init__(f"Set Property '{new_property or 'None'}'")
        self._ctrl = controller
        self._old = old
        self._new = new_property

    def execute(self):
        self._ctrl.set_active_property(self._new)

    def undo(self):
        self._ctrl.set_active_property(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class SetColormapCommand(UndoableCommand):
    """Change the colormap."""

    def __init__(self, controller: 'AppController', new_cmap: str):
        super().__init__(f"Set Colormap '{new_cmap}'")
        self._ctrl = controller
        self._old = controller.s.color_map
        self._new = new_cmap

    def execute(self):
        self._ctrl.set_colormap(self._new)

    def undo(self):
        self._ctrl.set_colormap(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new

    @property
    def merge_id(self) -> Optional[str]:
        return "colormap"

    def merge_with(self, newer: 'SetColormapCommand') -> bool:
        self._new = newer._new
        self._description = f"Set Colormap '{self._new}'"
        return True


class SetTransparencyCommand(UndoableCommand):
    """Change global transparency/opacity."""

    def __init__(self, controller: 'AppController', new_alpha: float):
        super().__init__(f"Set Transparency {new_alpha:.0%}")
        self._ctrl = controller
        self._old = controller.s.transparency
        self._new = new_alpha

    def execute(self):
        self._ctrl.set_transparency(self._new)

    def undo(self):
        self._ctrl.set_transparency(self._old)

    def is_obsolete(self) -> bool:
        return abs(self._old - self._new) < 1e-6

    @property
    def merge_id(self) -> Optional[str]:
        return "transparency"

    def merge_with(self, newer: 'SetTransparencyCommand') -> bool:
        self._new = newer._new
        self._description = f"Set Transparency {self._new:.0%}"
        return True


class SetGlobalOpacityCommand(UndoableCommand):
    """Change global opacity."""

    def __init__(self, controller: 'AppController', new_opacity: float):
        super().__init__(f"Set Opacity {new_opacity:.0%}")
        self._ctrl = controller
        self._old = getattr(controller.s, 'global_opacity', 1.0)
        self._new = new_opacity

    def execute(self):
        self._ctrl.set_global_opacity(self._new)

    def undo(self):
        self._ctrl.set_global_opacity(self._old)

    def is_obsolete(self) -> bool:
        return abs(self._old - self._new) < 1e-6

    @property
    def merge_id(self) -> Optional[str]:
        return "global_opacity"

    def merge_with(self, newer: 'SetGlobalOpacityCommand') -> bool:
        self._new = newer._new
        self._description = f"Set Opacity {self._new:.0%}"
        return True


class SetBackgroundColorCommand(UndoableCommand):
    """Change viewport background color."""

    def __init__(self, controller: 'AppController', new_color: tuple):
        super().__init__("Change Background Color")
        self._ctrl = controller
        self._old = getattr(controller.s, 'background_color', (0.1, 0.1, 0.1))
        self._new = new_color

    def execute(self):
        self._ctrl.set_background_color(self._new)

    def undo(self):
        self._ctrl.set_background_color(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class SetEdgeVisibilityCommand(UndoableCommand):
    """Toggle edge visibility."""

    def __init__(self, controller: 'AppController', visible: bool):
        state = "Show" if visible else "Hide"
        super().__init__(f"{state} Block Edges")
        self._ctrl = controller
        self._old = getattr(controller.s, 'edge_visible', False)
        self._new = visible

    def execute(self):
        self._ctrl.set_edge_visibility(self._new)

    def undo(self):
        self._ctrl.set_edge_visibility(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class SetEdgeColorCommand(UndoableCommand):
    """Change block edge color."""

    def __init__(self, controller: 'AppController', new_color: tuple):
        super().__init__("Change Edge Color")
        self._ctrl = controller
        self._old = getattr(controller.s, 'edge_color', (0, 0, 0))
        self._new = new_color

    def execute(self):
        self._ctrl.set_edge_color(self._new)

    def undo(self):
        self._ctrl.set_edge_color(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class SetLightingCommand(UndoableCommand):
    """Toggle lighting."""

    def __init__(self, controller: 'AppController', enabled: bool):
        state = "Enable" if enabled else "Disable"
        super().__init__(f"{state} Lighting")
        self._ctrl = controller
        self._old = getattr(controller.s, 'lighting_enabled', True)
        self._new = enabled

    def execute(self):
        self._ctrl.set_lighting_enabled(self._new)

    def undo(self):
        self._ctrl.set_lighting_enabled(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


# ═══════════════════════════════════════════════════════════════════
# FILTER / SLICE COMMANDS
# ═══════════════════════════════════════════════════════════════════

class ApplyFiltersCommand(UndoableCommand):
    """Apply property filters (dict of property → (min, max) tuples)."""

    def __init__(self, controller: 'AppController', new_filters: Dict[str, tuple]):
        super().__init__("Apply Filters")
        self._ctrl = controller
        # Deep copy to avoid mutation
        self._old = copy.deepcopy(getattr(controller.s, 'active_filters', {}))
        self._new = copy.deepcopy(new_filters)

    def execute(self):
        self._ctrl.apply_filters(self._new)

    def undo(self):
        self._ctrl.apply_filters(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class ApplySliceCommand(UndoableCommand):
    """Apply a spatial slice along an axis."""

    def __init__(self, controller: 'AppController', axis: str, position: float):
        super().__init__(f"Slice {axis.upper()} at {position:.1f}")
        self._ctrl = controller
        self._axis = axis
        self._new_pos = position
        # Capture current slice state
        slices = getattr(controller.s, 'active_slices', {})
        self._old_axis = axis
        self._old_pos = slices.get(axis)

    def execute(self):
        self._ctrl.apply_slice(self._axis, self._new_pos)

    def undo(self):
        if self._old_pos is not None:
            self._ctrl.apply_slice(self._old_axis, self._old_pos)
        else:
            # No previous slice — clear by applying neutral position
            self._ctrl.refresh_scene()

    @property
    def merge_id(self) -> Optional[str]:
        return f"slice:{self._axis}"

    def merge_with(self, newer: 'ApplySliceCommand') -> bool:
        if newer._axis != self._axis:
            return False
        self._new_pos = newer._new_pos
        self._description = f"Slice {self._axis.upper()} at {self._new_pos:.1f}"
        return True


# ═══════════════════════════════════════════════════════════════════
# LAYER COMMANDS
# ═══════════════════════════════════════════════════════════════════

class SetLayerVisibilityCommand(UndoableCommand):
    """Toggle a layer's visibility."""

    def __init__(self, controller: 'AppController', layer_name: str, visible: bool):
        state = "Show" if visible else "Hide"
        super().__init__(f"{state} Layer '{layer_name}'")
        self._ctrl = controller
        self._layer = layer_name
        self._new = visible
        self._old = not visible  # toggling from current state

    def execute(self):
        self._ctrl.set_layer_visibility(self._layer, self._new)

    def undo(self):
        self._ctrl.set_layer_visibility(self._layer, self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class SetLayerOpacityCommand(UndoableCommand):
    """Change a layer's opacity."""

    def __init__(self, controller: 'AppController', layer_name: str,
                 new_opacity: float, old_opacity: float = 1.0):
        super().__init__(f"Set '{layer_name}' Opacity {new_opacity:.0%}")
        self._ctrl = controller
        self._layer = layer_name
        self._new = new_opacity
        self._old = old_opacity

    def execute(self):
        self._ctrl.set_layer_opacity(self._layer, self._new)

    def undo(self):
        self._ctrl.set_layer_opacity(self._layer, self._old)

    def is_obsolete(self) -> bool:
        return abs(self._old - self._new) < 1e-6

    @property
    def merge_id(self) -> Optional[str]:
        return f"layer_opacity:{self._layer}"

    def merge_with(self, newer: 'SetLayerOpacityCommand') -> bool:
        if newer._layer != self._layer:
            return False
        self._new = newer._new
        self._description = f"Set '{self._layer}' Opacity {self._new:.0%}"
        return True


class SetActiveLayerCommand(UndoableCommand):
    """Switch the active layer."""

    def __init__(self, controller: 'AppController', new_layer: str,
                 old_layer: Optional[str] = None):
        super().__init__(f"Activate Layer '{new_layer}'")
        self._ctrl = controller
        self._new = new_layer
        self._old = old_layer

    def execute(self):
        self._ctrl.set_active_layer(self._new)

    def undo(self):
        if self._old is not None:
            self._ctrl.set_active_layer(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


# ═══════════════════════════════════════════════════════════════════
# LEGEND COMMANDS
# ═══════════════════════════════════════════════════════════════════

class SetLegendVisibilityCommand(UndoableCommand):
    """Toggle legend visibility."""

    def __init__(self, controller: 'AppController', visible: bool):
        state = "Show" if visible else "Hide"
        super().__init__(f"{state} Legend")
        self._ctrl = controller
        self._new = visible
        self._old = not visible

    def execute(self):
        self._ctrl.set_legend_visibility(self._new)

    def undo(self):
        self._ctrl.set_legend_visibility(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


class SetLegendOrientationCommand(UndoableCommand):
    """Change legend orientation."""

    def __init__(self, controller: 'AppController', orientation: str,
                 old_orientation: str = "vertical"):
        super().__init__(f"Set Legend {orientation.title()}")
        self._ctrl = controller
        self._new = orientation
        self._old = old_orientation

    def execute(self):
        self._ctrl.set_legend_orientation(self._new)

    def undo(self):
        self._ctrl.set_legend_orientation(self._old)

    def is_obsolete(self) -> bool:
        return self._old == self._new


# ═══════════════════════════════════════════════════════════════════
# OVERLAY / GRID COMMANDS
# ═══════════════════════════════════════════════════════════════════

class SetAxesVisibleCommand(UndoableCommand):
    """Toggle axes visibility."""

    def __init__(self, controller: 'AppController', visible: bool):
        state = "Show" if visible else "Hide"
        super().__init__(f"{state} Axes")
        self._ctrl = controller
        self._new = visible
        self._old = not visible

    def execute(self):
        self._ctrl.set_axes_visible(self._new)

    def undo(self):
        self._ctrl.set_axes_visible(self._old)


class SetGroundGridVisibleCommand(UndoableCommand):
    """Toggle ground grid visibility."""

    def __init__(self, controller: 'AppController', visible: bool):
        state = "Show" if visible else "Hide"
        super().__init__(f"{state} Ground Grid")
        self._ctrl = controller
        self._new = visible
        self._old = not visible

    def execute(self):
        self._ctrl.set_ground_grid_visible(self._new)

    def undo(self):
        self._ctrl.set_ground_grid_visible(self._old)


class SetGroundGridSpacingCommand(UndoableCommand):
    """Change ground grid spacing."""

    def __init__(self, controller: 'AppController', spacing: float,
                 old_spacing: float = 100.0):
        super().__init__(f"Set Grid Spacing {spacing:.0f}")
        self._ctrl = controller
        self._new = spacing
        self._old = old_spacing

    def execute(self):
        self._ctrl.set_ground_grid_spacing(self._new)

    def undo(self):
        self._ctrl.set_ground_grid_spacing(self._old)

    @property
    def merge_id(self) -> Optional[str]:
        return "grid_spacing"

    def merge_with(self, newer: 'SetGroundGridSpacingCommand') -> bool:
        self._new = newer._new
        self._description = f"Set Grid Spacing {self._new:.0f}"
        return True


# ═══════════════════════════════════════════════════════════════════
# GENERIC PARAMETER COMMAND
# ═══════════════════════════════════════════════════════════════════

class ParameterChangeCommand(UndoableCommand):
    """
    Generic command for any panel parameter change.

    Use this when a panel's parameter is changed and you want it
    to be undoable without writing a dedicated command class.

    Args:
        target: The object whose attribute is being changed
        attr_name: Attribute name on target
        new_value: New value to set
        description: Human-readable description
        apply_fn: Optional callable(value) to apply the change
                   (if None, uses setattr on target)
    """

    def __init__(self, target: Any, attr_name: str, new_value: Any,
                 description: str = "", apply_fn=None):
        self._target = target
        self._attr = attr_name
        self._new = new_value
        self._old = getattr(target, attr_name, None)
        self._apply_fn = apply_fn
        desc = description or f"Change {attr_name}"
        super().__init__(desc)

    def execute(self):
        self._apply(self._new)

    def undo(self):
        self._apply(self._old)

    def _apply(self, value):
        if self._apply_fn:
            self._apply_fn(value)
        else:
            setattr(self._target, self._attr, value)

    def is_obsolete(self) -> bool:
        return self._old == self._new

    @property
    def merge_id(self) -> Optional[str]:
        return f"param:{id(self._target)}:{self._attr}"

    def merge_with(self, newer: 'ParameterChangeCommand') -> bool:
        if (newer._target is not self._target or
                newer._attr != self._attr):
            return False
        self._new = newer._new
        self._description = newer._description
        return True


class CallableCommand(UndoableCommand):
    """
    Wraps an arbitrary do/undo callable pair as an undoable command.

    Use sparingly — prefer dedicated command classes for type safety.

    Args:
        do_fn: Callable to execute/redo
        undo_fn: Callable to undo
        description: Human-readable description
    """

    def __init__(self, do_fn, undo_fn, description: str = "Action"):
        super().__init__(description)
        self._do = do_fn
        self._undo = undo_fn

    def execute(self):
        self._do()

    def undo(self):
        self._undo()
