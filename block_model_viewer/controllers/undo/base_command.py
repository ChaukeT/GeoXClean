"""
Base Command — Abstract undoable command (Command Pattern).

Every undoable action in GeoX is a Command subclass. Commands capture
the state needed to undo AND redo. They are pushed onto the UndoManager
stack after execution.

Usage:
    class SetColormapCommand(UndoableCommand):
        def __init__(self, controller, new_cmap):
            super().__init__("Change Colormap")
            self._ctrl = controller
            self._new = new_cmap
            self._old = controller.s.color_map  # capture before

        def execute(self):
            self._ctrl.set_colormap(self._new)

        def undo(self):
            self._ctrl.set_colormap(self._old)

Commands can be merged when they represent incremental adjustments
to the same property (e.g. dragging a slider). Override `merge_with()`
and `merge_id` to enable this.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class UndoableCommand(ABC):
    """
    Abstract base for all undoable commands.

    Subclasses MUST implement:
        execute()  — perform the action (called once on first do)
        undo()     — reverse the action

    Subclasses MAY override:
        redo()     — re-perform (defaults to calling execute())
        merge_with(other) — merge consecutive similar commands
        merge_id   — string key for merge grouping (None = no merge)
        is_obsolete() — True if command has no effect (skip it)
    """

    def __init__(self, description: str = ""):
        self._description = description
        self._timestamp = time.time()

    # ── Required overrides ───────────────────────────────────────

    @abstractmethod
    def execute(self) -> None:
        """Perform the action. Called once when the command is first created."""
        ...

    @abstractmethod
    def undo(self) -> None:
        """Reverse the action, restoring previous state."""
        ...

    # ── Optional overrides ───────────────────────────────────────

    def redo(self) -> None:
        """
        Re-perform the action after an undo.

        Default implementation calls execute(). Override only if
        redo differs from the initial execution.
        """
        self.execute()

    @property
    def merge_id(self) -> Optional[str]:
        """
        Return a string key for merge grouping, or None to disable merging.

        Commands with the same merge_id that arrive within the merge
        window (default 500ms) will be merged into a single undo step.

        Example: "colormap", "transparency", "opacity:layer_name"
        """
        return None

    def merge_with(self, newer: 'UndoableCommand') -> bool:
        """
        Attempt to absorb `newer` into this command.

        Called when `newer` has the same `merge_id` and arrives within
        the merge window. If this returns True, `newer` is discarded
        and this command's "new value" is updated to `newer`'s.

        Returns:
            True if merge succeeded (newer is discarded).
            False if merge failed (newer becomes a separate undo step).
        """
        return False

    def is_obsolete(self) -> bool:
        """
        Return True if this command has no net effect.

        Example: setting colormap to the value it already has.
        Obsolete commands are silently discarded.
        """
        return False

    # ── Properties ───────────────────────────────────────────────

    @property
    def description(self) -> str:
        return self._description

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._description}>"


class MacroCommand(UndoableCommand):
    """
    A compound command that groups multiple sub-commands into one undo step.

    Usage:
        macro = MacroCommand("Apply Analysis Settings", [
            SetColormapCommand(ctrl, "plasma"),
            SetTransparencyCommand(ctrl, 0.8),
            ApplyFilterCommand(ctrl, filters),
        ])
        undo_manager.execute(macro)
        # Single Ctrl+Z undoes all three
    """

    def __init__(self, description: str, commands: list[UndoableCommand]):
        super().__init__(description)
        self._commands = list(commands)

    def execute(self) -> None:
        for cmd in self._commands:
            cmd.execute()

    def undo(self) -> None:
        # Undo in reverse order
        for cmd in reversed(self._commands):
            cmd.undo()

    def redo(self) -> None:
        for cmd in self._commands:
            cmd.redo()

    def is_obsolete(self) -> bool:
        return all(cmd.is_obsolete() for cmd in self._commands)

    @property
    def commands(self) -> list[UndoableCommand]:
        return list(self._commands)

    def __repr__(self) -> str:
        return (
            f"<MacroCommand: {self._description} "
            f"({len(self._commands)} sub-commands)>"
        )
