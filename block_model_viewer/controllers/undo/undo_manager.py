"""
Undo Manager — Stack-based undo/redo with command merging.

Central manager for the entire application. Owns two stacks (undo, redo),
handles execution of commands, merging of rapid incremental changes,
and emits signals so the Edit menu can update Undo/Redo labels.

Integration:
    # In AppController.__init__:
    from .undo.undo_manager import UndoManager
    self.undo_manager = UndoManager(max_history=200)
    self.undo_manager.state_changed.connect(self.signals.undo_state_changed)

    # Executing an undoable action:
    from .undo.commands import SetColormapCommand
    cmd = SetColormapCommand(self, "plasma")
    self.undo_manager.execute(cmd)

    # In MainWindow (Edit menu already wired):
    self.controller.undo_manager.undo()
    self.controller.undo_manager.redo()
"""

import logging
import time
from typing import List, Optional

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    QObject = object
    def pyqtSignal(*args, **kwargs):
        return None

from .base_command import UndoableCommand

logger = logging.getLogger(__name__)


class UndoManager(QObject):
    """
    Manages undo/redo stacks and command execution.

    Features:
        - Bounded history (default 200 commands)
        - Command merging within a configurable time window
        - Signals for UI state updates (enable/disable Undo/Redo)
        - MacroCommand support (group multiple commands)
        - Clear history on major state changes (e.g. new project)
    """

    # Emitted whenever undo/redo availability changes.
    # Args: (can_undo: bool, can_redo: bool, undo_text: str, redo_text: str)
    state_changed = pyqtSignal(bool, bool, str, str)

    # Emitted after any command is executed, undone, or redone.
    # Args: (action: str, description: str)
    # action is one of: "execute", "undo", "redo"
    command_executed = pyqtSignal(str, str)

    # Default merge window in seconds
    MERGE_WINDOW = 0.5  # 500ms

    def __init__(self, max_history: int = 200, parent=None):
        super().__init__(parent)
        self._undo_stack: List[UndoableCommand] = []
        self._redo_stack: List[UndoableCommand] = []
        self._max_history = max_history
        self._enabled = True
        self._inside_undo_redo = False  # guard against re-entrant pushes

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def execute(self, command: UndoableCommand) -> bool:
        """
        Execute a command and push it onto the undo stack.

        The command's execute() is called immediately. If the command
        is obsolete (no-op), it is silently discarded.

        Clears the redo stack (forward history is lost after a new action).

        Returns:
            True if the command was executed and pushed.
            False if it was discarded (obsolete or disabled).
        """
        if not self._enabled:
            # Still execute the action, just don't track it
            command.execute()
            return False

        if self._inside_undo_redo:
            # Commands triggered by undo/redo should not be re-pushed
            return False

        # Check if command is a no-op
        if command.is_obsolete():
            logger.debug("Discarding obsolete command: %s", command)
            return False

        # Try to merge with the top of the undo stack
        if self._try_merge(command):
            self._emit_state()
            return True

        # Execute the command
        try:
            command.execute()
        except Exception as e:
            logger.error("Command execution failed: %s — %s", command, e, exc_info=True)
            return False

        # Push onto undo stack
        self._undo_stack.append(command)

        # Trim history if over limit
        while len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

        # Clear redo stack (new action invalidates forward history)
        self._redo_stack.clear()

        self._emit_state()
        self.command_executed.emit("execute", command.description)
        logger.debug("Executed: %s (stack: %d)", command, len(self._undo_stack))
        return True

    def undo(self) -> bool:
        """
        Undo the most recent command.

        Returns:
            True if a command was undone, False if stack is empty.
        """
        if not self.can_undo:
            return False

        command = self._undo_stack.pop()
        self._inside_undo_redo = True
        try:
            command.undo()
        except Exception as e:
            logger.error("Undo failed: %s — %s", command, e, exc_info=True)
            # Push it back so state isn't corrupted
            self._undo_stack.append(command)
            return False
        finally:
            self._inside_undo_redo = False

        self._redo_stack.append(command)

        self._emit_state()
        self.command_executed.emit("undo", command.description)
        logger.debug("Undone: %s (undo: %d, redo: %d)",
                      command, len(self._undo_stack), len(self._redo_stack))
        return True

    def redo(self) -> bool:
        """
        Redo the most recently undone command.

        Returns:
            True if a command was redone, False if redo stack is empty.
        """
        if not self.can_redo:
            return False

        command = self._redo_stack.pop()
        self._inside_undo_redo = True
        try:
            command.redo()
        except Exception as e:
            logger.error("Redo failed: %s — %s", command, e, exc_info=True)
            self._redo_stack.append(command)
            return False
        finally:
            self._inside_undo_redo = False

        self._undo_stack.append(command)

        self._emit_state()
        self.command_executed.emit("redo", command.description)
        logger.debug("Redone: %s (undo: %d, redo: %d)",
                      command, len(self._undo_stack), len(self._redo_stack))
        return True

    def clear(self) -> None:
        """
        Clear all history. Use on major state changes like New Project.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_state()
        logger.info("Undo history cleared")

    # ═══════════════════════════════════════════════════════════════
    # STATE QUERIES
    # ═══════════════════════════════════════════════════════════════

    @property
    def can_undo(self) -> bool:
        return self._enabled and len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return self._enabled and len(self._redo_stack) > 0

    @property
    def undo_text(self) -> str:
        """Description of the next undo action (for menu label)."""
        if self._undo_stack:
            return f"Undo {self._undo_stack[-1].description}"
        return "Undo"

    @property
    def redo_text(self) -> str:
        """Description of the next redo action (for menu label)."""
        if self._redo_stack:
            return f"Redo {self._redo_stack[-1].description}"
        return "Redo"

    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        self._emit_state()

    @property
    def history(self) -> List[str]:
        """Return list of undo command descriptions (oldest first)."""
        return [cmd.description for cmd in self._undo_stack]

    # ═══════════════════════════════════════════════════════════════
    # INTERNALS
    # ═══════════════════════════════════════════════════════════════

    def _try_merge(self, command: UndoableCommand) -> bool:
        """
        Try to merge command with the top of the undo stack.

        Merging collapses rapid incremental changes (e.g. slider drags)
        into a single undo step.

        Returns True if merged (command was absorbed, not pushed separately).
        """
        if not self._undo_stack:
            return False

        top = self._undo_stack[-1]
        merge_id = command.merge_id

        if merge_id is None or top.merge_id != merge_id:
            return False

        # Check time window
        elapsed = command.timestamp - top.timestamp
        if elapsed > self.MERGE_WINDOW:
            return False

        # Attempt merge
        if top.merge_with(command):
            logger.debug("Merged command into: %s", top)
            return True

        return False

    def _emit_state(self) -> None:
        """Emit state_changed signal for UI updates."""
        try:
            self.state_changed.emit(
                self.can_undo, self.can_redo,
                self.undo_text, self.redo_text,
            )
        except Exception:
            pass  # signal may not be connected yet
