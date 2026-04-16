"""
Shortcut Registry — validates all keyboard shortcuts at startup.

Responsibilities:
  1. Walk every QAction in the menu bar and catalog its shortcut
  2. Detect collisions (same key sequence bound to multiple actions)
  3. Log warnings for collisions and missing shortcuts
  4. Provide runtime lookup: shortcut → action, action → shortcut
  5. Generate the keyboard shortcuts reference dialog

Usage:
    registry = ShortcutRegistry(main_window)
    registry.validate()          # logs warnings, returns collision list
    registry.get_action("Ctrl+D")  # returns QAction or None
    registry.get_shortcut(action)  # returns key string or None
    registry.all_shortcuts()       # returns sorted list of (key, menu, label)
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu, QMenuBar

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class ShortcutEntry:
    """One registered shortcut."""
    __slots__ = ('key_string', 'action', 'menu_path', 'label', 'status_tip')

    def __init__(self, key_string: str, action: QAction, menu_path: str,
                 label: str, status_tip: str):
        self.key_string = key_string
        self.action = action
        self.menu_path = menu_path
        self.label = label
        self.status_tip = status_tip

    def __repr__(self):
        return f"<Shortcut {self.key_string!r} → {self.menu_path} > {self.label}>"


class ShortcutRegistry:
    """
    Central registry for all keyboard shortcuts in the application.

    Walks the menu bar on validate(), catalogs every QAction with a
    non-empty shortcut, and reports collisions.
    """

    def __init__(self, main_window: 'MainWindow'):
        self._main_window = main_window
        self._entries: List[ShortcutEntry] = []
        self._by_key: Dict[str, List[ShortcutEntry]] = {}
        self._by_action: Dict[int, ShortcutEntry] = {}
        self._collisions: List[Tuple[str, List[ShortcutEntry]]] = []

    # ── Public API ───────────────────────────────────────────────

    def validate(self) -> List[Tuple[str, List[ShortcutEntry]]]:
        """
        Walk all menus, catalog shortcuts, detect collisions.

        Returns list of (key_string, [entries]) for each collision.
        Also logs warnings.
        """
        self._entries.clear()
        self._by_key.clear()
        self._by_action.clear()
        self._collisions.clear()

        menubar = self._main_window.menuBar()
        if not menubar:
            logger.warning("ShortcutRegistry: no menu bar found")
            return []

        # Walk every menu
        for menu_action in menubar.actions():
            menu = menu_action.menu()
            if menu:
                self._walk_menu(menu, menu_action.text().replace("&", ""))

        # Detect collisions
        for key, entries in self._by_key.items():
            if len(entries) > 1:
                self._collisions.append((key, entries))
                paths = [f"{e.menu_path} > {e.label}" for e in entries]
                logger.warning(
                    "Shortcut collision: %s bound to %d actions: %s",
                    key, len(entries), " | ".join(paths)
                )

        # Summary
        total = len(self._entries)
        collisions = len(self._collisions)
        if collisions:
            logger.warning(
                "ShortcutRegistry: %d shortcuts, %d COLLISIONS", total, collisions
            )
        else:
            logger.info("ShortcutRegistry: %d shortcuts, 0 collisions ✓", total)

        return self._collisions

    def get_action(self, key_string: str) -> Optional[QAction]:
        """Look up QAction by shortcut string (e.g. 'Ctrl+D')."""
        entries = self._by_key.get(key_string)
        if entries:
            return entries[0].action
        return None

    def get_shortcut(self, action: QAction) -> Optional[str]:
        """Look up shortcut string for a QAction."""
        entry = self._by_action.get(id(action))
        if entry:
            return entry.key_string
        return None

    def get_entry(self, key_string: str) -> Optional[ShortcutEntry]:
        """Get full entry by shortcut string."""
        entries = self._by_key.get(key_string)
        if entries:
            return entries[0]
        return None

    def all_shortcuts(self) -> List[ShortcutEntry]:
        """Return all registered shortcuts sorted by key."""
        return sorted(self._entries, key=lambda e: e.key_string)

    def all_actions(self) -> List[ShortcutEntry]:
        """Return all registered entries sorted by menu path."""
        return sorted(self._entries, key=lambda e: (e.menu_path, e.label))

    @property
    def collisions(self) -> List[Tuple[str, List[ShortcutEntry]]]:
        """Collision list from last validate() call."""
        return list(self._collisions)

    @property
    def count(self) -> int:
        return len(self._entries)

    def format_reference(self) -> str:
        """Generate a human-readable shortcut reference string."""
        if not self._entries:
            return "No shortcuts registered. Run validate() first."

        lines = []
        current_menu = ""
        for entry in sorted(self._entries, key=lambda e: (e.menu_path, e.key_string)):
            if entry.menu_path != current_menu:
                if current_menu:
                    lines.append("")
                current_menu = entry.menu_path
                lines.append(f"─── {current_menu} ───")

            key_display = entry.key_string.ljust(20)
            lines.append(f"  {key_display} {entry.label}")

        header = f"GeoX Keyboard Shortcuts ({self.count} total)\n{'═' * 45}\n"
        return header + "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────

    def _walk_menu(self, menu: QMenu, parent_path: str):
        """Recursively walk a QMenu and register all actions."""
        for action in menu.actions():
            if action.isSeparator():
                continue

            submenu = action.menu()
            if submenu:
                child_path = f"{parent_path} > {action.text().replace('&', '')}"
                self._walk_menu(submenu, child_path)
                continue

            # Leaf action — check for shortcut
            shortcut = action.shortcut()
            if shortcut.isEmpty():
                continue

            key_string = shortcut.toString(QKeySequence.SequenceFormat.NativeText)
            if not key_string:
                continue

            label = action.text().replace("&", "")
            status_tip = action.statusTip() or ""

            entry = ShortcutEntry(
                key_string=key_string,
                action=action,
                menu_path=parent_path,
                label=label,
                status_tip=status_tip,
            )

            self._entries.append(entry)
            self._by_key.setdefault(key_string, []).append(entry)
            self._by_action[id(action)] = entry
