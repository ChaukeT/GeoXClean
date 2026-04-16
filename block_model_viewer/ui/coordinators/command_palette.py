"""
Command Palette — fuzzy search across all 168+ menu actions.

Opened via Ctrl+Shift+P (Tools > Command Palette).
Matches VS Code / Figma / Blender convention.

Features:
  - Fuzzy substring matching on action name, menu path, and status tip
  - Keyboard-driven: type to filter, Up/Down to navigate, Enter to execute
  - Shows shortcut badge next to each result
  - Highlights matching characters
  - Remembers recent commands (session only, not persisted)

Usage:
    palette = CommandPalette(main_window)
    palette.show()  # or main_window._search_modules()
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QHBoxLayout, QLabel, QWidget, QMenu, QMenuBar,
)

from ..modern_styles import ModernColors

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class ActionEntry:
    """One searchable action."""
    __slots__ = ('action', 'label', 'menu_path', 'shortcut', 'status_tip',
                 'search_text')

    def __init__(self, action: QAction, menu_path: str):
        self.action = action
        self.label = action.text().replace("&", "")
        self.menu_path = menu_path
        shortcut = action.shortcut()
        self.shortcut = shortcut.toString(
            QKeySequence.SequenceFormat.NativeText
        ) if not shortcut.isEmpty() else ""
        self.status_tip = action.statusTip() or ""

        # Pre-compute lowercase search text
        self.search_text = (
            f"{self.label} {self.menu_path} {self.status_tip}"
        ).lower()


class CommandPalette(QDialog):
    """
    Modal fuzzy-search dialog for all application commands.

    The palette walks the menu bar on first show, catalogs every
    non-separator leaf QAction, and enables instant search.
    """

    MAX_RESULTS = 25
    RECENT_LIMIT = 5

    def __init__(self, main_window: 'MainWindow'):
        super().__init__(main_window)
        self._main_window = main_window
        self._all_entries: List[ActionEntry] = []
        self._recent: List[ActionEntry] = []
        self._cataloged = False

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Command Palette")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup
        )
        self.setMinimumWidth(550)
        self.setMaximumHeight(450)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Search input
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Type a command name...")
        self._search_input.setClearButtonEnabled(True)
        font = self._search_input.font()
        font.setPointSize(font.pointSize() + 2)
        self._search_input.setFont(font)
        self._search_input.setStyleSheet(
            f"QLineEdit {{ padding: 8px; border: 1px solid {ModernColors.BORDER}; "
            f"border-radius: 4px; }}"
        )
        self._search_input.textChanged.connect(self._on_text_changed)
        self._search_input.returnPressed.connect(self._on_execute)
        layout.addWidget(self._search_input)

        # Results list
        self._results_list = QListWidget()
        self._results_list.setSpacing(2)
        self._results_list.setStyleSheet(
            "QListWidget { border: none; }"
            "QListWidget::item { padding: 6px 8px; border-radius: 3px; }"
            f"QListWidget::item:selected {{ background: {ModernColors.ACCENT_PRIMARY}; color: white; }}"
        )
        self._results_list.itemActivated.connect(self._on_item_activated)
        layout.addWidget(self._results_list)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 11px;")
        layout.addWidget(self._status_label)

    # ── Public API ───────────────────────────────────────────────

    def show(self):
        """Show the palette, catalog actions if needed."""
        if not self._cataloged:
            self._catalog_actions()

        self._search_input.clear()
        self._show_recent_or_all()

        # Position: centered near top of main window
        mw = self._main_window
        geo = mw.geometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + 80
        self.move(x, y)

        super().show()
        self._search_input.setFocus()

    # ── Event handling ───────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_Down:
            row = self._results_list.currentRow()
            if row < self._results_list.count() - 1:
                self._results_list.setCurrentRow(row + 1)
        elif key == Qt.Key.Key_Up:
            row = self._results_list.currentRow()
            if row > 0:
                self._results_list.setCurrentRow(row - 1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_execute()
        else:
            super().keyPressEvent(event)

    # ── Catalog ──────────────────────────────────────────────────

    def _catalog_actions(self):
        """Walk all menus and build the searchable action list."""
        self._all_entries.clear()

        menubar = self._main_window.menuBar()
        if not menubar:
            return

        for menu_action in menubar.actions():
            menu = menu_action.menu()
            if menu:
                menu_name = menu_action.text().replace("&", "")
                self._walk_menu(menu, menu_name)

        self._cataloged = True
        logger.info("CommandPalette: cataloged %d actions", len(self._all_entries))

    def _walk_menu(self, menu: QMenu, parent_path: str):
        """Recursively walk a QMenu tree."""
        for action in menu.actions():
            if action.isSeparator():
                continue

            submenu = action.menu()
            if submenu:
                child_path = f"{parent_path} > {action.text().replace('&', '')}"
                self._walk_menu(submenu, child_path)
                continue

            # Leaf action
            label = action.text().replace("&", "").strip()
            if not label:
                continue

            entry = ActionEntry(action, parent_path)
            self._all_entries.append(entry)

    # ── Search ───────────────────────────────────────────────────

    def _on_text_changed(self, text: str):
        if not text.strip():
            self._show_recent_or_all()
            return

        query = text.lower().strip()
        scored = []

        for entry in self._all_entries:
            if not entry.action.isEnabled():
                continue
            score = self._score_match(query, entry)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        self._populate_results([e for _, e in scored[:self.MAX_RESULTS]])
        self._status_label.setText(
            f"{min(len(scored), self.MAX_RESULTS)} of {len(scored)} matches"
        )

    def _score_match(self, query: str, entry: ActionEntry) -> int:
        """
        Score how well query matches entry. Higher = better.
        Returns 0 for no match.
        """
        search = entry.search_text

        # Exact label match (highest priority)
        label_lower = entry.label.lower()
        if query == label_lower:
            return 1000

        # Label starts with query
        if label_lower.startswith(query):
            return 800

        # Query words all present in label
        words = query.split()
        if all(w in label_lower for w in words):
            return 600

        # All query words present in full search text
        if all(w in search for w in words):
            return 400

        # Substring in label
        if query in label_lower:
            return 300

        # Substring in search text
        if query in search:
            return 200

        # Fuzzy: all chars of query appear in order in label
        if self._fuzzy_match(query, label_lower):
            return 100

        return 0

    @staticmethod
    def _fuzzy_match(query: str, text: str) -> bool:
        """Check if all characters of query appear in text in order."""
        qi = 0
        for ch in text:
            if qi < len(query) and ch == query[qi]:
                qi += 1
        return qi == len(query)

    # ── Display ──────────────────────────────────────────────────

    def _show_recent_or_all(self):
        """Show recent commands, or first N if no recent."""
        if self._recent:
            self._populate_results(self._recent[:self.RECENT_LIMIT])
            self._status_label.setText("Recent commands")
        else:
            enabled = [e for e in self._all_entries if e.action.isEnabled()]
            self._populate_results(enabled[:self.MAX_RESULTS])
            self._status_label.setText(
                f"{len(self._all_entries)} commands available — type to search"
            )

    def _populate_results(self, entries: List[ActionEntry]):
        """Fill the results list widget."""
        self._results_list.clear()

        for entry in entries:
            item = QListWidgetItem()
            widget = self._create_result_widget(entry)
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, id(entry))
            self._results_list.addItem(item)
            self._results_list.setItemWidget(item, widget)

        if self._results_list.count() > 0:
            self._results_list.setCurrentRow(0)

    def _create_result_widget(self, entry: ActionEntry) -> QWidget:
        """Create a single result row widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # Command name (bold)
        name_label = QLabel(entry.label)
        font = name_label.font()
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)

        # Menu path (dimmed)
        path_label = QLabel(entry.menu_path)
        path_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT};")
        layout.addWidget(path_label)

        layout.addStretch()

        # Shortcut badge
        if entry.shortcut:
            shortcut_label = QLabel(entry.shortcut)
            shortcut_label.setStyleSheet(
                f"background: {ModernColors.ELEVATED_BG}; color: {ModernColors.TEXT_SECONDARY}; padding: 2px 6px; "
                f"border-radius: 3px; font-size: 11px; font-family: monospace;"
            )
            layout.addWidget(shortcut_label)

        return widget

    # ── Execute ──────────────────────────────────────────────────

    def _on_execute(self):
        """Execute the currently selected command."""
        item = self._results_list.currentItem()
        if not item:
            return

        entry_id = item.data(Qt.ItemDataRole.UserRole)
        entry = self._find_entry_by_id(entry_id)
        if not entry:
            return

        # Add to recent (deduplicate)
        self._recent = [e for e in self._recent if id(e) != id(entry)]
        self._recent.insert(0, entry)
        self._recent = self._recent[:self.RECENT_LIMIT]

        self.close()

        # Trigger the action after dialog closes
        QTimer.singleShot(50, entry.action.trigger)

    def _on_item_activated(self, item: QListWidgetItem):
        self._on_execute()

    def _find_entry_by_id(self, entry_id: int) -> Optional[ActionEntry]:
        """Find entry by Python id."""
        for entry in self._all_entries:
            if id(entry) == entry_id:
                return entry
        return None

    # ── Re-catalog on menu changes ───────────────────────────────

    def invalidate(self):
        """Force re-catalog on next show (call after menu rebuild)."""
        self._cataloged = False
