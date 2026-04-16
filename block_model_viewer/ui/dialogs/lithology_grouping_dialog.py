"""
Grouped Category Column — Leapfrog-style lithology grouping dialog.
====================================================================

Matches the Leapfrog Geo "Group Lithologies" workflow:

  ┌──────────────────────────┬──────────────────────────┐
  │  Ungrouped Categories    │  Groups                  │
  │  ⊙  Code     Colour     │  ⊘  Code     Colour     │
  │  ⊙ Alluvium   ■          │  ▸ Sandstone             │
  │  ⊙ Aluvium    ■          │     SS_FINE   ■          │
  │  ⊙ Gravel     ■          │     SS_COARSE ■          │
  │  ...                     │                    ∧  ∨  │
  ├──────────────────────────┴──────────────────────────┤
  │  [Auto Group ▾]  [New Group]              🗑         │
  │                             [Cancel]  [OK]          │
  └─────────────────────────────────────────────────────┘

Workflow:
  1. Select lithologies on the left
  2. Click "New Group" → group created from first selected lithology,
     all selected lithologies become members
  3. Drag additional lithologies onto a group (or double-click)
  4. "Auto Group" menu: One group per value / First N letters / Last N letters
  5. Double-click grouped code → returns to ungrouped
  6. 🗑 deletes selected group → codes return to ungrouped

Returns dict: {group_name: [code1, code2, ...]}
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QMimeData, QSize, Qt
from PyQt6.QtGui import QColor, QDrag, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QMenu, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget, QWidgetAction,
)

logger = logging.getLogger(__name__)

# ── Deterministic colour palette for lithology codes ──────────────

_PALETTE = [
    "#4CAF50", "#FFC107", "#2196F3", "#E91E63", "#9C27B0",
    "#FF5722", "#00BCD4", "#8BC34A", "#FF9800", "#607D8B",
    "#3F51B5", "#CDDC39", "#795548", "#009688", "#F44336",
    "#673AB7", "#03A9F4", "#FFEB3B", "#9E9E9E", "#00E676",
    "#FF6D00", "#AA00FF", "#0091EA", "#64DD17", "#D50000",
    "#304FFE", "#C6FF00", "#6D4C41", "#00BFA5", "#DD2C00",
]


def _color_for_code(code: str) -> QColor:
    """Stable colour per lithology code (deterministic hash)."""
    idx = int(hashlib.md5(code.encode()).hexdigest(), 16) % len(_PALETTE)
    return QColor(_PALETTE[idx])


def _color_icon(color: QColor, size: int = 16) -> QIcon:
    """Small square colour swatch icon."""
    px = QPixmap(size, size)
    px.fill(color)
    return QIcon(px)


_ROLE_CODE = Qt.ItemDataRole.UserRole
_ROLE_IS_GROUP = Qt.ItemDataRole.UserRole + 1


# ══════════════════════════════════════════════════════════════════
# Drop-target tree for the Groups panel
# ══════════════════════════════════════════════════════════════════

class _GroupTree(QTreeWidget):
    """QTreeWidget that accepts drops from the ungrouped tree."""

    def __init__(self, dialog: "LithologyGroupingDialog", parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    # ── Drop handling ────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.source() is self._dialog.ungrouped_tree:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.source() is self._dialog.ungrouped_tree:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.source() is not self._dialog.ungrouped_tree:
            event.ignore()
            return

        # Find which group the drop landed on
        target_item = self.itemAt(event.position().toPoint())
        if target_item is None:
            event.ignore()
            return

        # Navigate to group header
        while target_item.parent() is not None:
            target_item = target_item.parent()
        if not target_item.data(0, _ROLE_IS_GROUP):
            event.ignore()
            return

        # Move selected ungrouped codes into this group
        self._dialog._assign_to_group(target_item)
        event.acceptProposedAction()


# ══════════════════════════════════════════════════════════════════
# Drag-source tree for the Ungrouped panel
# ══════════════════════════════════════════════════════════════════

class _UngroupedTree(QTreeWidget):
    """QTreeWidget that supports dragging items to the group tree."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)

    def startDrag(self, supportedActions):
        drag = QDrag(self)
        mime = QMimeData()
        codes = []
        for item in self.selectedItems():
            code = item.data(0, _ROLE_CODE)
            if code:
                codes.append(code)
        mime.setText(",".join(codes))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)


# ══════════════════════════════════════════════════════════════════
# Main Dialog
# ══════════════════════════════════════════════════════════════════

class LithologyGroupingDialog(QDialog):
    """Leapfrog-style grouped category dialog.

    Parameters
    ----------
    raw_codes : list[str]
        All unique lithology codes from drillhole data.
    existing_grouping : dict | None
        Pre-existing grouping to edit ({group_name: [codes]}).
    parent : QWidget | None
        Parent widget.
    """

    def __init__(
        self,
        raw_codes: List[str],
        existing_grouping: Optional[Dict[str, List[str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Grouped Category Column - Lithology Grouping")
        self.setMinimumSize(680, 480)
        self.resize(720, 520)

        self._raw_codes = sorted(set(str(c) for c in raw_codes if str(c).strip()))
        self._code_colors: Dict[str, QColor] = {
            c: _color_for_code(c) for c in self._raw_codes
        }
        self._grouping: Dict[str, List[str]] = {}

        self._build_ui()

        if existing_grouping and not self._is_identity_grouping(existing_grouping):
            self._load_grouping(existing_grouping)
        else:
            self._populate_ungrouped(self._raw_codes)

    @staticmethod
    def _is_identity_grouping(grouping: Dict[str, List[str]]) -> bool:
        """True if every group has exactly 1 code matching its name."""
        if not grouping:
            return True
        return all(
            len(codes) == 1 and codes[0] == name
            for name, codes in grouping.items()
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Two-panel splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel — Ungrouped Categories
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        lbl_left = QLabel("Ungrouped Categories")
        lbl_left.setStyleSheet("font-weight: bold; color: #1976D2;")
        left_lay.addWidget(lbl_left)

        self.ungrouped_tree = _UngroupedTree()
        self.ungrouped_tree.setHeaderLabels(["", "Code", "Colour"])
        self.ungrouped_tree.setRootIsDecorated(False)
        self.ungrouped_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self.ungrouped_tree.setColumnWidth(0, 28)
        self.ungrouped_tree.setColumnWidth(2, 50)
        self.ungrouped_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch,
        )
        self.ungrouped_tree.setIconSize(QSize(16, 16))
        left_lay.addWidget(self.ungrouped_tree)
        splitter.addWidget(left)

        # Right panel — Groups
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        # Header row with label + up/down arrows
        right_header = QHBoxLayout()
        lbl_right = QLabel("Groups")
        lbl_right.setStyleSheet("font-weight: bold; color: #1976D2;")
        right_header.addWidget(lbl_right)
        right_header.addStretch()

        btn_up = QPushButton("\u2227")  # ∧
        btn_up.setFixedSize(28, 28)
        btn_up.setToolTip("Move group up")
        btn_up.clicked.connect(self._on_move_group_up)
        right_header.addWidget(btn_up)

        btn_down = QPushButton("\u2228")  # ∨
        btn_down.setFixedSize(28, 28)
        btn_down.setToolTip("Move group down")
        btn_down.clicked.connect(self._on_move_group_down)
        right_header.addWidget(btn_down)

        right_lay.addLayout(right_header)

        self.group_tree = _GroupTree(self)
        self.group_tree.setHeaderLabels(["", "Code", "Colour"])
        self.group_tree.setRootIsDecorated(True)
        self.group_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self.group_tree.setColumnWidth(0, 28)
        self.group_tree.setColumnWidth(2, 50)
        self.group_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch,
        )
        self.group_tree.setIconSize(QSize(16, 16))
        # Double-click child code → unassign; double-click group → rename
        self.group_tree.itemDoubleClicked.connect(self._on_group_double_click)
        right_lay.addWidget(self.group_tree)

        splitter.addWidget(right)
        splitter.setSizes([340, 340])
        root.addWidget(splitter, stretch=1)

        # ── Bottom action bar ─────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        # Auto Group — dropdown menu like Leapfrog
        self.btn_auto = QPushButton("Auto Group")
        auto_menu = QMenu(self)
        auto_menu.addAction(
            "One group per value", self._on_auto_one_per_value,
        )
        auto_menu.addSeparator()
        # First N letters action with inline spinbox
        self._first_n_spin = QSpinBox()
        self._first_n_spin.setRange(1, 20)
        self._first_n_spin.setValue(2)
        self._first_n_spin.setPrefix("First ")
        self._first_n_spin.setSuffix(" letter(s)")
        first_n_action = QWidgetAction(auto_menu)
        first_n_action.setDefaultWidget(self._first_n_spin)
        auto_menu.addAction(first_n_action)
        auto_menu.addAction(
            "Group by first letters", self._on_auto_first_letters,
        )
        auto_menu.addSeparator()
        # Last N letters action with inline spinbox
        self._last_n_spin = QSpinBox()
        self._last_n_spin.setRange(1, 20)
        self._last_n_spin.setValue(2)
        self._last_n_spin.setPrefix("Last ")
        self._last_n_spin.setSuffix(" letter(s)")
        last_n_action = QWidgetAction(auto_menu)
        last_n_action.setDefaultWidget(self._last_n_spin)
        auto_menu.addAction(last_n_action)
        auto_menu.addAction(
            "Group by last letters", self._on_auto_last_letters,
        )
        self.btn_auto.setMenu(auto_menu)
        action_row.addWidget(self.btn_auto)

        # New Group — creates group from selected ungrouped lithologies
        btn_new = QPushButton("New Group")
        btn_new.setToolTip(
            "Create a new group from the selected lithologies.\n"
            "The first selected lithology becomes the group name."
        )
        btn_new.clicked.connect(self._on_new_group)
        action_row.addWidget(btn_new)

        action_row.addStretch()

        # Remove from group — returns selected codes to ungrouped
        btn_remove = QPushButton("\u2190 Remove")
        btn_remove.setToolTip(
            "Remove selected lithologies from their group\n"
            "and return them to the ungrouped list"
        )
        btn_remove.clicked.connect(self._on_remove_from_group)
        action_row.addWidget(btn_remove)

        # Delete group
        btn_delete = QPushButton()
        btn_delete.setText("\U0001F5D1")  # 🗑
        btn_delete.setFixedSize(32, 32)
        btn_delete.setToolTip(
            "Delete selected group (all codes return to ungrouped)"
        )
        btn_delete.clicked.connect(self._on_delete_group)
        action_row.addWidget(btn_delete)

        root.addLayout(action_row)

        # ── Help text ─────────────────────────────────────────────
        help_lbl = QLabel(
            "Select lithologies, then click New Group. "
            "Drag lithologies onto a group to merge them. "
            "Select grouped lithologies and click \u2190 Remove to ungroup."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(help_lbl)

        # ── OK / Cancel ──────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok,
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------
    # Tree item helpers
    # ------------------------------------------------------------------

    def _make_code_item(self, code: str) -> QTreeWidgetItem:
        """Tree item for a lithology code with colour swatch."""
        color = self._code_colors.get(code, QColor("#808080"))
        item = QTreeWidgetItem()
        item.setIcon(0, _color_icon(color))
        item.setText(1, code)
        item.setIcon(2, _color_icon(color))
        item.setData(0, _ROLE_CODE, code)
        item.setData(0, _ROLE_IS_GROUP, False)
        return item

    def _make_group_item(self, name: str) -> QTreeWidgetItem:
        """Top-level group header (bold, no colour swatch)."""
        item = QTreeWidgetItem()
        item.setText(1, name)
        font = item.font(1)
        font.setBold(True)
        item.setFont(1, font)
        item.setData(0, _ROLE_CODE, name)
        item.setData(0, _ROLE_IS_GROUP, True)
        return item

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_ungrouped(self, codes: List[str]) -> None:
        self.ungrouped_tree.clear()
        for code in sorted(codes, key=str.lower):
            self.ungrouped_tree.addTopLevelItem(self._make_code_item(code))

    def _load_grouping(self, grouping: Dict[str, List[str]]) -> None:
        self._grouping = {}
        assigned: set = set()
        self.group_tree.clear()

        for group_name, codes in grouping.items():
            parent = self._make_group_item(group_name)
            self.group_tree.addTopLevelItem(parent)
            valid = []
            for code in codes:
                code = str(code).strip()
                if code and code in self._raw_codes:
                    parent.addChild(self._make_code_item(code))
                    assigned.add(code)
                    valid.append(code)
            self._grouping[group_name] = valid
            parent.setExpanded(True)

        ungrouped = [c for c in self._raw_codes if c not in assigned]
        self._populate_ungrouped(ungrouped)

    def _refresh_group_tree(self) -> None:
        self.group_tree.clear()
        for group_name, codes in self._grouping.items():
            parent = self._make_group_item(group_name)
            for code in sorted(codes, key=str.lower):
                parent.addChild(self._make_code_item(code))
            self.group_tree.addTopLevelItem(parent)
            parent.setExpanded(True)

    # ------------------------------------------------------------------
    # Core assign / unassign
    # ------------------------------------------------------------------

    def _assign_to_group(self, group_item: QTreeWidgetItem) -> None:
        """Move all selected ungrouped codes into the given group."""
        group_name = group_item.text(1)
        selected = self.ungrouped_tree.selectedItems()
        if not selected:
            return

        for item in selected:
            code = item.data(0, _ROLE_CODE)
            if not code:
                continue
            self._grouping.setdefault(group_name, []).append(code)
            group_item.addChild(self._make_code_item(code))
            idx = self.ungrouped_tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self.ungrouped_tree.takeTopLevelItem(idx)

        group_item.setExpanded(True)

    def _unassign_item(self, item: QTreeWidgetItem) -> None:
        """Move a grouped code back to ungrouped."""
        if item.parent() is None:
            return
        code = item.data(0, _ROLE_CODE)
        if not code:
            return
        group_item = item.parent()
        group_name = group_item.text(1)
        if group_name in self._grouping:
            try:
                self._grouping[group_name].remove(code)
            except ValueError:
                pass
        group_item.removeChild(item)
        self.ungrouped_tree.addTopLevelItem(self._make_code_item(code))
        self.ungrouped_tree.sortItems(1, Qt.SortOrder.AscendingOrder)

    def _on_remove_from_group(self) -> None:
        """Remove selected lithologies from their groups back to ungrouped.

        Works on both child codes (unassigns them) and group headers
        (dissolves the entire group).
        """
        selected = self.group_tree.selectedItems()
        if not selected:
            return

        # Separate into child codes and group headers
        children = []
        groups = []
        for item in selected:
            if item.parent() is not None:
                children.append(item)
            elif item.data(0, _ROLE_IS_GROUP):
                groups.append(item)

        # Unassign individual child codes first
        for item in children:
            self._unassign_item(item)

        # Then dissolve entire groups
        for item in groups:
            group_name = item.text(1)
            codes = self._grouping.pop(group_name, [])
            for code in codes:
                self.ungrouped_tree.addTopLevelItem(self._make_code_item(code))
            idx = self.group_tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self.group_tree.takeTopLevelItem(idx)

        self.ungrouped_tree.sortItems(1, Qt.SortOrder.AscendingOrder)

    # ------------------------------------------------------------------
    # New Group — Leapfrog-style: from selected lithologies
    # ------------------------------------------------------------------

    def _on_new_group(self) -> None:
        """Create a group from selected ungrouped lithologies.

        The first selected lithology becomes the group name.
        All selected lithologies become members of the group.
        If nothing is selected, prompt for a name and create empty.
        """
        selected = self.ungrouped_tree.selectedItems()

        if not selected:
            # Nothing selected — prompt for name (fallback)
            name, ok = QInputDialog.getText(
                self, "New Group", "Group name:")
            if not ok or not name.strip():
                return
            name = name.strip()
            if name in self._grouping:
                QMessageBox.warning(
                    self, "Duplicate", f"'{name}' already exists.")
                return
            self._grouping[name] = []
            parent = self._make_group_item(name)
            self.group_tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            self.group_tree.setCurrentItem(parent)
            return

        # Use first selected code as the group name
        first_code = selected[0].data(0, _ROLE_CODE)
        group_name = first_code or "New_Group"

        # Handle duplicate group name
        base_name = group_name
        counter = 2
        while group_name in self._grouping:
            group_name = f"{base_name}_{counter}"
            counter += 1

        # Create group and move all selected codes into it
        self._grouping[group_name] = []
        parent = self._make_group_item(group_name)
        self.group_tree.addTopLevelItem(parent)

        for item in selected:
            code = item.data(0, _ROLE_CODE)
            if not code:
                continue
            self._grouping[group_name].append(code)
            parent.addChild(self._make_code_item(code))
            idx = self.ungrouped_tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self.ungrouped_tree.takeTopLevelItem(idx)

        parent.setExpanded(True)
        self.group_tree.setCurrentItem(parent)

    # ------------------------------------------------------------------
    # Auto Group options
    # ------------------------------------------------------------------

    def _get_ungrouped_codes(self) -> List[str]:
        """Collect all codes currently in the ungrouped list."""
        codes = []
        for i in range(self.ungrouped_tree.topLevelItemCount()):
            code = self.ungrouped_tree.topLevelItem(i).data(0, _ROLE_CODE)
            if code:
                codes.append(code)
        return codes

    def _on_auto_one_per_value(self) -> None:
        """One group per ungrouped code (identity mapping)."""
        for code in self._get_ungrouped_codes():
            if code not in self._grouping:
                self._grouping[code] = [code]
        self.ungrouped_tree.clear()
        self._refresh_group_tree()

    def _on_auto_first_letters(self) -> None:
        """Group by the first N letters of each code name."""
        n = self._first_n_spin.value()
        self._auto_group_by_key(lambda c: c[:n].upper())

    def _on_auto_last_letters(self) -> None:
        """Group by the last N letters of each code name."""
        n = self._last_n_spin.value()
        self._auto_group_by_key(lambda c: c[-n:].upper() if len(c) >= n else c.upper())

    def _auto_group_by_key(self, key_fn) -> None:
        """Group ungrouped codes using a key function."""
        codes = self._get_ungrouped_codes()
        if not codes:
            return

        # Build groups by key
        groups: Dict[str, List[str]] = {}
        for code in codes:
            k = key_fn(code)
            groups.setdefault(k, []).append(code)

        # Merge into existing grouping
        for group_name, members in groups.items():
            if group_name in self._grouping:
                self._grouping[group_name].extend(members)
            else:
                self._grouping[group_name] = members

        self.ungrouped_tree.clear()
        self._refresh_group_tree()

    # ------------------------------------------------------------------
    # Delete group
    # ------------------------------------------------------------------

    def _on_delete_group(self) -> None:
        """Delete selected group(s) — codes return to ungrouped."""
        to_delete = []
        for item in self.group_tree.selectedItems():
            if item.parent() is None and item.data(0, _ROLE_IS_GROUP):
                to_delete.append(item)

        for item in to_delete:
            group_name = item.text(1)
            codes = self._grouping.pop(group_name, [])
            for code in codes:
                self.ungrouped_tree.addTopLevelItem(self._make_code_item(code))
            idx = self.group_tree.indexOfTopLevelItem(item)
            self.group_tree.takeTopLevelItem(idx)

        self.ungrouped_tree.sortItems(1, Qt.SortOrder.AscendingOrder)

    # ------------------------------------------------------------------
    # Group double-click → unassign child / rename group
    # ------------------------------------------------------------------

    def _on_group_double_click(self, item: QTreeWidgetItem, col: int) -> None:
        if item.parent() is not None:
            # Child code → unassign
            self._unassign_item(item)
        elif item.data(0, _ROLE_IS_GROUP):
            # Group header → rename
            self._rename_group(item)

    def _rename_group(self, item: QTreeWidgetItem) -> None:
        old_name = item.text(1)
        new_name, ok = QInputDialog.getText(
            self, "Rename Group", "New name:", text=old_name,
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name in self._grouping and new_name != old_name:
            QMessageBox.warning(self, "Duplicate", f"'{new_name}' already exists.")
            return
        codes = self._grouping.pop(old_name, [])
        self._grouping[new_name] = codes
        item.setText(1, new_name)
        item.setData(0, _ROLE_CODE, new_name)

    # ------------------------------------------------------------------
    # Move group up / down
    # ------------------------------------------------------------------

    def _on_move_group_up(self) -> None:
        item = self._get_selected_group()
        if item is None:
            return
        idx = self.group_tree.indexOfTopLevelItem(item)
        if idx <= 0:
            return
        self.group_tree.takeTopLevelItem(idx)
        self.group_tree.insertTopLevelItem(idx - 1, item)
        self.group_tree.setCurrentItem(item)
        self._sync_grouping_order()

    def _on_move_group_down(self) -> None:
        item = self._get_selected_group()
        if item is None:
            return
        idx = self.group_tree.indexOfTopLevelItem(item)
        if idx < 0 or idx >= self.group_tree.topLevelItemCount() - 1:
            return
        self.group_tree.takeTopLevelItem(idx)
        self.group_tree.insertTopLevelItem(idx + 1, item)
        self.group_tree.setCurrentItem(item)
        self._sync_grouping_order()

    def _get_selected_group(self) -> Optional[QTreeWidgetItem]:
        item = self.group_tree.currentItem()
        if item is None:
            return None
        while item.parent() is not None:
            item = item.parent()
        if item.data(0, _ROLE_IS_GROUP):
            return item
        return None

    def _sync_grouping_order(self) -> None:
        """Rebuild _grouping dict in visual order."""
        new_grouping = {}
        for i in range(self.group_tree.topLevelItemCount()):
            gi = self.group_tree.topLevelItem(i)
            name = gi.text(1)
            codes = []
            for j in range(gi.childCount()):
                code = gi.child(j).data(0, _ROLE_CODE)
                if code:
                    codes.append(code)
            new_grouping[name] = codes
        self._grouping = new_grouping

    # ------------------------------------------------------------------
    # Accept / Validate
    # ------------------------------------------------------------------

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        a, b = a.lower(), b.lower()
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                                prev[j - 1] + (ca != cb)))
            prev = curr
        return prev[-1]

    def _on_accept(self) -> None:
        ungrouped_count = self.ungrouped_tree.topLevelItemCount()
        if ungrouped_count > 0:
            reply = QMessageBox.question(
                self, "Ungrouped Lithologies",
                f"{ungrouped_count} lithologies are not assigned to any group.\n"
                "Continue anyway? (Ungrouped lithologies will be excluded "
                "from the new column.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self._sync_grouping_order()
        # Remove empty groups
        self._grouping = {k: v for k, v in self._grouping.items() if v}

        # ── Typo detection: warn on suspiciously similar group names ──
        import re as _re
        names = list(self._grouping.keys())
        suspicious: list[tuple[str, str]] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                dist = self._levenshtein(a, b)
                # Flag pairs that are very close but not identical
                threshold = max(2, min(len(a), len(b)) // 4)
                if 0 < dist <= threshold:
                    # Exclude intentional numbering/lettering sequences:
                    # "Group 1" vs "Group 2", "Unit A" vs "Unit B"
                    # Strip trailing digits, spaces, or single letters
                    base_a = _re.sub(r"[\s\d]+$|[\s][a-z]$", "", a.lower()).strip()
                    base_b = _re.sub(r"[\s\d]+$|[\s][a-z]$", "", b.lower()).strip()
                    if base_a and base_a == base_b:
                        continue
                    suspicious.append((a, b))

        if suspicious:
            pairs_text = "\n".join(f'  • "{a}"  ↔  "{b}"' for a, b in suspicious)
            reply = QMessageBox.warning(
                self, "Possible Typos in Group Names",
                "The following group names look suspiciously similar — possible typos:\n\n"
                f"{pairs_text}\n\n"
                "Click Cancel to go back and fix them, or OK to continue anyway.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return

        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_grouping(self) -> Dict[str, List[str]]:
        """Return the lithology grouping dict."""
        return dict(self._grouping)
