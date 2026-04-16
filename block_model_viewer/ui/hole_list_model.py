"""
Hole List Model — Scalable drillhole selection.

Replaces the 100-checkbox cap with a proper Qt Model/View that handles
50,000+ holes without UI lag.

Usage:
    from .hole_list_model import HoleListView

    # In your panel setup_ui():
    self.hole_list = HoleListView()
    self.hole_list.holeToggled.connect(self._on_hole_toggled)
    layout.addWidget(self.hole_list)

    # When data loads:
    self.hole_list.set_holes(["BH001", "BH002", ...])  # any size list

    # Query checked holes:
    visible = self.hole_list.get_checked_holes()   # returns set[str]
    count = self.hole_list.checked_count()          # fast O(1)
"""

from __future__ import annotations

from typing import List, Optional, Set
import logging

from PyQt6.QtCore import (
    Qt, QAbstractListModel, QModelIndex, QSortFilterProxyModel,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListView, QFrame,
)

from .panel_toolkit import action_button, hint_label, SECTION_SPACING

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# MODEL — stores hole IDs + check state
# ═══════════════════════════════════════════════════════════════════

class HoleListModel(QAbstractListModel):
    """
    List model for drillhole IDs with checkable items.

    Stores hole IDs and their checked state. Supports bulk operations
    (check all / uncheck all) without per-item signal overhead.
    """

    # Emitted when a single hole's check state changes: (hole_id, checked)
    holeToggled = pyqtSignal(str, bool)

    # Emitted after bulk operations (select all / deselect all)
    bulkToggled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._holes: List[str] = []
        self._checked: Set[str] = set()

    # ── Qt Model Interface ──────────────────────────────────────

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._holes)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._holes):
            return None

        hole_id = self._holes[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return hole_id
        elif role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if hole_id in self._checked else Qt.CheckState.Unchecked
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.CheckStateRole:
            return False

        hole_id = self._holes[index.row()]
        checked = value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked

        if checked:
            self._checked.add(hole_id)
        else:
            self._checked.discard(hole_id)

        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        self.holeToggled.emit(hole_id, checked)
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        default = super().flags(index)
        if index.isValid():
            return default | Qt.ItemFlag.ItemIsUserCheckable
        return default

    # ── Public API ──────────────────────────────────────────────

    def set_holes(self, hole_ids: List[str], all_checked: bool = True) -> None:
        """
        Replace the hole list. Resets check state.

        Args:
            hole_ids: List of hole ID strings.
            all_checked: If True, all holes start checked.
        """
        self.beginResetModel()
        self._holes = list(hole_ids)
        self._checked = set(hole_ids) if all_checked else set()
        self.endResetModel()

    def check_all(self, visible_only: List[str] = None) -> None:
        """
        Check all holes (or only the given subset).

        Args:
            visible_only: If provided, only check these IDs.
        """
        if visible_only is not None:
            self._checked.update(visible_only)
        else:
            self._checked = set(self._holes)

        self._emit_full_refresh()
        self.bulkToggled.emit()

    def uncheck_all(self, visible_only: List[str] = None) -> None:
        """
        Uncheck all holes (or only the given subset).

        Args:
            visible_only: If provided, only uncheck these IDs.
        """
        if visible_only is not None:
            self._checked -= set(visible_only)
        else:
            self._checked.clear()

        self._emit_full_refresh()
        self.bulkToggled.emit()

    def get_checked(self) -> Set[str]:
        """Return the set of checked hole IDs."""
        return self._checked.copy()

    def checked_count(self) -> int:
        """Return number of checked holes (O(1))."""
        return len(self._checked)

    def total_count(self) -> int:
        """Return total number of holes."""
        return len(self._holes)

    def _emit_full_refresh(self) -> None:
        """Notify views that all check states may have changed."""
        if self._holes:
            self.dataChanged.emit(
                self.index(0),
                self.index(len(self._holes) - 1),
                [Qt.ItemDataRole.CheckStateRole],
            )


# ═══════════════════════════════════════════════════════════════════
# FILTER PROXY — text search on hole IDs
# ═══════════════════════════════════════════════════════════════════

class HoleFilterProxy(QSortFilterProxyModel):
    """Case-insensitive substring filter for hole IDs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def get_visible_hole_ids(self) -> List[str]:
        """Return list of hole IDs currently passing the filter."""
        ids = []
        for row in range(self.rowCount()):
            index = self.index(row, 0)
            ids.append(index.data(Qt.ItemDataRole.DisplayRole))
        return ids


# ═══════════════════════════════════════════════════════════════════
# WIDGET — Complete hole list with search + select/deselect buttons
# ═══════════════════════════════════════════════════════════════════

class HoleListView(QWidget):
    """
    Complete hole selection widget.

    Contains:
    - Search bar (filters in real-time)
    - Select All / Deselect All buttons (operate on filtered results)
    - QListView with checkable items (handles 50k+ holes)
    - Summary label showing "X / Y selected"

    Signals:
        holeToggled(str, bool): Single hole check changed.
        selectionChanged(): Any change to selection (single or bulk).
    """

    holeToggled = pyqtSignal(str, bool)
    selectionChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Model
        self._model = HoleListModel()
        self._proxy = HoleFilterProxy()
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterRole(Qt.ItemDataRole.DisplayRole)

        # Forward signals
        self._model.holeToggled.connect(self._on_hole_toggled)
        self._model.bulkToggled.connect(self._on_bulk_toggled)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SECTION_SPACING)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter holes...")
        self.search_bar.setObjectName("panelCombo")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_bar)

        # Select / Deselect buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_select_all = action_button("Select All", "secondary")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_deselect_all = action_button("Deselect All", "secondary")
        self.btn_deselect_all.clicked.connect(self._on_deselect_all)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_deselect_all)
        layout.addLayout(btn_row)

        # List view
        self.list_view = QListView()
        self.list_view.setModel(self._proxy)
        self.list_view.setMinimumHeight(120)
        self.list_view.setMaximumHeight(250)
        self.list_view.setFrameShape(QFrame.Shape.NoFrame)
        self.list_view.setUniformItemSizes(True)  # Performance: all rows same height
        layout.addWidget(self.list_view)

        # Summary label
        self.summary_label = hint_label("No holes loaded")
        layout.addWidget(self.summary_label)

    # ── Event Handlers ──────────────────────────────────────────

    def _on_search_changed(self, text: str):
        self._proxy.setFilterFixedString(text)

    def _on_select_all(self):
        """Select all currently visible (filtered) holes."""
        visible = self._proxy.get_visible_hole_ids()
        self._model.check_all(visible_only=visible)
        self._update_summary()

    def _on_deselect_all(self):
        """Deselect all currently visible (filtered) holes."""
        visible = self._proxy.get_visible_hole_ids()
        self._model.uncheck_all(visible_only=visible)
        self._update_summary()

    def _on_hole_toggled(self, hole_id: str, checked: bool):
        self.holeToggled.emit(hole_id, checked)
        self._update_summary()
        self.selectionChanged.emit()

    def _on_bulk_toggled(self):
        self._update_summary()
        self.selectionChanged.emit()

    def _update_summary(self):
        checked = self._model.checked_count()
        total = self._model.total_count()
        self.summary_label.setText(f"{checked} / {total} holes selected")

    # ── Public API ──────────────────────────────────────────────

    def set_holes(self, hole_ids: List[str], all_checked: bool = True) -> None:
        """
        Load hole IDs into the list.

        Args:
            hole_ids: List of hole ID strings (any length).
            all_checked: If True, all holes start checked.
        """
        self._model.set_holes(hole_ids, all_checked=all_checked)
        self._update_summary()
        logger.info(f"HoleListView loaded {len(hole_ids)} holes")

    def get_checked_holes(self) -> Set[str]:
        """Return the set of currently checked hole IDs."""
        return self._model.get_checked()

    def checked_count(self) -> int:
        """Return number of checked holes."""
        return self._model.checked_count()

    def total_count(self) -> int:
        """Return total number of holes."""
        return self._model.total_count()

    def clear(self) -> None:
        """Clear all holes."""
        self._model.set_holes([])
        self._update_summary()
