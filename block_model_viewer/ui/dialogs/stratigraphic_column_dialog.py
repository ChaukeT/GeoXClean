"""
Stratigraphic Column Dialog — vertical reorder with contact types.
===================================================================

Lets the user define the stratigraphic order (youngest on top) and
the type of contact between each pair of adjacent units.

Returns list of dicts:
  [{"name": "UNIT_A", "contact_below": "conformable"}, ...]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

_CONTACT_TYPES = ["conformable", "unconformity", "disconformity", "fault"]


class StratigraphicColumnDialog(QDialog):
    """Interactive stratigraphic column editor.

    Parameters
    ----------
    unit_names : list[str]
        Modelling unit names (from lithology grouping).
    existing_column : list[dict] | None
        Pre-existing column definition to edit.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(
        self,
        unit_names: List[str],
        existing_column: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Stratigraphic Column")
        self.setMinimumSize(500, 400)
        self.resize(550, 450)

        self._unit_names = list(unit_names)
        self._build_ui()

        if existing_column:
            self._load_column(existing_column)
        else:
            self._populate_from_names(self._unit_names)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        header = QLabel(
            "Order units from youngest (top) to oldest (bottom). "
            "Set the contact type between each pair."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        # Table: [Unit Name] [Contact Below]
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Unit (top → bottom)", "Contact Below"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents,
        )
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows,
        )
        self.table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection,
        )
        root.addWidget(self.table, stretch=1)

        # Reorder buttons
        btn_row = QHBoxLayout()

        btn_up = QPushButton("Move Up")
        btn_up.clicked.connect(self._move_up)
        btn_row.addWidget(btn_up)

        btn_down = QPushButton("Move Down")
        btn_down.clicked.connect(self._move_down)
        btn_row.addWidget(btn_down)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # Dialog buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_from_names(self, names: List[str]) -> None:
        self.table.setRowCount(len(names))
        for i, name in enumerate(names):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            self._set_contact_combo(i, "conformable")

    def _load_column(self, column: List[Dict[str, Any]]) -> None:
        self.table.setRowCount(len(column))
        for i, entry in enumerate(column):
            name = entry.get("name", "")
            contact = entry.get("contact_below", "conformable")
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            self._set_contact_combo(i, contact)

    def _set_contact_combo(self, row: int, value: str) -> None:
        combo = QComboBox()
        combo.addItems(_CONTACT_TYPES)
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        # Last row: no contact below (it's the oldest unit)
        if row == self.table.rowCount() - 1:
            combo.setEnabled(False)
            combo.setCurrentText("conformable")
        self.table.setCellWidget(row, 1, combo)

    # ------------------------------------------------------------------
    # Reorder
    # ------------------------------------------------------------------

    def _move_up(self) -> None:
        row = self.table.currentRow()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self.table.setCurrentCell(row - 1, 0)

    def _move_down(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self.table.setCurrentCell(row + 1, 0)

    def _swap_rows(self, a: int, b: int) -> None:
        name_a = self.table.item(a, 0).text()
        name_b = self.table.item(b, 0).text()

        combo_a = self.table.cellWidget(a, 1)
        combo_b = self.table.cellWidget(b, 1)
        contact_a = combo_a.currentText() if combo_a else "conformable"
        contact_b = combo_b.currentText() if combo_b else "conformable"

        # Swap names
        self.table.item(a, 0).setText(name_b)
        self.table.item(b, 0).setText(name_a)

        # Swap contacts
        self._set_contact_combo(a, contact_b)
        self._set_contact_combo(b, contact_a)

        # Update last-row disable state
        self._update_last_row_state()

    def _update_last_row_state(self) -> None:
        n = self.table.rowCount()
        for i in range(n):
            combo = self.table.cellWidget(i, 1)
            if combo:
                combo.setEnabled(i < n - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_column(self) -> List[Dict[str, str]]:
        """Return the stratigraphic column as a list of dicts."""
        result = []
        for i in range(self.table.rowCount()):
            name_item = self.table.item(i, 0)
            combo = self.table.cellWidget(i, 1)
            result.append({
                "name": name_item.text() if name_item else "",
                "contact_below": combo.currentText() if combo else "conformable",
            })
        return result

    def get_unit_order(self) -> List[str]:
        """Return just the unit names in stratigraphic order."""
        return [
            self.table.item(i, 0).text()
            for i in range(self.table.rowCount())
            if self.table.item(i, 0)
        ]
