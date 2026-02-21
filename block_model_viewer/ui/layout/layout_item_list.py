"""
Layout Item List Widget for GeoX Layout Composer.

Provides a list view of layout items with visibility toggles,
lock controls, and drag-to-reorder functionality.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QAbstractItemView, QMenu
)

from ...layout.layout_document import LayoutDocument, LayoutItem

logger = logging.getLogger(__name__)


class LayoutItemList(QWidget):
    """
    List widget for managing layout items.

    Features:
    - Item visibility toggle
    - Item lock toggle
    - Drag to reorder
    - Delete items
    - Item selection sync with canvas
    """

    # Signals
    item_selected = pyqtSignal(str)  # item_id
    visibility_changed = pyqtSignal(str, bool)  # item_id, visible
    lock_changed = pyqtSignal(str, bool)  # item_id, locked
    item_deleted = pyqtSignal(str)  # item_id
    order_changed = pyqtSignal(list)  # list of item_ids in new order

    def __init__(self, document: LayoutDocument, parent=None):
        super().__init__(parent)
        self._document = document
        self._updating = False

        self._setup_ui()
        self._connect_signals()
        self.refresh()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # List widget
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDragEnabled(True)
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Styling
        self._list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d30;
                color: #f0f0f0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #3c3c3c;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover {
                background-color: #3c3c3c;
            }
        """)

        layout.addWidget(self._list)

        # Button bar
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)

        self._move_up_btn = QPushButton("Up")
        self._move_up_btn.setToolTip("Move item up (higher z-order)")
        self._move_up_btn.setFixedWidth(50)
        button_layout.addWidget(self._move_up_btn)

        self._move_down_btn = QPushButton("Down")
        self._move_down_btn.setToolTip("Move item down (lower z-order)")
        self._move_down_btn.setFixedWidth(50)
        button_layout.addWidget(self._move_down_btn)

        button_layout.addStretch()

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Delete selected item")
        self._delete_btn.setFixedWidth(60)
        button_layout.addWidget(self._delete_btn)

        layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.model().rowsMoved.connect(self._on_rows_moved)

        self._move_up_btn.clicked.connect(self._move_up)
        self._move_down_btn.clicked.connect(self._move_down)
        self._delete_btn.clicked.connect(self._delete_selected)

    def set_document(self, document: LayoutDocument) -> None:
        """Set the document and refresh the list."""
        self._document = document
        self.refresh()

    def refresh(self) -> None:
        """Refresh the list from the document."""
        self._updating = True
        self._list.clear()

        # Sort items by z-order (highest first in list = front)
        sorted_items = sorted(
            self._document.items,
            key=lambda x: x.z_order,
            reverse=True
        )

        for item in sorted_items:
            list_item = QListWidgetItem()
            list_item.setText(self._get_item_display_text(item))
            list_item.setData(Qt.ItemDataRole.UserRole, item.id)

            # Checkable for visibility
            list_item.setFlags(
                list_item.flags() |
                Qt.ItemFlag.ItemIsUserCheckable |
                Qt.ItemFlag.ItemIsDragEnabled
            )
            list_item.setCheckState(
                Qt.CheckState.Checked if item.visible else Qt.CheckState.Unchecked
            )

            # Icon based on type
            icon_text = self._get_type_icon(item.item_type)
            list_item.setToolTip(f"{item.item_type.title()}: {item.name}")

            self._list.addItem(list_item)

        self._updating = False

    def _get_item_display_text(self, item: LayoutItem) -> str:
        """Get display text for an item."""
        prefix = ""
        if item.locked:
            prefix = "[L] "
        return f"{prefix}{item.name or item.item_type.title()}"

    def _get_type_icon(self, item_type: str) -> str:
        """Get icon character for item type."""
        icons = {
            "viewport": "V",
            "legend": "L",
            "scale_bar": "S",
            "north_arrow": "N",
            "text": "T",
            "image": "I",
            "metadata": "M",
        }
        return icons.get(item_type, "?")

    def select_item(self, item_id: str) -> None:
        """Select an item by ID."""
        for i in range(self._list.count()):
            list_item = self._list.item(i)
            if list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                self._updating = True
                self._list.setCurrentItem(list_item)
                self._updating = False
                break

    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        if self._updating:
            return

        current = self._list.currentItem()
        if current:
            item_id = current.data(Qt.ItemDataRole.UserRole)
            self.item_selected.emit(item_id)

    def _on_item_changed(self, list_item: QListWidgetItem) -> None:
        """Handle item check state change (visibility toggle)."""
        if self._updating:
            return

        item_id = list_item.data(Qt.ItemDataRole.UserRole)
        visible = list_item.checkState() == Qt.CheckState.Checked
        self.visibility_changed.emit(item_id, visible)

    def _on_rows_moved(self, parent, start, end, destination, row) -> None:
        """Handle drag-drop reorder."""
        if self._updating:
            return

        # Get new order of item IDs
        new_order = []
        for i in range(self._list.count()):
            list_item = self._list.item(i)
            new_order.append(list_item.data(Qt.ItemDataRole.UserRole))

        # Update z-orders (first in list = highest z-order)
        for i, item_id in enumerate(new_order):
            item = self._document.get_item(item_id)
            if item:
                item.z_order = len(new_order) - i

        self.order_changed.emit(new_order)

    def _move_up(self) -> None:
        """Move selected item up (increase z-order)."""
        current = self._list.currentItem()
        if not current:
            return

        item_id = current.data(Qt.ItemDataRole.UserRole)
        self._document.move_item_to_front(item_id)
        self.refresh()
        self.select_item(item_id)
        self.order_changed.emit([])

    def _move_down(self) -> None:
        """Move selected item down (decrease z-order)."""
        current = self._list.currentItem()
        if not current:
            return

        item_id = current.data(Qt.ItemDataRole.UserRole)
        self._document.move_item_to_back(item_id)
        self.refresh()
        self.select_item(item_id)
        self.order_changed.emit([])

    def _delete_selected(self) -> None:
        """Delete the selected item."""
        current = self._list.currentItem()
        if not current:
            return

        item_id = current.data(Qt.ItemDataRole.UserRole)
        self.item_deleted.emit(item_id)
        self.refresh()

    def _show_context_menu(self, pos) -> None:
        """Show context menu for item."""
        item = self._list.itemAt(pos)
        if not item:
            return

        item_id = item.data(Qt.ItemDataRole.UserRole)
        layout_item = self._document.get_item(item_id)
        if not layout_item:
            return

        menu = QMenu(self)

        # Visibility toggle
        vis_action = menu.addAction(
            "Hide" if layout_item.visible else "Show"
        )
        vis_action.triggered.connect(
            lambda: self.visibility_changed.emit(item_id, not layout_item.visible)
        )

        # Lock toggle
        lock_action = menu.addAction(
            "Unlock" if layout_item.locked else "Lock"
        )
        lock_action.triggered.connect(
            lambda: self._toggle_lock(item_id)
        )

        menu.addSeparator()

        # Move actions
        front_action = menu.addAction("Bring to Front")
        front_action.triggered.connect(lambda: self._bring_to_front(item_id))

        back_action = menu.addAction("Send to Back")
        back_action.triggered.connect(lambda: self._send_to_back(item_id))

        menu.addSeparator()

        # Delete
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self.item_deleted.emit(item_id))

        menu.exec(self._list.mapToGlobal(pos))

    def _toggle_lock(self, item_id: str) -> None:
        """Toggle item lock state."""
        item = self._document.get_item(item_id)
        if item:
            item.locked = not item.locked
            self.lock_changed.emit(item_id, item.locked)
            self.refresh()

    def _bring_to_front(self, item_id: str) -> None:
        """Bring item to front."""
        self._document.move_item_to_front(item_id)
        self.refresh()
        self.select_item(item_id)
        self.order_changed.emit([])

    def _send_to_back(self, item_id: str) -> None:
        """Send item to back."""
        self._document.move_item_to_back(item_id)
        self.refresh()
        self.select_item(item_id)
        self.order_changed.emit([])
