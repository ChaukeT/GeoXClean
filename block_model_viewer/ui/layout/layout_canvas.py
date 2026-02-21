"""
Layout Canvas for GeoX Layout Composer.

QGraphicsView-based canvas for visual layout editing with page display,
margin guides, grid overlay, and item manipulation support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QWheelEvent, QMouseEvent,
    QKeyEvent, QContextMenuEvent
)
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem,
    QMenu, QApplication
)

from ...layout.layout_document import LayoutDocument, LayoutItem

if TYPE_CHECKING:
    from .layout_item_widgets import BaseLayoutGraphicsItem

logger = logging.getLogger(__name__)


class LayoutCanvas(QGraphicsView):
    """
    Canvas for visual layout editing.

    Provides:
    - Page display with shadow and white background
    - Margin guides (dashed blue lines)
    - Optional grid overlay
    - Zoom and pan navigation
    - Item selection and manipulation
    - Snap-to-grid functionality
    """

    # Signals
    selection_changed = pyqtSignal(list)  # List of selected item IDs
    item_moved = pyqtSignal(str, float, float)  # id, x_mm, y_mm
    item_resized = pyqtSignal(str, float, float)  # id, width_mm, height_mm
    item_double_clicked = pyqtSignal(str)  # id
    canvas_context_menu = pyqtSignal(QPointF)  # scene position for context menu

    # Display constants
    SCREEN_DPI = 96.0
    MM_TO_INCH = 25.4

    def __init__(self, document: LayoutDocument, parent=None):
        super().__init__(parent)
        self._document = document
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Rendering settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Background
        self.setBackgroundBrush(QBrush(QColor(80, 80, 85)))

        # View state
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 4.0

        # Grid and snapping
        self._show_grid = True
        self._snap_enabled = True
        self._grid_spacing_mm = 5.0
        self._snap_threshold_mm = 2.5

        # Page graphics
        self._page_shadow: Optional[QGraphicsRectItem] = None
        self._page_rect: Optional[QGraphicsRectItem] = None
        self._margin_lines: List[QGraphicsLineItem] = []

        # Item graphics mapping
        self._item_graphics: Dict[str, "BaseLayoutGraphicsItem"] = {}

        # Setup page
        self._setup_page()

    @property
    def document(self) -> LayoutDocument:
        """Get the current document."""
        return self._document

    def set_document(self, document: LayoutDocument) -> None:
        """Set a new document and refresh the canvas."""
        self._document = document
        self._setup_page()

    def mm_to_px(self, mm: float) -> float:
        """Convert millimeters to display pixels."""
        return mm / self.MM_TO_INCH * self.SCREEN_DPI

    def px_to_mm(self, px: float) -> float:
        """Convert display pixels to millimeters."""
        return px * self.MM_TO_INCH / self.SCREEN_DPI

    def _setup_page(self) -> None:
        """Initialize page rectangle and margin guides."""
        self._scene.clear()
        self._item_graphics.clear()
        self._margin_lines.clear()

        page = self._document.page
        width_px = self.mm_to_px(page.width_mm)
        height_px = self.mm_to_px(page.height_mm)

        # Page shadow
        shadow_offset = 6
        self._page_shadow = QGraphicsRectItem(
            shadow_offset, shadow_offset, width_px, height_px
        )
        self._page_shadow.setBrush(QBrush(QColor(0, 0, 0, 80)))
        self._page_shadow.setPen(QPen(Qt.PenStyle.NoPen))
        self._page_shadow.setZValue(-2)
        self._scene.addItem(self._page_shadow)

        # Page background
        self._page_rect = QGraphicsRectItem(0, 0, width_px, height_px)
        self._page_rect.setBrush(QBrush(QColor(255, 255, 255)))
        self._page_rect.setPen(QPen(QColor(180, 180, 180), 1))
        self._page_rect.setZValue(-1)
        self._scene.addItem(self._page_rect)

        # Margin guides
        margin_pen = QPen(QColor(180, 180, 220), 1, Qt.PenStyle.DashLine)
        left = self.mm_to_px(page.margin_left_mm)
        right = width_px - self.mm_to_px(page.margin_right_mm)
        top = self.mm_to_px(page.margin_top_mm)
        bottom = height_px - self.mm_to_px(page.margin_bottom_mm)

        # Vertical margin lines
        left_line = self._scene.addLine(left, 0, left, height_px, margin_pen)
        right_line = self._scene.addLine(right, 0, right, height_px, margin_pen)
        # Horizontal margin lines
        top_line = self._scene.addLine(0, top, width_px, top, margin_pen)
        bottom_line = self._scene.addLine(0, bottom, width_px, bottom, margin_pen)

        self._margin_lines = [left_line, right_line, top_line, bottom_line]
        for line in self._margin_lines:
            line.setZValue(1000)  # Above items

        # Set scene rect with padding
        padding = 100
        self._scene.setSceneRect(
            -padding, -padding,
            width_px + 2 * padding, height_px + 2 * padding
        )

        # Fit page in view
        self.fit_page_in_view()

    def fit_page_in_view(self) -> None:
        """Zoom to fit the entire page in the view."""
        page = self._document.page
        page_rect = QRectF(
            -20, -20,
            self.mm_to_px(page.width_mm) + 40,
            self.mm_to_px(page.height_mm) + 40
        )
        self.fitInView(page_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_zoom_factor()

    def _update_zoom_factor(self) -> None:
        """Update the stored zoom factor from the current transform."""
        transform = self.transform()
        self._zoom_factor = transform.m11()

    def add_item_graphic(self, item: LayoutItem) -> "BaseLayoutGraphicsItem":
        """
        Add a graphics item for a layout item.

        Args:
            item: The layout item to visualize

        Returns:
            The created graphics item
        """
        from .layout_item_widgets import create_graphics_item

        graphic = create_graphics_item(item, self.mm_to_px)
        graphic.setZValue(item.z_order)
        self._scene.addItem(graphic)
        self._item_graphics[item.id] = graphic

        # Connect signals for item changes
        graphic.setFlag(
            QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True
        )

        return graphic

    def remove_item_graphic(self, item_id: str) -> None:
        """Remove the graphics item for a layout item."""
        if item_id in self._item_graphics:
            graphic = self._item_graphics.pop(item_id)
            self._scene.removeItem(graphic)

    def update_item_graphic(self, item: LayoutItem) -> None:
        """Update the graphics item for a layout item."""
        if item.id in self._item_graphics:
            graphic = self._item_graphics[item.id]
            # Update position and size
            x = self.mm_to_px(item.x_mm)
            y = self.mm_to_px(item.y_mm)
            w = self.mm_to_px(item.width_mm)
            h = self.mm_to_px(item.height_mm)
            graphic.setRect(0, 0, w, h)
            graphic.setPos(x, y)
            graphic.setZValue(item.z_order)
            graphic.setVisible(item.visible)
            graphic.update()

    def refresh_all_items(self) -> None:
        """Refresh all item graphics from the document."""
        # Remove existing items
        for item_id in list(self._item_graphics.keys()):
            self.remove_item_graphic(item_id)

        # Add items from document
        for item in self._document.items:
            self.add_item_graphic(item)

    def get_selected_item_ids(self) -> List[str]:
        """Get IDs of currently selected items."""
        selected = []
        for item_id, graphic in self._item_graphics.items():
            if graphic.isSelected():
                selected.append(item_id)
        return selected

    def select_item(self, item_id: str, clear_others: bool = True) -> None:
        """Select an item by ID."""
        if clear_others:
            self._scene.clearSelection()

        if item_id in self._item_graphics:
            self._item_graphics[item_id].setSelected(True)
            self.selection_changed.emit(self.get_selected_item_ids())

    def clear_selection(self) -> None:
        """Clear all selection."""
        self._scene.clearSelection()
        self.selection_changed.emit([])

    def snap_to_grid(self, pos_mm: QPointF) -> QPointF:
        """
        Snap a position to the grid if snapping is enabled.

        Args:
            pos_mm: Position in millimeters

        Returns:
            Snapped position in millimeters
        """
        if not self._snap_enabled:
            return pos_mm

        grid = self._grid_spacing_mm
        x = round(pos_mm.x() / grid) * grid
        y = round(pos_mm.y() / grid) * grid
        return QPointF(x, y)

    def set_grid_visible(self, visible: bool) -> None:
        """Show or hide the grid."""
        self._show_grid = visible
        self.viewport().update()

    def set_snap_enabled(self, enabled: bool) -> None:
        """Enable or disable snap to grid."""
        self._snap_enabled = enabled

    def set_grid_spacing(self, spacing_mm: float) -> None:
        """Set the grid spacing in millimeters."""
        self._grid_spacing_mm = max(1.0, spacing_mm)
        self.viewport().update()

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw canvas background and grid."""
        super().drawBackground(painter, rect)

        if not self._show_grid or not self._page_rect:
            return

        # Draw grid within page bounds
        page = self._document.page
        width_px = self.mm_to_px(page.width_mm)
        height_px = self.mm_to_px(page.height_mm)
        spacing_px = self.mm_to_px(self._grid_spacing_mm)

        # Only draw grid if zoom level is high enough
        if spacing_px * self._zoom_factor < 5:
            return

        painter.setPen(QPen(QColor(235, 235, 235), 0.5))

        # Vertical lines
        x = 0.0
        while x <= width_px:
            painter.drawLine(int(x), 0, int(x), int(height_px))
            x += spacing_px

        # Horizontal lines
        y = 0.0
        while y <= height_px:
            painter.drawLine(0, int(y), int(width_px), int(y))
            y += spacing_px

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        # Zoom with Ctrl+Wheel
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15

            new_zoom = self._zoom_factor * factor
            if self._min_zoom <= new_zoom <= self._max_zoom:
                self.scale(factor, factor)
                self._zoom_factor = new_zoom
        else:
            # Normal scroll
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Delete:
            # Delete selected items
            selected = self.get_selected_item_ids()
            for item_id in selected:
                self._document.remove_item(item_id)
                self.remove_item_graphic(item_id)
            self.selection_changed.emit([])

        elif event.key() == Qt.Key.Key_Escape:
            self.clear_selection()

        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Select all
            for graphic in self._item_graphics.values():
                graphic.setSelected(True)
            self.selection_changed.emit(self.get_selected_item_ids())

        elif event.key() == Qt.Key.Key_0 and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Reset zoom
            self.fit_page_in_view()

        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Handle context menu."""
        scene_pos = self.mapToScene(event.pos())

        # Check if clicking on an item
        item = self._scene.itemAt(scene_pos, self.transform())
        if item and item != self._page_rect and item != self._page_shadow:
            # Item context menu will be handled by the item
            super().contextMenuEvent(event)
        else:
            # Canvas context menu
            self.canvas_context_menu.emit(scene_pos)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Handle double-click on items."""
        scene_pos = self.mapToScene(event.pos())
        item = self._scene.itemAt(scene_pos, self.transform())

        # Find which layout item was clicked
        for item_id, graphic in self._item_graphics.items():
            if graphic == item or graphic.isAncestorOf(item):
                self.item_double_clicked.emit(item_id)
                return

        super().mouseDoubleClickEvent(event)
