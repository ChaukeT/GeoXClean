"""
Layout Item Graphics Widgets for GeoX Layout Composer.

QGraphicsItem subclasses for visualizing and interacting with
layout items on the canvas.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TYPE_CHECKING

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
    QPixmap, QLinearGradient, QPainterPath
)
from PyQt6.QtWidgets import (
    QGraphicsRectItem, QGraphicsItem, QStyleOptionGraphicsItem, QWidget,
    QGraphicsSceneMouseEvent
)

from ...layout.layout_document import (
    LayoutItem, ViewportItem, LegendItem, ScaleBarItem,
    NorthArrowItem, TextItem, ImageItem, MetadataItem
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BaseLayoutGraphicsItem(QGraphicsRectItem):
    """
    Base graphics item for layout items with selection handles and resize support.
    """

    HANDLE_SIZE = 8
    HANDLE_HALF = 4
    MIN_SIZE = 10  # Minimum size in pixels

    def __init__(
        self,
        layout_item: LayoutItem,
        mm_to_px: Callable[[float], float],
        parent=None
    ):
        self._layout_item = layout_item
        self._mm_to_px = mm_to_px

        x = mm_to_px(layout_item.x_mm)
        y = mm_to_px(layout_item.y_mm)
        w = mm_to_px(layout_item.width_mm)
        h = mm_to_px(layout_item.height_mm)

        super().__init__(0, 0, w, h, parent)
        self.setPos(x, y)

        # Enable interaction
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Resize state
        self._resizing = False
        self._resize_handle: Optional[str] = None
        self._resize_start_rect: Optional[QRectF] = None
        self._resize_start_pos: Optional[QPointF] = None

    def refresh_theme(self):
        """Update colors when theme changes."""
        self.update()

    @property
    def layout_item(self) -> LayoutItem:
        """Get the associated layout item."""
        return self._layout_item

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        """Draw the item with selection handles."""
        # Draw content
        self._paint_content(painter)

        # Draw selection indicator
        if self.isSelected():
            # Selection border
            painter.setPen(QPen(QColor(0, 120, 215), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect())

            # Resize handles at corners and edges
            handle_brush = QBrush(QColor(0, 120, 215))
            handle_pen = QPen(QColor(255, 255, 255), 1)
            painter.setBrush(handle_brush)
            painter.setPen(handle_pen)

            for handle_rect in self._get_handle_rects().values():
                painter.drawRect(handle_rect)

        # Draw lock indicator
        if self._layout_item.locked:
            painter.setPen(QPen(QColor(200, 100, 100), 2))
            lock_size = 12
            lock_x = self.rect().right() - lock_size - 4
            lock_y = 4
            painter.drawRect(int(lock_x), int(lock_y), lock_size, lock_size)
            painter.drawLine(
                int(lock_x + 3), int(lock_y),
                int(lock_x + 3), int(lock_y - 4)
            )
            painter.drawLine(
                int(lock_x + lock_size - 3), int(lock_y),
                int(lock_x + lock_size - 3), int(lock_y - 4)
            )

    def _paint_content(self, painter: QPainter) -> None:
        """Override in subclasses to draw specific content."""
        # Default: light gray box with item type label
        painter.fillRect(self.rect(), QBrush(QColor(245, 245, 245)))
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawRect(self.rect())

        painter.setPen(QColor(120, 120, 120))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            f"[{self._layout_item.item_type}]"
        )

    def _get_handle_rects(self) -> dict:
        """Get rectangles for all resize handles."""
        rect = self.rect()
        hs = self.HANDLE_SIZE
        hh = self.HANDLE_HALF

        return {
            "top_left": QRectF(rect.left() - hh, rect.top() - hh, hs, hs),
            "top_right": QRectF(rect.right() - hh, rect.top() - hh, hs, hs),
            "bottom_left": QRectF(rect.left() - hh, rect.bottom() - hh, hs, hs),
            "bottom_right": QRectF(rect.right() - hh, rect.bottom() - hh, hs, hs),
            "top": QRectF(rect.center().x() - hh, rect.top() - hh, hs, hs),
            "bottom": QRectF(rect.center().x() - hh, rect.bottom() - hh, hs, hs),
            "left": QRectF(rect.left() - hh, rect.center().y() - hh, hs, hs),
            "right": QRectF(rect.right() - hh, rect.center().y() - hh, hs, hs),
        }

    def _get_handle_at(self, pos: QPointF) -> Optional[str]:
        """Get the handle at the given position, if any."""
        for name, rect in self._get_handle_rects().items():
            if rect.contains(pos):
                return name
        return None

    def hoverMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Update cursor based on hover position."""
        if self._layout_item.locked:
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            return

        handle = self._get_handle_at(event.pos())
        if handle and self.isSelected():
            cursors = {
                "top_left": Qt.CursorShape.SizeFDiagCursor,
                "top_right": Qt.CursorShape.SizeBDiagCursor,
                "bottom_left": Qt.CursorShape.SizeBDiagCursor,
                "bottom_right": Qt.CursorShape.SizeFDiagCursor,
                "top": Qt.CursorShape.SizeVerCursor,
                "bottom": Qt.CursorShape.SizeVerCursor,
                "left": Qt.CursorShape.SizeHorCursor,
                "right": Qt.CursorShape.SizeHorCursor,
            }
            self.setCursor(cursors.get(handle, Qt.CursorShape.ArrowCursor))
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle mouse press for resizing."""
        if self._layout_item.locked:
            event.ignore()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            handle = self._get_handle_at(event.pos())
            if handle:
                self._resizing = True
                self._resize_handle = handle
                self._resize_start_rect = self.rect()
                self._resize_start_pos = event.pos()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle mouse move for resizing."""
        if self._resizing and self._resize_handle:
            delta = event.pos() - self._resize_start_pos
            new_rect = QRectF(self._resize_start_rect)

            # Adjust rect based on which handle is being dragged
            handle = self._resize_handle
            if "left" in handle:
                new_rect.setLeft(new_rect.left() + delta.x())
            if "right" in handle:
                new_rect.setRight(new_rect.right() + delta.x())
            if "top" in handle:
                new_rect.setTop(new_rect.top() + delta.y())
            if "bottom" in handle:
                new_rect.setBottom(new_rect.bottom() + delta.y())

            # Enforce minimum size
            if new_rect.width() >= self.MIN_SIZE and new_rect.height() >= self.MIN_SIZE:
                # Adjust position if left/top changed
                if new_rect.left() != self._resize_start_rect.left():
                    self.setX(self.x() + new_rect.left())
                    new_rect.moveLeft(0)
                if new_rect.top() != self._resize_start_rect.top():
                    self.setY(self.y() + new_rect.top())
                    new_rect.moveTop(0)

                self.setRect(0, 0, new_rect.width(), new_rect.height())
                self.update()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handle mouse release."""
        if self._resizing:
            self._resizing = False
            self._resize_handle = None
            # Update layout item with new dimensions
            self._sync_to_layout_item()

        super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Handle item changes like position."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._sync_to_layout_item()
        return super().itemChange(change, value)

    def _sync_to_layout_item(self) -> None:
        """Sync graphics item state back to layout item."""
        # Convert pixels back to mm
        mm_per_px = 25.4 / 96.0  # Inverse of mm_to_px
        self._layout_item.x_mm = self.x() * mm_per_px
        self._layout_item.y_mm = self.y() * mm_per_px
        self._layout_item.width_mm = self.rect().width() * mm_per_px
        self._layout_item.height_mm = self.rect().height() * mm_per_px


class ViewportGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for 3D viewport."""

    def __init__(self, layout_item: ViewportItem, mm_to_px: Callable[[float], float]):
        super().__init__(layout_item, mm_to_px)
        self._viewport_pixmap: Optional[QPixmap] = None

    def set_viewport_pixmap(self, pixmap: QPixmap) -> None:
        """Set the captured viewport image."""
        self._viewport_pixmap = pixmap
        self.update()

    def _paint_content(self, painter: QPainter) -> None:
        item = self._layout_item
        rect = self.rect()

        # Background
        bg_color = QColor(item.background_color) if hasattr(item, 'background_color') else QColor(30, 30, 40)
        painter.fillRect(rect, bg_color)

        if self._viewport_pixmap and not self._viewport_pixmap.isNull():
            # Scale pixmap to fit
            scaled = self._viewport_pixmap.scaled(
                int(rect.width()), int(rect.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            # Center
            x = (rect.width() - scaled.width()) / 2
            y = (rect.height() - scaled.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled)
        else:
            # Placeholder
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "[Viewport]")
            # Draw camera icon
            icon_size = min(rect.width(), rect.height()) * 0.3
            icon_rect = QRectF(
                rect.center().x() - icon_size / 2,
                rect.center().y() - icon_size / 2 - 15,
                icon_size, icon_size * 0.7
            )
            painter.setPen(QPen(QColor(80, 80, 80), 2))
            painter.drawRect(icon_rect)


class LegendGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for legend.

    This paint code is the "ground truth" for how the legend looks in the
    software.  The export renderer (layout_renderer.py) mirrors this logic
    with DPI-scaled fonts so the exported image matches what the user sees
    on the canvas.
    """

    def _paint_content(self, painter: QPainter) -> None:
        item: LegendItem = self._layout_item
        rect = self.rect()

        # Background
        bg_color = QColor(item.background_color) if hasattr(item, 'background_color') else QColor(30, 30, 35, 230)
        painter.fillRect(rect, bg_color)

        text_color = QColor(item.text_color) if hasattr(item, 'text_color') else QColor(240, 240, 240)
        padding = 8

        # Use font_family from the item (falls back to "Segoe UI" via dataclass default)
        font_family = getattr(item, 'font_family', "Segoe UI")

        # Title font
        title_font = QFont(font_family, item.font_size)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(text_color)

        legend_state = item.legend_state or {}
        title = legend_state.get("property", legend_state.get("title", "Legend"))
        title_metrics = QFontMetrics(title_font)

        if item.show_title and title:
            painter.drawText(padding, padding + title_metrics.ascent(), str(title))

        # Content starts below title
        bar_top = padding + title_metrics.height() + 8 if item.show_title else padding
        bar_width = 20

        # Get colormap from matplotlib
        cmap_name = legend_state.get("colormap", "viridis")
        try:
            from matplotlib import cm
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap("viridis")

        # Categorical vs continuous
        categories = legend_state.get("categories") or []
        is_categorical = bool(categories)

        if is_categorical:
            self._draw_categorical_legend(painter, item, legend_state, categories,
                                          padding, bar_top, bar_width, text_color,
                                          font_family)
        else:
            # Continuous gradient legend
            bar_height = rect.height() - bar_top - padding - 20

            if bar_height > 20:
                # Draw gradient using actual colormap
                gradient = QLinearGradient(padding, bar_top + bar_height, padding, bar_top)

                for i in range(11):
                    t = i / 10.0
                    rgba = cmap(t)
                    color = QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]))
                    gradient.setColorAt(t, color)

                painter.fillRect(int(padding), int(bar_top), bar_width, int(bar_height), gradient)
                painter.setPen(QPen(QColor(60, 60, 60), 1))
                painter.drawRect(int(padding), int(bar_top), bar_width, int(bar_height))

                # Label font — smaller than title
                label_font = QFont(font_family, max(item.font_size - 2, 6))
                painter.setFont(label_font)
                painter.setPen(text_color)
                label_metrics = QFontMetrics(label_font)

                vmin = legend_state.get("vmin", 0)
                vmax = legend_state.get("vmax", 1)
                label_x = int(padding + bar_width + 5)

                # Top (vmax) and bottom (vmin) labels
                painter.drawText(label_x, int(bar_top + label_metrics.ascent()), f"{vmax:.4g}")
                painter.drawText(label_x, int(bar_top + bar_height), f"{vmin:.4g}")

                # Intermediate tick labels at 1/4 intervals
                for i in range(1, 4):
                    t = i / 4.0
                    val = vmin + t * (vmax - vmin)
                    y_pos = bar_top + bar_height * (1 - t)
                    text_y = y_pos + label_metrics.ascent() // 2
                    painter.drawText(label_x, int(text_y), f"{val:.3g}")

    def _draw_categorical_legend(self, painter: QPainter, item, legend_state: dict,
                                  categories: list, padding: int, bar_top: int,
                                  bar_width: int, text_color: QColor,
                                  font_family: str) -> None:
        """Draw categorical/discrete legend."""
        rect = self.rect()
        category_colors = legend_state.get("category_colors", {})

        cmap_name = legend_state.get("colormap", "tab10")
        try:
            from matplotlib import cm
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap("tab10")

        swatch_size = 14
        row_height = 20
        max_categories = int((rect.height() - bar_top - 10) / row_height)

        label_font = QFont(font_family, max(item.font_size - 1, 6))
        painter.setFont(label_font)
        label_metrics = QFontMetrics(label_font)

        for i, category in enumerate(categories[:max_categories]):
            row_y = bar_top + i * row_height

            # Get color for this category
            cat_key = str(category)
            if cat_key in category_colors:
                rgba = category_colors[cat_key]
                if isinstance(rgba, (list, tuple)) and len(rgba) >= 3:
                    color = QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]))
                else:
                    color = QColor(150, 150, 150)
            else:
                rgba = cmap(i / max(len(categories) - 1, 1))
                color = QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]))

            # Draw color swatch
            painter.fillRect(int(padding), int(row_y), swatch_size, swatch_size, color)
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            painter.drawRect(int(padding), int(row_y), swatch_size, swatch_size)

            # Draw label — use font metrics for proper vertical alignment
            painter.setPen(text_color)
            label = str(category)[:20]
            painter.drawText(
                int(padding + swatch_size + 5),
                int(row_y + label_metrics.ascent()),
                label
            )

        # Truncation indicator
        if len(categories) > max_categories:
            painter.setPen(text_color)
            painter.drawText(
                int(padding),
                int(bar_top + max_categories * row_height + label_metrics.ascent()),
                "…"
            )


class ScaleBarGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for scale bar."""

    def _paint_content(self, painter: QPainter) -> None:
        item: ScaleBarItem = self._layout_item
        rect = self.rect()

        # Background if set
        if item.background_color:
            painter.fillRect(rect, QColor(item.background_color))

        bar_color = QColor(item.bar_color)
        alt_color = QColor(item.alt_bar_color)
        text_color = QColor(item.text_color)

        font_family = getattr(item, 'font_family', "Segoe UI")

        # Calculate bar dimensions
        bar_height = self._mm_to_px(item.bar_height_mm)
        font = QFont(font_family, item.font_size)
        metrics = QFontMetrics(font)

        bar_y = (rect.height() - bar_height - metrics.height() - 5) / 2
        bar_width = rect.width() - 20
        segment_width = bar_width / item.num_segments

        # Draw alternating segments
        for i in range(item.num_segments):
            seg_x = 10 + i * segment_width
            color = bar_color if i % 2 == 0 else alt_color
            painter.fillRect(int(seg_x), int(bar_y), int(segment_width), int(bar_height), color)

        # Outline
        painter.setPen(QPen(bar_color, 1))
        painter.drawRect(10, int(bar_y), int(bar_width), int(bar_height))

        # Label
        painter.setFont(font)
        painter.setPen(text_color)

        label = f"100 {item.units}"
        if item.is_approximate:
            label += " (approx)"

        label_width = metrics.horizontalAdvance(label)
        label_x = 10 + (bar_width - label_width) / 2
        painter.drawText(int(label_x), int(bar_y + bar_height + 5 + metrics.ascent()), label)


class NorthArrowGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for north arrow."""

    def _paint_content(self, painter: QPainter) -> None:
        item: NorthArrowItem = self._layout_item
        rect = self.rect()

        fill_color = QColor(item.fill_color)
        outline_color = QColor(item.outline_color)

        font_family = getattr(item, 'font_family', "Segoe UI")

        center_x = rect.width() / 2
        center_y = rect.height() / 2 + 5

        # Arrow size
        arrow_size = min(rect.width(), rect.height() - 20) * 0.7
        half_size = arrow_size / 2

        painter.save()
        painter.translate(center_x, center_y)

        # Apply rotation if set
        rotation = item.rotation_override or 0.0
        painter.rotate(rotation)

        # Draw arrow
        arrow_path = QPainterPath()
        arrow_path.moveTo(0, -half_size)  # Top point
        arrow_path.lineTo(half_size * 0.35, half_size * 0.4)
        arrow_path.lineTo(0, half_size * 0.15)
        arrow_path.lineTo(-half_size * 0.35, half_size * 0.4)
        arrow_path.closeSubpath()

        painter.fillPath(arrow_path, QBrush(fill_color))
        painter.setPen(QPen(outline_color, 1.5))
        painter.drawPath(arrow_path)

        painter.restore()

        # Draw "N" label
        if item.show_label:
            font = QFont(font_family, item.font_size)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(fill_color)

            metrics = QFontMetrics(font)
            label_width = metrics.horizontalAdvance(item.label_text)
            painter.drawText(
                int(center_x - label_width / 2),
                int(12 + metrics.ascent()),
                item.label_text
            )

        # Warning indicator
        if not item.has_crs:
            painter.setPen(QPen(QColor(200, 150, 50), 1))
            painter.drawText(5, int(rect.height() - 5), "!")


class TextGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for text."""

    def _paint_content(self, painter: QPainter) -> None:
        item: TextItem = self._layout_item
        rect = self.rect()

        # Background
        if item.background_color:
            painter.fillRect(rect, QColor(item.background_color))

        # Border
        if item.border_color:
            border_width = self._mm_to_px(item.border_width_mm)
            painter.setPen(QPen(QColor(item.border_color), border_width))
            painter.drawRect(rect)

        # Text
        font = QFont(item.font_family, item.font_size)
        font.setBold(item.font_bold)
        font.setItalic(item.font_italic)
        painter.setFont(font)
        painter.setPen(QColor(item.text_color))

        padding = self._mm_to_px(item.padding_mm)
        text_rect = rect.adjusted(padding, padding, -padding, -padding)

        # Alignment
        h_align = {
            "left": Qt.AlignmentFlag.AlignLeft,
            "center": Qt.AlignmentFlag.AlignHCenter,
            "right": Qt.AlignmentFlag.AlignRight,
        }.get(item.alignment, Qt.AlignmentFlag.AlignLeft)

        v_align = {
            "top": Qt.AlignmentFlag.AlignTop,
            "middle": Qt.AlignmentFlag.AlignVCenter,
            "bottom": Qt.AlignmentFlag.AlignBottom,
        }.get(item.vertical_alignment, Qt.AlignmentFlag.AlignTop)

        flags = h_align | v_align
        if item.word_wrap:
            flags |= Qt.TextFlag.TextWordWrap

        painter.drawText(text_rect, flags, item.text)


class ImageGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for image/logo."""

    def __init__(self, layout_item: ImageItem, mm_to_px: Callable[[float], float]):
        super().__init__(layout_item, mm_to_px)
        self._image_pixmap: Optional[QPixmap] = None
        self._load_image()

    def _load_image(self) -> None:
        """Load image from path or embedded data."""
        item: ImageItem = self._layout_item

        if item.image_path:
            from pathlib import Path
            path = Path(item.image_path)
            if path.exists():
                self._image_pixmap = QPixmap(str(path))
                return

        if item.image_data_base64:
            import base64
            try:
                data = base64.b64decode(item.image_data_base64)
                self._image_pixmap = QPixmap()
                self._image_pixmap.loadFromData(data)
            except Exception as e:
                logger.warning(f"Failed to decode embedded image: {e}")

    def _paint_content(self, painter: QPainter) -> None:
        item: ImageItem = self._layout_item
        rect = self.rect()

        if self._image_pixmap and not self._image_pixmap.isNull():
            if item.maintain_aspect:
                scaled = self._image_pixmap.scaled(
                    int(rect.width()), int(rect.height()),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                x = (rect.width() - scaled.width()) / 2
                y = (rect.height() - scaled.height()) / 2
            else:
                scaled = self._image_pixmap.scaled(
                    int(rect.width()), int(rect.height()),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                x, y = 0, 0

            if item.opacity < 1.0:
                painter.setOpacity(item.opacity)

            painter.drawPixmap(int(x), int(y), scaled)
            painter.setOpacity(1.0)
        else:
            # Placeholder
            painter.fillRect(rect, QColor(240, 240, 240))
            painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.PenStyle.DashLine))
            painter.drawRect(rect)
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "[Image]")

        # Border
        if item.border_color and item.border_width_mm > 0:
            border_width = self._mm_to_px(item.border_width_mm)
            painter.setPen(QPen(QColor(item.border_color), border_width))
            painter.drawRect(rect)


class MetadataGraphicsItem(BaseLayoutGraphicsItem):
    """Graphics item for metadata block."""

    def _paint_content(self, painter: QPainter) -> None:
        item: MetadataItem = self._layout_item
        rect = self.rect()

        # Background
        if item.background_color:
            painter.fillRect(rect, QColor(item.background_color))

        font = QFont(item.font_family, item.font_size)
        painter.setFont(font)
        painter.setPen(QColor(item.text_color))

        metrics = QFontMetrics(font)
        line_height = max(
            int(metrics.height() * item.line_spacing),
            metrics.height() + 2
        )
        label_width = self._mm_to_px(item.label_width_mm)
        padding = 5

        field_labels = {
            "project_name": "Project:",
            "date": "Date:",
            "author": "Author:",
            "crs": "CRS:",
            "software_version": "Software:",
            "dataset_name": "Dataset:",
            "domain": "Domain:",
        }

        ascent = metrics.ascent()
        y = padding
        for field in item.fields[:8]:  # Limit displayed fields
            if y + ascent > rect.height() - padding:
                break

            if item.show_labels:
                label = field_labels.get(field, f"{field}:")
                painter.drawText(int(padding), int(y + ascent), label)

                value = item.custom_values.get(field, f"[{field}]")
                painter.drawText(int(padding + label_width), int(y + ascent), str(value))
            else:
                value = item.custom_values.get(field, f"[{field}]")
                painter.drawText(int(padding), int(y + ascent), str(value))

            y += line_height


# Factory function to create the appropriate graphics item
def create_graphics_item(
    item: LayoutItem,
    mm_to_px: Callable[[float], float]
) -> BaseLayoutGraphicsItem:
    """
    Create the appropriate graphics item for a layout item.

    Args:
        item: The layout item
        mm_to_px: Conversion function from mm to pixels

    Returns:
        The created graphics item
    """
    item_classes = {
        "viewport": ViewportGraphicsItem,
        "legend": LegendGraphicsItem,
        "scale_bar": ScaleBarGraphicsItem,
        "north_arrow": NorthArrowGraphicsItem,
        "text": TextGraphicsItem,
        "image": ImageGraphicsItem,
        "metadata": MetadataGraphicsItem,
    }

    item_cls = item_classes.get(item.item_type, BaseLayoutGraphicsItem)
    return item_cls(item, mm_to_px)
