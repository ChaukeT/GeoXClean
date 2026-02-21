"""
Layout Renderer for GeoX Layout Composer.

DPI-independent rendering engine that converts layout documents
to QImage at any resolution. Text and vector elements are rendered
at target DPI for sharp output; viewport captures are scaled appropriately.

Key design note — font scaling:
  To render a font at N points on an image at D DPI, we compute the
  pixel size as  pixels = points × D / 72  and use font.setPixelSize().
  We must NOT use font.setPointSizeF(points × D / 72) because Qt will
  apply its own points-to-pixels conversion on top, double-scaling the text.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QFont, QFontMetrics, QPen, QBrush,
    QLinearGradient, QPainterPath, QPixmap, QPolygonF
)

from .layout_document import (
    LayoutDocument, LayoutItem, ViewportItem, LegendItem, ScaleBarItem,
    NorthArrowItem, TextItem, ImageItem, MetadataItem
)

if TYPE_CHECKING:
    from ..ui.viewer_widget import ViewerWidget

logger = logging.getLogger(__name__)


# DPI presets for common use cases
DPI_PRESETS = {
    "screen": 96,
    "presentation": 150,
    "publication": 300,
    "poster": 600,
}

# Maximum image dimension to prevent memory issues
MAX_DIMENSION_PX = 16000  # ~100 megapixels for A0 at 300 DPI


def _make_font(family: str, point_size: float, dpi: int,
               bold: bool = False, italic: bool = False) -> QFont:
    """Create a QFont with correct pixel size for the target DPI.

    Args:
        family: Font family name.
        point_size: Desired size in typographic points (1 pt = 1/72 inch).
        dpi: Target rendering DPI.
        bold: Bold flag.
        italic: Italic flag.

    Returns:
        QFont configured with setPixelSize so it renders at the correct
        physical size on an image at *dpi* resolution.
    """
    font = QFont(family)
    pixel_size = max(1, int(round(point_size * dpi / 72.0)))
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    font.setItalic(italic)
    return font


class LayoutRenderer:
    """
    Resolution-independent renderer for layout documents.

    Renders to QImage at specified DPI, suitable for export to
    PDF, PNG, or TIFF formats.
    """

    MM_TO_INCH = 25.4

    def __init__(self, document: LayoutDocument, viewer_widget: Optional["ViewerWidget"] = None):
        self._document = document
        self._viewer = viewer_widget
        self._viewport_cache: Dict[str, QImage] = {}
        self._metadata_values: Dict[str, str] = {}

    def set_metadata_values(self, values: Dict[str, str]) -> None:
        """Set dynamic metadata values for rendering."""
        self._metadata_values = values

    def mm_to_px(self, mm: float, dpi: int) -> int:
        """Convert millimeters to pixels at given DPI."""
        return int(mm / self.MM_TO_INCH * dpi)

    def px_to_mm(self, px: int, dpi: int) -> float:
        """Convert pixels to millimeters at given DPI."""
        return px * self.MM_TO_INCH / dpi

    def render_to_image(self, dpi: int = 300, transparent: bool = False) -> QImage:
        """Render complete layout to QImage at specified DPI."""
        page = self._document.page

        width_px = self.mm_to_px(page.width_mm, dpi)
        height_px = self.mm_to_px(page.height_mm, dpi)

        # Clamp to sane dimensions
        if width_px > MAX_DIMENSION_PX or height_px > MAX_DIMENSION_PX:
            logger.warning(
                f"Layout dimensions ({width_px}x{height_px}px) exceed maximum. "
                f"Clamping to {MAX_DIMENSION_PX}px."
            )
            scale = min(MAX_DIMENSION_PX / width_px, MAX_DIMENSION_PX / height_px)
            width_px = int(width_px * scale)
            height_px = int(height_px * scale)
            dpi = int(dpi * scale)

        if transparent:
            image = QImage(width_px, height_px, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
        else:
            image = QImage(width_px, height_px, QImage.Format.Format_RGB32)
            image.fill(QColor(255, 255, 255))

        # DPI metadata (informational; QPainter does NOT auto-scale fonts to this)
        image.setDotsPerMeterX(int(dpi / self.MM_TO_INCH * 1000))
        image.setDotsPerMeterY(int(dpi / self.MM_TO_INCH * 1000))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        for item in self._document.get_items_sorted():
            if not item.visible:
                continue
            self._render_item(painter, item, dpi)

        if self._document.show_footer:
            self._render_footer(painter, dpi)

        painter.end()
        return image

    # ------------------------------------------------------------------
    # Item dispatch
    # ------------------------------------------------------------------

    def _render_item(self, painter: QPainter, item: LayoutItem, dpi: int) -> None:
        x = self.mm_to_px(item.x_mm, dpi)
        y = self.mm_to_px(item.y_mm, dpi)
        w = self.mm_to_px(item.width_mm, dpi)
        h = self.mm_to_px(item.height_mm, dpi)

        painter.save()
        painter.translate(x, y)

        if item.rotation_deg != 0:
            painter.translate(w / 2, h / 2)
            painter.rotate(item.rotation_deg)
            painter.translate(-w / 2, -h / 2)

        if isinstance(item, ViewportItem):
            self._render_viewport(painter, item, w, h, dpi)
        elif isinstance(item, LegendItem):
            self._render_legend(painter, item, w, h, dpi)
        elif isinstance(item, ScaleBarItem):
            self._render_scale_bar(painter, item, w, h, dpi)
        elif isinstance(item, NorthArrowItem):
            self._render_north_arrow(painter, item, w, h, dpi)
        elif isinstance(item, TextItem):
            self._render_text(painter, item, w, h, dpi)
        elif isinstance(item, ImageItem):
            self._render_image(painter, item, w, h, dpi)
        elif isinstance(item, MetadataItem):
            self._render_metadata(painter, item, w, h, dpi)
        else:
            self._render_placeholder(painter, item, w, h, dpi)

        painter.restore()

    # ------------------------------------------------------------------
    # Viewport
    # ------------------------------------------------------------------

    def _render_viewport(self, painter: QPainter, item: ViewportItem,
                         width: int, height: int, dpi: int) -> None:
        painter.fillRect(0, 0, width, height, QColor(item.background_color))

        if self._viewer is None:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(
                QRectF(0, 0, width, height),
                Qt.AlignmentFlag.AlignCenter,
                "[Viewport - No viewer connected]"
            )
            return

        try:
            renderer = getattr(self._viewer, 'renderer', None)
            if renderer is None:
                raise ValueError("No renderer available")

            plotter = getattr(renderer, 'plotter', None)
            if plotter is None:
                raise ValueError("No plotter available")

            current_size = plotter.window_size
            if current_size[0] == 0 or current_size[1] == 0:
                raise ValueError("Invalid plotter window size")

            scale_x = width / current_size[0]
            scale_y = height / current_size[1]
            scale = max(int(max(scale_x, scale_y)), 1)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = Path(f.name)

            try:
                plotter.screenshot(str(temp_path), scale=scale)

                viewport_img = QImage(str(temp_path))
                if viewport_img.isNull():
                    raise ValueError("Failed to load screenshot")

                scaled = viewport_img.scaled(
                    width, height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                x_off = (width - scaled.width()) // 2
                y_off = (height - scaled.height()) // 2
                painter.drawImage(x_off, y_off, scaled)
            finally:
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to render viewport: {e}")
            painter.setPen(QPen(QColor(200, 100, 100)))
            painter.drawText(
                QRectF(0, 0, width, height),
                Qt.AlignmentFlag.AlignCenter,
                f"[Viewport Error: {str(e)[:50]}]"
            )

    # ------------------------------------------------------------------
    # Legend  (mirrors LegendGraphicsItem._paint_content for consistency)
    # ------------------------------------------------------------------

    def _render_legend(self, painter: QPainter, item: LegendItem,
                       width: int, height: int, dpi: int) -> None:
        """Render legend — matches the canvas widget's visual output."""
        painter.fillRect(0, 0, width, height, QColor(item.background_color))

        legend_state = item.legend_state or {}
        title = legend_state.get("property", legend_state.get("title", "Legend"))
        text_color = QColor(item.text_color)

        padding = int(8 * dpi / 96)
        bar_width = int(20 * dpi / 96)

        # Title font
        title_font = _make_font(item.font_family, item.font_size, dpi, bold=True)
        label_font = _make_font(item.font_family, max(item.font_size - 2, 6), dpi)

        # Draw title
        painter.setFont(title_font)
        painter.setPen(text_color)
        title_metrics = QFontMetrics(title_font)

        if item.show_title and title:
            painter.drawText(padding, padding + title_metrics.ascent(), str(title))

        bar_top = padding + title_metrics.height() + int(8 * dpi / 96) if item.show_title else padding

        # Categorical vs continuous
        categories = legend_state.get("categories") or []
        if categories:
            self._render_discrete_legend(
                painter, item, legend_state, categories,
                padding, bar_top, width, height,
                label_font, text_color, dpi
            )
        else:
            self._render_continuous_legend(
                painter, item, legend_state,
                padding, bar_top, bar_width, width, height,
                label_font, text_color, dpi
            )

    def _render_continuous_legend(self, painter: QPainter, item: LegendItem,
                                  legend_state: dict,
                                  padding: int, bar_top: int, bar_width: int,
                                  total_width: int, total_height: int,
                                  font: QFont, text_color: QColor, dpi: int) -> None:
        """Render continuous gradient legend (matches canvas widget)."""
        vmin = legend_state.get("vmin", 0.0)
        vmax = legend_state.get("vmax", 1.0)
        cmap_name = legend_state.get("colormap", "viridis")

        try:
            from matplotlib import cm
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap("viridis")

        bar_height = total_height - bar_top - padding - int(20 * dpi / 96)
        if bar_height <= 20:
            return

        # Draw gradient bar
        gradient = QLinearGradient(padding, bar_top + bar_height, padding, bar_top)
        for i in range(11):
            t = i / 10.0
            rgba = cmap(t)
            gradient.setColorAt(t, QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2])))

        painter.fillRect(padding, bar_top, bar_width, bar_height, gradient)
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(padding, bar_top, bar_width, bar_height)

        # Tick labels
        painter.setFont(font)
        painter.setPen(text_color)
        metrics = QFontMetrics(font)
        label_x = padding + bar_width + int(5 * dpi / 96)

        # Top (vmax) and bottom (vmin) labels
        painter.drawText(label_x, bar_top + metrics.ascent(), f"{vmax:.4g}")
        painter.drawText(label_x, bar_top + bar_height, f"{vmin:.4g}")

        # Intermediate ticks at 1/4 intervals
        for i in range(1, 4):
            t = i / 4.0
            val = vmin + t * (vmax - vmin)
            y_pos = bar_top + int(bar_height * (1 - t))
            # Centre text on tick mark
            text_y = y_pos + metrics.ascent() // 2
            painter.drawText(label_x, text_y, f"{val:.3g}")

            # Tick mark
            painter.drawLine(
                padding + bar_width, y_pos,
                padding + bar_width + int(3 * dpi / 96), y_pos
            )

    def _render_discrete_legend(self, painter: QPainter, item: LegendItem,
                                legend_state: dict, categories: list,
                                padding: int, bar_top: int,
                                total_width: int, total_height: int,
                                font: QFont, text_color: QColor, dpi: int) -> None:
        """Render discrete categorical legend (matches canvas widget)."""
        category_colors = legend_state.get("category_colors", {})

        cmap_name = legend_state.get("colormap", "tab10")
        try:
            from matplotlib import cm
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap("tab10")

        swatch_size = int(14 * dpi / 96)
        row_height = int(20 * dpi / 96)
        max_cats = int((total_height - bar_top - int(10 * dpi / 96)) / row_height)

        painter.setFont(font)
        metrics = QFontMetrics(font)

        for i, category in enumerate(categories[:max_cats]):
            row_y = bar_top + i * row_height

            # Colour for category
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

            # Swatch
            painter.fillRect(padding, row_y, swatch_size, swatch_size, color)
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            painter.drawRect(padding, row_y, swatch_size, swatch_size)

            # Label
            painter.setPen(text_color)
            label = str(category)[:20]
            painter.drawText(
                padding + swatch_size + int(5 * dpi / 96),
                row_y + metrics.ascent(),
                label
            )

        # Truncation indicator
        if len(categories) > max_cats:
            painter.setPen(text_color)
            painter.drawText(padding, bar_top + max_cats * row_height + metrics.ascent(), "…")

    # ------------------------------------------------------------------
    # Scale bar
    # ------------------------------------------------------------------

    def _render_scale_bar(self, painter: QPainter, item: ScaleBarItem,
                          width: int, height: int, dpi: int) -> None:
        if item.background_color:
            painter.fillRect(0, 0, width, height, QColor(item.background_color))

        bar_height = self.mm_to_px(item.bar_height_mm, dpi)
        font = _make_font(item.font_family, item.font_size, dpi)
        metrics = QFontMetrics(font)

        label_height = metrics.height()
        bar_y = (height - bar_height - label_height - 5) // 2
        bar_width = width - 20

        # Determine scale value
        scale_value = item.scale_value or 100.0
        total_length = scale_value * (bar_width * self.MM_TO_INCH / dpi)

        # Round to nice number
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        nice_length = min(nice_values, key=lambda v: abs(v - total_length / 2))
        segment_width = bar_width / item.num_segments

        # Draw alternating segments
        for i in range(item.num_segments):
            seg_x = 10 + int(i * segment_width)
            color = QColor(item.bar_color) if i % 2 == 0 else QColor(item.alt_bar_color)
            painter.fillRect(seg_x, bar_y, int(segment_width), bar_height, color)

        # Outline
        painter.setPen(QPen(QColor(item.bar_color), 1))
        painter.drawRect(10, bar_y, bar_width, bar_height)

        # Label
        painter.setFont(font)
        painter.setPen(QColor(item.text_color))

        label = f"{nice_length} {item.units}"
        if item.is_approximate:
            label += " (approx)"

        label_x = 10 + bar_width // 2 - metrics.horizontalAdvance(label) // 2
        label_y = bar_y + bar_height + 5 + metrics.ascent()
        painter.drawText(label_x, label_y, label)

    # ------------------------------------------------------------------
    # North arrow
    # ------------------------------------------------------------------

    def _render_north_arrow(self, painter: QPainter, item: NorthArrowItem,
                            width: int, height: int, dpi: int) -> None:
        center_x = width // 2
        center_y = height // 2 + int(5 * dpi / 96)

        arrow_size = min(width, height - int(20 * dpi / 96)) * 0.8
        half_size = arrow_size / 2

        rotation = item.rotation_override or 0.0

        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(rotation)

        arrow_path = QPainterPath()

        if item.style == "simple":
            arrow_path.moveTo(0, -half_size)
            arrow_path.lineTo(half_size * 0.4, half_size * 0.5)
            arrow_path.lineTo(0, half_size * 0.2)
            arrow_path.lineTo(-half_size * 0.4, half_size * 0.5)
            arrow_path.closeSubpath()

            painter.fillPath(arrow_path, QBrush(QColor(item.fill_color)))
            painter.setPen(QPen(QColor(item.outline_color), 2))
            painter.drawPath(arrow_path)

        elif item.style == "compass":
            north_path = QPainterPath()
            north_path.moveTo(0, -half_size)
            north_path.lineTo(half_size * 0.2, 0)
            north_path.lineTo(0, half_size * 0.15)
            north_path.lineTo(-half_size * 0.2, 0)
            north_path.closeSubpath()

            painter.fillPath(north_path, QBrush(QColor(item.fill_color)))
            painter.setPen(QPen(QColor(item.outline_color), 1))
            painter.drawPath(north_path)

            south_path = QPainterPath()
            south_path.moveTo(0, half_size)
            south_path.lineTo(half_size * 0.2, 0)
            south_path.lineTo(0, -half_size * 0.15)
            south_path.lineTo(-half_size * 0.2, 0)
            south_path.closeSubpath()

            painter.setPen(QPen(QColor(item.fill_color), 1))
            painter.drawPath(south_path)

        painter.restore()

        # Label
        if item.show_label:
            label_font = _make_font(item.font_family, item.font_size, dpi, bold=True)
            painter.setFont(label_font)
            painter.setPen(QColor(item.fill_color))

            metrics = QFontMetrics(label_font)
            lw = metrics.horizontalAdvance(item.label_text)
            painter.drawText(
                center_x - lw // 2,
                int(10 * dpi / 96) + metrics.ascent(),
                item.label_text
            )

        # CRS warning
        if not item.has_crs and item.warning_message:
            warn_font = _make_font(item.font_family, 7, dpi)
            painter.setPen(QColor(200, 150, 50))
            painter.setFont(warn_font)
            painter.drawText(0, height - 5, item.warning_message[:30])

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def _render_text(self, painter: QPainter, item: TextItem,
                     width: int, height: int, dpi: int) -> None:
        # Background
        if item.background_color:
            painter.fillRect(0, 0, width, height, QColor(item.background_color))

        # Border
        if item.border_color:
            border_width = self.mm_to_px(item.border_width_mm, dpi)
            painter.setPen(QPen(QColor(item.border_color), border_width))
            painter.drawRect(0, 0, width, height)

        # Font
        font = _make_font(
            item.font_family, item.font_size, dpi,
            bold=item.font_bold, italic=item.font_italic
        )
        painter.setFont(font)
        painter.setPen(QColor(item.text_color))

        # Text area with padding
        padding = self.mm_to_px(item.padding_mm, dpi)
        text_rect = QRectF(padding, padding, width - 2 * padding, height - 2 * padding)

        # Alignment flags
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

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def _render_image(self, painter: QPainter, item: ImageItem,
                      width: int, height: int, dpi: int) -> None:
        image = None

        if item.image_path:
            path = Path(item.image_path)
            if path.exists():
                image = QImage(str(path))

        if image is None and item.image_data_base64:
            import base64
            try:
                data = base64.b64decode(item.image_data_base64)
                image = QImage()
                image.loadFromData(data)
            except Exception as e:
                logger.warning(f"Failed to decode embedded image: {e}")

        if image is None or image.isNull():
            painter.fillRect(0, 0, width, height, QColor(240, 240, 240))
            painter.setPen(QColor(150, 150, 150))
            painter.drawRect(0, 0, width - 1, height - 1)
            painter.drawText(
                QRectF(0, 0, width, height),
                Qt.AlignmentFlag.AlignCenter,
                "[Image]"
            )
            return

        if item.maintain_aspect:
            scaled = image.scaled(
                width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x_offset = (width - scaled.width()) // 2
            y_offset = (height - scaled.height()) // 2
        else:
            scaled = image.scaled(
                width, height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x_offset = 0
            y_offset = 0

        if item.opacity < 1.0:
            painter.setOpacity(item.opacity)

        painter.drawImage(x_offset, y_offset, scaled)
        painter.setOpacity(1.0)

        if item.border_color and item.border_width_mm > 0:
            border_width = self.mm_to_px(item.border_width_mm, dpi)
            painter.setPen(QPen(QColor(item.border_color), border_width))
            painter.drawRect(0, 0, width, height)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _render_metadata(self, painter: QPainter, item: MetadataItem,
                         width: int, height: int, dpi: int) -> None:
        if item.background_color:
            painter.fillRect(0, 0, width, height, QColor(item.background_color))

        font = _make_font(item.font_family, item.font_size, dpi)
        painter.setFont(font)
        painter.setPen(QColor(item.text_color))
        metrics = painter.fontMetrics()

        # Line height from actual font metrics
        line_height = max(
            int(metrics.height() * item.line_spacing),
            metrics.height() + 2    # absolute minimum: font height + 2px gap
        )

        label_width = self.mm_to_px(item.label_width_mm, dpi)
        # Ensure labels don't squash values
        min_label_width = int(60 * dpi / 96)
        label_width = max(label_width, min_label_width)

        padding = int(5 * dpi / 96)

        field_labels = {
            "project_name": "Project",
            "date": "Date",
            "author": "Author",
            "crs": "CRS",
            "software_version": "Software",
            "dataset_name": "Dataset",
            "domain": "Domain",
            "run_id": "Run ID",
            "variogram_signature": "Variogram",
            "export_dpi": "DPI",
            "page_size": "Page Size",
        }

        ascent = metrics.ascent()
        y = padding

        for field_name in item.fields:
            # Stop before overflowing
            if y + ascent > height - padding:
                break

            # Resolve value
            if field_name in item.custom_values:
                value = item.custom_values[field_name]
            elif field_name in self._metadata_values:
                value = self._metadata_values[field_name]
            else:
                value = f"[{field_name}]"

            if item.show_labels:
                label = field_labels.get(field_name, field_name) + ":"
                painter.drawText(padding, y + ascent, label)
                painter.drawText(padding + label_width, y + ascent, str(value))
            else:
                painter.drawText(padding, y + ascent, str(value))

            y += line_height

    # ------------------------------------------------------------------
    # Placeholder / Footer
    # ------------------------------------------------------------------

    def _render_placeholder(self, painter: QPainter, item: LayoutItem,
                            width: int, height: int, dpi: int) -> None:
        painter.fillRect(0, 0, width, height, QColor(220, 220, 220))
        painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
        painter.drawRect(0, 0, width - 1, height - 1)
        painter.setPen(QColor(100, 100, 100))
        painter.drawText(
            QRectF(0, 0, width, height),
            Qt.AlignmentFlag.AlignCenter,
            f"[{item.item_type}]"
        )

    def _render_footer(self, painter: QPainter, dpi: int) -> None:
        page = self._document.page
        width_px = self.mm_to_px(page.width_mm, dpi)
        height_px = self.mm_to_px(page.height_mm, dpi)

        font = _make_font("Segoe UI", 8, dpi)
        painter.setFont(font)
        painter.setPen(QColor(150, 150, 150))

        metrics = QFontMetrics(font)
        footer_text = self._document.footer_text
        text_width = metrics.horizontalAdvance(footer_text)

        x = (width_px - text_width) // 2
        y = height_px - self.mm_to_px(5, dpi)

        painter.drawText(x, y, footer_text)
