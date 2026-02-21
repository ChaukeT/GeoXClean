from __future__ import annotations

from typing import Optional
from PyQt6.QtWidgets import QWidget
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
from PyQt6.QtCore import Qt, QRectF, QSize


class ScaleBarWidget(QWidget):
    """
    Simple high-DPI scale bar overlay with a modern card look.

    set_scale(pixel_length: float, max_value: float, units: str)
      - pixel_length: visual length of the bar in device pixels
      - max_value: world-space length represented by the bar
      - units: text representing the measurement units (e.g., "m", "ft")
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._px_len: float = 160.0
        self._max_value: float = 100.0
        self._unit_label: str = "m"
        self._start_label: str = "0"
        self._mid_label: str = "50"
        self._end_label: str = "100"
        self._padding = 10
        self._bar_height = 8
        self._corner_radius = 12.0
        self._bg = QColor(30, 30, 30, 200)
        self._fg = QColor(240, 240, 240)
        self._outline = QColor(85, 85, 85)
        self._tick_color = QColor(230, 230, 230)
        self._tick_width = 1.6
        # Use cross-platform font stack with fallbacks (UX-006 fix)
        self._font = QFont()
        self._font.setFamilies(["Segoe UI", "Roboto", "SF Pro Display", "Helvetica Neue", "Arial", "sans-serif"])
        self._font.setPointSize(10)
        self._font.setBold(True)
        self._theme_name = "white"
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Don't set window flags - widget should be a child of the viewer, not a separate window
        self.setFixedHeight(56)
        self.resize(240, 56)
        # Drag state
        self._dragging = False
        self._drag_offset = None
        # Remember if user has manually placed the widget; renderer won't auto-reposition when True
        self._user_placed = False
        self._default_margin = 16



    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def set_scale(self, pixel_length: float, max_value: float, units: str) -> None:
        self._px_len = max(40.0, float(pixel_length))
        self._max_value = max(0.0, float(max_value))
        self._unit_label = (units or "").strip()
        self._start_label = "0"
        self._mid_label = self._format_value(self._max_value / 2.0)
        self._end_label = self._format_value(self._max_value)
        # Resize width to fit bar + padding
        total_w = int(self._px_len + 2 * self._padding)
        self.setFixedWidth(max(140, total_w))
        self.update()

    def set_theme(self, theme: str, font_size: int) -> None:
        """Update basic theming (foreground/background colors and font size)."""
        name = (theme or "").lower()
        self._theme_name = name
        if name.startswith("black"):
            self._bg = QColor(245, 245, 245, 230)
            self._fg = QColor(15, 15, 15)
            self._tick_color = QColor(25, 25, 25)
            self._outline = QColor(160, 160, 160)
        elif name.startswith("gray"):
            self._bg = QColor(80, 80, 80, 210)
            self._fg = QColor(235, 235, 235)
            self._tick_color = QColor(220, 220, 220)
            self._outline = QColor(110, 110, 110)
        else:
            self._bg = QColor(30, 30, 30, 200)
            self._fg = QColor(240, 240, 240)
            self._tick_color = QColor(230, 230, 230)
            self._outline = QColor(85, 85, 85)
        self._font.setPointSize(max(8, min(int(font_size or 10), 32)))
        self._font.setBold(True)
        self.update()

    def set_anchor(self, anchor: str, margin: Optional[int] = None) -> None:
        """Place widget at a preset anchor unless the user already moved it."""
        if self.has_user_placement():
            return
        margin = self._default_margin if margin is None else margin
        anchor_name = (anchor or "").lower()
        if anchor_name in ("bottom_left", "bottom-left"):
            self.place_bottom_left(margin)
        elif anchor_name in ("top_left", "top-left"):
            self.place_top_left(margin)
        elif anchor_name in ("top_right", "top-right"):
            self.place_top_right(margin)
        elif anchor_name == "bottom_center":
            self.place_bottom_center(margin)
        else:
            self.place_bottom_right(margin)
        self.update()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(max(140, int(self._px_len + 2 * self._padding)), 56)

    def paintEvent(self, _):
        p = None
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = self.rect()
            # Card background
            path = QPainterPath()
            r = self._corner_radius
            path.addRoundedRect(QRectF(rect.adjusted(0, 0, -1, -1)), r, r)
            p.fillPath(path, QBrush(self._bg))
            p.setPen(QPen(self._outline, 1.4))
            p.drawPath(path)

            # Units label (top)
            p.setFont(self._font)
            p.setPen(QPen(self._fg))
            display_label = self._unit_label or ""
            p.drawText(
                rect.adjusted(0, 6, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                display_label
            )

            # Scale bar
            bar_top = rect.top() + 26
            bar_left = rect.left() + self._padding
            bar_rect = QRectF(bar_left, bar_top, self._px_len, self._bar_height)

            # Draw main bar rounded
            bar_path = QPainterPath()
            bar_path.addRoundedRect(bar_rect, self._bar_height / 2.0, self._bar_height / 2.0)
            p.fillPath(bar_path, QBrush(self._fg))

            # Ticks at 0, 25%, 50%, 75%, 100%
            p.setPen(QPen(self._tick_color, self._tick_width))
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x = bar_rect.left() + t * bar_rect.width()
                p.drawLine(int(x), int(bar_rect.top() - 6), int(x), int(bar_rect.bottom() + 6))

            # Labels below ticks (three labels with subtle spacing)
            tick_gap = 6
            label_rect_y = bar_rect.bottom() + tick_gap
            value_font = QFont(self._font)
            value_font.setPointSize(max(8, self._font.pointSize() - 1))
            value_font.setBold(True)
            p.setFont(value_font)
            p.setPen(QPen(self._fg))
            p.drawText(
                QRectF(bar_rect.left() - 30, label_rect_y, 50, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._start_label
            )
            p.drawText(
                QRectF(bar_rect.left(), label_rect_y, bar_rect.width(), 18),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                self._mid_label
            )
            p.drawText(
                QRectF(bar_rect.right() - 40, label_rect_y, 40, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._end_label
            )
        finally:
            if p is not None:
                p.end()

    # ---------------- Dragging support ----------------
    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._dragging and self.parent() is not None:
            try:
                parent = self.parent()
                # Calculate new top-left based on drag
                new_pos = self.mapToParent(event.pos() - self._drag_offset)
                # Constrain to parent rect with small margin
                margin = 4
                x = max(margin, min(new_pos.x(), parent.width() - self.width() - margin))
                y = max(margin, min(new_pos.y(), parent.height() - self.height() - margin))
                self.move(int(x), int(y))
                self._user_placed = True
                self.update()
            except Exception:
                pass
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_offset = None
            self._user_placed = True
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def place_bottom_right(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(parent.width() - self.width() - margin), int(parent.height() - self.height() - margin))

    # New placement helpers for flexible positioning
    def place_bottom_left(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(margin), int(parent.height() - self.height() - margin))

    def place_top_left(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(margin), int(margin))

    def place_top_right(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(parent.width() - self.width() - margin), int(margin))

    def place_bottom_center(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            x = int((parent.width() - self.width()) / 2)
            y = int(parent.height() - self.height() - margin)
            self.move(x, y)

    # ------------- helpers for integration -------------
    def has_user_placement(self) -> bool:
        return bool(self._user_placed)

    def clear_user_placement(self) -> None:
        self._user_placed = False

    # -------- AxisManager / overlay integration helpers --------

    def set_visible(self, visible: bool) -> None:
        """
        Simple visibility toggle used by AxisManager / overlay controllers.

        This intentionally wraps QWidget.setVisible so external managers do not
        have to know about the widget's internal state.
        """
        super().setVisible(bool(visible))

    def update_bounds(self, bounds) -> None:
        """
        Optional hint from AxisManager when scene bounds change.

        The HUD scale bar already recomputes its scale based on camera and
        world-per-pixel metrics inside the renderer, so this method is a
        lightweight no-op kept for API compatibility and future extension.
        """
        # API stub – no behaviour required for current HUD implementation.
        _ = bounds

    def _format_value(self, value: float) -> str:
        val = abs(value)
        if val >= 1000:
            text = f"{value:.0f}"
        elif val >= 100:
            text = f"{value:.0f}"
        elif val >= 10:
            text = f"{value:.1f}"
        elif val >= 1:
            text = f"{value:.2f}"
        else:
            text = f"{value:.3f}"
        text = text.rstrip('0').rstrip('.') if '.' in text else text
        return text or "0"
