"""
North Arrow Widget - Simple compass overlay for 3D viewer.

Displays a minimal arrow pointing north with 'N' label.
Rotates based on camera azimuth to always indicate true north direction.
"""

from __future__ import annotations

import math
from typing import Optional
from PyQt6.QtWidgets import QWidget
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QPolygonF
from PyQt6.QtCore import Qt, QRectF, QSize, QPointF


class NorthArrowWidget(QWidget):
    """
    Simple north arrow overlay showing geographic orientation.

    Displays minimal arrow pointing north with 'N' label.
    Rotates based on camera azimuth.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rotation_degrees: float = 0.0  # 0 = north up
        self._size: int = 60
        self._padding = 8
        self._corner_radius = 8.0

        # Colors
        self._bg = QColor(30, 30, 30, 200)
        self._fg = QColor(240, 240, 240)
        self._arrow_color = QColor(220, 60, 60)  # Red for north arrow
        self._outline = QColor(85, 85, 85)

        # Font for 'N' label
        self._font = QFont()
        self._font.setFamilies(["Segoe UI", "Roboto", "SF Pro Display", "Helvetica Neue", "Arial", "sans-serif"])
        self._font.setPointSize(12)
        self._font.setBold(True)

        self._theme_name = "white"
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Don't set window flags - widget should be a child of the viewer, not a separate window
        self.setFixedSize(self._size + 2 * self._padding, self._size + 2 * self._padding)

        # Drag state
        self._dragging = False
        self._drag_offset = None
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
    def set_rotation(self, degrees: float) -> None:
        """
        Set the rotation angle for the north arrow.

        Args:
            degrees: Rotation angle in degrees. 0 = north up (arrow points up),
                    90 = north to the right, etc.
        """
        self._rotation_degrees = degrees % 360.0
        self.update()

    def set_theme(self, theme: str, font_size: int = 12) -> None:
        """Update basic theming (foreground/background colors and font size)."""
        name = (theme or "").lower()
        self._theme_name = name
        if name.startswith("black"):
            self._bg = QColor(245, 245, 245, 230)
            self._fg = QColor(15, 15, 15)
            self._arrow_color = QColor(180, 40, 40)
            self._outline = QColor(160, 160, 160)
        elif name.startswith("gray"):
            self._bg = QColor(80, 80, 80, 210)
            self._fg = QColor(235, 235, 235)
            self._arrow_color = QColor(220, 70, 70)
            self._outline = QColor(110, 110, 110)
        else:  # white/default
            self._bg = QColor(30, 30, 30, 200)
            self._fg = QColor(240, 240, 240)
            self._arrow_color = QColor(220, 60, 60)
            self._outline = QColor(85, 85, 85)
        self._font.setPointSize(max(8, min(int(font_size or 12), 20)))
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
        elif anchor_name in ("bottom_right", "bottom-right"):
            self.place_bottom_right(margin)
        else:  # Default to top right
            self.place_top_right(margin)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._size + 2 * self._padding, self._size + 2 * self._padding)

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

            # Center of widget
            cx = rect.width() / 2.0
            cy = rect.height() / 2.0

            # Apply rotation transform
            p.translate(cx, cy)
            p.rotate(-self._rotation_degrees)  # Negative because Qt rotates clockwise

            # Arrow parameters
            arrow_length = self._size * 0.35
            arrow_width = self._size * 0.15
            stem_width = self._size * 0.06

            # Draw arrow pointing up (north)
            # Arrow head (triangle)
            arrow_head = QPolygonF([
                QPointF(0, -arrow_length),  # Tip
                QPointF(-arrow_width, -arrow_length * 0.3),  # Left base
                QPointF(arrow_width, -arrow_length * 0.3),   # Right base
            ])
            p.setBrush(QBrush(self._arrow_color))
            p.setPen(QPen(self._arrow_color.darker(120), 1.0))
            p.drawPolygon(arrow_head)

            # Arrow stem (rectangle)
            stem_rect = QRectF(
                -stem_width / 2, -arrow_length * 0.3,
                stem_width, arrow_length * 0.6
            )
            p.fillRect(stem_rect, QBrush(self._fg))

            # Small circle at center
            center_radius = self._size * 0.05
            p.setBrush(QBrush(self._fg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(0, 0), center_radius, center_radius)

            # 'N' label at arrow tip
            p.setFont(self._font)
            p.setPen(QPen(self._fg))

            # Position 'N' above the arrow tip
            n_offset = arrow_length + 12
            n_rect = QRectF(-15, -n_offset - 10, 30, 20)
            p.drawText(n_rect, Qt.AlignmentFlag.AlignCenter, "N")

        finally:
            if p is not None:
                p.end()

    # ---------------- Dragging support ----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self.parent() is not None:
            try:
                parent = self.parent()
                new_pos = self.mapToParent(event.pos() - self._drag_offset)
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

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_offset = None
            self._user_placed = True
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ---------------- Placement helpers ----------------
    def place_top_right(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(parent.width() - self.width() - margin), int(margin))

    def place_top_left(self, margin: int = 16):
        if self.parent() is None:
            return
        self.move(int(margin), int(margin))

    def place_bottom_right(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(parent.width() - self.width() - margin), int(parent.height() - self.height() - margin))

    def place_bottom_left(self, margin: int = 16):
        if self.parent() is None:
            return
        parent = self.parent()
        if hasattr(parent, 'width'):
            self.move(int(margin), int(parent.height() - self.height() - margin))

    # ------------- helpers for integration -------------
    def has_user_placement(self) -> bool:
        return bool(self._user_placed)

    def clear_user_placement(self) -> None:
        self._user_placed = False

    def set_visible(self, visible: bool) -> None:
        """Simple visibility toggle for overlay managers."""
        super().setVisible(bool(visible))
