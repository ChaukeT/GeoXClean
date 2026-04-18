"""
Panel Micro-Widgets — Small reusable UI components for docked panels.

Contents:
    RadiusPreview           — Circle that scales with a radius value
    CountBadgeButton        — QPushButton with a count badge overlay
    ConfirmButton           — Requires double-click for destructive actions
    ActiveIndicatorButton   — Button with "active" visual state
    BusyOverlay             — Semi-transparent spinner for long operations
"""

from __future__ import annotations

from typing import Optional
import logging

from PyQt6.QtCore import (
    Qt, QRectF, QRect, QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QFont, QPaintEvent,
)
from PyQt6.QtWidgets import (
    QWidget, QPushButton,
)

from .modern_styles import ModernColors

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# RADIUS PREVIEW — scaled circle that tracks a slider value
# ═══════════════════════════════════════════════════════════════════

class RadiusPreview(QWidget):
    """
    Small preview showing a filled circle proportional to the radius value.

    The circle scales between min_radius and max_radius within a fixed
    widget size (40x40px default).

    Usage:
        preview = RadiusPreview(min_radius=0.1, max_radius=5.0)
        slider.valueChanged.connect(preview.set_radius)
    """

    def __init__(
        self,
        min_radius: float = 0.1,
        max_radius: float = 5.0,
        size: int = 40,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._min = min_radius
        self._max = max_radius
        self._radius = min_radius
        self._widget_size = size

        self.setFixedSize(size, size)
        self.setToolTip("Drillhole radius preview")

    def set_radius(self, value: float) -> None:
        """Update the displayed radius."""
        self._radius = max(self._min, min(self._max, value))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        bg = QColor(ModernColors.ELEVATED_BG)
        painter.setBrush(bg)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 6, 6)

        # Circle — scale radius to widget size
        t = (self._radius - self._min) / max(1e-9, self._max - self._min)
        min_px = 3.0
        max_px = (self._widget_size - 8) / 2.0
        circle_r = min_px + t * (max_px - min_px)

        cx = self.width() / 2.0
        cy = self.height() / 2.0

        # Outer ring (semi-transparent fill)
        accent = QColor(ModernColors.ACCENT_PRIMARY)
        fill = QColor(accent)
        fill.setAlpha(64)
        painter.setPen(QPen(accent, 1.0))
        painter.setBrush(fill)
        painter.drawEllipse(QRectF(cx - circle_r, cy - circle_r, circle_r * 2, circle_r * 2))

        # Center dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(QRectF(cx - 2, cy - 2, 4, 4))

        painter.end()


# ═══════════════════════════════════════════════════════════════════
# COUNT BADGE BUTTON — button with selection count in text
# ═══════════════════════════════════════════════════════════════════

class CountBadgeButton(QPushButton):
    """
    QPushButton that shows a count alongside its label.

    Displays as "Plot 3D (247)" when count > 0, plain "Plot 3D" when 0.

    Usage:
        btn = CountBadgeButton("Plot 3D", style="primary")
        btn.set_count(247)
    """

    def __init__(
        self,
        text: str = "",
        style: str = "primary",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._base_text = text
        self._count: int = 0

        self.setObjectName(f"btn_{style}")
        self.setMinimumHeight(32 if style == "primary" else 28)

    def set_count(self, count: int) -> None:
        """Update the badge count. Hidden when count is 0."""
        self._count = count
        if count > 0:
            self.setText(f"{self._base_text} ({count})")
        else:
            self.setText(self._base_text)


# ═══════════════════════════════════════════════════════════════════
# CONFIRM BUTTON — double-action for destructive operations
# ═══════════════════════════════════════════════════════════════════

class ConfirmButton(QPushButton):
    """
    Button that requires confirmation before executing.

    First click: changes text to confirm prompt with danger styling.
    Second click within timeout: fires confirmed signal.
    Timeout: auto-resets to normal state.

    Usage:
        btn = ConfirmButton("Clear All", confirm_text="Confirm Clear?")
        btn.confirmed.connect(self._on_clear)
    """

    confirmed = pyqtSignal()

    def __init__(
        self,
        text: str = "",
        confirm_text: Optional[str] = None,
        timeout_ms: int = 3000,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)
        self._base_text = text
        self._confirm_text = confirm_text or f"{text}?"
        self._timeout_ms = timeout_ms
        self._armed = False

        self.setObjectName("btn_danger")
        self.setMinimumHeight(28)

        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self._disarm)

        self.clicked.connect(self._on_click)

    def _on_click(self):
        if self._armed:
            self._reset_timer.stop()
            self._disarm()
            self.confirmed.emit()
        else:
            self._armed = True
            self.setText(self._confirm_text)
            self.setObjectName("btn_danger_armed")
            self.style().unpolish(self)
            self.style().polish(self)
            self._reset_timer.start(self._timeout_ms)

    def _disarm(self):
        self._armed = False
        self.setText(self._base_text)
        self.setObjectName("btn_danger")
        self.style().unpolish(self)
        self.style().polish(self)


# ═══════════════════════════════════════════════════════════════════
# ACTIVE INDICATOR BUTTON — shows which preset is currently active
# ═══════════════════════════════════════════════════════════════════

class ActiveIndicatorButton(QPushButton):
    """
    QPushButton that visually shows active/inactive state.

    When active, uses the toggle checked style. When inactive, secondary.

    Usage:
        btn = ActiveIndicatorButton("Top")
        btn.set_active(True)   # accent-colored active state
        btn.set_active(False)  # normal secondary button
    """

    def __init__(self, text: str = "", tooltip: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self._active = False
        self.setObjectName("btn_secondary")
        self.setMinimumHeight(28)
        self.setCheckable(True)
        if tooltip:
            self.setToolTip(tooltip)

    def set_active(self, active: bool) -> None:
        """Mark this button as the currently active preset."""
        self._active = active
        self.setChecked(active)
        if active:
            self.setObjectName("btn_toggle")
        else:
            self.setObjectName("btn_secondary")
        self.style().unpolish(self)
        self.style().polish(self)

    def is_active(self) -> bool:
        return self._active


# ═══════════════════════════════════════════════════════════════════
# BUSY OVERLAY — spinner shown during long operations
# ═══════════════════════════════════════════════════════════════════

class BusyOverlay(QWidget):
    """
    Semi-transparent overlay with a spinning arc indicator.

    Place over any widget to indicate a long-running operation.
    Blocks mouse events to prevent interaction during processing.

    Usage:
        self.busy = BusyOverlay(parent=self)
        self.busy.show_busy("Updating colors...")
        # ... do work ...
        self.busy.hide_busy()
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._angle = 0
        self._message = ""
        self._spinning = False

        self.setVisible(False)
        # Block clicks from passing through
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(30)  # ~33 fps
        self._spin_timer.timeout.connect(self._tick)

    def show_busy(self, message: str = "Working...") -> None:
        """Show the overlay with a message and start spinning."""
        self._message = message
        self._spinning = True
        self._angle = 0

        if self.parent():
            self.setGeometry(self.parent().rect())

        self.setVisible(True)
        self.raise_()
        self._spin_timer.start()

    def hide_busy(self) -> None:
        """Hide the overlay and stop the spinner."""
        self._spinning = False
        self._spin_timer.stop()
        self.setVisible(False)

    def _tick(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        cx = self.width() / 2.0
        cy = self.height() / 2.0

        # Spinner arc
        size = 32
        rect = QRectF(cx - size / 2, cy - size / 2 - 10, size, size)
        pen = QPen(QColor(ModernColors.ACCENT_PRIMARY), 3.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, int(self._angle * 16), int(270 * 16))

        # Message
        if self._message:
            painter.setPen(QColor(ModernColors.TEXT_PRIMARY))
            font = QFont()
            font.setPixelSize(12)
            painter.setFont(font)
            text_rect = QRect(0, int(cy + size / 2), self.width(), 24)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._message)

        painter.end()


# ═══════════════════════════════════════════════════════════════════
# QSS for micro-widgets (append to global stylesheet)
# ═══════════════════════════════════════════════════════════════════

MICRO_WIDGET_QSS = f"""
/* Confirm button styles */
QPushButton#btn_danger {{
    background-color: transparent;
    color: {ModernColors.TEXT_SECONDARY};
    border: 1px solid {ModernColors.BORDER};
    border-radius: 4px;
    padding: 4px 12px;
}}
QPushButton#btn_danger:hover {{
    border-color: #e74c3c;
    color: #e74c3c;
}}
QPushButton#btn_danger_armed {{
    background-color: #e74c3c;
    color: white;
    border: 1px solid #c0392b;
    border-radius: 4px;
    padding: 4px 12px;
}}
QPushButton#btn_toggle:checked {{
    background-color: {ModernColors.ACCENT_PRIMARY};
    color: white;
    border: 1px solid {ModernColors.ACCENT_PRIMARY};
    border-radius: 4px;
}}
"""