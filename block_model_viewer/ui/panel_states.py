"""
Panel State Overlays — empty, loading, and error states for panels.

Provides overlay widgets that temporarily replace panel content to show
contextual feedback: no data loaded, analysis running, or error with retry.

Usage in BasePanel:
    # In BasePanel (add these methods):
    def show_empty_state(self, message="No data loaded", icon="📊"):
        self._state_overlay.show_empty(message, icon)

    def show_loading_state(self, message="Processing..."):
        self._state_overlay.show_loading(message)

    def show_error_state(self, message="An error occurred", retry_callback=None):
        self._state_overlay.show_error(message, retry_callback)

    def hide_state_overlay(self):
        self._state_overlay.hide()

    # In panel code:
    def on_run_clicked(self):
        self.show_loading_state("Running kriging estimation...")
        self.controller.run_task('kriging', params, callback=self._on_complete)

    def _on_complete(self, result):
        self.hide_state_overlay()
        self.display_results(result)

    def _on_error(self, msg):
        self.show_error_state(msg, retry_callback=self.on_run_clicked)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QMovie, QPainter, QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .modern_styles import ModernColors

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SPINNER WIDGET (pure Qt, no external assets)
# ═══════════════════════════════════════════════════════════════════

class _SpinnerWidget(QWidget):
    """Animated loading spinner using QPainter (no GIF/asset needed)."""

    def __init__(self, size: int = 32, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._num_dots = 8
        self._color = QColor(ModernColors.ACCENT_PRIMARY)

    def start(self):
        self._timer.start(80)

    def stop(self):
        self._timer.stop()

    def _rotate(self):
        self._angle = (self._angle + 1) % self._num_dots
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        radius = min(cx, cy) * 0.7
        dot_radius = min(cx, cy) * 0.12

        for i in range(self._num_dots):
            import math
            angle = 2 * math.pi * i / self._num_dots
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)

            # Fade dots based on distance from active position
            distance = (i - self._angle) % self._num_dots
            opacity = max(0.15, 1.0 - (distance / self._num_dots))

            color = QColor(self._color)
            color.setAlphaF(opacity)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(x - dot_radius), int(y - dot_radius),
                                int(dot_radius * 2), int(dot_radius * 2))
        painter.end()


# ═══════════════════════════════════════════════════════════════════
# STATE OVERLAY — the main widget that panels use
# ═══════════════════════════════════════════════════════════════════

class PanelStateOverlay(QFrame):
    """
    Semi-transparent overlay that covers panel content to show state.

    States:
    - EMPTY:   icon + message + optional action button
    - LOADING: spinner + message + optional cancel button
    - ERROR:   error icon + message + optional retry button

    Install over any QWidget:
        overlay = PanelStateOverlay(parent=my_panel)
        overlay.show_empty("No block model loaded")
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("panelStateOverlay")

        # Overlay covers entire parent
        if parent:
            self.setGeometry(parent.rect())

        # Semi-transparent background — built dynamically so theme changes apply
        self._apply_overlay_style()

        # Main layout (centered)
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.setSpacing(12)

        # Icon / spinner area
        self._icon_label = QLabel()
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setStyleSheet("font-size: 48px; background: transparent;")
        self._layout.addWidget(self._icon_label)

        self._spinner = _SpinnerWidget(40, self)
        self._spinner.hide()
        self._layout.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        # Message
        self._message_label = QLabel()
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet(
            f"font-size: 14px; color: {ModernColors.TEXT_SECONDARY}; background: transparent;"
            "padding: 0 24px;"
        )
        self._layout.addWidget(self._message_label)

        # Detail / sub-message
        self._detail_label = QLabel()
        self._detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            f"font-size: 12px; color: {ModernColors.TEXT_HINT}; background: transparent;"
            "padding: 0 24px;"
        )
        self._detail_label.hide()
        self._layout.addWidget(self._detail_label)

        # Action button
        self._action_button = QPushButton()
        self._action_button.setObjectName("stateOverlayButton")
        self._action_button.setStyleSheet(
            f"QPushButton#stateOverlayButton {{"
            f"  background-color: {ModernColors.ACCENT_PRIMARY}; color: {ModernColors.TEXT_PRIMARY};"
            f"  border: none; border-radius: 4px;"
            f"  padding: 8px 24px; font-size: 13px;"
            f"}}"
            f"QPushButton#stateOverlayButton:hover {{"
            f"  background-color: {ModernColors.ACCENT_HOVER};"
            f"}}"
        )
        self._action_button.hide()
        self._layout.addWidget(self._action_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Start hidden
        self.hide()
        self._current_callback = None

    # ── Public API ────────────────────────────────────────────────

    def show_empty(
        self,
        message: str = "No data loaded",
        icon: str = "📊",
        detail: str = "",
        action_text: str = "",
        action_callback: Optional[Callable] = None,
    ):
        """Show empty state — no data, nothing to display."""
        self._reset()
        self._icon_label.setText(icon)
        self._icon_label.show()
        self._message_label.setText(message)

        if detail:
            self._detail_label.setText(detail)
            self._detail_label.show()

        if action_text and action_callback:
            self._action_button.setText(action_text)
            self._current_callback = action_callback
            self._action_button.clicked.connect(action_callback)
            self._action_button.show()

        self._show_with_fade()

    def show_loading(
        self,
        message: str = "Processing...",
        detail: str = "",
        cancel_callback: Optional[Callable] = None,
    ):
        """Show loading state — spinner + message."""
        self._reset()
        self._icon_label.hide()
        self._spinner.show()
        self._spinner.start()
        self._message_label.setText(message)

        if detail:
            self._detail_label.setText(detail)
            self._detail_label.show()

        if cancel_callback:
            self._action_button.setText("Cancel")
            self._action_button.setStyleSheet(
                f"QPushButton#stateOverlayButton {{"
                f"  background-color: {ModernColors.BORDER}; color: {ModernColors.TEXT_PRIMARY};"
                f"  border: none; border-radius: 4px;"
                f"  padding: 8px 24px; font-size: 13px;"
                f"}}"
                f"QPushButton#stateOverlayButton:hover {{"
                f"  background-color: {ModernColors.BORDER_LIGHT};"
                f"}}"
            )
            self._current_callback = cancel_callback
            self._action_button.clicked.connect(cancel_callback)
            self._action_button.show()

        self._show_with_fade()

    def show_error(
        self,
        message: str = "An error occurred",
        detail: str = "",
        retry_callback: Optional[Callable] = None,
    ):
        """Show error state — error icon + message + optional retry."""
        self._reset()
        self._icon_label.setText("⚠️")
        self._icon_label.show()
        self._message_label.setText(message)
        self._message_label.setStyleSheet(
            f"font-size: 14px; color: {ModernColors.ERROR}; background: transparent;"
            "padding: 0 24px; font-weight: bold;"
        )

        if detail:
            self._detail_label.setText(detail)
            self._detail_label.show()

        if retry_callback:
            self._action_button.setText("Retry")
            self._action_button.setStyleSheet(
                f"QPushButton#stateOverlayButton {{"
                f"  background-color: {ModernColors.ERROR}; color: {ModernColors.TEXT_PRIMARY};"
                f"  border: none; border-radius: 4px;"
                f"  padding: 8px 24px; font-size: 13px;"
                f"}}"
                f"QPushButton#stateOverlayButton:hover {{"
                f"  background-color: {ModernColors.ERROR};"
                f"}}"
            )
            self._current_callback = retry_callback
            self._action_button.clicked.connect(retry_callback)
            self._action_button.show()

        self._show_with_fade()

    def update_loading_message(self, message: str):
        """Update message during loading (e.g., progress)."""
        self._message_label.setText(message)

    def update_loading_detail(self, detail: str):
        """Update detail during loading (e.g., "Step 3 of 5")."""
        self._detail_label.setText(detail)
        self._detail_label.show()

    # ── Override ──────────────────────────────────────────────────

    def hide(self):
        """Hide overlay and clean up."""
        self._spinner.stop()
        super().hide()

    def resizeEvent(self, event):
        """Keep overlay same size as parent."""
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)

    # ── Internal ──────────────────────────────────────────────────

    def _reset(self):
        """Reset all elements to default state."""
        self._spinner.stop()
        self._spinner.hide()
        self._icon_label.show()
        self._icon_label.setStyleSheet("font-size: 48px; background: transparent;")
        self._message_label.setStyleSheet(
            f"font-size: 14px; color: {ModernColors.TEXT_SECONDARY}; background: transparent;"
            "padding: 0 24px;"
        )
        self._detail_label.hide()
        self._detail_label.setText("")

        # Disconnect previous callback
        if self._current_callback:
            try:
                self._action_button.clicked.disconnect(self._current_callback)
            except Exception:
                pass
            self._current_callback = None
        self._action_button.hide()

    def _show_with_fade(self):
        """Show with a quick fade-in."""
        # Resize to parent
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()


# ═══════════════════════════════════════════════════════════════════
# BASE PANEL INTEGRATION MIXIN
# ═══════════════════════════════════════════════════════════════════

class PanelStateMixin:
    """
    Mixin for BasePanel to add state overlay support.

    Add to BasePanel's inheritance:
        class BasePanel(QWidget, PanelStateMixin):
            ...

    Then in __init__:
        self._init_state_overlay()

    Usage in any panel:
        self.show_empty_state("Load a block model to begin")
        self.show_loading_state("Running variogram analysis...")
        self.show_error_state("Failed to compute", retry_callback=self.run)
        self.hide_state_overlay()
    """

    _state_overlay: Optional[PanelStateOverlay] = None

    def _init_state_overlay(self):
        """Initialize the state overlay. Call in __init__."""
        self._state_overlay = PanelStateOverlay(parent=self)

    def show_empty_state(
        self,
        message: str = "No data loaded",
        icon: str = "📊",
        detail: str = "",
        action_text: str = "",
        action_callback: Optional[Callable] = None,
    ):
        """Show empty state overlay."""
        if self._state_overlay is None:
            self._init_state_overlay()
        self._state_overlay.show_empty(message, icon, detail, action_text, action_callback)

    def show_loading_state(
        self,
        message: str = "Processing...",
        detail: str = "",
        cancel_callback: Optional[Callable] = None,
    ):
        """Show loading state overlay with spinner."""
        if self._state_overlay is None:
            self._init_state_overlay()
        self._state_overlay.show_loading(message, detail, cancel_callback)

    def show_error_state(
        self,
        message: str = "An error occurred",
        detail: str = "",
        retry_callback: Optional[Callable] = None,
    ):
        """Show error state overlay with optional retry button."""
        if self._state_overlay is None:
            self._init_state_overlay()
        self._state_overlay.show_error(message, detail, retry_callback)

    def update_loading_message(self, message: str):
        """Update the loading overlay message (e.g., progress)."""
        if self._state_overlay:
            self._state_overlay.update_loading_message(message)

    def update_loading_detail(self, detail: str):
        """Update the loading overlay detail text."""
        if self._state_overlay:
            self._state_overlay.update_loading_detail(detail)

    def hide_state_overlay(self):
        """Hide any active state overlay, revealing panel content."""
        if self._state_overlay:
            self._state_overlay.hide()

    def resizeEvent(self, event):
        """Keep overlay sized to panel."""
        super().resizeEvent(event)
        if self._state_overlay and self._state_overlay.isVisible():
            self._state_overlay.setGeometry(self.rect())
