"""
Reusable header bar for persistent panels.

Provides a uniform title bar with minimise, refresh, clear, and close (hide) actions,
plus a stage/status label to indicate workflow provenance.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)


class PanelHeaderBar(QWidget):
    """Lightweight header that can optionally include controls; defaults to labels only."""

    minimise_toggled = pyqtSignal(bool)
    refresh_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, title: str, stage_text: str = "", parent: Optional[QWidget] = None, show_controls: bool = False):
        super().__init__(parent)
        self._minimised = False
        self.has_controls = show_controls

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 11pt;")
        self.title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.stage_label = QLabel(stage_text or "")
        self.stage_label.setStyleSheet("color: #777; font-size: 9pt; font-style: italic;")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.stage_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout.addWidget(self.title_label)
        layout.addWidget(self.stage_label)
        layout.addStretch(1)

        # Always create control buttons (visibility controlled by show_controls)
        def _make_btn(text: str, tooltip: str, slot):
            btn = QPushButton(text, self)
            btn.setToolTip(tooltip)
            btn.setFlat(True)
            btn.setFixedHeight(22)
            btn.setMinimumWidth(70)
            btn.setStyleSheet(
                """
                QPushButton { border: 1px solid #444; border-radius: 4px; padding: 2px 8px; color: #ddd; background: #2c2c2c; }
                QPushButton:hover { border-color: #888; color: #fff; background: #3a3a3a; }
                """
            )
            btn.clicked.connect(slot)
            return btn

        self.minimise_btn = _make_btn("Collapse", "Collapse/expand panel content", self._toggle_minimise)
        self.refresh_btn = _make_btn("Refresh", "Refresh / reload panel data", self.refresh_requested.emit)
        self.clear_btn = _make_btn("Clear", "Clear panel inputs (UI only)", self.clear_requested.emit)
        self.close_btn = _make_btn("Hide", "Hide panel", self.close_requested.emit)

        layout.addWidget(self.minimise_btn)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.close_btn)

        # Set initial visibility based on show_controls
        self.set_controls_visible(show_controls)



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
    def _toggle_minimise(self):
        if not self.minimise_btn:
            return
        self._minimised = not self._minimised
        self.minimise_btn.setText("Expand" if self._minimised else "Collapse")
        self.minimise_toggled.emit(self._minimised)

    def set_controls_visible(self, visible: bool):
        """Show/hide control buttons dynamically without restart."""
        self.has_controls = visible
        for btn in [self.minimise_btn, self.refresh_btn, self.clear_btn, self.close_btn]:
            if btn:
                btn.setVisible(visible)

    def set_stage_text(self, text: str):
        self.stage_label.setText(text or "")

    def set_refresh_visible(self, visible: bool):
        if self.refresh_btn:
            self.refresh_btn.setVisible(visible)

# Backwards-compatible alias
PanelHeader = PanelHeaderBar
