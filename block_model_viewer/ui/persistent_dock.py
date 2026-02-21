"""
Persistent, workflow-aware dock wrapper for drillhole panels.

Responsibilities:
- Keep a single persistent instance of each panel (hide on close).
- Provide a consistent header bar with minimise, refresh, clear, close actions.
- Wrap content in a scrollable area with standard margins.
- Remember geometry per-dock in a simple user_settings.json file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QEvent, QSettings
from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QSizePolicy,
)

from .panel_header import PanelHeaderBar


SETTINGS_PATH = Path("user_settings.json")


def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass


class PersistentDockWidget(QDockWidget):
    """Dock widget that hides instead of destroying and wraps a panel with a standard header."""

    def __init__(self, key: str, title: str, content: QWidget, stage_text: str = "", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self.setObjectName(key)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        # Enable native title bar controls (close/min/max on float) and allow docking/floating
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        # Request native min/max/close buttons when floating
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self.content_widget = content

        # Standard container with header + scroll area
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Read preference for showing header controls (default: True for better UX)
        s_panels = QSettings("GeoX", "Panels")
        show_header_controls = s_panels.value("show_header_controls", True, type=bool)

        self.header = PanelHeaderBar(title, stage_text, show_controls=show_header_controls)
        # Always connect signals (buttons are always created, just visibility varies)
        self.header.minimise_toggled.connect(self._on_minimise_toggled)
        self.header.clear_requested.connect(self._on_clear_requested)
        self.header.refresh_requested.connect(self._on_refresh_requested)
        self.header.close_requested.connect(self._on_close_requested)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setWidget(content)

        layout.addWidget(self.header)
        layout.addWidget(scroll)

        self.setWidget(container)

        # Apply stored geometry if available
        self._restore_geometry()

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QEvent):
        # Hide instead of closing/destroying
        self.hide()
        event.ignore()
        self._save_geometry()

    def _on_close_requested(self):
        self.hide()
        self._save_geometry()

    def _on_minimise_toggled(self, minimised: bool):
        if self.content_widget:
            self.content_widget.setVisible(not minimised)

    def _on_clear_requested(self):
        # Allow panels to implement their own UI-only clear logic
        target = getattr(self, "content_widget", None)
        if target is None:
            return
        for attr in ("clear_panel", "clear_ui", "clear"):
            fn = getattr(target, attr, None)
            if callable(fn):
                try:
                    fn()  # UI-only reset expected
                except Exception:
                    pass
                break

    def _on_refresh_requested(self):
        target = getattr(self, "content_widget", None)
        if target is None:
            return
        for attr in ("refresh_panel", "refresh_ui", "reload"):
            fn = getattr(target, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _save_geometry(self):
        settings = _load_settings()
        settings.setdefault("docks", {})
        settings["docks"][self.objectName()] = {
            "floating": self.isFloating(),
            "geometry": bytes(self.saveGeometry().toHex()).decode("ascii"),
        }
        _save_settings(settings)

    def _restore_geometry(self):
        settings = _load_settings().get("docks", {})
        info = settings.get(self.objectName())
        if not info:
            return
        try:
            geom_hex = info.get("geometry")
            if geom_hex:
                self.restoreGeometry(bytes.fromhex(geom_hex))
            if info.get("floating"):
                self.setFloating(True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Dynamic control visibility
    # ------------------------------------------------------------------ #
    def set_header_controls_visible(self, visible: bool):
        """Toggle header controls visibility without restart."""
        if self.header and hasattr(self.header, 'set_controls_visible'):
            self.header.set_controls_visible(visible)
