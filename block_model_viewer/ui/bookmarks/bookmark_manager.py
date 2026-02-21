"""
BookmarkManager - Centralized view bookmark management for GeoX.

Responsibilities:
- Save/load view bookmarks (camera positions)
- Persist bookmarks to QSettings
- Apply bookmarks to viewer renderer
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtWidgets import QStatusBar

logger = logging.getLogger(__name__)

SETTINGS_ORG = "GeoX"
SETTINGS_APP = "ViewBookmarks"


class BookmarkManager(QObject):
    """Manages view bookmarks for the 3D viewer."""

    def __init__(self, viewer=None, status_bar: Optional[QStatusBar] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._viewer = viewer
        self._status_bar = status_bar
        self._bookmarks: Dict[int, Dict[str, Any]] = {}

    @property
    def bookmarks(self) -> Dict[int, Dict[str, Any]]:
        """Access the bookmark dictionary (read/write)."""
        return self._bookmarks

    def bind_viewer(self, viewer) -> None:
        """Bind or rebind the viewer widget."""
        self._viewer = viewer

    def bind_status_bar(self, status_bar: QStatusBar) -> None:
        """Bind the status bar for messages."""
        self._status_bar = status_bar

    def save_bookmark(self, bookmark_num: int) -> None:
        """Save current camera position as a view bookmark."""
        if not self._viewer or not getattr(self._viewer, "renderer", None):
            self._show_status(f"Cannot save bookmark {bookmark_num}: No active view", 3000)
            return

        try:
            camera_info = self._viewer.renderer.get_camera_info()
            self._bookmarks[bookmark_num] = camera_info
            self.persist_bookmarks()
            self._show_status(
                f"✓ View bookmark {bookmark_num} saved (Ctrl+Shift+{bookmark_num})",
                3000,
            )
            logger.info(f"Saved view bookmark {bookmark_num}")
        except Exception as e:
            logger.error(f"Error saving bookmark {bookmark_num}: {e}")
            self._show_status(f"Error saving bookmark {bookmark_num}", 3000)

    def load_bookmark(self, bookmark_num: int) -> None:
        """Load and restore a saved view bookmark."""
        if bookmark_num not in self._bookmarks:
            self._show_status(
                f"Bookmark {bookmark_num} is empty (save with Ctrl+Shift+{bookmark_num})",
                3000,
            )
            return

        if not self._viewer or not getattr(self._viewer, "renderer", None):
            self._show_status(f"Cannot load bookmark {bookmark_num}: No active view", 3000)
            return

        try:
            camera_info = self._bookmarks[bookmark_num]
            self._viewer.renderer.set_camera_position(
                camera_info["position"],
                camera_info["focal_point"],
                camera_info.get("view_up", [0, 0, 1]),
            )
            self._show_status(
                f"✓ View bookmark {bookmark_num} restored (Ctrl+Shift+F{bookmark_num})",
                3000,
            )
            logger.info(f"Loaded view bookmark {bookmark_num}")
        except Exception as e:
            logger.error(f"Error loading bookmark {bookmark_num}: {e}")
            self._show_status(f"Error loading bookmark {bookmark_num}", 3000)

    def load_from_settings(self) -> None:
        """Load saved view bookmarks from persistent storage."""
        try:
            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            for i in range(1, 10):
                bookmark_data = settings.value(f"bookmark_{i}")
                if bookmark_data:
                    self._bookmarks[i] = bookmark_data
            logger.info(f"Loaded {len(self._bookmarks)} view bookmarks from storage")
        except Exception as e:
            logger.warning(f"Could not load view bookmarks: {e}")

    def persist_bookmarks(self) -> None:
        """Save view bookmarks to persistent storage."""
        try:
            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            for bookmark_num, camera_info in self._bookmarks.items():
                settings.setValue(f"bookmark_{bookmark_num}", camera_info)
            logger.debug("Persisted view bookmarks to storage")
        except Exception as e:
            logger.warning(f"Could not persist view bookmarks: {e}")

    def set_bookmarks(self, bookmarks: Dict[int, Dict[str, Any]]) -> None:
        """Replace bookmarks in memory and persist them."""
        if not isinstance(bookmarks, dict):
            return
        self._bookmarks = bookmarks
        self.persist_bookmarks()

    def get_bookmarks(self) -> Dict[int, Dict[str, Any]]:
        """Return a shallow copy of bookmarks."""
        return dict(self._bookmarks)

    def _show_status(self, message: str, timeout_ms: int = 3000) -> None:
        """Show a status bar message."""
        if self._status_bar is not None:
            try:
                self._status_bar.showMessage(message, timeout_ms)
            except Exception:
                pass

