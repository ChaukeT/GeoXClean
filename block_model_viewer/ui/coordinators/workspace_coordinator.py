"""
Workspace Coordinator — layout, bookmarks, dialog persistence, sessions.

Extracted from MainWindow. Owns all state persistence: window geometry,
dock layout, view bookmarks, dialog positions, and session save/restore.

Usage:
    # In MainWindow.__init__:
    self._workspace_coordinator = WorkspaceCoordinator(self)

    # Delegates:
    def _restore_state(self):
        self._workspace_coordinator.restore_state()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class WorkspaceCoordinator(QObject):
    """
    Owns all workspace/layout persistence for MainWindow.

    Responsibilities:
    - Window geometry save/restore
    - Dock widget state save/restore
    - View bookmarks (camera positions)
    - Dialog geometry persistence (via DialogManager)
    - Session save/restore (last file, camera state)
    """

    def __init__(self, main_window: 'MainWindow'):
        super().__init__(main_window)
        self._mw = main_window

    # ═══════════════════════════════════════════════════════════════
    # WINDOW STATE
    # ═══════════════════════════════════════════════════════════════

    def restore_state(self):
        """Restore window geometry and dock layout from QSettings."""
        try:
            settings = QSettings("GeoX", "Layout")
            geometry = settings.value("geometry")
            state = settings.value("state")

            if geometry:
                self._mw.restoreGeometry(geometry)
            if state:
                self._mw.restoreState(state)

            logger.info("Restored window layout")
        except Exception as e:
            logger.warning(f"Failed to restore window layout: {e}")

    def save_state(self):
        """Save window geometry and dock layout to QSettings."""
        try:
            settings = QSettings("GeoX", "Layout")
            settings.setValue("geometry", self._mw.saveGeometry())
            settings.setValue("state", self._mw.saveState())

            # Save dialog geometries
            self.save_dialog_geometries()

            logger.info("Saved window layout")
        except Exception as e:
            logger.warning(f"Failed to save window layout: {e}")

    def save_layout(self):
        """Save window geometry and dock layout (explicit call)."""
        self.save_state()

    # ═══════════════════════════════════════════════════════════════
    # BOOKMARKS
    # ═══════════════════════════════════════════════════════════════

    def save_bookmark(self, bookmark_num: int):
        """Save current camera position as a view bookmark."""
        mw = self._mw
        if mw.bookmarks is not None:
            mw.bookmarks.save_bookmark(bookmark_num)

    def load_bookmark(self, bookmark_num: int):
        """Load and restore a saved view bookmark."""
        mw = self._mw
        if mw.bookmarks is not None:
            mw.bookmarks.load_bookmark(bookmark_num)

    def load_bookmarks(self):
        """Load saved view bookmarks from persistent storage."""
        mw = self._mw
        if mw.bookmarks is not None:
            mw.bookmarks.load_from_settings()

    def persist_bookmarks(self):
        """Save view bookmarks to persistent storage."""
        mw = self._mw
        if mw.bookmarks is not None:
            mw.bookmarks.persist_bookmarks()

    # ═══════════════════════════════════════════════════════════════
    # DIALOG PERSISTENCE
    # ═══════════════════════════════════════════════════════════════

    def save_dialog_geometries(self):
        """Save geometry for all dialog windows."""
        mw = self._mw
        if mw.dialogs is not None:
            mw.dialogs.save_all_geometries()

    def restore_dialog_geometries(self):
        """Restore geometry for dialog windows (auto-restored on open)."""
        logger.debug("Dialog geometry restoration ready")

    def is_dialog_valid(self, dialog) -> bool:
        """Check if a dialog widget is still valid."""
        mw = self._mw
        if mw.dialogs is not None:
            return mw.dialogs.is_valid(dialog)
        if dialog is None:
            return False
        try:
            _ = dialog.isVisible()
            _ = dialog.windowTitle()
            return True
        except (RuntimeError, AttributeError):
            return False

    def show_or_create_dialog(self, dialog_attr_name: str, create_callback):
        """Show existing dialog or create new one via DialogManager."""
        mw = self._mw
        if mw.dialogs is not None:
            return mw.dialogs.show_or_create(
                dialog_attr_name,
                create_callback,
                attr_holder=mw,
                attr_name=dialog_attr_name,
            )
        return create_callback()

    def setup_dialog_persistence(self, dialog, dialog_name: str,
                                  panel_name: str = None):
        """Setup persistence for a dialog."""
        mw = self._mw
        if mw.dialogs is not None:
            mw.dialogs.setup_persistence(dialog, dialog_name, panel_name)
            if dialog not in mw._open_panels:
                mw._open_panels.append(dialog)
        else:
            if dialog not in mw._open_panels:
                mw._open_panels.append(dialog)

