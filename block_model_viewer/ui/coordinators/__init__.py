"""
Coordinators — extracted from MainWindow to break the god object.

Each coordinator owns a focused slice of MainWindow's responsibilities:
    MenuCoordinator      — menu bar build, shortcut validation, command palette
    SignalCoordinator     — signal wiring between UI ↔ controller ↔ panels
    WorkspaceCoordinator  — layout persistence, bookmarks, session templates
    FileCoordinator       — file open/load/save, recent files, export, clear scene
"""

from .menu_coordinator import MenuCoordinator
from .signal_coordinator import SignalCoordinator
from .workspace_coordinator import WorkspaceCoordinator
from .file_coordinator import FileCoordinator

__all__ = [
    'MenuCoordinator',
    'SignalCoordinator',
    'WorkspaceCoordinator',
    'FileCoordinator',
]
