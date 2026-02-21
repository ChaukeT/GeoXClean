"""Layout management module for MainWindow dock setup and workspace management."""

from .dock_setup import setup_docks, setup_toolbar
from .workspace import (
    load_workspace_layout,
    reset_workspace_layout,
    save_workspace_layout,
    load_workspace_layout_file,
)

__all__ = [
    'setup_docks',
    'setup_toolbar',
    'load_workspace_layout',
    'reset_workspace_layout',
    'save_workspace_layout',
    'load_workspace_layout_file',
]

