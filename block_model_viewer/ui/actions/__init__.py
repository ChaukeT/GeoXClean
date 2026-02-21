"""Action handlers for MainWindow menu actions.

This module contains action handlers extracted from MainWindow.
Handlers delegate to AppController for business logic and only handle UI concerns.
"""

# File actions
from .file_actions import (
    handle_open_file,
    handle_load_file,
    handle_new_project,
    handle_open_project,
    handle_save_project,
    handle_save_project_as,
    handle_export_screenshot,
    handle_export_filtered_data,
    handle_export_model,
    handle_clear_scene,
)

__all__ = [
    'handle_open_file',
    'handle_load_file',
    'handle_new_project',
    'handle_open_project',
    'handle_save_project',
    'handle_save_project_as',
    'handle_export_screenshot',
    'handle_export_filtered_data',
    'handle_export_model',
    'handle_clear_scene',
]

