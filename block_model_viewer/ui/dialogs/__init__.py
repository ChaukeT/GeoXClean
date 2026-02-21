"""
Dialog management module for GeoX.

Centralizes dialog lifecycle management:
- Dialog tracking
- Geometry persistence (save/restore)
- Unsaved changes protection
- Show/hide/create helpers
"""

from .dialog_manager import DialogManager

__all__ = ['DialogManager']

