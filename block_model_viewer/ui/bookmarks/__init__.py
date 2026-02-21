"""
Bookmarks module for GeoX.

Centralizes view bookmark management:
- Save/load camera bookmarks
- Persistence to QSettings
- Apply saved camera positions to the renderer
"""

from .bookmark_manager import BookmarkManager

__all__ = ["BookmarkManager"]
