"""
Theme Manager for BlockModelViewer.

Handles loading and applying QSS themes, color palettes, and theme switching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


class ThemeManager(QObject):
    """
    Manages application themes, color palettes, and QSS stylesheets.
    
    Provides:
    - Theme loading (light/dark)
    - QSS stylesheet application
    - Color palette access for visualization
    - Theme change notifications
    """
    
    theme_changed = pyqtSignal(str)  # Emits theme name when changed
    
    def __init__(self, app: Optional[QApplication] = None):
        """
        Initialize ThemeManager.
        
        Args:
            app: QApplication instance (optional, can be set later)
        """
        super().__init__()
        self.app = app
        
        # Get assets directory
        self._assets_dir = Path(__file__).parent.parent / "assets"
        self._themes_dir = self._assets_dir / "themes"
        self._branding_dir = self._assets_dir / "branding"
        
        # Current theme state
        self._current_theme: str = "light"
        self._color_palette: Dict[str, List[str]] = {}
        
        # Load color palette
        self._load_color_palette()
        
        logger.info(f"ThemeManager initialized (themes_dir={self._themes_dir})")
    
    def _load_color_palette(self) -> None:
        """Load color palette from JSON file."""
        palette_file = self._themes_dir / "color_palette.json"
        try:
            if palette_file.exists():
                with open(palette_file, 'r', encoding='utf-8') as f:
                    self._color_palette = json.load(f)
                logger.info(f"Loaded color palette from {palette_file}")
            else:
                # Use defaults if file doesn't exist
                self._color_palette = {
                    "geology": ["#d8b365", "#f5f5dc", "#8c510a"],
                    "resource": ["#1f78b4", "#33a02c", "#e31a1c"],
                    "uncertainty": ["#54278f", "#756bb1", "#bcbddc"],
                    "esg": ["#238b45", "#78c679", "#c2e699"],
                    "pit": ["#b2182b", "#ef8a62", "#fddbc7"],
                    "underground": ["#2c7fb8", "#7fcdbb", "#edf8b1"]
                }
                logger.warning(f"Color palette file not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading color palette: {e}", exc_info=True)
            # Use defaults on error
            self._color_palette = {
                "geology": ["#d8b365", "#f5f5dc", "#8c510a"],
                "resource": ["#1f78b4", "#33a02c", "#e31a1c"],
                "uncertainty": ["#54278f", "#756bb1", "#bcbddc"],
                "esg": ["#238b45", "#78c679", "#c2e699"],
                "pit": ["#b2182b", "#ef8a62", "#fddbc7"],
                "underground": ["#2c7fb8", "#7fcdbb", "#edf8b1"]
            }
    
    def load_theme(self, name: str) -> None:
        """
        Load a theme by name.

        Args:
            name: Theme name ('light' or 'dark')
        """
        if name not in ['light', 'dark']:
            logger.warning(f"Unknown theme '{name}', defaulting to 'light'")
            name = 'light'

        self._current_theme = name
        logger.info(f"Loading theme: {name}")

        # Sync modern_styles module with new theme
        try:
            from .modern_styles import set_current_theme
            set_current_theme(name)
            logger.debug(f"Synced modern_styles to theme: {name}")
        except ImportError:
            logger.warning("Could not import modern_styles for theme sync")

        # Emit signal for theme change
        self.theme_changed.emit(name)
    
    def apply_theme(self, app: Optional[QApplication] = None) -> None:
        """
        Apply the current theme's QSS stylesheet to the application.
        
        Args:
            app: QApplication instance (uses self.app if not provided)
        """
        if app is None:
            app = self.app
        
        if app is None:
            logger.warning("No QApplication instance available for theme application")
            return
        
        qss_file = self._themes_dir / f"{self._current_theme}.qss"
        
        try:
            if qss_file.exists():
                with open(qss_file, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
                
                # Replace relative paths with absolute paths for resources
                stylesheet = self._resolve_resource_paths(stylesheet)
                
                app.setStyleSheet(stylesheet)
                logger.info(f"Applied theme '{self._current_theme}' from {qss_file}")
            else:
                logger.warning(f"QSS file not found: {qss_file}")
                app.setStyleSheet("")  # Clear stylesheet
        except Exception as e:
            logger.error(f"Error applying theme: {e}", exc_info=True)
            app.setStyleSheet("")  # Clear stylesheet on error
    
    def _resolve_resource_paths(self, stylesheet: str) -> str:
        """
        Resolve relative resource paths in QSS to absolute paths.
        
        Args:
            stylesheet: QSS stylesheet content
            
        Returns:
            Stylesheet with resolved paths
        """
        # For now, just return as-is
        # In a full implementation, you might want to resolve url() references
        return stylesheet
    
    def get_color(self, category: str, index: int = 0) -> QColor:
        """
        Get a color from the palette by category and index.
        
        Args:
            category: Color category ('geology', 'resource', 'uncertainty', 'esg', 'pit', 'underground')
            index: Index within the category (default: 0)
            
        Returns:
            QColor instance
        """
        if category not in self._color_palette:
            logger.warning(f"Unknown color category '{category}', using default")
            return QColor(128, 128, 128)  # Gray default
        
        colors = self._color_palette[category]
        if index < 0 or index >= len(colors):
            logger.warning(f"Index {index} out of range for category '{category}', using index 0")
            index = 0
        
        color_str = colors[index]
        return QColor(color_str)
    
    def get_color_list(self, category: str) -> List[QColor]:
        """
        Get all colors for a category.
        
        Args:
            category: Color category name
            
        Returns:
            List of QColor instances
        """
        if category not in self._color_palette:
            logger.warning(f"Unknown color category '{category}'")
            return [QColor(128, 128, 128)]
        
        return [QColor(c) for c in self._color_palette[category]]
    
    def current_theme(self) -> str:
        """
        Get the current theme name.
        
        Returns:
            Current theme name ('light' or 'dark')
        """
        return self._current_theme
    
    def get_color_palette(self) -> Dict[str, List[str]]:
        """
        Get the full color palette dictionary.
        
        Returns:
            Dictionary mapping category names to color lists
        """
        return self._color_palette.copy()
    
    # Convenience properties for common color categories
    @property
    def geology_colors(self) -> List[QColor]:
        """Get geology color palette."""
        return self.get_color_list("geology")
    
    @property
    def resource_colors(self) -> List[QColor]:
        """Get resource color palette."""
        return self.get_color_list("resource")
    
    @property
    def uncertainty_colors(self) -> List[QColor]:
        """Get uncertainty color palette."""
        return self.get_color_list("uncertainty")
    
    @property
    def esg_colors(self) -> List[QColor]:
        """Get ESG color palette."""
        return self.get_color_list("esg")
    
    @property
    def pit_colors(self) -> List[QColor]:
        """Get pit color palette."""
        return self.get_color_list("pit")
    
    @property
    def underground_colors(self) -> List[QColor]:
        """Get underground color palette."""
        return self.get_color_list("underground")
    
    def set_application(self, app: QApplication) -> None:
        """
        Set the QApplication instance.
        
        Args:
            app: QApplication instance
        """
        self.app = app

