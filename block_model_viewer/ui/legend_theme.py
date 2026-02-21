"""
Legend Theme - Single source of truth for legend and overlay styling.

Centralizes all visual styling rules for legends, axes, scale bars, and coordinate displays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from PyQt6.QtGui import QColor


@dataclass
class LegendTheme:
    """Centralized theme configuration for legends."""
    
    # Font settings
    font_family: str = "Segoe UI"
    font_size: int = 13
    title_font_size: int = 14
    title_bold: bool = True
    
    # Gradient bar settings
    gradient_bar_height: int = 200
    gradient_bar_width: int = 30
    bar_corner_radius: float = 10.0
    bar_outline_width: float = 2.0
    
    # Category swatch settings
    swatch_size: int = 20
    swatch_corner_radius: float = 6.0
    swatch_spacing: int = 8
    
    # Card/container settings
    card_corner_radius: float = 20.0
    padding: int = 12
    border_width: float = 2.0
    
    # Colors
    background_color: QColor = field(default_factory=lambda: QColor(15, 15, 20, 255))  # Fully opaque
    text_color: QColor = field(default_factory=lambda: QColor(245, 245, 245))
    border_color: QColor = field(default_factory=lambda: QColor(60, 60, 60))
    bar_outline_color: QColor = field(default_factory=lambda: QColor(20, 20, 20))
    tick_color: QColor = field(default_factory=lambda: QColor(235, 235, 235))
    
    # Effects
    enable_shadow: bool = True
    shadow_blur: int = 8
    shadow_offset: int = 2
    shadow_color: QColor = field(default_factory=lambda: QColor(0, 0, 0, 100))
    
    # Opacity (for overlays)
    background_opacity: float = 1.0  # Fully opaque by default
    
    # Label formatting
    label_decimals: Optional[int] = None  # Auto-format
    label_thousands_sep: bool = False
    
    # Colormap brightness (100 = original)
    colormap_brightness: int = 160


@dataclass
class AxisTheme:
    """Centralized theme configuration for axes."""
    
    # Font settings
    font_family: str = "Arial"
    font_size: int = 12
    font_bold: bool = True
    font_color: QColor = field(default_factory=lambda: QColor(231, 76, 60))  # Red
    
    # Formatting
    decimals: int = 0
    thousands_sep: bool = False
    
    # Visibility
    show_labels: bool = True
    show_ticks: bool = True


@dataclass
class ScaleBarTheme:
    """Centralized theme configuration for scale bars."""
    
    # Font settings
    font_family: str = "Arial"
    font_size: int = 11
    font_color: QColor = field(default_factory=lambda: QColor(0, 0, 0))
    
    # Bar settings
    bar_height: int = 4
    bar_color: QColor = field(default_factory=lambda: QColor(0, 0, 0))
    background_color: QColor = field(default_factory=lambda: QColor(255, 255, 255, 200))
    
    # Positioning
    margin: int = 18
    position: str = "bottom_center"  # bottom_center, bottom_right, etc.


@dataclass
class CoordinateDisplayTheme:
    """Centralized theme configuration for coordinate displays."""
    
    # Font settings
    font_family: str = "Consolas"
    font_size: int = 11
    font_color: QColor = field(default_factory=lambda: QColor(255, 255, 255))
    
    # Background
    background_color: QColor = field(default_factory=lambda: QColor(0, 0, 0, 180))
    padding: int = 6
    corner_radius: float = 4.0


# Global theme instances
LEGEND_THEME = LegendTheme()
AXIS_THEME = AxisTheme()
SCALE_BAR_THEME = ScaleBarTheme()
COORDINATE_DISPLAY_THEME = CoordinateDisplayTheme()


def get_legend_theme() -> LegendTheme:
    """Get the current legend theme."""
    return LEGEND_THEME


def get_axis_theme() -> AxisTheme:
    """Get the current axis theme."""
    return AXIS_THEME


def get_scale_bar_theme() -> ScaleBarTheme:
    """Get the current scale bar theme."""
    return SCALE_BAR_THEME


def get_coordinate_display_theme() -> CoordinateDisplayTheme:
    """Get the current coordinate display theme."""
    return COORDINATE_DISPLAY_THEME

