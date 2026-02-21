"""
Qt-based axes overlay widget that draws axes lines and labels using QPainter.

This is a screen-space overlay (like the legend and scale bar) that doesn't use PyVista.
It projects 3D world coordinates to screen space and draws axes lines and labels.
"""

from __future__ import annotations

from typing import Optional, Tuple
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetrics
from PyQt6.QtCore import Qt, QRectF, QPointF
import logging
from .modern_styles import get_theme_colors, get_current_theme

logger = logging.getLogger(__name__)


class AxesWidget(QWidget):
    """
    Qt-based axes overlay widget.
    
    Draws X, Y, Z axes lines and labels in screen space, similar to how
    the scale bar and legend work. This is a pure Qt overlay, not PyVista actors.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AxesWidget")
        
        # Widget properties
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # Axes state
        self._bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self._visible: bool = False

        # Style
        self._font = QFont("Arial", 12)
        self._font.setBold(True)
        self._line_width = 2.0
        self._label_offset = 10  # pixels offset from axis end

        # Initialize theme-aware colors
        self._update_colors()
        
        # World-to-screen projection (set by renderer/viewer)
        self._projection_func = None  # func(world_pos) -> (screen_x, screen_y) or None
        
        # Size hint
        self.setMinimumSize(200, 200)
        self.resize(300, 300)
    
    def set_bounds(self, bounds: Optional[Tuple[float, float, float, float, float, float]]):
        """Set scene bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
        self._bounds = bounds
        self.update()
    
    def set_visible(self, visible: bool):
        """Set visibility."""
        if visible != self._visible:
            self._visible = visible
            if visible:
                self.show()
                self.raise_()
            else:
                self.hide()
            self.update()
    
    def set_projection_function(self, func):
        """
        Set function to project 3D world coordinates to screen space.
        
        Args:
            func: Callable that takes (x, y, z) tuple and returns (screen_x, screen_y) tuple or None
        """
        self._projection_func = func
    
    def paintEvent(self, event):
        """Draw axes lines and labels."""
        if not self._visible or self._bounds is None or self._projection_func is None:
            return
        
        painter = None
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            
            xmin, xmax, ymin, ymax, zmin, zmax = self._bounds
            
            # Origin point (bottom-left-front corner)
            origin_3d = (xmin, ymin, zmin)
            origin_screen = self._projection_func(origin_3d)
            
            if origin_screen is None:
                return  # Let finally block handle painter cleanup
            
            # End points for each axis
            x_end_3d = (xmax, ymin, zmin)
            y_end_3d = (xmin, ymax, zmin)
            z_end_3d = (xmin, ymin, zmax)
            
            x_end_screen = self._projection_func(x_end_3d)
            y_end_screen = self._projection_func(y_end_3d)
            z_end_screen = self._projection_func(z_end_3d)
            
            if None in (x_end_screen, y_end_screen, z_end_screen):
                return  # Let finally block handle painter cleanup
            
            # Draw axes lines
            pen = QPen()
            pen.setWidthF(self._line_width)
            
            # X axis (red)
            pen.setColor(self._colors['x'])
            painter.setPen(pen)
            painter.drawLine(
                QPointF(origin_screen[0], origin_screen[1]),
                QPointF(x_end_screen[0], x_end_screen[1])
            )
            
            # Y axis (green)
            pen.setColor(self._colors['y'])
            painter.setPen(pen)
            painter.drawLine(
                QPointF(origin_screen[0], origin_screen[1]),
                QPointF(y_end_screen[0], y_end_screen[1])
            )
            
            # Z axis (blue)
            pen.setColor(self._colors['z'])
            painter.setPen(pen)
            painter.drawLine(
                QPointF(origin_screen[0], origin_screen[1]),
                QPointF(z_end_screen[0], z_end_screen[1])
            )
            
            # Draw labels
            painter.setFont(self._font)
            fm = QFontMetrics(self._font)
            
            # X label
            pen.setColor(self._colors['x'])
            painter.setPen(pen)
            x_label = "X"
            label_rect = fm.boundingRect(x_label)
            label_x = x_end_screen[0] + self._label_offset
            label_y = x_end_screen[1] - label_rect.height() / 2
            painter.drawText(int(label_x), int(label_y), x_label)
            
            # Y label
            pen.setColor(self._colors['y'])
            painter.setPen(pen)
            y_label = "Y"
            label_rect = fm.boundingRect(y_label)
            label_x = y_end_screen[0] + self._label_offset
            label_y = y_end_screen[1] - label_rect.height() / 2
            painter.drawText(int(label_x), int(label_y), y_label)
            
            # Z label
            pen.setColor(self._colors['z'])
            painter.setPen(pen)
            z_label = "Z"
            label_rect = fm.boundingRect(z_label)
            label_x = z_end_screen[0] + self._label_offset
            label_y = z_end_screen[1] - label_rect.height() / 2
            painter.drawText(int(label_x), int(label_y), z_label)
            
        except Exception as e:
            logger.warning(f"AxesWidget.paintEvent failed: {e}", exc_info=True)
        finally:
            if painter is not None:
                painter.end()

    def _update_colors(self):
        """
        Update axis colors for current theme.

        Axes use standard XYZ colors (X=red, Y=green, Z=blue) that are recognizable
        across 3D applications. We adjust brightness slightly for theme compatibility
        while maintaining the distinctive colors.
        """
        is_light_theme = get_current_theme() == "light"

        if is_light_theme:
            # Slightly darker/more saturated for light backgrounds
            self._colors = {
                'x': QColor(192, 57, 43),   # Darker red for light theme
                'y': QColor(39, 174, 96),   # Darker green for light theme
                'z': QColor(41, 128, 185),  # Darker blue for light theme
            }
        else:
            # Slightly brighter for dark backgrounds
            self._colors = {
                'x': QColor(231, 76, 60),   # Brighter red for dark theme
                'y': QColor(46, 204, 113),  # Brighter green for dark theme
                'z': QColor(52, 152, 219),  # Brighter blue for dark theme
            }

    def refresh_theme(self):
        """Refresh colors when theme changes."""
        self._update_colors()
        if self._visible:
            self.update()  # Trigger repaint with new colors

