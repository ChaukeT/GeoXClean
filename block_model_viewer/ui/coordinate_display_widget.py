"""
Coordinate Display Widget - Shows camera position and picked point coordinates.
"""

from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QFont, QColor, QPen
import logging
from .modern_styles import get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class CoordinateDisplayWidget(QWidget):
    """
    Overlay widget displaying coordinate information in the scene.
    Shows:
    - Camera position (X, Y, Z)
    - Picked point coordinates when user clicks
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # State
        self.camera_position = None
        self.picked_point = None
        self.visible_enabled = False  # Start hidden

        # Styling - theme-aware
        self.font_size = 8
        self.padding = 6
        self._update_colors()

        # Make widget transparent for custom painting
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Set compact size
        self.setFixedSize(240, 55)

        # Start hidden
        self.hide()

        logger.debug("CoordinateDisplayWidget initialized")

    def _update_colors(self):
        """Update colors from current theme."""
        colors = get_theme_colors()
        # Text color - primary text from theme
        self.text_color = QColor(colors.TEXT_PRIMARY)
        # Background - semi-transparent card background
        bg_rgb = self._hex_to_rgb(colors.ELEVATED_BG)
        self.background_color = QColor(bg_rgb[0], bg_rgb[1], bg_rgb[2], 200)
        # Highlight color for picked point
        highlight_rgb = self._hex_to_rgb(colors.INFO)
        self.highlight_color = QColor(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2])

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def set_camera_position(self, x: float, y: float, z: float):
        """Update camera position display."""
        self.camera_position = (x, y, z)
        if self.visible_enabled:
            self.update()
    
    def set_picked_point(self, x: float, y: float, z: float):
        """Update picked point display."""
        self.picked_point = (x, y, z)
        if self.visible_enabled:
            self.update()
    
    def clear_picked_point(self):
        """Clear the picked point display."""
        self.picked_point = None
        if self.visible_enabled:
            self.update()
    
    def set_visible(self, visible: bool):
        """Show or hide the coordinate display."""
        self.visible_enabled = visible
        if visible:
            self.show()
            self.raise_()
        else:
            self.hide()
    
    def place_top_left(self, margin: int = 10):
        """Position widget at top-left corner of parent."""
        if self.parent():
            self.move(margin, margin)
            if self.visible_enabled:
                self.raise_()
    
    def paintEvent(self, event):
        """Custom paint for clean, minimal overlay."""
        if not self.visible_enabled:
            return
            
        painter = None
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw rounded background card
            rect = self.rect()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.background_color)
            painter.drawRoundedRect(rect, 4, 4)
            
            # Setup text rendering
            painter.setPen(self.text_color)
            font = QFont("Consolas", self.font_size)
            painter.setFont(font)
            
            y_offset = self.padding + 11
            
            # Camera position (compact format)
            if self.camera_position:
                x, y, z = self.camera_position
                cam_text = f"Cam: {x:,.0f}, {y:,.0f}, {z:,.0f}"
                painter.drawText(self.padding, y_offset, cam_text)
            else:
                painter.drawText(self.padding, y_offset, "Camera: —")
            y_offset += 14
            
            # Picked point (if any, compact format)
            if self.picked_point:
                x, y, z = self.picked_point
                pick_text = f"Pick: {x:,.0f}, {y:,.0f}, {z:,.0f}"
                painter.setPen(self.highlight_color)  # Highlight with theme color
                painter.drawText(self.padding, y_offset, pick_text)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"CoordinateDisplayWidget paint error: {e}", exc_info=True)
        finally:
            if painter is not None:
                try:
                    painter.end()  # Always close painter, even on exception
                except Exception:
                    pass  # Ignore errors during painter cleanup

    def refresh_theme(self):
        """Refresh colors when theme changes."""
        self._update_colors()
        if self.visible_enabled:
            self.update()  # Trigger repaint with new colors
