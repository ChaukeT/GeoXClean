"""
Splash Screen for GeoX.

Displays a branded splash screen during application startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QSplashScreen, QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QFont, QColor

logger = logging.getLogger(__name__)


class SplashScreen(QSplashScreen):
    """
    Branded splash screen displayed during application startup.
    
    Shows the application logo and loading message.
    """
    
    def __init__(self, pixmap: Optional[QPixmap] = None):
        """
        Initialize splash screen.
        
        Args:
            pixmap: Optional QPixmap for splash image (loads default if None)
        """
        if pixmap is None:
            pixmap = self._load_default_splash()
        
        super().__init__(pixmap)
        
        # Set window flags
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SplashScreen |
            Qt.WindowType.FramelessWindowHint
        )
        
        logger.info("Splash screen initialized")
    
    def _load_default_splash(self) -> QPixmap:
        """
        Load default splash screen image or create a placeholder.
        
        Returns:
            QPixmap for splash screen
        """
        # Try to load from assets
        assets_dir = Path(__file__).parent.parent / "assets" / "branding"
        splash_file = assets_dir / "splash.png"
        
        if splash_file.exists():
            try:
                pixmap = QPixmap(str(splash_file))
                if not pixmap.isNull():
                    logger.info(f"Loaded splash image from {splash_file}")
                    return pixmap
            except Exception as e:
                logger.warning(f"Error loading splash image: {e}")
        
        # Create a placeholder splash screen
        logger.info("Creating placeholder splash screen")
        return self._create_placeholder_splash()
    
    def _create_placeholder_splash(self) -> QPixmap:
        """
        Create a placeholder splash screen programmatically.
        
        Returns:
            QPixmap with placeholder splash design
        """
        width, height = 800, 600
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(30, 30, 40))  # Dark background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw gradient background
        from PyQt6.QtGui import QLinearGradient
        gradient = QLinearGradient(0, 0, width, height)
        gradient.setColorAt(0, QColor(41, 128, 185))  # #2980B9
        gradient.setColorAt(1, QColor(26, 188, 156))  # #1ABC9C
        painter.fillRect(0, 0, width, height, gradient)
        
        # Draw title text
        font = QFont("Arial", 48, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            width // 2 - 200,
            height // 2 - 50,
            400,
            100,
            Qt.AlignmentFlag.AlignCenter,
            "GeoX"
        )
        
        # Draw subtitle
        font = QFont("Arial", 18)
        painter.setFont(font)
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(
            width // 2 - 200,
            height // 2 + 50,
            400,
            50,
            Qt.AlignmentFlag.AlignCenter,
            "Geoscience Visualization Platform"
        )
        
        # Draw version info at bottom
        font = QFont("Arial", 12)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(
            0,
            height - 40,
            width,
            30,
            Qt.AlignmentFlag.AlignCenter,
            "Loading..."
        )
        
        painter.end()
        
        return pixmap
    
    def show_message(self, message: str, alignment: Qt.AlignmentFlag = None):
        """
        Show a message on the splash screen.
        
        Args:
            message: Message text to display
            alignment: Text alignment (defaults to bottom center)
        """
        if alignment is None:
            alignment = Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        
        super().showMessage(message, alignment)
        QApplication.processEvents()  # Process events to update display
    
    def finish(self, widget):
        """
        Finish splash screen and show main window.
        
        Args:
            widget: Main window widget to show
        """
        super().finish(widget)
        logger.info("Splash screen finished")

