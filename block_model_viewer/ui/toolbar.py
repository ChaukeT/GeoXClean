"""
Modern Toolbar with icons, dropdowns, and status strip for BlockModelViewer.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QToolButton, QLabel, 
    QSpacerItem, QSizePolicy, QMenu, QComboBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QAction

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class Toolbar(QWidget):
    """
    Toolbar with quick access buttons for common operations.
    
    Provides buttons for file operations, view controls, and export functions.
    """
    
    # Signals
    open_file_requested = pyqtSignal()
    reset_view_requested = pyqtSignal()
    fit_view_requested = pyqtSignal()
    toggle_projection_requested = pyqtSignal()
    export_screenshot_requested = pyqtSignal()
    export_data_requested = pyqtSignal()
    
    # New signals for modern toolbar
    scene_action_requested = pyqtSignal(str)  # "reset", "fit"
    view_action_requested = pyqtSignal(str)  # "block_data", "drillhole_data", "statistics"
    panel_action_requested = pyqtSignal(str)  # "axes_panel"
    
    def __init__(self):
        super().__init__()
        
        # UI Elements
        self.open_file_btn: Optional[QPushButton] = None
        self.reset_view_btn: Optional[QPushButton] = None
        self.fit_view_btn: Optional[QPushButton] = None
        self.toggle_projection_btn: Optional[QPushButton] = None
        self.export_screenshot_btn: Optional[QPushButton] = None
        self.export_data_btn: Optional[QPushButton] = None
        
        # Status strip elements
        self.status_selection_label: Optional[QLabel] = None
        self.status_property_label: Optional[QLabel] = None
        self.status_camera_label: Optional[QLabel] = None
        self.status_theme_label: Optional[QLabel] = None
        
        self._setup_ui()
        logger.info("Initialized modern toolbar")
    


    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _setup_ui(self):
        """Setup the modern toolbar UI with icons, dropdowns, and status strip."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top toolbar row
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar_layout.setSpacing(5)
        
        # Get icons directory
        from pathlib import Path
        icons_dir = Path(__file__).parent.parent / "assets" / "icons"
        
        # Helper function to load icon
        def load_icon(name: str) -> QIcon:
            icon_path = icons_dir / f"{name}.svg"
            if icon_path.exists():
                return QIcon(str(icon_path))
            return QIcon()
        
        # File operations
        self.open_file_btn = QToolButton()
        open_icon = load_icon("open")
        if not open_icon.isNull():
            self.open_file_btn.setIcon(open_icon)
        self.open_file_btn.setToolTip("Open a block model file (Ctrl+O)")
        self.open_file_btn.clicked.connect(self.open_file_requested.emit)
        toolbar_layout.addWidget(self.open_file_btn)
        
        # Add separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar_layout.addWidget(separator1)
        
        # Scene dropdown
        scene_btn = QToolButton()
        scene_btn.setText("Scene")
        scene_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        scene_menu = QMenu(scene_btn)
        
        reset_action = QAction("Reset View", scene_menu)
        refresh_icon = load_icon("refresh")
        if not refresh_icon.isNull():
            reset_action.setIcon(refresh_icon)
        reset_action.triggered.connect(lambda: self.scene_action_requested.emit("reset"))
        scene_menu.addAction(reset_action)
        
        scene_btn.setMenu(scene_menu)
        toolbar_layout.addWidget(scene_btn)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar_layout.addWidget(separator2)
        
        # View dropdown
        view_btn = QToolButton()
        view_btn.setText("View")
        view_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        view_menu = QMenu(view_btn)
        
        block_data_action = QAction("Block Model Data", view_menu)
        table_icon = load_icon("table")
        if not table_icon.isNull():
            block_data_action.setIcon(table_icon)
        block_data_action.triggered.connect(lambda: self.view_action_requested.emit("block_data"))
        view_menu.addAction(block_data_action)
        
        drillhole_data_action = QAction("Drillhole Data", view_menu)
        drillhole_icon = load_icon("drillhole")
        if not drillhole_icon.isNull():
            drillhole_data_action.setIcon(drillhole_icon)
        drillhole_data_action.triggered.connect(lambda: self.view_action_requested.emit("drillhole_data"))
        view_menu.addAction(drillhole_data_action)
        
        statistics_action = QAction("Statistics", view_menu)
        chart_icon = load_icon("chart")
        if not chart_icon.isNull():
            statistics_action.setIcon(chart_icon)
        statistics_action.triggered.connect(lambda: self.view_action_requested.emit("statistics"))
        view_menu.addAction(statistics_action)
        
        view_btn.setMenu(view_menu)
        toolbar_layout.addWidget(view_btn)
        
        # Add separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar_layout.addWidget(separator3)
        
        # Axes/Scale Bar Panel button
        axes_panel_btn = QToolButton()
        axes_panel_btn.setToolTip("Axes & Scale Bar Panel")
        axes_panel_btn.setText("Axes Panel")
        axes_panel_btn.clicked.connect(lambda: self.panel_action_requested.emit("axes_panel"))
        toolbar_layout.addWidget(axes_panel_btn)
        
        # Add stretch
        toolbar_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        
        main_layout.addLayout(toolbar_layout)
        
        # Status strip at bottom
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setStyleSheet(f"QFrame {{ border-top: 1px solid {ModernColors.BORDER}; background-color: {ModernColors.CARD_BG}; }}")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)
        status_layout.setSpacing(15)
        
        self.status_selection_label = QLabel("Selected: 0")
        self.status_property_label = QLabel("Property: None")
        self.status_camera_label = QLabel("Camera: (0, 0, 0)")
        self.status_theme_label = QLabel("Theme: Dark")
        
        status_layout.addWidget(self.status_selection_label)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.status_property_label)
        status_layout.addWidget(QLabel("|"))
        status_layout.addWidget(self.status_camera_label)
        status_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        status_layout.addWidget(self.status_theme_label)
        
        main_layout.addWidget(status_frame)
    
    def set_status(self, message: str):
        """Set status message in toolbar (legacy compatibility)."""
        # This method is kept for backward compatibility
        pass
    
    def update_status_strip(self, selection_count: int = 0, property_name: str = "", 
                           camera_pos: tuple = (0, 0, 0), theme_name: str = "dark"):
        """
        Update status strip information.
        
        Args:
            selection_count: Number of selected elements
            property_name: Active property name
            camera_pos: Camera position tuple (x, y, z)
            theme_name: Current theme name
        """
        if self.status_selection_label:
            self.status_selection_label.setText(f"Selected: {selection_count}")
        if self.status_property_label:
            prop_text = property_name if property_name else "None"
            self.status_property_label.setText(f"Property: {prop_text}")
        if self.status_camera_label:
            x, y, z = camera_pos
            self.status_camera_label.setText(f"Camera: ({x:.1f}, {y:.1f}, {z:.1f})")
        if self.status_theme_label:
            self.status_theme_label.setText(f"Theme: {theme_name.capitalize()}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable all toolbar buttons."""
        buttons = [
            self.open_file_btn,
            self.reset_view_btn,
            self.fit_view_btn,
            self.toggle_projection_btn,
            self.export_screenshot_btn,
            self.export_data_btn
        ]
        
        for btn in buttons:
            if btn:
                btn.setEnabled(enabled)
    
    def set_file_operations_enabled(self, enabled: bool):
        """Enable or disable file operation buttons."""
        if self.open_file_btn:
            self.open_file_btn.setEnabled(enabled)
    
    def set_view_operations_enabled(self, enabled: bool):
        """Enable or disable view operation buttons."""
        buttons = [self.reset_view_btn, self.fit_view_btn, self.toggle_projection_btn]
        for btn in buttons:
            if btn:
                btn.setEnabled(enabled)
    
    def set_export_operations_enabled(self, enabled: bool):
        """Enable or disable export operation buttons."""
        buttons = [self.export_screenshot_btn, self.export_data_btn]
        for btn in buttons:
            if btn:
                btn.setEnabled(enabled)
