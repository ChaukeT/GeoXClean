"""
Collapsible group widget for modern UI panels.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PyQt6.QtGui import QIcon, QFont
from .modern_styles import get_theme_colors


class CollapsibleGroup(QWidget):
    """
    A collapsible group box widget with smooth animation.
    
    Provides a title bar with expand/collapse button and animated content area.
    Similar to QGroupBox but with collapsible functionality.
    """
    
    collapsed_changed = pyqtSignal(bool)  # True when collapsed
    
    def __init__(self, title: str = "", icon_name: str = "", collapsed: bool = False, parent=None):
        super().__init__(parent)
        
        self.title = title
        self.icon_name = icon_name
        self._collapsed = collapsed
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 8)
        main_layout.setSpacing(0)

        # Title bar with modern styling
        self.title_frame = QFrame()
        self.title_frame.setObjectName("CollapsibleGroupTitle")
        self.title_frame.setFixedHeight(42)
        self.title_frame.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_title_frame_style()
        
        title_layout = QHBoxLayout(self.title_frame)
        title_layout.setContentsMargins(12, 6, 12, 6)
        title_layout.setSpacing(10)
        
        # Expand/collapse button with modern icon
        self.toggle_button = QPushButton()
        self.toggle_button.setFixedSize(28, 28)
        self.toggle_button.setFlat(True)
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_toggle_button_style()
        self._update_toggle_icon()
        title_layout.addWidget(self.toggle_button)
        
        # Icon with label (if provided)
        if self.icon_name:
            try:
                icon = QIcon.fromTheme(self.icon_name)
                if not icon.isNull():
                    icon_label = QLabel()
                    icon_label.setPixmap(icon.pixmap(18, 18))
                    title_layout.addWidget(icon_label)
            except:
                pass  # Icon loading failed, continue without icon
        
        # Title label with modern typography
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("CollapsibleGroupTitleLabel")
        self.title_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_title_label_style()
        # Make title label clickable
        self.title_label.mousePressEvent = lambda e: self.toggle_collapsed()
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()

        # Add a subtle hint that this is collapsible
        self.hint_label = QLabel("click to toggle")
        self.hint_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_hint_label_style()
        # Make hint label clickable too
        self.hint_label.mousePressEvent = lambda e: self.toggle_collapsed()
        title_layout.addWidget(self.hint_label)
        
        main_layout.addWidget(self.title_frame)
        
        # Content area with modern styling
        self.content_widget = QWidget()
        self.content_widget.setObjectName("CollapsibleGroupContent")
        self._apply_content_widget_style()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(8)
        
        main_layout.addWidget(self.content_widget)
        
        # Animation for smooth collapse/expand
        self.animation = QPropertyAnimation(self.content_widget, b"maximumHeight")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        
        # Set initial state
        if self._collapsed:
            self.content_widget.setMaximumHeight(0)
            self.content_widget.setVisible(False)
        else:
            self.content_widget.setMaximumHeight(16777215)  # Max height
            self.content_widget.setVisible(True)
    
    def _update_toggle_icon(self):
        """Update the toggle button icon based on collapsed state."""
        if self._collapsed:
            self.toggle_button.setText("▶")  # Right-pointing triangle (highly visible)
        else:
            self.toggle_button.setText("▼")  # Down-pointing triangle (highly visible)
    
    def toggle_collapsed(self):
        """Toggle the collapsed state."""
        self.set_collapsed(not self._collapsed)
    
    def set_collapsed(self, collapsed: bool):
        """
        Set the collapsed state with smooth animation.
        
        Args:
            collapsed: True to collapse, False to expand
        """
        if self._collapsed == collapsed:
            return
        
        self._collapsed = collapsed
        self._update_toggle_icon()
        
        # Stop any running animation and disconnect previous handlers
        # to prevent handler accumulation (UX-001 fix)
        if hasattr(self, 'animation'):
            self.animation.stop()
            try:
                self.animation.finished.disconnect()
            except TypeError:
                pass  # No handlers connected
        
        if collapsed:
            # Collapse with animation
            start_height = self.content_widget.height()
            self.animation.setStartValue(start_height)
            self.animation.setEndValue(0)
            self.animation.finished.connect(self._on_collapse_finished)
            self.animation.start()
        else:
            # Expand with animation
            self.content_widget.setVisible(True)
            self.content_widget.setMaximumHeight(16777215)
            content_height = self.content_widget.sizeHint().height()
            
            self.animation.setStartValue(0)
            self.animation.setEndValue(content_height if content_height > 0 else 200)
            self.animation.finished.connect(self._on_expand_finished)
            self.animation.start()
        
        self.collapsed_changed.emit(collapsed)
    
    def _on_collapse_finished(self):
        """Handler for collapse animation completion."""
        self.content_widget.setVisible(False)
    
    def _on_expand_finished(self):
        """Handler for expand animation completion."""
        self.content_widget.setMaximumHeight(16777215)
    
    def is_collapsed(self) -> bool:
        """Return whether the group is collapsed."""
        return self._collapsed
    
    def set_title(self, title: str):
        """Set the title text."""
        self.title = title
        self.title_label.setText(title)
    
    def add_widget(self, widget: QWidget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)

    def _apply_title_frame_style(self):
        """Apply current theme style to title frame."""
        colors = get_theme_colors()
        self.title_frame.setStyleSheet(f"""
            QFrame#CollapsibleGroupTitle {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {colors.CARD_BG},
                    stop:1 {colors.ELEVATED_BG}
                );
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
            }}
            QFrame#CollapsibleGroupTitle:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {colors.CARD_HOVER},
                    stop:1 {colors.ELEVATED_BG}
                );
                border-color: {colors.ACCENT_PRIMARY};
            }}
        """)

    def _apply_toggle_button_style(self):
        """Apply current theme style to toggle button."""
        colors = get_theme_colors()
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.ELEVATED_BG};
                border: 1px solid {colors.BORDER};
                border-radius: 4px;
                color: {colors.ACCENT_PRIMARY};
                font-size: 16px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {colors.ACCENT_PRIMARY};
                color: white;
                border-color: {colors.ACCENT_PRIMARY};
            }}
        """)

    def _apply_title_label_style(self):
        """Apply current theme style to title label."""
        colors = get_theme_colors()
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {colors.TEXT_PRIMARY};
                font-weight: 600;
                font-size: 13px;
                background: transparent;
                border: none;
            }}
        """)

    def _apply_hint_label_style(self):
        """Apply current theme style to hint label."""
        colors = get_theme_colors()
        self.hint_label.setStyleSheet(f"""
            QLabel {{
                color: {colors.TEXT_HINT};
                font-size: 9px;
                font-style: italic;
                background: transparent;
                border: none;
            }}
        """)

    def _apply_content_widget_style(self):
        """Apply current theme style to content widget."""
        colors = get_theme_colors()
        self.content_widget.setStyleSheet(f"""
            QWidget#CollapsibleGroupContent {{
                background-color: {colors.CARD_BG};
                border: 1px solid {colors.BORDER};
                border-top: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                padding: 4px;
            }}
        """)

    def refresh_theme(self):
        """Refresh all styles when theme changes."""
        self._apply_title_frame_style()
        self._apply_toggle_button_style()
        self._apply_title_label_style()
        self._apply_hint_label_style()
        self._apply_content_widget_style()

