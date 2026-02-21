"""
GeoX Error Dialog
=================

A user-friendly error dialog that displays:
- Error code and message
- Expandable technical details
- Suggestions for resolution
- Copy-to-clipboard functionality
- Report issue button

Usage:
    from block_model_viewer.ui.error_dialog import show_error_dialog, ErrorDialog
    
    # Show a GeoXError
    show_error_dialog(error)
    
    # Or create dialog directly
    dialog = ErrorDialog(error, parent=self)
    dialog.exec()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFrame,
    QApplication,
    QWidget,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QClipboard

if TYPE_CHECKING:
    from ..core.errors import GeoXError

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


# =============================================================================
# STYLE CONSTANTS
# =============================================================================

COLORS = {
    "error_bg": "#2d1f1f",
    "error_border": "#ff4444",
    "warning_bg": "#2d2a1f",
    "warning_border": "#ffaa00",
    "info_bg": "#1f2d2d",
    "info_border": "#44aaff",
    "text_primary": f"{ModernColors.TEXT_PRIMARY}",
    "text_secondary": "#a0a0a0",
    "text_muted": "#707070",
    "button_primary": "#4a90d9",
    "button_hover": "#5aa0e9",
    "code_bg": "#1a1a1a",
    "suggestion_bg": "#1f2f1f",
}


def get_severity_colors(severity: str) -> tuple:
    """Get background and border colors based on severity."""
    if severity == "critical" or severity == "error":
        return COLORS["error_bg"], COLORS["error_border"]
    elif severity == "warning":
        return COLORS["warning_bg"], COLORS["warning_border"]
    else:
        return COLORS["info_bg"], COLORS["info_border"]


# =============================================================================
# COLLAPSIBLE SECTION
# =============================================================================

class CollapsibleSection(QFrame):
    """A collapsible section with header and content."""
    
    toggled = pyqtSignal(bool)
    
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._collapsed = True
        self._title = title
        self._setup_ui()
    


    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""
        
                    QDialog {{
                        background-color: {ModernColors.PANEL_BG};
                        color: {{COLORS['text_primary']}};
                    }}
                    QLabel {{
                        color: {{COLORS['text_primary']}};
                    }}
                
        """

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            # Rebuild stylesheet with new theme colors
            self.setStyleSheet(self._get_stylesheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header button
        self._header = QPushButton(f"▶ {self._title}")
        self._header.setFlat(True)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 8px 12px;
                color: {COLORS['text_secondary']};
                font-weight: 500;
                border: none;
                background: transparent;
            }}
            QPushButton:hover {{
                color: {COLORS['text_primary']};
                background: rgba(255, 255, 255, 0.05);
            }}
        """)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)
        
        # Content frame
        self._content = QFrame()
        self._content.setVisible(False)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 8, 12, 12)
        layout.addWidget(self._content)
    
    def _toggle(self):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        arrow = "▼" if not self._collapsed else "▶"
        self._header.setText(f"{arrow} {self._title}")
        self.toggled.emit(not self._collapsed)
    
    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)
    
    def set_expanded(self, expanded: bool):
        if expanded != (not self._collapsed):
            self._toggle()


# =============================================================================
# ERROR DIALOG
# =============================================================================

class ErrorDialog(QDialog):
    """
    A comprehensive error dialog for displaying GeoXError instances.
    
    Features:
    - Clear error code and message display
    - Collapsible technical details section
    - Suggestions for resolution
    - Copy-to-clipboard functionality
    - Report issue button (configurable)
    """
    
    report_requested = pyqtSignal(dict)  # Emitted when user clicks "Report Issue"
    
    def __init__(
        self,
        error: "GeoXError",
        parent: Optional[QWidget] = None,
        show_report_button: bool = True,
    ):
        super().__init__(parent)
        self._error = error
        self._show_report_button = show_report_button
        self._setup_ui()
    
    def _setup_ui(self):
        self.setWindowTitle("Error")
        self.setMinimumWidth(500)
        self.setMaximumWidth(700)
        
        # Get colors based on severity
        severity = getattr(self._error, 'severity', None)
        severity_str = severity.value if severity else "error"
        bg_color, border_color = get_severity_colors(severity_str)
        
        # Main stylesheet
        self.setStyleSheet(self._get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with error code and icon
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        # Error icon (using text emoji for simplicity)
        icon_label = QLabel("⚠️" if severity_str == "warning" else "❌")
        icon_label.setStyleSheet("font-size: 32px;")
        header_layout.addWidget(icon_label)
        
        # Error code and title
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        
        code_label = QLabel(self._error.code)
        code_label.setStyleSheet(f"""
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            font-weight: bold;
            color: {border_color};
            padding: 2px 8px;
            background: {COLORS['code_bg']};
            border-radius: 4px;
        """)
        code_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        title_layout.addWidget(code_label)
        
        message_label = QLabel(self._error.message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 500;
            color: {COLORS['text_primary']};
        """)
        title_layout.addWidget(message_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background: {COLORS['text_muted']}; max-height: 1px;")
        layout.addWidget(separator)
        
        # Suggestions section (if any)
        suggestions = getattr(self._error, 'suggestions', [])
        if suggestions:
            suggestions_frame = QFrame()
            suggestions_frame.setStyleSheet(f"""
                QFrame {{
                    background: {COLORS['suggestion_bg']};
                    border-radius: 6px;
                    padding: 12px;
                }}
            """)
            suggestions_layout = QVBoxLayout(suggestions_frame)
            suggestions_layout.setSpacing(8)
            
            suggestions_title = QLabel("💡 Suggestions")
            suggestions_title.setStyleSheet(f"""
                font-weight: 600;
                color: #88cc88;
                font-size: 13px;
            """)
            suggestions_layout.addWidget(suggestions_title)
            
            for suggestion in suggestions:
                suggestion_label = QLabel(f"  • {suggestion}")
                suggestion_label.setWordWrap(True)
                suggestion_label.setStyleSheet(f"""
                    color: {COLORS['text_secondary']};
                    font-size: 12px;
                """)
                suggestions_layout.addWidget(suggestion_label)
            
            layout.addWidget(suggestions_frame)
        
        # Technical details (collapsible)
        details = getattr(self._error, 'details', '')
        context = getattr(self._error, 'context', {})
        
        if details or context:
            details_section = CollapsibleSection("Technical Details")
            
            details_text = QTextEdit()
            details_text.setReadOnly(True)
            details_text.setMaximumHeight(200)
            details_text.setStyleSheet(f"""
                QTextEdit {{
                    background: {COLORS['code_bg']};
                    color: {COLORS['text_secondary']};
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 11px;
                    border: 1px solid {COLORS['text_muted']};
                    border-radius: 4px;
                    padding: 8px;
                }}
            """)
            
            # Build details content
            content = ""
            if details:
                content += f"Details:\n{details}\n\n"
            if context:
                content += "Context:\n"
                for key, value in context.items():
                    if key == "traceback":
                        content += f"\nTraceback:\n{value}\n"
                    else:
                        content += f"  {key}: {value}\n"
            
            details_text.setPlainText(content.strip())
            details_section.add_widget(details_text)
            
            layout.addWidget(details_section)
        
        # Timestamp
        timestamp = getattr(self._error, 'timestamp', None)
        if timestamp:
            timestamp_label = QLabel(f"Occurred at: {timestamp}")
            timestamp_label.setStyleSheet(f"""
                color: {COLORS['text_muted']};
                font-size: 10px;
            """)
            layout.addWidget(timestamp_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Copy to clipboard button
        copy_btn = QPushButton("📋 Copy Error Info")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 16px;
                background: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['text_muted']};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
                color: {COLORS['text_primary']};
            }}
        """)
        copy_btn.clicked.connect(self._copy_to_clipboard)
        button_layout.addWidget(copy_btn)
        
        # Report issue button (optional)
        if self._show_report_button:
            report_btn = QPushButton("🐛 Report Issue")
            report_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            report_btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 8px 16px;
                    background: transparent;
                    color: {COLORS['text_secondary']};
                    border: 1px solid {COLORS['text_muted']};
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background: rgba(255, 255, 255, 0.05);
                    color: {COLORS['text_primary']};
                }}
            """)
            report_btn.clicked.connect(self._report_issue)
            button_layout.addWidget(report_btn)
        
        button_layout.addStretch()
        
        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 24px;
                background: {COLORS['button_primary']};
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {COLORS['button_hover']};
            }}
        """)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def _copy_to_clipboard(self):
        """Copy error information to clipboard."""
        error_info = f"""GeoX Error Report
==================
Code: {self._error.code}
Message: {self._error.message}
Timestamp: {getattr(self._error, 'timestamp', 'N/A')}

Details:
{getattr(self._error, 'details', 'N/A')}

Context:
{getattr(self._error, 'context', {})}

Suggestions:
{chr(10).join('- ' + s for s in getattr(self._error, 'suggestions', []))}
"""
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(error_info)
            logger.info("Error info copied to clipboard")
    
    def _report_issue(self):
        """Handle report issue button click."""
        self.report_requested.emit(self._error.to_dict())
        # Could open a web browser to issue tracker, send telemetry, etc.
        logger.info(f"Issue report requested for error: {self._error.code}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def show_error_dialog(
    error: "GeoXError",
    parent: Optional[QWidget] = None,
    show_report_button: bool = True,
) -> None:
    """
    Show an error dialog for a GeoXError.
    
    Args:
        error: The GeoXError to display
        parent: Parent widget for the dialog
        show_report_button: Whether to show the "Report Issue" button
    """
    dialog = ErrorDialog(error, parent, show_report_button)
    dialog.exec()


def show_error_message(
    code: str,
    message: str,
    details: str = "",
    parent: Optional[QWidget] = None,
) -> None:
    """
    Convenience function to show a simple error message.
    
    Creates a GeoXError and shows a dialog for it.
    """
    from ..core.errors import GeoXError
    
    error = GeoXError(code=code, message=message, details=details)
    show_error_dialog(error, parent)


# =============================================================================
# REGISTER HANDLER
# =============================================================================

def register_error_dialog_handler():
    """
    Register the error dialog as the global error handler.
    
    Call this during application startup after Qt is initialized.
    """
    from ..core.errors import set_error_dialog_handler
    set_error_dialog_handler(show_error_dialog)
    logger.info("Error dialog handler registered")

