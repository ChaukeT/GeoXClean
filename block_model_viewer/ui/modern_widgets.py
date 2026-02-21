"""
Modern UI Widgets for GeoX.

Reusable, modern-styled Qt widgets for a polished user experience.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QSizePolicy, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


# --- MODERN COLOR PALETTE ---
class _ColorsMeta(type):
    """
    Metaclass that enables dynamic color access based on current theme.

    This ensures Colors.BG_PRIMARY always returns the current theme's color,
    not a frozen value from module import time.
    """

    # Attribute mapping from Colors to ModernColors
    _ATTR_MAP = {
        # Backgrounds
        'BG_PRIMARY': 'PANEL_BG',
        'BG_SURFACE': 'CARD_BG',
        'BG_HOVER': 'CARD_HOVER',
        'BG_SELECTED': 'ACCENT_PRIMARY',

        # Accents
        'PRIMARY': 'ACCENT_PRIMARY',
        'PRIMARY_HOVER': 'ACCENT_HOVER',
        'PRIMARY_DARK': 'ACCENT_PRESSED',
        'SUCCESS': 'SUCCESS',
        'SUCCESS_LIGHT': 'SUCCESS',
        'WARNING': 'WARNING',
        'WARNING_LIGHT': 'WARNING',
        'ERROR': 'ERROR',
        'ERROR_LIGHT': 'ERROR',

        # Text
        'TEXT_PRIMARY': 'TEXT_PRIMARY',
        'TEXT_SECONDARY': 'TEXT_SECONDARY',
        'TEXT_MUTED': 'TEXT_HINT',

        # Borders
        'BORDER': 'BORDER',
        'BORDER_HOVER': 'BORDER_LIGHT',
        'BORDER_FOCUS': 'ACCENT_PRIMARY',
    }

    def __getattr__(cls, name: str) -> str:
        """Get color attribute from current theme dynamically."""
        if name in cls._ATTR_MAP:
            modern_colors_attr = cls._ATTR_MAP[name]
            return getattr(ModernColors, modern_colors_attr)
        raise AttributeError(f"'Colors' has no attribute '{name}'")


class Colors(metaclass=_ColorsMeta):
    """
    Modern color palette for consistent styling.

    This class dynamically returns colors from the current theme.
    All attribute access goes through the metaclass to ensure
    colors update when the theme changes.
    """
    pass


class FileInputCard(QFrame):
    """
    A modern card widget for file selection.
    
    Features:
    - Clean, minimal design with clear visual states
    - File name display with truncation
    - File size and row count display
    - Browse button with hover effects
    - Clear button to remove selection
    """
    
    fileSelected = pyqtSignal(str)  # Emits file path when selected
    fileCleared = pyqtSignal()  # Emits when file is cleared
    
    def __init__(
        self,
        label: str,
        file_filter: str = "CSV Files (*.csv);;All Files (*)",
        required: bool = False,
        icon: str = "📄",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.label = label
        self.file_filter = file_filter
        self.required = required
        self.icon = icon
        self._file_path: Optional[str] = None
        self._file_info: Dict[str, Any] = {}

        self._setup_ui()
        self._apply_empty_style()

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply styles
        self._apply_empty_style() if not self._file_path else self._apply_selected_style()
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
    
    def _setup_ui(self):
        """Setup the card UI."""
        self.setMinimumHeight(80)
        self.setMaximumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        
        # Icon
        self.icon_label = QLabel(self.icon)
        self.icon_label.setStyleSheet("font-size: 24px;")
        self.icon_label.setFixedWidth(32)
        layout.addWidget(self.icon_label)
        
        # Content section
        content = QVBoxLayout()
        content.setSpacing(4)
        
        # Header row with label and required badge
        header = QHBoxLayout()
        header.setSpacing(8)
        
        self.label_widget = QLabel(self.label)
        label_font = QFont()
        label_font.setBold(True)
        label_font.setPointSize(10)
        self.label_widget.setFont(label_font)
        header.addWidget(self.label_widget)
        
        if self.required:
            required_badge = QLabel("Required")
            required_badge.setStyleSheet(f"""
                background-color: {Colors.ERROR_LIGHT};
                color: {Colors.ERROR};
                font-size: 8pt;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 4px;
            """)
            header.addWidget(required_badge)
        
        header.addStretch()
        content.addLayout(header)
        
        # Status/filename row
        self.status_label = QLabel("No file selected")
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
        content.addWidget(self.status_label)
        
        # File info row (hidden until file selected)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 8pt;")
        self.info_label.hide()
        content.addWidget(self.info_label)
        
        layout.addLayout(content, 1)
        
        # Action buttons
        buttons = QVBoxLayout()
        buttons.setSpacing(4)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.browse_btn.setFixedSize(70, 28)
        self.browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 9pt;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {Colors.PRIMARY_DARK};
            }}
        """)
        self.browse_btn.clicked.connect(self._on_browse)
        buttons.addWidget(self.browse_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.clear_btn.setFixedSize(70, 24)
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-size: 8pt;
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_HOVER};
                border-color: {Colors.BORDER_HOVER};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        self.clear_btn.clicked.connect(self._on_clear)
        self.clear_btn.hide()
        buttons.addWidget(self.clear_btn)
        
        layout.addLayout(buttons)
    
    def _apply_empty_style(self):
        """Style for empty/no file selected state."""
        self.setStyleSheet(f"""
            FileInputCard {{
                background-color: {Colors.BG_SURFACE};
                border: 2px dashed {Colors.BORDER};
                border-radius: 10px;
            }}
            FileInputCard:hover {{
                border-color: {Colors.PRIMARY};
                background-color: {Colors.BG_SELECTED};
            }}
        """)
        self.label_widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
    
    def _apply_loaded_style(self):
        """Style for file loaded state."""
        self.setStyleSheet(f"""
            FileInputCard {{
                background-color: {Colors.SUCCESS_LIGHT};
                border: 2px solid {Colors.SUCCESS};
                border-radius: 10px;
            }}
        """)
        self.label_widget.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")
    
    def _apply_pending_style(self):
        """Style for auto-detected file pending confirmation."""
        self.setStyleSheet(f"""
            FileInputCard {{
                background-color: {Colors.WARNING_LIGHT};
                border: 2px solid {Colors.WARNING};
                border-radius: 10px;
            }}
            FileInputCard:hover {{
                background-color: {Colors.WARNING_LIGHT};
                border-color: {Colors.WARNING};
            }}
        """)
        self.label_widget.setStyleSheet(f"color: {Colors.WARNING}; font-weight: bold;")
    
    def _on_browse(self):
        """Open file dialog to select a file, or confirm auto-detected file."""
        from PyQt6.QtWidgets import QFileDialog
        
        # Check if we have an auto-detected file pending confirmation
        # (indicated by having a file path but the pending style)
        if self._file_path and self.icon_label.text() == "🔍":
            # Confirm the auto-detected file
            pending_path = self._file_path
            self.icon_label.setText("✅")
            self._apply_loaded_style()
            self.info_label.setText(self.info_label.text().replace(" • Click to confirm", ""))
            self.fileSelected.emit(pending_path)
            return
        
        # Open file dialog to select a new file
        # Use the folder of the current file as starting directory if available
        start_dir = ""
        if self._file_path:
            start_dir = str(Path(self._file_path).parent)
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.label}",
            start_dir,
            self.file_filter
        )
        
        if path:
            self.set_file(path)
    
    def _on_clear(self):
        """Clear the selected file."""
        self._file_path = None
        self._file_info = {}
        
        self.status_label.setText("No file selected")
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9pt;")
        self.info_label.hide()
        self.clear_btn.hide()
        self.icon_label.setText(self.icon)
        
        self._apply_empty_style()
        self.fileCleared.emit()
    
    def set_file(self, path: str, row_count: Optional[int] = None, emit_signal: bool = True):
        """
        Set the selected file.
        
        Args:
            path: File path
            row_count: Optional row count to display
            emit_signal: Whether to emit fileSelected signal (default True).
                         Set to False for auto-detected files that need user confirmation.
        """
        self._file_path = path
        file_path = Path(path)
        
        # Truncate filename if too long
        filename = file_path.name
        if len(filename) > 35:
            filename = filename[:32] + "..."
        
        self.status_label.setText(filename)
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 9pt; font-weight: 500;")
        self.status_label.setToolTip(path)
        
        # Build info text
        info_parts = []
        try:
            size = file_path.stat().st_size
            if size < 1024:
                info_parts.append(f"{size} B")
            elif size < 1024 * 1024:
                info_parts.append(f"{size / 1024:.1f} KB")
            else:
                info_parts.append(f"{size / (1024*1024):.1f} MB")
        except:
            pass
        
        if row_count is not None:
            info_parts.append(f"{row_count:,} rows")
        
        # For auto-detected files, show "Click to confirm" hint
        if not emit_signal:
            info_parts.append("Click to confirm")
        
        if info_parts:
            self.info_label.setText(" • ".join(info_parts))
            self.info_label.show()
        
        self.icon_label.setText("🔍" if not emit_signal else "✅")
        self.clear_btn.show()
        self._apply_loaded_style() if emit_signal else self._apply_pending_style()
        
        if emit_signal:
            self.fileSelected.emit(path)
    
    def get_file_path(self) -> Optional[str]:
        """Get the selected file path."""
        return self._file_path
    
    def set_row_count(self, count: int):
        """Update the row count display without emitting fileSelected signal."""
        if self._file_path:
            # Update info label directly without calling set_file to avoid re-emitting signal
            file_path = Path(self._file_path)
            info_parts = []
            try:
                size = file_path.stat().st_size
                if size < 1024:
                    info_parts.append(f"{size} B")
                elif size < 1024 * 1024:
                    info_parts.append(f"{size / 1024:.1f} KB")
                else:
                    info_parts.append(f"{size / (1024*1024):.1f} MB")
            except:
                pass
            
            info_parts.append(f"{count:,} rows")
            self.info_label.setText(" • ".join(info_parts))
            self.info_label.show()


class ModernProgressBar(QFrame):
    """
    A modern, styled progress bar with label and percentage.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet(f"""
            ModernProgressBar {{
                background-color: {Colors.BG_SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        # Header with label and percentage
        header = QHBoxLayout()
        
        self.label = QLabel("Processing...")
        self.label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 9pt; font-weight: 500;")
        header.addWidget(self.label)
        
        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet(f"color: {Colors.PRIMARY}; font-size: 9pt; font-weight: 600;")
        header.addWidget(self.percent_label)
        
        layout.addLayout(header)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_PRIMARY};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress)
    
    def setValue(self, value: int):
        """Set progress value (0-100)."""
        self.progress.setValue(value)
        self.percent_label.setText(f"{value}%")
    
    def setLabel(self, text: str):
        """Set the progress label text."""
        self.label.setText(text)
    
    def setRange(self, min_val: int, max_val: int):
        """Set progress range."""
        self.progress.setRange(min_val, max_val)
        if min_val == 0 and max_val == 0:
            # Indeterminate mode
            self.percent_label.setText("")
        else:
            self.percent_label.setText("0%")


class SectionHeader(QFrame):
    """
    A modern section header with title and optional subtitle.
    """
    
    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._setup_ui(title, subtitle)
    
    def _setup_ui(self, title: str, subtitle: Optional[str]):
        self.setStyleSheet(f"""
            SectionHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        self.setFixedHeight(70 if subtitle else 50)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 16px;
            font-weight: 700;
        """)
        layout.addWidget(self.title_label)
        
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setStyleSheet(f"""
                color: {Colors.TEXT_SECONDARY};
                font-size: 12px;
            """)
            layout.addWidget(self.subtitle_label)


class StatusBadge(QLabel):
    """
    A modern status badge with different states.
    """
    
    class State:
        NEUTRAL = "neutral"
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"
        INFO = "info"
    
    def __init__(self, text: str = "", state: str = "neutral", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.set_state(state)
    
    def set_state(self, state: str):
        """Set the badge state (neutral, success, warning, error, info)."""
        styles = {
            "neutral": ("#444444", "#cccccc", "#666666"),
            "success": ("#1a4d3a", "#4ade80", f"{ModernColors.SUCCESS}"),
            "warning": ("#4d3d1a", "#fbbf24", f"{ModernColors.WARNING}"),
            "error": ("#4d1a1a", "#f87171", f"{ModernColors.ERROR}"),
            "info": ("#1a3a4d", "#60a5fa", f"{ModernColors.ACCENT_PRIMARY}"),
        }
        
        bg, text, border = styles.get(state, styles["neutral"])
        
        self.setStyleSheet(f"""
            background-color: {bg};
            color: {text};
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid {border};
        """)


class ActionButton(QPushButton):
    """
    A modern action button with different variants.
    """
    
    class Variant:
        PRIMARY = "primary"
        SECONDARY = "secondary"
        SUCCESS = "success"
        DANGER = "danger"
    
    def __init__(
        self,
        text: str,
        variant: str = "primary",
        icon: Optional[str] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(text, parent)
        self._variant = variant
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(40)
        self._apply_style()
        
        if icon:
            self.setText(f"{icon}  {text}")
    
    def _apply_style(self):
        styles = {
            "primary": f"""
                QPushButton {{
                    background-color: {Colors.PRIMARY};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 13px;
                    padding: 0 20px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.PRIMARY_HOVER};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.PRIMARY_DARK};
                }}
                QPushButton:disabled {{
                    background-color: #666666;
                    color: #888888;
                }}
            """,
            "secondary": f"""
                QPushButton {{
                    background-color: #444444;
                    color: {ModernColors.TEXT_PRIMARY};
                    border: 1px solid #666666;
                    border-radius: 8px;
                    font-weight: 500;
                    font-size: 13px;
                    padding: 0 20px;
                }}
                QPushButton:hover {{
                    background-color: #555555;
                    border-color: {{Colors.PRIMARY}};
                }}
                QPushButton:pressed {{
                    background-color: #666666;
                }}
                QPushButton:disabled {{
                    background-color: #333333;
                    color: #888888;
                }}
            """,
            "success": f"""
                QPushButton {{
                    background-color: {Colors.SUCCESS};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 13px;
                    padding: 0 20px;
                }}
                QPushButton:hover {{
                    background-color: #059669;
                }}
                QPushButton:disabled {{
                    background-color: #666666;
                    color: #888888;
                }}
            """,
            "danger": f"""
                QPushButton {{
                    background-color: {{Colors.ERROR}};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 13px;
                    padding: 0 20px;
                }}
                QPushButton:hover {{
                    background-color: {ModernColors.ERROR};
                }}
                QPushButton:disabled {{
                    background-color: #666666;
                    color: #888888;
                }}
            """,
        }
        
        self.setStyleSheet(styles.get(self._variant, styles["primary"]))

