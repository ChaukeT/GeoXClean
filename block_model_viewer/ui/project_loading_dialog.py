"""
Modern Project Loading Progress Dialog for GeoX.

Shows detailed progress when opening a project with a polished, dark-themed UI.
"""

import logging
from typing import List, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QFrame, QApplication, QScrollArea, QWidget,
    QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QPalette

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


# ============================================================================
# MODERN DARK THEME COLORS
# ============================================================================
class ThemeColors:
    """Modern dark theme color palette."""
    # Backgrounds
    BG_DARK = "#1a1a1a"
    BG_SURFACE = "#242424"
    BG_ELEVATED = f"{ModernColors.CARD_BG}"
    BG_HOVER = "#353535"
    
    # Accents
    PRIMARY = f"{ModernColors.ACCENT_PRIMARY}"  # Blue
    PRIMARY_GLOW = "#60a5fa"
    SUCCESS = f"{ModernColors.SUCCESS}"  # Green
    SUCCESS_GLOW = "#34d399"
    WARNING = f"{ModernColors.WARNING}"  # Amber
    ERROR = f"{ModernColors.ERROR}"    # Red
    
    # Text
    TEXT_PRIMARY = f"{ModernColors.TEXT_PRIMARY}"
    TEXT_SECONDARY = f"{ModernColors.TEXT_SECONDARY}"
    TEXT_MUTED = f"{ModernColors.TEXT_HINT}"
    
    # Borders
    BORDER = f"{ModernColors.BORDER}"
    BORDER_ACCENT = "#4d4d4d"


class ModernStepIndicator(QFrame):
    """Individual step indicator with icon and label."""
    
    def __init__(self, step_number: int, label: str, parent=None):
        super().__init__(parent)
        self.step_number = step_number
        self.label_text = label
        self._state = "pending"  # pending, active, complete
        self._setup_ui()
    


    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""
        
                    ProcessListItem {{
                        background-color: {ThemeColors.BG_ELEVATED};
                        border-radius: 6px;
                        margin: 2px 0;
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
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(10)
        
        # Step circle
        self.circle = QLabel()
        self.circle.setFixedSize(24, 24)
        self.circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.circle)
        
        # Step label
        self.label = QLabel(self.label_text)
        self.label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.label, 1)
        
        self._update_style()
    
    def set_state(self, state: str):
        """Set state: pending, active, or complete."""
        self._state = state
        self._update_style()
    
    def _update_style(self):
        if self._state == "complete":
            self.circle.setText("✓")
            self.circle.setStyleSheet(f"""
                background-color: {ThemeColors.SUCCESS};
                border-radius: 12px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            """)
            self.label.setStyleSheet(f"color: {ThemeColors.SUCCESS};")
        elif self._state == "active":
            self.circle.setText(str(self.step_number))
            self.circle.setStyleSheet(f"""
                background-color: {ThemeColors.PRIMARY};
                border-radius: 12px;
                color: white;
                font-size: 11px;
                font-weight: bold;
            """)
            self.label.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY}; font-weight: 600;")
        else:  # pending
            self.circle.setText(str(self.step_number))
            self.circle.setStyleSheet(f"""
                background-color: {ThemeColors.BG_ELEVATED};
                border: 2px solid {ThemeColors.BORDER};
                border-radius: 12px;
                color: {ThemeColors.TEXT_MUTED};
                font-size: 11px;
            """)
            self.label.setStyleSheet(f"color: {ThemeColors.TEXT_MUTED};")


class ProcessListItem(QFrame):
    """Single process item in the process history list."""
    
    def __init__(self, process_data: dict, index: int, parent=None):
        super().__init__(parent)
        self._setup_ui(process_data, index)
    
    def _setup_ui(self, process: dict, index: int):
        self.setFixedHeight(42)
        self.setStyleSheet(self._get_stylesheet())
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)
        
        # Status icon
        status = process.get('status', 'unknown')
        status_icon = QLabel()
        status_icon.setFixedSize(20, 20)
        status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if status == 'success':
            status_icon.setText("✓")
            status_icon.setStyleSheet(f"""
                background-color: {ThemeColors.SUCCESS};
                border-radius: 10px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            """)
        elif status == 'failed':
            status_icon.setText("✗")
            status_icon.setStyleSheet(f"""
                background-color: {ThemeColors.ERROR};
                border-radius: 10px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            """)
        else:
            status_icon.setText("○")
            status_icon.setStyleSheet(f"""
                background-color: {ThemeColors.BG_HOVER};
                border-radius: 10px;
                color: {ThemeColors.TEXT_MUTED};
                font-size: 12px;
            """)
        
        layout.addWidget(status_icon)
        
        # Process name
        task_name = process.get('task_name', 'Unknown Process')
        name_label = QLabel(task_name)
        name_label.setFont(QFont("Segoe UI", 10))
        name_label.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY};")
        layout.addWidget(name_label, 1)
        
        # Duration (if available)
        duration = process.get('duration_seconds')
        if duration and isinstance(duration, (int, float)):
            if duration < 1:
                duration_text = f"{duration:.2f}s"
            else:
                duration_text = f"{duration:.1f}s"
            duration_label = QLabel(duration_text)
            duration_label.setFont(QFont("Segoe UI", 9))
            duration_label.setStyleSheet(f"color: {ThemeColors.TEXT_MUTED};")
            layout.addWidget(duration_label)


class ModernProgressBar(QFrame):
    """Modern styled progress bar with glow effect."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setFixedHeight(8)
        self.setStyleSheet(f"""
            ModernProgressBar {{
                background-color: {ThemeColors.BG_ELEVATED};
                border-radius: 4px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.bar = QProgressBar()
        self.bar.setFixedHeight(8)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ThemeColors.PRIMARY},
                    stop:1 {ThemeColors.PRIMARY_GLOW});
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.bar)
    
    def setValue(self, value: int):
        self.bar.setValue(value)
    
    def setRange(self, min_val: int, max_val: int):
        self.bar.setRange(min_val, max_val)


class ProjectLoadingDialog(QDialog):
    """
    Modern progress dialog for project loading operations.

    Features a sleek dark theme, step indicators, animated progress,
    and a polished process history display.
    """

    cancelled = pyqtSignal()

    def __init__(self, project_name: str, parent=None):
        super().__init__(parent)

        self.project_name = project_name
        self.current_step = 0
        self.total_steps = 0
        self.is_cancelled = False
        self.step_indicators: List[ModernStepIndicator] = []

        self.setWindowTitle(f"Opening Project")
        self.setModal(True)
        self.setFixedSize(550, 620)

        # Modern window appearance
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._setup_ui()
        self._center_on_screen()
        self._start_animations()

        logger.info(f"Project loading dialog initialized for: {project_name}")

    def _setup_ui(self):
        """Setup the modern user interface."""
        # Outer layout (for shadow/margin)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        
        # Main container with shadow
        container = QFrame()
        container.setObjectName("MainContainer")
        container.setStyleSheet(f"""
            #MainContainer {{
                background-color: {ThemeColors.BG_DARK};
                border: 1px solid {ThemeColors.BORDER};
                border-radius: 16px;
            }}
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 4)
        container.setGraphicsEffect(shadow)
        
        outer_layout.addWidget(container)
        
        # Container layout
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)

        # ===== Header Section =====
        header = QFrame()
        header.setStyleSheet(f"""
            background-color: {ThemeColors.BG_SURFACE};
            border-radius: 12px;
            padding: 12px;
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(8)
        
        # Project icon + name row
        title_row = QHBoxLayout()
        
        # Animated folder icon
        self.icon_label = QLabel("📂")
        self.icon_label.setStyleSheet("font-size: 32px;")
        title_row.addWidget(self.icon_label)
        
        title_text = QVBoxLayout()
        title_text.setSpacing(2)
        
        opening_label = QLabel("Opening Project")
        opening_label.setFont(QFont("Segoe UI", 11))
        opening_label.setStyleSheet(f"color: {ThemeColors.TEXT_SECONDARY};")
        title_text.addWidget(opening_label)
        
        self.project_label = QLabel(self.project_name)
        project_font = QFont("Segoe UI Semibold", 16)
        project_font.setBold(True)
        self.project_label.setFont(project_font)
        self.project_label.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY};")
        title_text.addWidget(self.project_label)
        
        title_row.addLayout(title_text, 1)
        header_layout.addLayout(title_row)
        
        layout.addWidget(header)

        # ===== Progress Section =====
        progress_section = QFrame()
        progress_section.setStyleSheet(f"""
            background-color: {ThemeColors.BG_SURFACE};
            border-radius: 12px;
        """)
        progress_layout = QVBoxLayout(progress_section)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.setSpacing(12)
        
        # Current step label
        self.step_label = QLabel("Initializing...")
        self.step_label.setFont(QFont("Segoe UI", 11))
        self.step_label.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY};")
        progress_layout.addWidget(self.step_label)
        
        # Progress bar
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        # Percentage label
        progress_info = QHBoxLayout()
        self.percent_label = QLabel("0%")
        self.percent_label.setFont(QFont("Segoe UI Semibold", 10))
        self.percent_label.setStyleSheet(f"color: {ThemeColors.PRIMARY};")
        progress_info.addWidget(self.percent_label)
        progress_info.addStretch()
        progress_layout.addLayout(progress_info)
        
        # Step indicators
        steps_container = QFrame()
        steps_container.setStyleSheet(f"""
            background-color: {ThemeColors.BG_DARK};
            border-radius: 8px;
            padding: 4px;
        """)
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(4, 8, 4, 8)
        steps_layout.setSpacing(2)
        
        default_steps = [
            "Reading project file",
            "Loading metadata",
            "Loading process history",
            "Preparing data",
            "Loading block model",
            "Loading drillholes",
            "Restoring analysis",
            "Finalizing"
        ]
        
        for i, step_text in enumerate(default_steps, 1):
            indicator = ModernStepIndicator(i, step_text)
            self.step_indicators.append(indicator)
            steps_layout.addWidget(indicator)
        
        progress_layout.addWidget(steps_container)
        layout.addWidget(progress_section)

        # ===== Process History Section =====
        history_section = QFrame()
        history_section.setStyleSheet(f"""
            background-color: {ThemeColors.BG_SURFACE};
            border-radius: 12px;
        """)
        history_layout = QVBoxLayout(history_section)
        history_layout.setContentsMargins(16, 12, 16, 12)
        history_layout.setSpacing(8)
        
        # Header
        history_header = QHBoxLayout()
        history_title = QLabel("Process History")
        history_title.setFont(QFont("Segoe UI Semibold", 11))
        history_title.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY};")
        history_header.addWidget(history_title)
        
        self.process_count_label = QLabel("0 processes")
        self.process_count_label.setFont(QFont("Segoe UI", 10))
        self.process_count_label.setStyleSheet(f"color: {ThemeColors.TEXT_MUTED};")
        history_header.addStretch()
        history_header.addWidget(self.process_count_label)
        history_layout.addLayout(history_header)
        
        # Process list scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {ThemeColors.BG_DARK};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {ThemeColors.BORDER_ACCENT};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        scroll_area.setFixedHeight(100)
        
        self.process_container = QWidget()
        self.process_layout = QVBoxLayout(self.process_container)
        self.process_layout.setContentsMargins(0, 0, 4, 0)
        self.process_layout.setSpacing(4)
        self.process_layout.addStretch()
        
        scroll_area.setWidget(self.process_container)
        history_layout.addWidget(scroll_area)
        
        # Empty state message
        self.empty_message = QLabel("No processes found in this project")
        self.empty_message.setFont(QFont("Segoe UI", 10))
        self.empty_message.setStyleSheet(f"color: {ThemeColors.TEXT_MUTED};")
        self.empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.process_layout.insertWidget(0, self.empty_message)
        
        layout.addWidget(history_section)

        # ===== Footer =====
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 8, 0, 0)
        footer.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedSize(100, 36)
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ThemeColors.BG_ELEVATED};
                color: {ThemeColors.TEXT_SECONDARY};
                border: 1px solid {ThemeColors.BORDER};
                border-radius: 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {ThemeColors.BG_HOVER};
                border-color: {ThemeColors.BORDER_ACCENT};
                color: {ThemeColors.TEXT_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {ThemeColors.BG_DARK};
            }}
        """)
        footer.addWidget(self.cancel_button)
        
        layout.addLayout(footer)

    def _center_on_screen(self):
        """Center the dialog on the screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        dialog_geometry = self.frameGeometry()
        center_point = screen.center()
        dialog_geometry.moveCenter(center_point)
        self.move(dialog_geometry.topLeft())

    def _start_animations(self):
        """Start subtle UI animations."""
        # Pulse the icon periodically
        self._icon_timer = QTimer(self)
        self._icon_timer.timeout.connect(self._pulse_icon)
        self._icon_timer.start(800)
        self._icon_state = False
    
    def _pulse_icon(self):
        """Subtle icon animation."""
        self._icon_state = not self._icon_state
        if self._icon_state:
            self.icon_label.setText("📁")
        else:
            self.icon_label.setText("📂")

    def set_total_steps(self, total_steps: int):
        """Set the total number of loading steps."""
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def set_detailed_progress(self, current: int, total: int, message: str):
        """Set detailed progress with current/total and message."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.percent_label.setText(f"{percent}%")
        
        self.step_label.setText(message)
        
        # Update step indicators
        for i, indicator in enumerate(self.step_indicators):
            if i < current - 1:
                indicator.set_state("complete")
            elif i == current - 1:
                indicator.set_state("active")
            else:
                indicator.set_state("pending")

        QApplication.processEvents()

    def update_progress(self, step_name: str, increment: bool = True):
        """Update the progress with a new step."""
        if increment:
            self.current_step += 1

        self.step_label.setText(step_name)

        if self.total_steps > 0:
            percent = int((self.current_step / self.total_steps) * 100)
            self.progress_bar.setValue(percent)
            self.percent_label.setText(f"{percent}%")

        QApplication.processEvents()
        logger.debug(f"Project loading progress: {step_name} ({self.current_step}/{self.total_steps})")

    def set_process_history(self, processes: List[dict]):
        """Set the list of processes to display."""
        # Clear existing items
        while self.process_layout.count() > 1:  # Keep stretch at end
            item = self.process_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not processes:
            self.empty_message = QLabel("No processes found in this project")
            self.empty_message.setFont(QFont("Segoe UI", 10))
            self.empty_message.setStyleSheet(f"color: {ThemeColors.TEXT_MUTED};")
            self.empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.process_layout.insertWidget(0, self.empty_message)
            self.process_count_label.setText("0 processes")
            return
        
        # Add process items
        for i, process in enumerate(processes):
            item = ProcessListItem(process, i)
            self.process_layout.insertWidget(i, item)
        
        # Update count
        successful = sum(1 for p in processes if p.get('status') == 'success')
        self.process_count_label.setText(f"{len(processes)} processes ({successful} ✓)")
        
        logger.info(f"Displayed {len(processes)} processes in loading dialog")

    def add_process_info(self, process_info: str):
        """Add information about a process being loaded."""
        # Create a simple process dict for display
        process = {'task_name': process_info, 'status': 'pending'}
        idx = self.process_layout.count() - 1  # Before stretch
        item = ProcessListItem(process, idx)
        self.process_layout.insertWidget(idx, item)
        QApplication.processEvents()

    def mark_process_complete(self, process_info: str):
        """Mark a process as completed in the display."""
        # Update existing item (simplified - could be enhanced)
        QApplication.processEvents()

    def complete_loading(self):
        """Mark the loading as complete."""
        self.step_label.setText("✓ Project opened successfully!")
        self.step_label.setStyleSheet(f"color: {ThemeColors.SUCCESS}; font-weight: 600;")
        self.progress_bar.setValue(100)
        self.percent_label.setText("100%")
        self.percent_label.setStyleSheet(f"color: {ThemeColors.SUCCESS};")
        
        # Update icon
        self.icon_label.setText("✅")
        if hasattr(self, '_icon_timer'):
            self._icon_timer.stop()
        
        # Mark all steps complete
        for indicator in self.step_indicators:
            indicator.set_state("complete")
        
        # Change button to Close
        self.cancel_button.setText("Close")
        self.cancel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ThemeColors.SUCCESS};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {ThemeColors.SUCCESS_GLOW};
            }}
        """)

        # Auto-close after a short delay
        QTimer.singleShot(1200, self.accept)

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        if hasattr(self, '_icon_timer'):
            self._icon_timer.stop()
        self.is_cancelled = True
        self.cancelled.emit()
        self.reject()

    def was_cancelled(self) -> bool:
        """Check if the loading was cancelled by the user."""
        return self.is_cancelled
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if hasattr(self, '_icon_timer'):
            self._icon_timer.stop()
        super().closeEvent(event)
