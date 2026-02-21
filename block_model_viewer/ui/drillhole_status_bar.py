"""
Modern Status Bar System for Drillhole Processing Operations.

Provides a reusable status bar component with:
- Multi-step progress tracking
- Visual indicators for each processing stage
- Real-time status updates
- Modern styling consistent with GeoX design
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Callable, Any
from enum import Enum
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ProcessStage(Enum):
    """Process stages for drillhole operations."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


class StatusLevel(Enum):
    """Status message levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ProcessStep:
    """Individual step in a multi-step process."""
    id: str
    name: str
    description: str
    estimated_duration: float = 1.0  # seconds
    completed: bool = False
    error: Optional[str] = None


class ModernStatusBar(QFrame):
    """
    Modern status bar for drillhole processing operations.
    
    Features:
    - Multi-step progress tracking
    - Real-time status updates
    - Visual step indicators
    - Modern styling with animations
    """
    
    # Signals
    cancel_requested = pyqtSignal()
    step_completed = pyqtSignal(str)  # step_id
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_animations()
        
        # State
        self._current_stage = ProcessStage.IDLE
        self._current_step_index = 0
        self._steps: List[ProcessStep] = []
        self._start_time: Optional[float] = None
        self._cancellable = False
        
        # Progress tracking
        self._overall_progress = 0
        self._step_progress = 0
        
        # Animation timer
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_animation)
        


    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""

                    ModernStatusBar {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #2a2a2a,
                            stop:1 {ModernColors.PANEL_BG});
                        border: 1px solid #444444;
                        border-radius: 8px;
                        margin: 2px;
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
        """Setup the user interface."""
        self.setFixedHeight(100)
        self.setStyleSheet(self._get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        # Header row with status and cancel button
        header_layout = QHBoxLayout()
        
        # Status icon and text
        self._status_icon = QLabel("●")
        self._status_icon.setFixedSize(12, 12)
        self._status_icon.setStyleSheet("color: #888888; font-size: 14px;")
        header_layout.addWidget(self._status_icon)
        
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"""
            color: {ModernColors.TEXT_PRIMARY};
            font-size: 11px;
            font-weight: 600;
        """)
        header_layout.addWidget(self._status_label)
        
        header_layout.addStretch()
        
        # Time elapsed
        self._time_label = QLabel("")
        self._time_label.setStyleSheet("""
            color: #999999;
            font-size: 9px;
        """)
        header_layout.addWidget(self._time_label)
        
        # Cancel button (hidden by default)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedSize(60, 22)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 9px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #c62828;
            }
        """)
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._cancel_btn.setVisible(False)
        header_layout.addWidget(self._cancel_btn)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #333333;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50,
                    stop:1 #81C784);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar)
        
        # Current step details
        step_layout = QHBoxLayout()
        
        self._step_label = QLabel("")
        self._step_label.setStyleSheet("""
            color: #cccccc;
            font-size: 10px;
        """)
        step_layout.addWidget(self._step_label)
        
        step_layout.addStretch()
        
        self._step_counter = QLabel("")
        self._step_counter.setStyleSheet("""
            color: #888888;
            font-size: 9px;
        """)
        step_layout.addWidget(self._step_counter)
        
        layout.addLayout(step_layout)
        
    def _setup_animations(self):
        """Setup UI animations."""
        self._opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_animation.setDuration(200)
        self._opacity_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        
    def set_process_steps(self, steps: List[ProcessStep]):
        """Set the steps for the current process."""
        self._steps = steps
        self._current_step_index = 0
        self._update_display()
        
    def start_process(self, cancellable: bool = True):
        """Start the process with the configured steps."""
        import time
        
        self._start_time = time.time()
        self._current_stage = ProcessStage.INITIALIZING
        self._current_step_index = 0
        self._cancellable = cancellable
        
        self._cancel_btn.setVisible(cancellable)
        self._pulse_timer.start(500)  # Pulse every 500ms
        
        self._update_display()
        logger.debug(f"Started process with {len(self._steps)} steps")
        
    def advance_step(self, step_progress: float = 1.0):
        """Advance to the next step or update current step progress."""
        if not self._steps:
            return
            
        if step_progress >= 1.0 and self._current_step_index < len(self._steps):
            # Complete current step
            if self._current_step_index < len(self._steps):
                self._steps[self._current_step_index].completed = True
                self.step_completed.emit(self._steps[self._current_step_index].id)
            
            # Move to next step
            self._current_step_index += 1
            
        # Update progress
        self._step_progress = step_progress
        self._calculate_overall_progress()
        self._update_display()
        
        # Check if all steps completed
        if self._current_step_index >= len(self._steps):
            self.finish_process(success=True)
            
    def set_step_error(self, error_message: str):
        """Mark the current step as having an error."""
        if self._current_step_index < len(self._steps):
            self._steps[self._current_step_index].error = error_message
        
        self._current_stage = ProcessStage.ERROR
        self._pulse_timer.stop()
        self._update_display()
        
    def finish_process(self, success: bool = True, message: str = ""):
        """Finish the current process."""
        import time
        
        if success:
            self._current_stage = ProcessStage.COMPLETED
            self._overall_progress = 100
        else:
            self._current_stage = ProcessStage.ERROR
            
        self._pulse_timer.stop()
        self._cancel_btn.setVisible(False)
        
        if self._start_time:
            elapsed = time.time() - self._start_time
            self._time_label.setText(f"Completed in {elapsed:.1f}s")
            
        if message:
            self._status_label.setText(message)
            
        self._update_display()
        
    def update_status(self, message: str, level: StatusLevel = StatusLevel.INFO):
        """Update the status message."""
        self._status_label.setText(message)
        
        # Update status icon color based on level
        colors = {
            StatusLevel.INFO: "#4CAF50",
            StatusLevel.WARNING: "#FF9800", 
            StatusLevel.ERROR: "#f44336",
            StatusLevel.SUCCESS: "#4CAF50"
        }
        
        color = colors.get(level, "#888888")
        self._status_icon.setStyleSheet(f"color: {color}; font-size: 14px;")
        
    def _calculate_overall_progress(self):
        """Calculate the overall progress percentage."""
        if not self._steps:
            self._overall_progress = 0
            return
            
        completed_steps = sum(1 for step in self._steps if step.completed)
        
        # Add partial progress from current step
        current_step_contribution = 0
        if self._current_step_index < len(self._steps):
            current_step_contribution = self._step_progress
            
        total_progress = completed_steps + current_step_contribution
        self._overall_progress = int((total_progress / len(self._steps)) * 100)
        
    def _update_display(self):
        """Update the visual display based on current state."""
        # Update progress bar
        self._progress_bar.setValue(self._overall_progress)
        
        # Update step counter
        if self._steps:
            current = min(self._current_step_index + 1, len(self._steps))
            total = len(self._steps)
            self._step_counter.setText(f"Step {current} of {total}")
        else:
            self._step_counter.setText("")
            
        # Update current step label
        if self._current_step_index < len(self._steps):
            current_step = self._steps[self._current_step_index]
            if current_step.error:
                self._step_label.setText(f"❌ {current_step.name}: {current_step.error}")
                self._step_label.setStyleSheet("color: #f44336; font-size: 10px;")
            else:
                self._step_label.setText(current_step.description)
                self._step_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        elif self._current_stage == ProcessStage.COMPLETED:
            self._step_label.setText("✅ Process completed successfully")
            self._step_label.setStyleSheet("color: #4CAF50; font-size: 10px;")
        else:
            self._step_label.setText("")
            
        # Update progress bar styling based on stage
        if self._current_stage == ProcessStage.ERROR:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #333333;
                    border: none;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background-color: #f44336;
                    border-radius: 4px;
                }
            """)
        else:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #333333;
                    border: none;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4CAF50,
                        stop:1 #81C784);
                    border-radius: 4px;
                }
            """)
            
    def _pulse_animation(self):
        """Animate the status icon during processing."""
        if self._current_stage in [ProcessStage.PROCESSING, ProcessStage.VALIDATING]:
            # Simple color pulse effect
            colors = ["#4CAF50", "#81C784", "#A5D6A7"]
            import time
            color_index = int(time.time() * 2) % len(colors)
            color = colors[color_index]
            self._status_icon.setStyleSheet(f"color: {color}; font-size: 14px;")


class DrillholeProcessStatusBar(ModernStatusBar):
    """
    Specialized status bar for drillhole processing operations.
    
    Provides predefined step templates for common operations like:
    - QC validation
    - Compositing
    - Declustering
    """
    
    @classmethod
    def create_for_qc(cls, parent: Optional[QWidget] = None) -> 'DrillholeProcessStatusBar':
        """Create status bar configured for QC operations."""
        status_bar = cls(parent)
        
        qc_steps = [
            ProcessStep("data_check", "Data Check", "Validating input data structure"),
            ProcessStep("collar_validation", "Collar Validation", "Checking collar coordinates and depths"),
            ProcessStep("survey_validation", "Survey Validation", "Validating survey data and trajectories"),  
            ProcessStep("assay_validation", "Assay Validation", "Checking sample intervals and grades"),
            ProcessStep("lithology_validation", "Lithology Validation", "Validating geological intervals"),
            ProcessStep("cross_validation", "Cross Validation", "Checking consistency across tables"),
            ProcessStep("report_generation", "Report Generation", "Generating validation report"),
        ]
        
        status_bar.set_process_steps(qc_steps)
        return status_bar
        
    @classmethod
    def create_for_compositing(cls, parent: Optional[QWidget] = None) -> 'DrillholeProcessStatusBar':
        """Create status bar configured for compositing operations."""
        status_bar = cls(parent)
        
        compositing_steps = [
            ProcessStep("validation_check", "Validation Check", "Checking data validation status"),
            ProcessStep("data_preparation", "Data Preparation", "Preparing intervals for compositing"),
            ProcessStep("composite_calculation", "Composite Calculation", "Computing composite intervals"),
            ProcessStep("grade_calculation", "Grade Calculation", "Calculating weighted averages"),
            ProcessStep("quality_check", "Quality Check", "Validating composite results"),
            ProcessStep("output_generation", "Output Generation", "Generating final composites"),
        ]
        
        status_bar.set_process_steps(compositing_steps)
        return status_bar
        
    @classmethod  
    def create_for_declustering(cls, parent: Optional[QWidget] = None) -> 'DrillholeProcessStatusBar':
        """Create status bar configured for declustering operations."""
        status_bar = cls(parent)
        
        declustering_steps = [
            ProcessStep("lineage_check", "Lineage Check", "Validating data source and lineage"),
            ProcessStep("coordinate_validation", "Coordinate Validation", "Checking sample coordinates"),
            ProcessStep("cell_assignment", "Cell Assignment", "Assigning samples to declustering cells"),
            ProcessStep("weight_calculation", "Weight Calculation", "Computing declustering weights"),
            ProcessStep("statistics_calculation", "Statistics Calculation", "Generating summary statistics"),
            ProcessStep("diagnostics_generation", "Diagnostics Generation", "Creating diagnostic outputs"),
        ]
        
        status_bar.set_process_steps(declustering_steps)
        return status_bar


# Utility function for creating callback functions
def create_progress_callback(status_bar: ModernStatusBar) -> Callable[[float, str], None]:
    """Create a progress callback function for backend engines."""
    def progress_callback(fraction: float, message: str = ""):
        if message:
            status_bar.update_status(message)
        status_bar.advance_step(fraction)
    
    return progress_callback