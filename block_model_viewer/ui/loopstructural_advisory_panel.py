"""
StructuralAdvisoryWidget - AI-Assisted Fault Suggestion Panel.

Suggests structural edits to the geologist to fix JORC compliance issues
by analyzing systematic model errors.

GeoX Panel Safety Rules:
- Panels initialize private state only (self._attr = None)
- Controllers bind data via explicit methods
- No assignments to @property without setter
"""

from __future__ import annotations

import logging
from typing import Optional, List, Callable, TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QMessageBox, QGroupBox,
    QDoubleSpinBox, QSpinBox, QProgressBar, QFrame
)
from PyQt6.QtGui import QColor, QBrush

if TYPE_CHECKING:
    from ..geology.fault_detection import SuggestedFault, FaultDetectionEngine

logger = logging.getLogger(__name__)


class StructuralAdvisoryWidget(QWidget):
    """
    Suggests structural edits to the geologist to fix JORC compliance issues.
    
    This panel analyzes high-error regions in the geological model and suggests
    potential fault planes that could explain the systematic misfits.
    
    Features:
    - Automatic fault detection from error clustering
    - Visual confidence indicators
    - One-click fault insertion into the model
    """
    
    # Signals
    fault_selected = pyqtSignal(object)  # Emits SuggestedFault
    fault_applied = pyqtSignal(object)  # Emits SuggestedFault after application
    detection_completed = pyqtSignal(list)  # Emits list of SuggestedFault
    
    def __init__(
        self,
        apply_callback: Optional[Callable[["SuggestedFault"], None]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
        
        # Private state (GeoX Panel Safety Rules)
        self._apply_callback = apply_callback
        self._suggestions: List["SuggestedFault"] = []
        self._detection_engine: Optional["FaultDetectionEngine"] = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the modern panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 1. Compact Dark Header
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border-left: 4px solid {ModernColors.SUCCESS};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        header_frame_layout = QVBoxLayout(header_frame)
        header_frame_layout.setSpacing(8)
        
        header_label = QLabel("🤖 Structural Advisory")
        header_label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: {ModernColors.SUCCESS};
                font-size: 14px;
                font-weight: 700;
            }}
        """)
        header_frame_layout.addWidget(header_label)
        
        info_label = QLabel(
            "AI-assisted fault detection from model errors"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 11px;
            }}
        """)
        header_frame_layout.addWidget(info_label)
        
        layout.addWidget(header_frame)
        
        # 2. Dark Detection Parameters
        params_group = QGroupBox("⚙️ Detection Parameters")
        params_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(12)
        params_layout.setContentsMargins(16, 20, 16, 16)
        
        # Error threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Error Threshold (m):"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.5, 20.0)
        self._threshold_spin.setValue(3.0)
        self._threshold_spin.setSingleStep(0.5)
        self._threshold_spin.setToolTip("Minimum misfit to consider as systematic error")
        thresh_layout.addWidget(self._threshold_spin)
        thresh_layout.addStretch()
        params_layout.addLayout(thresh_layout)
        
        # Cluster distance
        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Cluster Distance (m):"))
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setRange(10.0, 200.0)
        self._eps_spin.setValue(50.0)
        self._eps_spin.setSingleStep(10.0)
        self._eps_spin.setToolTip("Maximum distance between clustered error points")
        eps_layout.addWidget(self._eps_spin)
        eps_layout.addStretch()
        params_layout.addLayout(eps_layout)
        
        # Minimum cluster size
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min. Cluster Size:"))
        self._min_samples_spin = QSpinBox()
        self._min_samples_spin.setRange(3, 20)
        self._min_samples_spin.setValue(4)
        self._min_samples_spin.setToolTip("Minimum number of points to form a cluster")
        min_layout.addWidget(self._min_samples_spin)
        min_layout.addStretch()
        params_layout.addLayout(min_layout)
        
        layout.addWidget(params_group)
        
        # 3. Modern Run Detection Button
        detect_layout = QHBoxLayout()
        detect_layout.setSpacing(12)
        
        self._detect_btn = QPushButton("🔍 Detect Faults from Errors")
        self._detect_btn.clicked.connect(self._on_detect_clicked)
        self._detect_btn.setStyleSheet(f"""
            QPushButton {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-weight: 700;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background: {ModernColors.ACCENT_PRESSED};
            }}
        """)
        self._detect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        detect_layout.addWidget(self._detect_btn)
        detect_layout.addStretch()
        layout.addLayout(detect_layout)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        # 4. Dark Suggestions List
        suggestions_group = QGroupBox("💡 Suggested Faults")
        suggestions_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background: {ModernColors.CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        suggestions_layout = QVBoxLayout(suggestions_group)
        suggestions_layout.setSpacing(12)
        suggestions_layout.setContentsMargins(16, 20, 16, 16)
        
        self._suggestion_list = QListWidget()
        self._suggestion_list.setMinimumHeight(150)
        self._suggestion_list.setStyleSheet(f"""
            QListWidget {{
                background: {ModernColors.PANEL_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                padding: 6px;
                font-size: 12px;
            }}
            QListWidget::item {{
                background: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
                padding: 10px;
                margin: 3px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {ModernColors.CARD_HOVER};
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QListWidget::item:hover {{
                background: {ModernColors.BORDER};
            }}
        """)
        self._suggestion_list.itemSelectionChanged.connect(self._on_selection_changed)
        self._suggestion_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        suggestions_layout.addWidget(self._suggestion_list)
        
        # Modern Selection info
        self._selection_label = QLabel("")
        self._selection_label.setWordWrap(True)
        self._selection_label.setStyleSheet(f"""
            QLabel {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
                padding: 8px;
                border-radius: 6px;
                border: 1px solid {ModernColors.BORDER};
            }}
        """)
        suggestions_layout.addWidget(self._selection_label)
        
        layout.addWidget(suggestions_group)
        
        # 5. Modern Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        success_btn_style = f"""
            QPushButton {{
                background: {ModernColors.SUCCESS};
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: #4caf50;
            }}
            QPushButton:pressed {{
                background: #2d8e47;
            }}
            QPushButton:disabled {{
                background: {ModernColors.BORDER};
                color: {ModernColors.TEXT_DISABLED};
            }}
        """
        
        secondary_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_SECONDARY};
                border: 2px solid {ModernColors.BORDER};
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: {ModernColors.ACCENT_PRIMARY};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
            QPushButton:disabled {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_DISABLED};
                border-color: {ModernColors.BORDER};
            }}
        """
        
        self._apply_btn = QPushButton("✓ Insert Selected Fault")
        self._apply_btn.clicked.connect(self._on_apply)
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet(success_btn_style)
        self._apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._apply_btn)
        
        self._preview_btn = QPushButton("👁 Preview in 3D")
        self._preview_btn.clicked.connect(self._on_preview)
        self._preview_btn.setEnabled(False)
        self._preview_btn.setStyleSheet(secondary_btn_style)
        self._preview_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._preview_btn)
        
        button_layout.addStretch()
        
        self._clear_btn = QPushButton("🗑 Clear")
        self._clear_btn.clicked.connect(self._on_clear)
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {ModernColors.TEXT_SECONDARY};
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                color: {ModernColors.ERROR};
            }}
            QPushButton:pressed {{
                background: {ModernColors.BORDER};
            }}
        """)
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._clear_btn)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
    
    def set_apply_callback(
        self,
        callback: Callable[["SuggestedFault"], None]
    ) -> None:
        """
        Set the callback for applying suggested faults.
        
        Controller method - called by parent panel/controller.
        """
        self._apply_callback = callback
    
    def populate_suggestions(self, suggestions: List["SuggestedFault"]) -> None:
        """
        Populate the list with fault suggestions.
        
        Args:
            suggestions: List of SuggestedFault objects
        """
        self._suggestions = suggestions
        self._suggestion_list.clear()
        
        for s in suggestions:
            # Create list item with formatted text
            text = (
                f"{s.name}: Dip {s.dip:.1f}° → {s.dip_dir:.1f}° "
                f"(Confidence: {s.confidence:.0%})"
            )
            item = QListWidgetItem(text)
            
            # Store the fault object
            item.setData(Qt.ItemDataRole.UserRole, s)
            
            # Color based on confidence
            if s.confidence >= 0.8:
                item.setBackground(QBrush(QColor("#e8f5e9")))  # Green
            elif s.confidence >= 0.6:
                item.setBackground(QBrush(QColor("#fff3e0")))  # Orange
            else:
                item.setBackground(QBrush(QColor("#ffebee")))  # Red
            
            self._suggestion_list.addItem(item)
        
        # Update status
        if len(suggestions) == 0:
            self._selection_label.setText(
                "No structural patterns detected in error distribution."
            )
        else:
            self._selection_label.setText(
                f"Found {len(suggestions)} potential fault(s). "
                "Select one to view details."
            )
        
        # Emit signal
        self.detection_completed.emit(suggestions)
    
    def run_detection(self, misfit_data) -> List["SuggestedFault"]:
        """
        Run fault detection on misfit data.
        
        Args:
            misfit_data: DataFrame with X, Y, Z, residual_m columns
            
        Returns:
            List of SuggestedFault objects
        """
        from ..geology.fault_detection import FaultDetectionEngine
        
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(10)
        
        try:
            # Create engine with current parameters
            self._detection_engine = FaultDetectionEngine(
                error_threshold_m=self._threshold_spin.value(),
                cluster_eps=self._eps_spin.value(),
                cluster_min_samples=self._min_samples_spin.value(),
            )
            
            self._progress_bar.setValue(30)
            
            # Run detection
            suggestions = self._detection_engine.detect_potential_faults(misfit_data)
            
            self._progress_bar.setValue(90)
            
            # Populate list
            self.populate_suggestions(suggestions)
            
            self._progress_bar.setValue(100)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Fault detection failed: {e}")
            QMessageBox.warning(
                self,
                "Detection Failed",
                f"Fault detection encountered an error:\n{e}"
            )
            return []
            
        finally:
            self._progress_bar.setVisible(False)
    
    def _on_detect_clicked(self) -> None:
        """Handle detect button click."""
        # This is emitted as a signal - the parent should provide misfit data
        logger.info("Fault detection requested - parent should provide misfit data")
        # In practice, the parent panel connects this and calls run_detection()
    
    def _on_selection_changed(self) -> None:
        """Handle selection change in the list."""
        selected = self._suggestion_list.currentItem()
        
        if selected:
            fault = selected.data(Qt.ItemDataRole.UserRole)
            self._apply_btn.setEnabled(True)
            self._preview_btn.setEnabled(True)
            
            # Show detailed info
            self._selection_label.setText(
                f"<b>{fault.name}</b><br>"
                f"Center: ({fault.center[0]:.1f}, {fault.center[1]:.1f}, {fault.center[2]:.1f})<br>"
                f"Dip: {fault.dip:.1f}° → {fault.dip_dir:.1f}° (Strike: {fault.strike:.1f}°)<br>"
                f"Confidence: {fault.confidence:.0%} based on {fault.n_points} error points<br>"
                f"Avg. Misfit: {fault.avg_misfit:.2f} m"
            )
            
            # Emit signal
            self.fault_selected.emit(fault)
        else:
            self._apply_btn.setEnabled(False)
            self._preview_btn.setEnabled(False)
            self._selection_label.setText("")
    
    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle double-click to apply fault."""
        fault = item.data(Qt.ItemDataRole.UserRole)
        if fault:
            self._apply_fault(fault)
    
    def _on_apply(self) -> None:
        """Handle apply button click."""
        selected = self._suggestion_list.currentItem()
        if selected:
            fault = selected.data(Qt.ItemDataRole.UserRole)
            self._apply_fault(fault)
    
    def _apply_fault(self, fault: "SuggestedFault") -> None:
        """Apply the selected fault to the model."""
        if self._apply_callback:
            self._apply_callback(fault)
            
            QMessageBox.information(
                self,
                "Audit Update",
                f"Fault '{fault.name}' added to Event Stack.\n\n"
                "Re-solve the model to verify compliance improvement."
            )
            
            # Emit signal
            self.fault_applied.emit(fault)
        else:
            QMessageBox.warning(
                self,
                "No Callback",
                "No apply callback configured. Cannot insert fault."
            )
    
    def _on_preview(self) -> None:
        """Preview the selected fault in 3D."""
        selected = self._suggestion_list.currentItem()
        if selected:
            fault = selected.data(Qt.ItemDataRole.UserRole)
            # Emit selection signal - parent should handle visualization
            self.fault_selected.emit(fault)
            logger.info(f"Preview requested for {fault.name}")
    
    def _on_clear(self) -> None:
        """Clear all suggestions."""
        self._suggestions = []
        self._suggestion_list.clear()
        self._selection_label.setText("")
        self._apply_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
    
    def get_selected_fault(self) -> Optional["SuggestedFault"]:
        """Get the currently selected fault, if any."""
        selected = self._suggestion_list.currentItem()
        if selected:
            return selected.data(Qt.ItemDataRole.UserRole)
        return None
    
    def get_all_suggestions(self) -> List["SuggestedFault"]:
        """Get all current suggestions."""
        return self._suggestions.copy()

