"""
Modern Toolbar with icons, dropdowns, and status strip for BlockModelViewer.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QToolButton, QLabel,
    QSpacerItem, QSizePolicy, QMenu, QComboBox, QFrame, QSlider, QSpinBox,
    QCheckBox, QButtonGroup, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
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

    # Quick-action icon button signals
    new_scene_requested = pyqtSignal()
    data_table_requested = pyqtSignal()
    refresh_requested = pyqtSignal()
    maximize_requested = pyqtSignal()
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    fit_view_icon_requested = pyqtSignal()

    # Grade cutoff signals
    grade_cutoff_changed = pyqtSignal(float)
    grade_cutoff_toggled = pyqtSignal(bool)
    grade_cutoff_reset = pyqtSignal()
    min_cluster_changed = pyqtSignal(int)

    # Mouse mode signals (from March Pictures)
    mouse_mode_requested = pyqtSignal(str)  # "select", "pan", "rotate", "zoom_box"
    # Alias for fit-view (dock_setup expects zoom_fit_requested per spec)
    zoom_fit_requested = pyqtSignal()
    
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

        # ── Quick-action icon buttons ──
        icon_defs = [
            ("+",  "New scene",       self.new_scene_requested),
            ("≡",  "Data table",      self.data_table_requested),
            ("↻",  "Refresh",         self.refresh_requested),
            ("□",  "Maximize viewer", self.maximize_requested),
            ("+",  "Zoom in",         self.zoom_in_requested),
            ("−",  "Zoom out",        self.zoom_out_requested),
            ("✕",  "Fit view",        self.fit_view_icon_requested),
        ]
        for label, tip, signal in icon_defs:
            btn = QToolButton()
            btn.setText(label)
            btn.setToolTip(tip)
            btn.setFixedSize(28, 28)
            btn.clicked.connect(signal.emit)
            toolbar_layout.addWidget(btn)

        # Separator before mouse mode buttons
        separator_mm = QFrame()
        separator_mm.setFrameShape(QFrame.Shape.VLine)
        separator_mm.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar_layout.addWidget(separator_mm)

        # ── Mouse Mode Buttons (from March Pictures) ─────────────────
        # Four mutually-exclusive buttons for Select / Pan / Rotate / ZoomBox
        self._mouse_mode_group = QButtonGroup(self)
        self._mouse_mode_group.setExclusive(True)
        self._mouse_mode_buttons = {}
        # Icon chars and tooltips match the March Pictures spec
        mode_defs = [
            ("select",   "\u271a", "Select / Pick (click blocks)"),
            ("pan",      "\u2630", "Pan (middle-click drag)"),
            ("rotate",   "\u21bb", "Rotate / Trackball (left-click drag)"),
            ("zoom_box", "\u25a1", "Zoom Box — drag a rectangle to zoom (Z)"),
        ]
        for mode, label, tip in mode_defs:
            btn = QToolButton()
            btn.setText(label)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(28, 28)
            if mode == "rotate":
                btn.setChecked(True)  # default mode
            # Capture mode via default arg to avoid late-binding in lambda
            btn.clicked.connect(lambda _=False, m=mode: self.mouse_mode_requested.emit(m))
            self._mouse_mode_group.addButton(btn)
            self._mouse_mode_buttons[mode] = btn
            toolbar_layout.addWidget(btn)
        # Expose named handles matching March Pictures spec for sync_mode()
        self.select_btn = self._mouse_mode_buttons["select"]
        self.pan_btn = self._mouse_mode_buttons["pan"]
        self.rotate_btn = self._mouse_mode_buttons["rotate"]
        self.zoom_box_btn = self._mouse_mode_buttons["zoom_box"]

        # Add separator before View
        separator2b = QFrame()
        separator2b.setFrameShape(QFrame.Shape.VLine)
        separator2b.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar_layout.addWidget(separator2b)

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

        # Add separator before cutoff
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.Shape.VLine)
        separator4.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar_layout.addWidget(separator4)

        # ── Grade cutoff (advanced, from March Pictures) ─────────────
        # Decimal support (QDoubleSpinBox), 300ms debounce, log-scale for
        # ranges >100×, reset button, formatted display (%, ppm, sci notation)
        cutoff_label = QLabel("Grade cutoff:")
        cutoff_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px;")
        toolbar_layout.addWidget(cutoff_label)

        self.grade_cutoff_toggle = QCheckBox("On")
        self.grade_cutoff_toggle.setChecked(False)
        self.grade_cutoff_toggle.setToolTip("Enable/disable grade cutoff filtering")
        self.grade_cutoff_toggle.toggled.connect(self._on_cutoff_toggled)
        toolbar_layout.addWidget(self.grade_cutoff_toggle)

        self.grade_cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self.grade_cutoff_slider.setMinimum(0)
        self.grade_cutoff_slider.setMaximum(1000)
        self.grade_cutoff_slider.setValue(0)
        self.grade_cutoff_slider.setFixedWidth(120)
        self.grade_cutoff_slider.setToolTip("Adjust grade cutoff value")
        self.grade_cutoff_slider.valueChanged.connect(self._on_cutoff_slider_changed)
        toolbar_layout.addWidget(self.grade_cutoff_slider)

        # Decimal spin box (replaces integer-only QSpinBox from Documents)
        self.grade_cutoff_spin = QDoubleSpinBox()
        self.grade_cutoff_spin.setDecimals(3)
        self.grade_cutoff_spin.setMinimum(0.0)
        self.grade_cutoff_spin.setMaximum(1e9)
        self.grade_cutoff_spin.setFixedWidth(90)
        self.grade_cutoff_spin.setToolTip("Grade cutoff value (decimal)")
        self.grade_cutoff_spin.valueChanged.connect(self._on_cutoff_spin_changed)
        toolbar_layout.addWidget(self.grade_cutoff_spin)

        # Formatted display label (%, ppm, sci notation)
        self.grade_cutoff_value_label = QLabel("0.00")
        self.grade_cutoff_value_label.setFixedWidth(60)
        self.grade_cutoff_value_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-size: 11px;")
        toolbar_layout.addWidget(self.grade_cutoff_value_label)

        # Reset button
        self.grade_cutoff_reset_btn = QToolButton()
        self.grade_cutoff_reset_btn.setText("⟲")
        self.grade_cutoff_reset_btn.setToolTip("Reset grade cutoff to show all data")
        self.grade_cutoff_reset_btn.setFixedSize(24, 24)
        self.grade_cutoff_reset_btn.clicked.connect(self._on_cutoff_reset)
        toolbar_layout.addWidget(self.grade_cutoff_reset_btn)

        # Debounce timer — prevents re-rendering on every tick
        self._cutoff_debounce = QTimer(self)
        self._cutoff_debounce.setSingleShot(True)
        self._cutoff_debounce.setInterval(300)  # 300ms
        self._cutoff_debounce.timeout.connect(self._emit_cutoff_value)
        self._syncing_mode = False  # guard against signal loops

        # ── Min cluster spinner ──
        cluster_label = QLabel("Min cluster:")
        cluster_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 11px;")
        toolbar_layout.addWidget(cluster_label)

        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setMinimum(1)
        self.min_cluster_spin.setMaximum(9999)
        self.min_cluster_spin.setValue(1)
        self.min_cluster_spin.setSuffix(" blks")
        self.min_cluster_spin.setToolTip("Minimum number of connected blocks to display")
        self.min_cluster_spin.valueChanged.connect(self.min_cluster_changed.emit)
        toolbar_layout.addWidget(self.min_cluster_spin)

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
    
    # ── Cutoff slider internals (advanced, from March Pictures) ───────

    _cutoff_min: float = 0.0
    _cutoff_max: float = 100.0
    _cutoff_log_scale: bool = False

    def _format_grade(self, value: float) -> str:
        """Format a grade value with appropriate units/notation."""
        if value == 0:
            return "0"
        abs_v = abs(value)
        if abs_v >= 1000:
            return f"{value:.0f}"
        if abs_v >= 1:
            return f"{value:.3f}"
        if abs_v >= 0.01:
            return f"{value:.4f}"
        if abs_v >= 1e-4:
            return f"{value*1e6:.1f} ppm"
        return f"{value:.2e}"  # scientific

    def _slider_to_grade(self, slider_value: int) -> float:
        """Convert slider integer position to actual grade (linear or log)."""
        frac = slider_value / 1000.0
        if self._cutoff_log_scale and self._cutoff_min > 0 and self._cutoff_max > 0:
            import math
            log_min = math.log10(max(self._cutoff_min, 1e-10))
            log_max = math.log10(self._cutoff_max)
            return 10 ** (log_min + frac * (log_max - log_min))
        return self._cutoff_min + frac * (self._cutoff_max - self._cutoff_min)

    def _grade_to_slider(self, grade: float) -> int:
        """Inverse: actual grade back to slider integer."""
        if self._cutoff_max == self._cutoff_min:
            return 0
        if self._cutoff_log_scale and self._cutoff_min > 0 and self._cutoff_max > 0:
            import math
            log_min = math.log10(max(self._cutoff_min, 1e-10))
            log_max = math.log10(self._cutoff_max)
            log_val = math.log10(max(grade, 1e-10))
            frac = (log_val - log_min) / (log_max - log_min)
        else:
            frac = (grade - self._cutoff_min) / (self._cutoff_max - self._cutoff_min)
        return int(max(0, min(1000, frac * 1000)))

    def _on_cutoff_toggled(self, checked: bool):
        self.grade_cutoff_toggle.setText("On" if checked else "Off")
        self.grade_cutoff_toggled.emit(checked)

    def _on_cutoff_slider_changed(self, value: int):
        if self._syncing_mode:
            return
        grade = self._slider_to_grade(value)
        self.grade_cutoff_value_label.setText(self._format_grade(grade))
        # Sync spin box without triggering its signal
        self._syncing_mode = True
        try:
            self.grade_cutoff_spin.setValue(grade)
        finally:
            self._syncing_mode = False
        self._cutoff_debounce.start()

    def _on_cutoff_spin_changed(self, grade: float):
        if self._syncing_mode:
            return
        self.grade_cutoff_value_label.setText(self._format_grade(grade))
        # Sync slider without triggering its signal
        self._syncing_mode = True
        try:
            self.grade_cutoff_slider.setValue(self._grade_to_slider(grade))
        finally:
            self._syncing_mode = False
        self._cutoff_debounce.start()

    def _on_cutoff_reset(self):
        """Reset cutoff to minimum (show all data) and emit reset signal."""
        self._syncing_mode = True
        try:
            self.grade_cutoff_slider.setValue(0)
            self.grade_cutoff_spin.setValue(self._cutoff_min)
            self.grade_cutoff_value_label.setText(self._format_grade(self._cutoff_min))
        finally:
            self._syncing_mode = False
        self.grade_cutoff_reset.emit()

    def _emit_cutoff_value(self):
        """Fired by debounce timer — emit the final value after rapid edits settle."""
        grade = self._slider_to_grade(self.grade_cutoff_slider.value())
        self.grade_cutoff_changed.emit(grade)

    def set_cutoff_range(self, min_val: float, max_val: float):
        """Set the grade cutoff range (called after data load).

        Automatically detects when log-scale mapping is appropriate (range
        exceeds 100×) for trace elements like ppm gold grades.
        """
        self._cutoff_min = float(min_val)
        self._cutoff_max = float(max_val)
        # Auto-detect log scale: useful when range is >100× (e.g. 0.01–10 g/t)
        self._cutoff_log_scale = (
            self._cutoff_min > 0
            and self._cutoff_max > 0
            and (self._cutoff_max / max(self._cutoff_min, 1e-10)) > 100
        )
        self._syncing_mode = True
        try:
            self.grade_cutoff_slider.setValue(0)
            self.grade_cutoff_spin.setRange(self._cutoff_min, self._cutoff_max)
            self.grade_cutoff_spin.setValue(self._cutoff_min)
            self.grade_cutoff_value_label.setText(self._format_grade(self._cutoff_min))
        finally:
            self._syncing_mode = False

    def sync_mode(self, mode: str):
        """Update the checked mouse-mode button to match an external change.

        Called by main_window when mouse mode changes via keyboard shortcut or
        menu.  Accepts both canonical and legacy mode names.
        """
        mapping = {
            "select":    self.select_btn,
            "click":     self.select_btn,
            "pan":       self.pan_btn,
            "rotate":    self.rotate_btn,
            "trackball": self.rotate_btn,
            "zoom_box":  self.zoom_box_btn,
            "zoom":      self.zoom_box_btn,
            "zoombox":   self.zoom_box_btn,
        }
        btn = mapping.get(mode)
        if btn is not None:
            btn.blockSignals(True)
            try:
                btn.setChecked(True)
            finally:
                btn.blockSignals(False)

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
