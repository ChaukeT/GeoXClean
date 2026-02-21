"""
Scene Inspector Panel for camera controls, view presets, and export tools.
"""

from __future__ import annotations

from typing import Optional, Sequence, Any, Dict

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QGridLayout, QFileDialog,
    QSpinBox, QLineEdit, QComboBox, QWidget, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSignalBlocker
from PyQt6.QtGui import QFont
import logging
from pathlib import Path

from .base_display_panel import BaseDisplayPanel
from .signals import UISignals
from .collapsible_group import CollapsibleGroup
from .modern_styles import (
    ModernColors, get_complete_panel_stylesheet, get_button_stylesheet,
    apply_modern_style
)

logger = logging.getLogger(__name__)


class SceneInspectorPanel(BaseDisplayPanel):
    """
    Scene Inspector panel for camera controls, view orientation, and display settings.
    
    Provides comprehensive scene navigation and export capabilities.
    """
    
    # Signals
    reset_view_requested = pyqtSignal()
    view_preset_requested = pyqtSignal(str)  # preset name
    projection_toggled = pyqtSignal(bool)  # True = orthographic, False = perspective
    # Emitted when the consolidated legend style/settings change. Payload is a dict.
    legend_style_changed = pyqtSignal(dict)
    scalar_bar_toggled = pyqtSignal(bool)
    
    def __init__(self, parent: Optional[object] = None, signals: Optional[UISignals] = None):
        # Current states
        self.is_orthographic = False
        self.show_scalar_bar = True
        self.show_axes = True
        self.show_grid = False

        # UISignals bus (for centralized signal emission)
        self.signals: Optional[UISignals] = signals

        super().__init__(parent=parent, panel_id="scene_inspector")

        logger.info("Initialized scene inspector panel")
    
    def set_signals(self, signals: UISignals):
        """Set the UISignals bus for centralized signal emission."""
        self.signals = signals
    
    def setup_ui(self):
        """Setup the UI layout with modern styling."""
        # Apply modern stylesheet
        self.setStyleSheet(get_complete_panel_stylesheet())
        
        # Use existing layout from BasePanel
        if self.main_layout is None:
            self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_area.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        
        # Title with modern styling
        title_label = QLabel("🎬 Scene Inspector")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 16px;
                font-weight: 700;
                padding: 8px 0;
                border-bottom: 2px solid {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        layout.addWidget(title_label)
        
        # Camera controls
        self._create_camera_controls(layout)
        
        # View presets
        self._create_view_presets(layout)
        
        # Projection toggle
        self._create_projection_controls(layout)
        
        # Display toggles
        self._create_display_toggles(layout)
        
        # Camera info display
        self._create_camera_info(layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()

        scroll_area.setWidget(content_widget)
        self.main_layout.addWidget(scroll_area)
        
        # Set max width for better layout
        self.setMaximumWidth(380)
        self.setMinimumWidth(300)
    
    def _create_camera_controls(self, layout: QVBoxLayout):
        """Create camera control buttons with modern styling."""
        group = CollapsibleGroup("📷 Camera Controls", collapsed=False)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        
        # Camera control buttons with modern styling
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        
        self.reset_view_btn = QPushButton("🔄 Reset View")
        self.reset_view_btn.setStyleSheet(get_button_stylesheet("secondary"))
        self.reset_view_btn.setMinimumHeight(40)
        self.reset_view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_view_btn.setToolTip("Reset camera to default position")
        self.reset_view_btn.clicked.connect(self._on_reset_view)
        row1.addWidget(self.reset_view_btn)
        
        group_layout.addLayout(row1)
        
        group.add_layout(group_layout)
        layout.addWidget(group)
    
    def _create_view_presets(self, layout: QVBoxLayout):
        """Create view preset buttons with modern styling."""
        group = CollapsibleGroup("🧭 View Orientation", collapsed=False)
        group_layout = QGridLayout()
        group_layout.setSpacing(10)
        
        # Button stylesheet for view presets
        view_btn_style = get_button_stylesheet("secondary")
        
        # Create view preset buttons with icons
        self.top_btn = QPushButton("⬆ Top")
        self.top_btn.setStyleSheet(view_btn_style)
        self.top_btn.setMinimumHeight(38)
        self.top_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.top_btn.setToolTip("View from top (XY plane)")
        self.top_btn.clicked.connect(lambda: self._on_view_preset("Top"))
        
        self.bottom_btn = QPushButton("⬇ Bottom")
        self.bottom_btn.setStyleSheet(view_btn_style)
        self.bottom_btn.setMinimumHeight(38)
        self.bottom_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bottom_btn.setToolTip("View from bottom")
        self.bottom_btn.clicked.connect(lambda: self._on_view_preset("Bottom"))
        
        self.front_btn = QPushButton("⬅ Front")
        self.front_btn.setStyleSheet(view_btn_style)
        self.front_btn.setMinimumHeight(38)
        self.front_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.front_btn.setToolTip("View from front (XZ plane)")
        self.front_btn.clicked.connect(lambda: self._on_view_preset("Front"))
        
        self.back_btn = QPushButton("➡ Back")
        self.back_btn.setStyleSheet(view_btn_style)
        self.back_btn.setMinimumHeight(38)
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.setToolTip("View from back")
        self.back_btn.clicked.connect(lambda: self._on_view_preset("Back"))
        
        self.right_btn = QPushButton("⮕ Right")
        self.right_btn.setStyleSheet(view_btn_style)
        self.right_btn.setMinimumHeight(38)
        self.right_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.right_btn.setToolTip("View from right (YZ plane)")
        self.right_btn.clicked.connect(lambda: self._on_view_preset("Right"))
        
        self.left_btn = QPushButton("⬅ Left")
        self.left_btn.setStyleSheet(view_btn_style)
        self.left_btn.setMinimumHeight(38)
        self.left_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.left_btn.setToolTip("View from left")
        self.left_btn.clicked.connect(lambda: self._on_view_preset("Left"))
        
        # Isometric button with primary styling for emphasis
        self.iso_btn = QPushButton("◈ Isometric")
        self.iso_btn.setStyleSheet(get_button_stylesheet("primary"))
        self.iso_btn.setMinimumHeight(42)
        self.iso_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.iso_btn.setToolTip("Isometric 3D view")
        self.iso_btn.clicked.connect(lambda: self._on_view_preset("Isometric"))
        
        # Add to grid with better layout
        group_layout.addWidget(self.top_btn, 0, 0)
        group_layout.addWidget(self.bottom_btn, 0, 1)
        group_layout.addWidget(self.front_btn, 1, 0)
        group_layout.addWidget(self.back_btn, 1, 1)
        group_layout.addWidget(self.right_btn, 2, 0)
        group_layout.addWidget(self.left_btn, 2, 1)
        group_layout.addWidget(self.iso_btn, 3, 0, 1, 2)
        
        group.add_layout(group_layout)
        layout.addWidget(group)
    
    def _create_projection_controls(self, layout: QVBoxLayout):
        """Create projection toggle controls with modern styling."""
        group = CollapsibleGroup("⚙️ Projection Mode", collapsed=True)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(12)
        
        self.projection_check = QCheckBox("📐 Orthographic View")
        self.projection_check.setToolTip("Toggle between perspective and orthographic projection")
        self.projection_check.setChecked(self.is_orthographic)
        self.projection_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.projection_check.stateChanged.connect(self._on_projection_toggled)
        group_layout.addWidget(self.projection_check)
        
        group.add_layout(group_layout)
        layout.addWidget(group)
    
    def _create_display_toggles(self, layout: QVBoxLayout):
        """Create display toggle checkboxes with modern styling."""
        group = CollapsibleGroup("👁️ Display Toggles", collapsed=False)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(12)

        self.toggle_scalar_bar = QCheckBox("📊 Show Legend (Scalar Bar)")
        self.toggle_scalar_bar.setChecked(self.show_scalar_bar)
        self.toggle_scalar_bar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_scalar_bar.stateChanged.connect(self._on_scalar_bar_toggled)
        group_layout.addWidget(self.toggle_scalar_bar)

        self.toggle_axes = QCheckBox("🎯 Show Axes")
        self.toggle_axes.setChecked(self.show_axes)
        self.toggle_axes.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_axes.setToolTip("Toggle 3D axes display")
        self.toggle_axes.stateChanged.connect(self._on_axes_toggled)
        group_layout.addWidget(self.toggle_axes)

        self.toggle_grid = QCheckBox("📏 Show Grid")
        self.toggle_grid.setChecked(self.show_grid)
        self.toggle_grid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_grid.setToolTip("Toggle ground grid display")
        self.toggle_grid.stateChanged.connect(self._on_grid_toggled)
        group_layout.addWidget(self.toggle_grid)

        group.add_layout(group_layout)
        layout.addWidget(group)
    
    def _create_camera_info(self, layout: QVBoxLayout):
        """Create camera information display with modern styling."""
        group = CollapsibleGroup("ℹ️ Camera Info", collapsed=True)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        
        # Position label
        self.camera_label = QLabel("Position: —")
        self.camera_label.setWordWrap(True)
        self.camera_label.setStyleSheet(f"""
            QLabel {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: {ModernColors.TEXT_SECONDARY};
                background-color: {ModernColors.ELEVATED_BG};
                padding: 8px;
                border-radius: 4px;
            }}
        """)
        group_layout.addWidget(self.camera_label)
        
        # Focus label
        self.focal_label = QLabel("Focus: —")
        self.focal_label.setWordWrap(True)
        self.focal_label.setStyleSheet(f"""
            QLabel {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: {ModernColors.TEXT_SECONDARY};
                background-color: {ModernColors.ELEVATED_BG};
                padding: 8px;
                border-radius: 4px;
            }}
        """)
        group_layout.addWidget(self.focal_label)
        
        group.add_layout(group_layout)
        layout.addWidget(group)
    
    # Event handlers
    def _on_reset_view(self):
        """Handle reset view button click."""
        try:
            if self.controller:
                self.controller.reset_scene()
        except Exception:
            logger.debug("Controller reset_scene failed", exc_info=True)
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.resetViewRequested.emit()
        else:
            self.reset_view_requested.emit()
        logger.info("Reset view requested")
    
    def _on_view_preset(self, preset: str):
        """Handle view preset button click."""
        try:
            if self.controller:
                self.controller.set_view_preset(preset)
        except Exception:
            logger.debug("Controller set_view_preset failed", exc_info=True)
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.viewPresetRequested.emit(preset)
        else:
            self.view_preset_requested.emit(preset)
        logger.info(f"View preset requested: {preset}")
    
    def _on_projection_toggled(self, state):
        """Handle projection toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.is_orthographic = enabled
        try:
            if self.controller:
                self.controller.set_projection_mode(enabled)
        except Exception:
            logger.debug("Controller set_projection_mode failed", exc_info=True)
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.projectionToggled.emit(enabled)
        else:
            self.projection_toggled.emit(enabled)
        logger.info(f"Projection: {'Orthographic' if enabled else 'Perspective'}")
    
    
    def _on_scalar_bar_toggled(self, state):
        """Handle scalar bar visibility toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.show_scalar_bar = enabled
        try:
            if self.controller:
                self.controller.set_legend_visibility(enabled)
        except Exception:
            logger.debug("Controller set_legend_visibility failed", exc_info=True)
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.scalarBarToggled.emit(enabled)
        else:
            self.scalar_bar_toggled.emit(enabled)
        logger.info(f"Scalar bar: {enabled}")

    def _on_axes_toggled(self, state):
        """Handle axes visibility toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.show_axes = enabled
        try:
            if self.controller:
                overlay_manager = getattr(self.controller, "overlay_manager", None)
                if overlay_manager and hasattr(overlay_manager, "set_overlay_visibility"):
                    overlay_manager.set_overlay_visibility("axes", enabled)
        except Exception:
            logger.debug("Controller axes toggle failed", exc_info=True)
        logger.info(f"Axes: {enabled}")

    def _on_grid_toggled(self, state):
        """Handle grid visibility toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.show_grid = enabled
        try:
            if self.controller:
                overlay_manager = getattr(self.controller, "overlay_manager", None)
                if overlay_manager and hasattr(overlay_manager, "set_overlay_visibility"):
                    overlay_manager.set_overlay_visibility("ground_grid", enabled)
        except Exception:
            logger.debug("Controller grid toggle failed", exc_info=True)
        logger.info(f"Grid: {enabled}")



    # Public methods
    def update_camera_info(self, position: tuple, focal_point: tuple):
        """
        Update camera information display.
        
        Args:
            position: Camera position (x, y, z)
            focal_point: Camera focal point (x, y, z)
        """
        try:
            if position and len(position) >= 3:
                pos_str = f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
                self.camera_label.setText(f"Position: {pos_str}")
            
            if focal_point and len(focal_point) >= 3:
                focal_str = f"({focal_point[0]:.1f}, {focal_point[1]:.1f}, {focal_point[2]:.1f})"
                self.focal_label.setText(f"Focus: {focal_str}")
        except (TypeError, IndexError, AttributeError) as e:
            # Silently handle malformed camera data
            logger.debug(f"Camera info display error: {e}")
    
    def clear_camera_info(self):
        """Clear camera information display."""
        self.camera_label.setText("Position: —")
        self.focal_label.setText("Focus: —")
        logger.debug("Cleared camera info")
    
    def set_projection_mode(self, orthographic: bool):
        """Set projection mode programmatically."""
        self.projection_check.setChecked(orthographic)
    
    def set_scalar_bar_visible(self, visible: bool):
        """Set scalar bar visibility programmatically."""
        self.toggle_scalar_bar.setChecked(visible)
    
    def set_axes_visible(self, visible: bool):
        """Set axes visibility programmatically."""
        self.toggle_axes.setChecked(visible)
    
    def set_grid_visible(self, visible: bool):
        """Set grid visibility programmatically."""
        self.toggle_grid.setChecked(visible)

    # ------------------------------------------------------------------
    # Controller binding & refresh helpers
    # ------------------------------------------------------------------
    def bind_controller(self, controller: Optional["AppController"]) -> None:
        super().bind_controller(controller)
        self.connect_layer_events()
        self.refresh()

    def connect_layer_events(self):
        controller = getattr(self, "controller", None)
        if controller is None:
            return
        legend = getattr(controller, "legend_manager", None)
        if legend is not None:
            try:
                legend.visibility_changed.connect(self._on_legend_visibility_update)
            except Exception as e:
                logger.error(f"Failed to connect legend visibility signal: {e}", exc_info=True)
            try:
                legend.legend_changed.connect(lambda _: self.refresh())
            except Exception as e:
                logger.error(f"Failed to connect legend changed signal: {e}", exc_info=True)
        overlay = getattr(controller, "overlay_manager", None)
        if overlay is not None and hasattr(overlay, "overlays_changed"):
            try:
                overlay.overlays_changed.connect(self._on_overlay_state_update)
            except Exception as e:
                logger.error(f"Failed to connect overlay changed signal: {e}", exc_info=True)

    def refresh(self):
        controller = getattr(self, "controller", None)
        if controller is None:
            return
        try:
            legend_state = {}
            legend = getattr(controller, "legend_manager", None)
            if legend and hasattr(legend, "get_state"):
                legend_state = legend.get_state()
            overlays_state = {}
            overlay_manager = getattr(controller, "overlay_manager", None)
            if overlay_manager and hasattr(overlay_manager, "get_state"):
                overlays_state = overlay_manager.get_state().get("overlays", {})

            with QSignalBlocker(self.toggle_scalar_bar):
                self.toggle_scalar_bar.setChecked(bool(legend_state.get("visible", True)))
            with QSignalBlocker(self.toggle_axes):
                self.toggle_axes.setChecked(bool(overlays_state.get("axes", True)))
            with QSignalBlocker(self.toggle_grid):
                self.toggle_grid.setChecked(bool(overlays_state.get("ground_grid", False)))
            
        except Exception:
            logger.debug("SceneInspectorPanel refresh failed", exc_info=True)

    def _on_legend_visibility_update(self, visible: bool) -> None:
        with QSignalBlocker(self.toggle_scalar_bar):
            self.toggle_scalar_bar.setChecked(bool(visible))
        self.show_scalar_bar = bool(visible)

    def _on_overlay_state_update(self, overlays: Dict[str, bool]) -> None:
        try:
            axes_state = bool(overlays.get("axes", True))
            grid_state = bool(overlays.get("ground_grid", False))
            with QSignalBlocker(self.toggle_axes):
                self.toggle_axes.setChecked(axes_state)
            with QSignalBlocker(self.toggle_grid):
                self.toggle_grid.setChecked(grid_state)
            self.show_axes = axes_state
            self.show_grid = grid_state
        except Exception:
            logger.debug("Overlay state update failed", exc_info=True)

