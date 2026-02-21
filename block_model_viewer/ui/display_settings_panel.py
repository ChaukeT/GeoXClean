"""
Display Settings Panel for controlling visual appearance of the 3D viewer.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QCheckBox, QFormLayout, QGroupBox, QColorDialog,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSignalBlocker
from PyQt6.QtGui import QColor
import logging

from .base_display_panel import BaseDisplayPanel

logger = logging.getLogger(__name__)


class DisplaySettingsPanel(BaseDisplayPanel):
    """
    Panel for controlling display settings (opacity, colors, lighting, etc.).
    
    Provides real-time visual control over the 3D rendering appearance.
    """
    
    # Signals
    opacity_changed = pyqtSignal(float)  # opacity 0-1
    edge_color_changed = pyqtSignal(tuple)  # RGB tuple
    edge_visibility_toggled = pyqtSignal(bool)  # show/hide edges
    background_color_changed = pyqtSignal(tuple)  # RGB tuple
    lighting_toggled = pyqtSignal(bool)  # enabled/disabled
    legend_font_size_changed = pyqtSignal(int)  # font size
    legend_visibility_toggled = pyqtSignal(bool)  # show/hide legend
    legend_orientation_changed = pyqtSignal(str)  # vertical/horizontal
    legend_position_reset = pyqtSignal()  # reset legend position
    reset_view_requested = pyqtSignal()
    # STEP 18: Performance settings signals
    performance_settings_changed = pyqtSignal(dict)  # performance settings dict
    
    def __init__(self, parent: Optional[object] = None):
        # Current values
        self.current_opacity = 1.0
        self.current_edge_color = (0, 0, 0)  # black
        self.current_edge_visibility = True  # edges visible by default
        self.current_background_color = (0.827, 0.827, 0.827)  # lightgrey
        self.current_lighting = True
        self.current_legend_size = 13
        self.current_legend_visible = True
        self.current_legend_orientation = 'vertical'
        
        # STEP 18: Performance settings defaults
        self.current_lod_quality = 0.7
        self.current_max_render_cells = 1_000_000
        self.current_downsample_large = True
        self.current_async_loading = True
        self.current_aggressive_compaction = False
        
        super().__init__(parent=parent, panel_id="display_settings")
        
        logger.info("Initialized display settings panel")
    
    # ------------------------------------------------------------------
    # BasePanel hooks
    # ------------------------------------------------------------------
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout or QVBoxLayout(self)
        self.main_layout = layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("<b>Display Settings</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Groups
        self._create_rendering_group(layout)
        self._create_colors_group(layout)
        self._create_legend_group(layout)
        self._create_performance_group(layout)  # STEP 18
        self._create_actions_group(layout)
        
        layout.addStretch()
        
        # Set max width for better layout
        self.setMaximumWidth(350)
        self.setMinimumWidth(280)
    
    def connect_signals(self):
        """Wire interactive controls to handlers."""
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.lighting_check.stateChanged.connect(self._on_lighting_toggled)
        self.edge_visibility_check.stateChanged.connect(self._on_edge_visibility_toggled)
        self.edge_color_btn.clicked.connect(self._on_edge_color_clicked)
        self.background_color_btn.clicked.connect(self._on_background_color_clicked)
        self.legend_visible_check.stateChanged.connect(self._on_legend_visibility_toggled)
        self.legend_vertical_btn.clicked.connect(
            lambda: self._on_legend_orientation_changed('vertical')
        )
        self.legend_horizontal_btn.clicked.connect(
            lambda: self._on_legend_orientation_changed('horizontal')
        )
        self.legend_slider.valueChanged.connect(self._on_legend_font_size_changed)
        self.legend_reset_btn.clicked.connect(self._on_legend_position_reset)
        self.reset_view_btn.clicked.connect(self._on_reset_view)
        self.fit_view_btn.clicked.connect(self._on_reset_view)
        # STEP 18: Performance settings signals
        self.lod_quality_slider.valueChanged.connect(self._on_performance_settings_changed)
        self.max_cells_spinbox.valueChanged.connect(self._on_performance_settings_changed)
        self.downsample_check.stateChanged.connect(self._on_performance_settings_changed)
        self.async_loading_check.stateChanged.connect(self._on_performance_settings_changed)
        self.compaction_check.stateChanged.connect(self._on_performance_settings_changed)
    
    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _create_rendering_group(self, layout: QVBoxLayout):
        group = QGroupBox("Rendering")
        group_layout = QFormLayout(group)
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(self.current_opacity * 100))
        self.opacity_slider.setToolTip(
            "Control block transparency (10% to 100%)\n"
            "Lower values make blocks more transparent, useful for seeing internal structure"
        )
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel(f"{self.current_opacity:.2f}")
        self.opacity_label.setMinimumWidth(40)
        opacity_layout.addWidget(self.opacity_label)
        
        group_layout.addRow("Opacity:", opacity_layout)
        
        # Lighting toggle
        self.lighting_check = QCheckBox("Enable Lighting")
        self.lighting_check.setChecked(self.current_lighting)
        self.lighting_check.setToolTip(
            "Enable realistic 3D lighting and shadows\n"
            "Disable for flat coloring to see data values more clearly"
        )
        group_layout.addRow(self.lighting_check)

        # Edge visibility toggle
        self.edge_visibility_check = QCheckBox("Show Block Edges")
        self.edge_visibility_check.setChecked(self.current_edge_visibility)
        self.edge_visibility_check.setToolTip(
            "Show or hide block edges/wireframes\n"
            "Disable for cleaner visualization of block models with many small cells\n"
            "Automatically disabled for large models (>50,000 cells) to prevent GPU timeouts"
        )
        group_layout.addRow(self.edge_visibility_check)

        layout.addWidget(group)
    
    def _create_colors_group(self, layout: QVBoxLayout):
        group = QGroupBox("Colors")
        group_layout = QVBoxLayout(group)
        
        # Edge color button
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Edge Color:"))
        self.edge_color_btn = QPushButton("Select")
        self.edge_color_btn.setToolTip(
            "Choose color for block edges/outlines\n"
            "Black or dark gray works well for most visualizations"
        )
        edge_layout.addWidget(self.edge_color_btn)
        
        # Background color button
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background:"))
        self.background_color_btn = QPushButton("Select")
        self.background_color_btn.setToolTip(
            "Choose color for the viewport background\n"
            "Light grey works well for high-contrast visualization"
        )
        bg_layout.addWidget(self.background_color_btn)
        
        group_layout.addLayout(edge_layout)
        group_layout.addLayout(bg_layout)
        
        layout.addWidget(group)
    
    def _create_legend_group(self, layout: QVBoxLayout):
        group = QGroupBox("Legend")
        group_layout = QVBoxLayout(group)
        
        self.legend_visible_check = QCheckBox("Show Legend Bar")
        self.legend_visible_check.setChecked(True)
        self.current_legend_visible = True
        self.legend_visible_check.setToolTip(
            "Show or hide the color legend bar\n"
            "The legend shows the mapping between colors and data values"
        )
        group_layout.addWidget(self.legend_visible_check)
        
        orientation_layout = QHBoxLayout()
        self.legend_vertical_btn = QPushButton("Vertical")
        self.legend_vertical_btn.setToolTip("Display legend as vertical bar (right side)")
        self.legend_horizontal_btn = QPushButton("Horizontal")
        self.legend_horizontal_btn.setToolTip("Display legend as horizontal bar (bottom)")
        self.legend_vertical_btn.setCheckable(True)
        self.legend_horizontal_btn.setCheckable(True)
        self.legend_vertical_btn.setChecked(self.current_legend_orientation == 'vertical')
        self.legend_horizontal_btn.setChecked(self.current_legend_orientation == 'horizontal')
        orientation_layout.addWidget(self.legend_vertical_btn)
        orientation_layout.addWidget(self.legend_horizontal_btn)
        group_layout.addWidget(QLabel("Orientation:"))
        group_layout.addLayout(orientation_layout)
        
        legend_font_layout = QHBoxLayout()
        self.legend_slider = QSlider(Qt.Orientation.Horizontal)
        self.legend_slider.setRange(8, 24)
        self.legend_slider.setValue(self.current_legend_size)
        self.legend_slider.setToolTip("Adjust legend font size")
        legend_font_layout.addWidget(self.legend_slider)
        
        self.legend_reset_btn = QPushButton("Reset Position")
        self.legend_reset_btn.setToolTip(
            "Reset legend to default location (right side, centered)"
        )
        legend_font_layout.addWidget(self.legend_reset_btn)
        
        group_layout.addLayout(legend_font_layout)
        layout.addWidget(group)
    
    def _create_actions_group(self, layout: QVBoxLayout):
        group = QGroupBox("View")
        group_layout = QVBoxLayout(group)
        
        row1 = QHBoxLayout()
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip("Reset camera to default position")
        row1.addWidget(self.reset_view_btn)
        
        self.fit_view_btn = QPushButton("Fit to Model")
        self.fit_view_btn.setToolTip("Fit entire model in view")
        row1.addWidget(self.fit_view_btn)
        group_layout.addLayout(row1)
        
        layout.addWidget(group)
    
    def _create_performance_group(self, layout: QVBoxLayout):
        """STEP 18: Create performance settings group."""
        group = QGroupBox("Performance")
        group_layout = QFormLayout(group)
        
        # LOD Quality slider
        lod_layout = QHBoxLayout()
        self.lod_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.lod_quality_slider.setRange(0, 100)
        self.lod_quality_slider.setValue(int(self.current_lod_quality * 100))
        self.lod_quality_slider.setToolTip(
            "Level of Detail quality (0=Low, 100=High)\n"
            "Lower quality improves performance for large models"
        )
        lod_layout.addWidget(self.lod_quality_slider)
        self.lod_quality_label = QLabel(f"{self.current_lod_quality:.1f}")
        self.lod_quality_label.setMinimumWidth(40)
        lod_layout.addWidget(self.lod_quality_label)
        group_layout.addRow("LOD Quality:", lod_layout)
        
        # Max Render Cells
        self.max_cells_spinbox = QSpinBox()
        self.max_cells_spinbox.setRange(10_000, 10_000_000)
        self.max_cells_spinbox.setValue(self.current_max_render_cells)
        self.max_cells_spinbox.setSingleStep(100_000)
        self.max_cells_spinbox.setToolTip(
            "Maximum number of cells to render\n"
            "Models exceeding this will use LOD rendering"
        )
        group_layout.addRow("Max Render Cells:", self.max_cells_spinbox)
        
        # Downsample for Large Models
        self.downsample_check = QCheckBox("Downsample Large Models")
        self.downsample_check.setChecked(self.current_downsample_large)
        self.downsample_check.setToolTip(
            "Automatically downsample very large models for better performance"
        )
        group_layout.addRow(self.downsample_check)
        
        # Enable Async Loading
        self.async_loading_check = QCheckBox("Enable Async Loading")
        self.async_loading_check.setChecked(self.current_async_loading)
        self.async_loading_check.setToolTip(
            "Load large models in chunks to avoid UI freezing"
        )
        group_layout.addRow(self.async_loading_check)
        
        # Aggressive Compaction
        self.compaction_check = QCheckBox("Aggressive Compaction")
        self.compaction_check.setChecked(self.current_aggressive_compaction)
        self.compaction_check.setToolTip(
            "Free memory aggressively after analysis tasks"
        )
        group_layout.addRow(self.compaction_check)
        
        layout.addWidget(group)
    
    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change."""
        opacity = value / 100.0
        self.current_opacity = opacity
        self.opacity_label.setText(f"{opacity:.2f}")
        self.opacity_changed.emit(opacity)
        try:
            if self.controller:
                self.controller.set_global_opacity(opacity)
        except Exception:
            logger.debug("Controller set_global_opacity failed", exc_info=True)
        logger.debug("Opacity changed to %.2f", opacity)
    
    def _on_lighting_toggled(self, state: int):
        """Handle lighting toggle."""
        enabled = state == Qt.CheckState.Checked
        self.current_lighting = enabled
        self.lighting_toggled.emit(enabled)
        try:
            if self.controller:
                self.controller.set_lighting_enabled(enabled)
        except Exception:
            logger.debug("Controller set_lighting_enabled failed", exc_info=True)
        logger.debug("Lighting toggled -> %s", enabled)

    def _on_edge_visibility_toggled(self, state: int):
        """Handle edge visibility toggle."""
        visible = state == Qt.CheckState.Checked
        self.current_edge_visibility = visible
        self.edge_visibility_toggled.emit(visible)
        try:
            if self.controller:
                self.controller.set_edge_visibility(visible)
        except Exception:
            logger.debug("Controller set_edge_visibility failed", exc_info=True)
        logger.info("Edge visibility toggled -> %s", visible)

    def _on_edge_color_clicked(self):
        """Handle edge color selection."""
        color = QColorDialog.getColor(
            QColor.fromRgbF(*self.current_edge_color),
            self,
            "Select Edge Color"
        )
        if color.isValid():
            rgb = (color.redF(), color.greenF(), color.blueF())
            self.current_edge_color = rgb
            self.edge_color_changed.emit(rgb)
            try:
                if self.controller:
                    self.controller.set_edge_color(rgb)
            except Exception:
                logger.debug("Controller set_edge_color failed", exc_info=True)
            logger.debug("Edge color changed -> %s", rgb)
    
    def _on_background_color_clicked(self):
        """Handle background color selection."""
        color = QColorDialog.getColor(
            QColor.fromRgbF(*self.current_background_color),
            self,
            "Select Background Color"
        )
        if color.isValid():
            rgb = (color.redF(), color.greenF(), color.blueF())
            self.current_background_color = rgb
            self.background_color_changed.emit(rgb)
            try:
                if self.controller:
                    self.controller.set_background_color(rgb)
            except Exception:
                logger.debug("Controller set_background_color failed", exc_info=True)
            logger.debug("Background color changed -> %s", rgb)
    
    def _on_legend_visibility_toggled(self, state: int):
        """Handle legend visibility toggle."""
        visible = state == Qt.CheckState.Checked
        self.current_legend_visible = visible
        self.legend_visibility_toggled.emit(visible)
        try:
            if self.controller:
                self.controller.set_legend_visibility(visible)
        except Exception:
            logger.debug("Controller set_legend_visibility failed", exc_info=True)
        logger.info("Legend visibility changed to: %s", visible)
    
    def _on_legend_orientation_changed(self, orientation: str):
        """Handle legend orientation change."""
        self.current_legend_orientation = orientation
        self.legend_vertical_btn.setChecked(orientation == 'vertical')
        self.legend_horizontal_btn.setChecked(orientation == 'horizontal')
        self.legend_orientation_changed.emit(orientation)
        try:
            if self.controller:
                self.controller.set_legend_orientation(orientation)
        except Exception:
            logger.debug("Controller set_legend_orientation failed", exc_info=True)
        logger.info("Legend orientation changed to: %s", orientation)
    
    def _on_legend_font_size_changed(self, size: int):
        """Handle legend font size slider."""
        self.current_legend_size = size
        self.legend_font_size_changed.emit(size)
        try:
            if self.controller:
                self.controller.set_legend_font_size(size)
        except Exception:
            logger.debug("Controller set_legend_font_size failed", exc_info=True)
        logger.debug("Legend font size changed -> %d", size)
    
    def _on_legend_position_reset(self):
        """Handle legend position reset button click."""
        self.legend_position_reset.emit()
        try:
            if self.controller:
                self.controller.reset_legend_position()
        except Exception:
            logger.debug("Controller reset_legend_position failed", exc_info=True)
        logger.info("Legend position reset requested")
    
    def _on_reset_view(self):
        """Handle reset/fit view button click."""
        self.reset_view_requested.emit()
        try:
            if self.controller:
                self.controller.reset_scene()
        except Exception:
            logger.debug("Controller reset_scene failed (display panel)", exc_info=True)
    
    def _on_performance_settings_changed(self):
        """STEP 18: Handle performance settings changes."""
        # Update current values
        self.current_lod_quality = self.lod_quality_slider.value() / 100.0
        self.lod_quality_label.setText(f"{self.current_lod_quality:.1f}")
        self.current_max_render_cells = self.max_cells_spinbox.value()
        self.current_downsample_large = self.downsample_check.isChecked()
        self.current_async_loading = self.async_loading_check.isChecked()
        self.current_aggressive_compaction = self.compaction_check.isChecked()
        
        # Emit settings dict
        settings = {
            'lod_quality': self.current_lod_quality,
            'max_render_cells': self.current_max_render_cells,
            'downsample_large_models': self.current_downsample_large,
            'enable_async_loading': self.current_async_loading,
            'aggressive_compaction': self.current_aggressive_compaction
        }
        self.performance_settings_changed.emit(settings)
    
    def _on_fit_view(self):
        """Handle fit view button click."""
        try:
            if self.controller:
                self.controller.fit_to_view()
        except Exception:
            logger.debug("Controller fit_to_view failed (display panel)", exc_info=True)
    
    # ------------------------------------------------------------------
    # External setters
    # ------------------------------------------------------------------
    def set_opacity(self, opacity: float):
        """Set opacity value programmatically."""
        self.opacity_slider.setValue(int(opacity * 100))

    def set_lighting(self, enabled: bool):
        """Set lighting state programmatically."""
        self.lighting_check.setChecked(enabled)

    def set_edge_visibility(self, visible: bool):
        """Set edge visibility state programmatically."""
        self.edge_visibility_check.setChecked(visible)

    def set_legend_font_size(self, size: int):
        """Set legend font size programmatically."""
        self.legend_slider.setValue(size)
    
    # ------------------------------------------------------------------
    # Controller binding and refresh
    # ------------------------------------------------------------------
    def bind_controller(self, controller: Optional["AppController"]) -> None:
        super().bind_controller(controller)
        self._connect_manager_signals()
        self.refresh()

    def _connect_manager_signals(self) -> None:
        controller = getattr(self, "controller", None)
        if controller is None:
            return
        legend = getattr(controller, "legend_manager", None)
        if legend is not None:
            try:
                legend.visibility_changed.connect(self._update_legend_visibility_checkbox)
            except Exception as e:
                logger.error(f"Failed to connect legend visibility signal: {e}", exc_info=True)
            try:
                legend.legend_changed.connect(lambda _: self.refresh())
            except Exception as e:
                logger.error(f"Failed to connect legend changed signal: {e}", exc_info=True)

    def refresh(self) -> None:
        controller = getattr(self, "controller", None)
        if controller is None:
            return
        renderer = getattr(controller, "renderer", None)
        try:
            if renderer is not None:
                with QSignalBlocker(self.opacity_slider):
                    self.opacity_slider.setValue(int(getattr(renderer, "current_opacity", self.current_opacity) * 100))
                with QSignalBlocker(self.lighting_check):
                    self.lighting_check.setChecked(bool(getattr(renderer, "lighting_enabled", self.current_lighting)))
                with QSignalBlocker(self.edge_visibility_check):
                    # Get edge visibility from renderer
                    edge_visible = renderer.get_edge_visibility() if hasattr(renderer, "get_edge_visibility") else self.current_edge_visibility
                    self.edge_visibility_check.setChecked(bool(edge_visible))
            legend_state: Dict[str, Any] = {}
            legend = getattr(controller, "legend_manager", None)
            if legend and hasattr(legend, "get_state"):
                legend_state = legend.get_state()
            visible = bool(legend_state.get("visible", self.current_legend_visible))
            with QSignalBlocker(self.legend_visible_check):
                self.legend_visible_check.setChecked(visible)
            self.current_legend_visible = visible
        except Exception:
            logger.debug("DisplaySettingsPanel refresh failed", exc_info=True)

    def _update_legend_visibility_checkbox(self, visible: bool) -> None:
        with QSignalBlocker(self.legend_visible_check):
            self.legend_visible_check.setChecked(bool(visible))
        self.current_legend_visible = bool(visible)
