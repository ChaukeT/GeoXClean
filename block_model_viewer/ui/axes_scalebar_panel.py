"""
Floating Axes & Scale Bar Control Panel.

Provides professional-grade controls for PyVista/VTK Floating Axes (Bounds) 
and Dynamic Scale Bars with enhanced styling and positioning capabilities.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QGroupBox, QSpinBox, QComboBox, QWidget,
    QDoubleSpinBox, QFrame, QGridLayout, QTabWidget,
    QPushButton, QColorDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import logging

from .base_display_panel import BaseDisplayPanel
from .modern_styles import get_theme_colors, ModernColors

logger = logging.getLogger(__name__)

# --- STYLING ---
def get_dark_theme() -> str:
    """Get dark theme stylesheet."""
    colors = get_theme_colors()
    return f"""
QWidget {{
    background-color: #2b2b2b;
    color: {colors.TEXT_PRIMARY};
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 12px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #007acc;
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: #3a3a3a;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px;
    color: white;
}}
QComboBox:hover, QSpinBox:hover {{
    border: 1px solid #007acc;
}}
QLabel {{
    color: #cccccc;
}}
/* Tabs */
QTabWidget::pane {{
    border: 1px solid #3a3a3a;
    background: #2b2b2b;
}}
QTabBar::tab {{
    background: {colors.PANEL_BG};
    color: #888;
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: #2b2b2b;
    color: white;
    border-top: 2px solid #007acc;
}}
QPushButton {{
    background-color: #444;
    border: none;
    border-radius: 3px;
    padding: 5px 10px;
}}
QPushButton:hover {{ background-color: #555; }}
QPushButton:pressed {{ background-color: #007acc; }}
"""

class AxesScaleBarPanel(BaseDisplayPanel):
    """
    Panel for controlling Floating Axes (Grid) and Dynamic Scale Bar.
    """
    
    def __init__(self, parent: Optional[object] = None):
        super().__init__(parent=parent, panel_id="axes_scalebar")
        self.setStyleSheet(get_dark_theme())
        logger.info("Initialized enhanced axes and scale bar panel")
    


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
    def setup_ui(self):
        """Setup the UI layout."""
        if self.main_layout is None:
            self.main_layout = QVBoxLayout(self)
        
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)
        
        # Header
        header = QLabel("Reference Systems")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        self.main_layout.addWidget(header)
        
        # Tabs for Axes, Scale Bar, and North Arrow
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_axes_tab(), "Grid_Axes")
        self.tabs.addTab(self._create_scalebar_tab(), "Scale Bar")
        self.tabs.addTab(self._create_north_arrow_tab(), "North Arrow")
        self.main_layout.addWidget(self.tabs)
        
        # Common Actions
        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("Force Refresh")
        self.btn_apply.clicked.connect(self._trigger_update)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_apply)
        self.main_layout.addLayout(btn_layout)
        
        self.main_layout.addStretch()

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION - AXES TAB
    # -------------------------------------------------------------------------
    def _create_axes_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # 1. Main Toggle & Visibility
        self.axes_enable_check = QCheckBox("Enable 3D Grid / Axes")
        self.axes_enable_check.setChecked(False)
        self.axes_enable_check.setStyleSheet("font-weight: bold;")
        self.axes_enable_check.toggled.connect(self._on_axes_changed)
        layout.addWidget(self.axes_enable_check)

        # 2. Appearance Group (Color, Width, Font)
        style_group = QGroupBox("Appearance")
        style_grid = QGridLayout(style_group)
        
        # Color Theme
        style_grid.addWidget(QLabel("Color Theme:"), 0, 0)
        self.axes_color_combo = QComboBox()
        self.axes_color_combo.addItems(["Black (High Contrast)", "White", "Gray"])
        self.axes_color_combo.currentIndexChanged.connect(self._on_axes_changed)
        style_grid.addWidget(self.axes_color_combo, 0, 1)

        # Line Width
        style_grid.addWidget(QLabel("Line Thickness:"), 1, 0)
        self.axes_width_spin = QSpinBox()
        self.axes_width_spin.setRange(1, 10)
        self.axes_width_spin.setValue(2) # Default to thick line
        self.axes_width_spin.valueChanged.connect(self._on_axes_changed)
        style_grid.addWidget(self.axes_width_spin, 1, 1)

        # Font Size
        style_grid.addWidget(QLabel("Font Size:"), 2, 0)
        self.axes_font_spin = QSpinBox()
        self.axes_font_spin.setRange(8, 48)
        self.axes_font_spin.setValue(12)
        self.axes_font_spin.valueChanged.connect(self._on_axes_changed)
        style_grid.addWidget(self.axes_font_spin, 2, 1)

        layout.addWidget(style_group)

        # 3. Geometry / Ticks
        geo_group = QGroupBox("Grid Spacing (m)")
        geo_layout = QGridLayout(geo_group)
        
        # Headers
        geo_layout.addWidget(QLabel("Axis"), 0, 0)
        geo_layout.addWidget(QLabel("Major"), 0, 1)
        geo_layout.addWidget(QLabel("Minor"), 0, 2)
        
        # X Axis
        geo_layout.addWidget(QLabel("X:"), 1, 0)
        self.x_major = self._create_spacing_spin(100)
        self.x_minor = self._create_spacing_spin(50)
        geo_layout.addWidget(self.x_major, 1, 1)
        geo_layout.addWidget(self.x_minor, 1, 2)

        # Y Axis
        geo_layout.addWidget(QLabel("Y:"), 2, 0)
        self.y_major = self._create_spacing_spin(100)
        self.y_minor = self._create_spacing_spin(50)
        geo_layout.addWidget(self.y_major, 2, 1)
        geo_layout.addWidget(self.y_minor, 2, 2)

        # Z Axis
        geo_layout.addWidget(QLabel("Z:"), 3, 0)
        self.z_major = self._create_spacing_spin(50)
        self.z_minor = self._create_spacing_spin(25)
        geo_layout.addWidget(self.z_major, 3, 1)
        geo_layout.addWidget(self.z_minor, 3, 2)

        layout.addWidget(geo_group)
        layout.addStretch()
        return container

    def _create_spacing_spin(self, default_val: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(1, 100000)
        sb.setValue(default_val)
        sb.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)  # Cleaner look
        sb.editingFinished.connect(self._on_axes_changed)
        return sb

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION - SCALE BAR TAB
    # -------------------------------------------------------------------------
    def _create_scalebar_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # Enable
        self.sb_enable_check = QCheckBox("Enable Scale Bar")
        self.sb_enable_check.setChecked(False)
        self.sb_enable_check.toggled.connect(self._on_scalebar_changed)
        layout.addWidget(self.sb_enable_check)

        # Settings
        sets_group = QGroupBox("Configuration")
        sets_layout = QGridLayout(sets_group)

        # Title / Unit
        sets_layout.addWidget(QLabel("Unit Label:"), 0, 0)
        self.sb_unit_edit = QComboBox()
        self.sb_unit_edit.addItems(["m", "km", "ft", "mm"])
        self.sb_unit_edit.setEditable(True)
        self.sb_unit_edit.currentTextChanged.connect(self._on_scalebar_changed)
        sets_layout.addWidget(self.sb_unit_edit, 0, 1)

        # Color
        sets_layout.addWidget(QLabel("Color:"), 1, 0)
        self.sb_color_combo = QComboBox()
        self.sb_color_combo.addItems(["Black", "White", "Gray"])
        self.sb_color_combo.currentIndexChanged.connect(self._on_scalebar_changed)
        sets_layout.addWidget(self.sb_color_combo, 1, 1)

        # Font Size
        sets_layout.addWidget(QLabel("Font Size:"), 2, 0)
        self.sb_font_spin = QSpinBox()
        self.sb_font_spin.setRange(8, 30)
        self.sb_font_spin.setValue(14)
        self.sb_font_spin.valueChanged.connect(self._on_scalebar_changed)
        sets_layout.addWidget(self.sb_font_spin, 2, 1)

        layout.addWidget(sets_group)

        # Positioning
        pos_group = QGroupBox("Position & Size")
        pos_layout = QGridLayout(pos_group)

        # Position presets
        pos_layout.addWidget(QLabel("Anchor:"), 0, 0)
        self.sb_pos_combo = QComboBox()
        self.sb_pos_combo.addItems(["Bottom Right", "Bottom Left", "Top Right", "Top Left"])
        self.sb_pos_combo.currentIndexChanged.connect(self._on_scalebar_changed)
        pos_layout.addWidget(self.sb_pos_combo, 0, 1)

        # Width Ratio
        pos_layout.addWidget(QLabel("Width %:"), 1, 0)
        self.sb_width_spin = QSpinBox()
        self.sb_width_spin.setRange(5, 90)
        self.sb_width_spin.setValue(20)
        self.sb_width_spin.setSuffix("%")
        self.sb_width_spin.valueChanged.connect(self._on_scalebar_changed)
        pos_layout.addWidget(self.sb_width_spin, 1, 1)

        layout.addWidget(pos_group)
        layout.addStretch()
        return container

    # -------------------------------------------------------------------------
    # UI CONSTRUCTION - NORTH ARROW TAB
    # -------------------------------------------------------------------------
    def _create_north_arrow_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # Enable
        self.na_enable_check = QCheckBox("Enable North Arrow")
        self.na_enable_check.setChecked(False)
        self.na_enable_check.toggled.connect(self._on_north_arrow_changed)
        layout.addWidget(self.na_enable_check)

        # Settings
        sets_group = QGroupBox("Appearance")
        sets_layout = QGridLayout(sets_group)

        # Color Theme
        sets_layout.addWidget(QLabel("Color:"), 0, 0)
        self.na_color_combo = QComboBox()
        self.na_color_combo.addItems(["White", "Black", "Gray"])
        self.na_color_combo.currentIndexChanged.connect(self._on_north_arrow_changed)
        sets_layout.addWidget(self.na_color_combo, 0, 1)

        # Font Size
        sets_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.na_font_spin = QSpinBox()
        self.na_font_spin.setRange(8, 20)
        self.na_font_spin.setValue(12)
        self.na_font_spin.valueChanged.connect(self._on_north_arrow_changed)
        sets_layout.addWidget(self.na_font_spin, 1, 1)

        layout.addWidget(sets_group)

        # Positioning
        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout(pos_group)

        # Position presets
        pos_layout.addWidget(QLabel("Anchor:"), 0, 0)
        self.na_pos_combo = QComboBox()
        self.na_pos_combo.addItems(["Top Right", "Top Left", "Bottom Right", "Bottom Left"])
        self.na_pos_combo.currentIndexChanged.connect(self._on_north_arrow_changed)
        pos_layout.addWidget(self.na_pos_combo, 0, 1)

        layout.addWidget(pos_group)
        layout.addStretch()
        return container

    # -------------------------------------------------------------------------
    # LOGIC & EVENTS
    # -------------------------------------------------------------------------

    def _get_renderer(self):
        """Robustly retrieve the renderer."""
        if not self.controller:
            return None
        # Common patterns in MV frameworks
        for attr in ['r', 'renderer', 'plotter']:
            if hasattr(self.controller, attr):
                return getattr(self.controller, attr)

        # Check child widgets
        if hasattr(self.controller, 'viewer_widget'):
            return getattr(self.controller.viewer_widget, 'renderer', None)

        return None

    def _get_viewer_widget(self):
        """Retrieve the viewer widget from controller."""
        if not self.controller:
            return None
        return getattr(self.controller, 'viewer_widget', None)

    def _ensure_overlay_widgets(self):
        """Ensure scale bar and north arrow widgets exist before using them."""
        viewer = self._get_viewer_widget()
        if viewer is None:
            return
        # Ensure scale bar widget
        if hasattr(viewer, '_ensure_scale_bar_widget'):
            viewer._ensure_scale_bar_widget()
        # Ensure north arrow widget
        if hasattr(viewer, '_ensure_north_arrow_widget'):
            viewer._ensure_north_arrow_widget()

    def _map_color(self, name: str) -> str:
        """Map combo box text to hex/string colors."""
        mapping = {
            "Black (High Contrast)": "black",
            "Black": "black",
            "White": "white",
            "Gray": "gray"
        }
        return mapping.get(name, "black")

    def _collect_axes_params(self) -> Dict[str, Any]:
        """Collect all axes parameters into a dictionary."""
        return {
            "enabled": self.axes_enable_check.isChecked(),
            "color": self._map_color(self.axes_color_combo.currentText()),
            "width": self.axes_width_spin.value(),
            "font_size": self.axes_font_spin.value(),
            "xtick_major": self.x_major.value(),
            "xtick_minor": self.x_minor.value(),
            "ytick_major": self.y_major.value(),
            "ytick_minor": self.y_minor.value(),
            "ztick_major": self.z_major.value(),
            "ztick_minor": self.z_minor.value(),
            "auto_spacing": True,
            # Implicit settings for 'Thick Box'
            "grid": True,
            "location": "all"
        }

    def _collect_scalebar_params(self) -> Dict[str, Any]:
        """Collect all scale bar parameters."""
        pos_text = self.sb_pos_combo.currentText()
        pos_x, pos_y = 0.8, 0.05
        anchor_key = "bottom_right"
        if "Bottom Left" in pos_text:
            pos_x, pos_y = 0.05, 0.05
            anchor_key = "bottom_left"
        elif "Top Right" in pos_text:
            pos_x, pos_y = 0.8, 0.85
            anchor_key = "top_right"
        elif "Top Left" in pos_text:
            pos_x, pos_y = 0.05, 0.85
            anchor_key = "top_left"
        
        return {
            "enabled": self.sb_enable_check.isChecked(),
            "units": self.sb_unit_edit.currentText(),
            "color": self._map_color(self.sb_color_combo.currentText()),
            "font_size": self.sb_font_spin.value(),
            "position_x": pos_x,
            "position_y": pos_y,
            "width_fraction": self.sb_width_spin.value() / 100.0,
            "anchor": anchor_key
        }

    def _on_axes_changed(self, *args):
        """Handle any axes parameter change."""
        params = self._collect_axes_params()
        self._apply_axes_settings(params)

    def _on_scalebar_changed(self, *args):
        """Handle any scalebar parameter change."""
        params = self._collect_scalebar_params()
        self._apply_scalebar_settings(params)

    def _on_north_arrow_changed(self, *args):
        """Handle any north arrow parameter change."""
        params = self._collect_north_arrow_params()
        self._apply_north_arrow_settings(params)

    def _collect_north_arrow_params(self) -> Dict[str, Any]:
        """Collect all north arrow parameters."""
        pos_text = self.na_pos_combo.currentText()
        anchor_key = "top_right"
        if "Top Left" in pos_text:
            anchor_key = "top_left"
        elif "Bottom Right" in pos_text:
            anchor_key = "bottom_right"
        elif "Bottom Left" in pos_text:
            anchor_key = "bottom_left"

        return {
            "enabled": self.na_enable_check.isChecked(),
            "color": self._map_color(self.na_color_combo.currentText()),
            "font_size": self.na_font_spin.value(),
            "anchor": anchor_key
        }

    def _trigger_update(self):
        """Force apply all settings."""
        self._on_axes_changed()
        self._on_scalebar_changed()
        self._on_north_arrow_changed()

    def _apply_axes_settings(self, params: Dict[str, Any]):
        """
        Passes params to the renderer. 
        Note: Checks for specific method signature of the renderer.
        """
        renderer = self._get_renderer()
        if not renderer:
            return

        try:
            # Check for specific wrapper method
            if hasattr(renderer, 'set_floating_axes_enabled'):
                # Legacy support or Wrapper support
                renderer.set_floating_axes_enabled(
                    enabled=params['enabled'],
                    x_major=params['xtick_major'],
                    x_minor=params['xtick_minor'],
                    y_major=params['ytick_major'],
                    y_minor=params['ytick_minor'],
                    z_major=params['ztick_major'],
                    z_minor=params['ztick_minor'],
                    draw_box=True, # Always draw box if enabled per new design
                    color=params['color'],
                    line_width=params['width'],
                    font_size=params['font_size'],
                    auto_spacing=params.get('auto_spacing', True),
                )
            # Fallback to direct PyVista access if available (renderer.plotter)
            elif hasattr(renderer, 'show_bounds'):
                if params['enabled']:
                    renderer.show_bounds(
                        grid='back',
                        location='outer',
                        ticks='both',
                        color=params['color'],
                        font_size=params['font_size'],
                        line_width=params['width']
                    )
                else:
                    renderer.remove_bounds()
                    
        except Exception as e:
            logger.error(f"Error applying axes settings: {e}")

    def _apply_scalebar_settings(self, params: Dict[str, Any]):
        renderer = self._get_renderer()
        if not renderer:
            return

        # Ensure widget exists before applying settings
        self._ensure_overlay_widgets()

        try:
            if hasattr(renderer, 'set_scale_bar_3d_enabled'):
                renderer.set_scale_bar_3d_enabled(
                    enabled=params['enabled'],
                    units=params['units'],
                    bar_fraction=params['width_fraction'],
                    color=params['color'],
                    font_size=params['font_size'],
                    position_x=params['position_x'],
                    position_y=params['position_y'],
                    anchor=params.get('anchor', 'bottom_right')
                )
        except Exception as e:
            logger.error(f"Error applying scalebar settings: {e}")

    def _apply_north_arrow_settings(self, params: Dict[str, Any]):
        renderer = self._get_renderer()
        if not renderer:
            return

        # Ensure widget exists before applying settings
        self._ensure_overlay_widgets()

        try:
            if hasattr(renderer, 'set_north_arrow_enabled'):
                renderer.set_north_arrow_enabled(
                    enabled=params['enabled'],
                    color=params['color'],
                    font_size=params['font_size'],
                    anchor=params.get('anchor', 'top_right')
                )
        except Exception as e:
            logger.error(f"Error applying north arrow settings: {e}")

    def refresh(self):
        """Refresh panel state from renderer (if bidirectional sync is needed)."""
        # In a complex app, we might want to read the renderer state and update UI.
        # For now, we assume the UI drives the renderer.
        pass
