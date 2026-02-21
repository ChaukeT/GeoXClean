"""
Property panel for displaying and controlling block model properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, ContextManager
import logging
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QDoubleSpinBox, QGroupBox, QScrollArea, QFormLayout, QCheckBox,
    QPushButton, QProgressBar, QTextEdit, QFontDialog, QColorDialog,
    QMessageBox, QLineEdit, QDialog, QDialogButtonBox, QSpinBox,
    QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSignalBlocker, QSize, QTimer
from PyQt6.QtGui import QFont, QColor, QImage, QPixmap

from ..models.block_model import BlockModel
from ..visualization import ColorMapper
from ..controllers.app_state import AppState, get_empty_state_message
from .collapsible_group import CollapsibleGroup
from .signals import UISignals
from .modern_styles import (
    get_theme_colors, ModernColors, get_complete_panel_stylesheet, get_button_stylesheet,
    apply_modern_style
)

logger = logging.getLogger(__name__)


@dataclass
class LegendStyleState:
    """Container for legend styling parameters pushed to the renderer."""
    orientation: str = "vertical"
    font_size: int = 13
    tick_count: int = 5
    decimals: int = 2
    shadow: bool = True
    outline: bool = True
    background: QColor = field(default_factory=lambda: QColor(32, 32, 32))
    background_opacity: float = 0.6
    mode: str = "continuous"

    def background_tuple(self) -> Tuple[float, float, float]:
        return (self.background.red() / 255.0,
                self.background.green() / 255.0,
                self.background.blue() / 255.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.tick_count),
            "decimals": int(self.decimals if self.mode != "discrete" else 0),
            "shadow": bool(self.shadow),
            "outline": bool(self.outline),
            "background": self.background_tuple(),
            "background_opacity": float(self.background_opacity),
            "orientation": self.orientation,
            "font_size": int(self.font_size),
            "mode": self.mode,
        }


class PropertyPanel(QWidget):
    """
    Property panel for displaying block model information and controls.
    """
    
    # Signals
    property_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    color_mode_changed = pyqtSignal(str)
    filter_changed = pyqtSignal(str, float, float)
    slice_changed = pyqtSignal(str, float)
    transparency_changed = pyqtSignal(float)
    block_size_changed = pyqtSignal(float, float, float)
    legend_settings_changed = pyqtSignal(str, int)
    legend_style_changed = pyqtSignal(dict)
    axis_font_changed = pyqtSignal(str, int)
    axis_color_changed = pyqtSignal(tuple)
    opacity_changed = pyqtSignal(float)
    request_visualization = pyqtSignal(object, str)  # (grid, layer_name) – ask main window to add a block model layer
    
    def __init__(self, parent: Optional[QWidget] = None, signals: Optional[UISignals] = None):
        super().__init__()

        # State
        self.color_mapper = ColorMapper()
        self.current_model: Optional[BlockModel] = None
        self.drillhole_data: Optional[Dict[str, Any]] = None
        self.renderer = None

        # UISignals bus (for centralized signal emission)
        self.signals: Optional[UISignals] = signals

        # Registry - initialize early so it's available in all methods
        self.registry = None  # Will be populated by get_registry() when needed

        # Legend State
        self.legend_style = LegendStyleState()
        self._legend_is_categorical = False
        self._legend_range: Tuple[Optional[float], Optional[float]] = (None, None)
        self._custom_discrete_colors: Dict[Tuple[str, str], Dict[Any, str]] = {}
        self._updating_from_legend = False

        # Block-layer cache: remembers every block-type grid that has been
        # in the renderer so we can offer it back when the renderer drops it.
        # { layer_name: pyvista_grid }
        self._block_layer_cache: Dict[str, object] = {}

        # Guard flag: when True, _on_active_layer_changed will NOT re-emit
        # cached grids.  Set by main_window before add_layer() to prevent
        # the removal callback from re-emitting a stale grid that would
        # overwrite the new classification/block-model being added.
        self._suppress_cache_reemit: bool = False

        # UI Elements (Init references)
        self.scroll_area: Optional[QScrollArea] = None
        self.main_content_widget: Optional[QWidget] = None

        # Application state tracking
        self._app_state: AppState = AppState.EMPTY

        self._setup_ui()

        # Get registry reference after UI setup
        self.registry = self.get_registry()

        # Apply initial EMPTY state
        self._apply_empty_state()

        logger.info("Initialized property panel")
    
    def set_signals(self, signals: UISignals):
        """Set the UISignals bus for centralized signal emission."""
        self.signals = signals
        
        # Subscribe to geological model updates to refresh layer controls
        if signals and hasattr(signals, 'geologicalModelUpdated'):
            try:
                signals.geologicalModelUpdated.connect(self._on_geological_model_updated)
                logger.debug("PropertyPanel subscribed to geologicalModelUpdated signal")
            except Exception as e:
                logger.debug(f"Could not subscribe to geologicalModelUpdated: {e}")
    
    def _on_geological_model_updated(self, model_result):
        """Handle geological model update - refresh layer controls."""
        logger.debug("PropertyPanel received geological model update")
        try:
            # Refresh layer controls to show new geology layers
            self.update_layer_controls()
            # Also update the geology toggle state - BLOCK SIGNALS to prevent redundant toggle
            if hasattr(self, 'geology_toggle'):
                with self._block_signal(self.geology_toggle):
                    self.geology_toggle.setChecked(True)
        except Exception as e:
            logger.debug(f"PropertyPanel refresh after geo model update failed: {e}")

    # =========================================================================
    # Application State Handling
    # =========================================================================
    
    def on_app_state_changed(self, state: int) -> None:
        """
        Handle application state changes.
        
        This method is called when AppController emits app_state_changed signal.
        UI visibility and enablement is gated by state.
        
        Args:
            state: AppState enum value (as int for signal compatibility)
        """
        try:
            new_state = AppState(state)
        except ValueError:
            logger.warning(f"Invalid app state value: {state}")
            return
        
        if self._app_state == new_state:
            return
        
        old_state = self._app_state
        self._app_state = new_state
        logger.debug(f"PropertyPanel: State changed {old_state.name} -> {new_state.name}")
        
        # Apply state-specific UI rules
        if new_state == AppState.EMPTY:
            self._apply_empty_state()
        elif new_state == AppState.DATA_LOADED:
            self._apply_data_loaded_state()
        elif new_state == AppState.RENDERED:
            self._apply_rendered_state()
        elif new_state == AppState.BUSY:
            self._apply_busy_state()
    
    def _apply_empty_state(self) -> None:
        """Apply EMPTY state: Disable property controls, show placeholder text."""
        # Disable property visualization controls
        if hasattr(self, 'property_group'):
            self.property_group.setEnabled(False)
        if hasattr(self, 'visualization_group'):
            self.visualization_group.setEnabled(False)
        if hasattr(self, 'block_size_group'):
            self.block_size_group.setEnabled(False)
        if hasattr(self, 'quick_toggle_group'):
            self.quick_toggle_group.setVisible(False)
        
        # Update file info to show empty state message
        if hasattr(self, 'file_name_label'):
            self.file_name_label.setText("No file loaded")
        if hasattr(self, 'block_count_label'):
            self.block_count_label.setText("-")
        if hasattr(self, 'bounds_label'):
            self.bounds_label.setText("-")
        
        # Update property combo to show helpful message
        if hasattr(self, 'property_combo'):
            with self._block_signal(self.property_combo):
                self.property_combo.clear()
                self.property_combo.addItem(get_empty_state_message("property_controls"))
        
        # Update active layer combo
        if hasattr(self, 'active_layer_combo'):
            with self._block_signal(self.active_layer_combo):
                self.active_layer_combo.clear()
                self.active_layer_combo.addItem(get_empty_state_message("active_layer_dropdown"))
    
    def _apply_data_loaded_state(self) -> None:
        """Apply DATA_LOADED state: Enable file info, keep vis controls disabled."""
        # File info group should be enabled
        if hasattr(self, 'file_info_group'):
            self.file_info_group.setEnabled(True)
        
        # Property controls still disabled until rendering
        if hasattr(self, 'property_group'):
            self.property_group.setEnabled(False)
        if hasattr(self, 'visualization_group'):
            self.visualization_group.setEnabled(False)
    
    def _apply_rendered_state(self) -> None:
        """Apply RENDERED state: Enable all controls."""
        if hasattr(self, 'file_info_group'):
            self.file_info_group.setEnabled(True)
        if hasattr(self, 'property_group'):
            self.property_group.setEnabled(True)
        if hasattr(self, 'visualization_group'):
            self.visualization_group.setEnabled(True)
        if hasattr(self, 'block_size_group'):
            self.block_size_group.setEnabled(True)
        # Quick toggle visibility based on layers
        if hasattr(self, '_update_quick_toggle_states'):
            self._update_quick_toggle_states()
    
    def _apply_busy_state(self) -> None:
        """Apply BUSY state: Disable interactive controls during processing."""
        if hasattr(self, 'property_group'):
            self.property_group.setEnabled(False)
        if hasattr(self, 'visualization_group'):
            self.visualization_group.setEnabled(False)
        if hasattr(self, 'block_size_group'):
            self.block_size_group.setEnabled(False)

    @contextmanager
    def _block_signal(self, widget: QWidget):
        """Helper to block signals safely."""
        if widget:
            blocker = QSignalBlocker(widget)
            yield
            del blocker
        else:
            yield

    def _block_signals(self, state: bool):
        """
        Block/unblock signals on property combo.
        Used by MainWindow to prevent recursive signal storms.
        """
        if self.property_combo:
            self.property_combo.blockSignals(state)

    def set_active_layer(self, layer_name: str, layer_data: dict):
        """
        Public entry point for MainWindow.
        Safely rebuilds property list for the active layer.

        Args:
            layer_name: Name of the layer (e.g., "drillholes", "SGSIM: FE_PCT", etc.)
            layer_data: Layer data dict or PyVista mesh
        """
        if layer_data is None:
            self.property_combo.clear()
            logger.debug(f"[PROPERTY PANEL] set_active_layer: cleared (layer_data is None)")
            return

        self._block_signals(True)

        try:
            self.property_combo.clear()
            properties = []

            # Drillholes
            if layer_name == "drillholes" or (isinstance(layer_data, dict) and
                                             ('hole_segment_lith' in layer_data or
                                              'hole_polys' in layer_data or
                                              'lith_to_index' in layer_data)):
                logger.debug(f"[PROPERTY PANEL] set_active_layer: handling drillholes")

                # Add Lithology only if lithology data exists
                if isinstance(layer_data, dict):
                    if "hole_segment_lith" in layer_data and layer_data["hole_segment_lith"]:
                        properties.append("Lithology")
                        logger.debug(f"[PROPERTY PANEL] Added Lithology (hole_segment_lith exists)")
                    elif "lith_to_index" in layer_data and layer_data["lith_to_index"]:
                        properties.append("Lithology")
                        logger.debug(f"[PROPERTY PANEL] Added Lithology (lith_to_index exists)")

                    # Add current assay field
                    if "assay_field" in layer_data and layer_data["assay_field"]:
                        properties.append(layer_data["assay_field"])
                        logger.debug(f"[PROPERTY PANEL] Added assay field: {layer_data['assay_field']}")

                    # Add all available assay columns from database
                    if "database" in layer_data:
                        db = layer_data["database"]
                        if hasattr(db, 'assays') and hasattr(db.assays, 'columns'):
                            assay_cols = [
                                c for c in db.assays.columns
                                if (c.upper().endswith("_PCT") or c.upper().endswith("_PPM") or
                                    c.upper().endswith("_PPB") or c.upper().endswith("_GPT"))
                                and c.upper() not in ['HOLE_ID', 'DEPTH_FROM', 'DEPTH_TO', 'X', 'Y', 'Z']
                            ]
                            properties.extend(sorted(set(assay_cols)))
                            logger.debug(f"[PROPERTY PANEL] Added {len(assay_cols)} assay columns from database")

            # Block model / kriging / sgsim
            # These are PyVista meshes with array_names or dict with property keys
            else:
                logger.debug(f"[PROPERTY PANEL] set_active_layer: handling block model/grid")

                # PyVista mesh (SGSIM, Kriging, Block Model)
                if hasattr(layer_data, 'array_names'):
                    properties = list(layer_data.array_names)
                    logger.debug(f"[PROPERTY PANEL] Extracted {len(properties)} properties from PyVista mesh")
                # Dict-based layer data
                elif isinstance(layer_data, dict):
                    for key in layer_data.keys():
                        if isinstance(layer_data[key], (list, tuple)):
                            continue
                        if key.lower() not in ["geometry", "mesh", "actors", "database", "hole_polys",
                                              "hole_segment_lith", "hole_segment_assay", "lith_to_index"]:
                            properties.append(key)
                    logger.debug(f"[PROPERTY PANEL] Extracted {len(properties)} properties from dict keys")

            # Remove duplicates and sort
            properties = sorted(set(properties))
            logger.info(f"[PROPERTY PANEL] set_active_layer('{layer_name}'): Populating {len(properties)} properties: {properties[:5]}...")

            for p in properties:
                self.property_combo.addItem(p)

        finally:
            self._block_signals(False)

    def _setup_ui(self):
        """Setup the UI layout with modern styling and proper scrolling architecture."""
        # Apply modern stylesheet to the entire panel
        self.setStyleSheet(get_complete_panel_stylesheet())
        
        # Root layout for the PropertyPanel widget
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        
        # 1. Scroll Area with modern styling
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # Ensure scroll bar is painted on top
        self.scroll_area.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        # 2. Main Content Widget (The thing that scrolls)
        self.main_content_widget = QWidget()
        self.main_content_widget.setObjectName("PanelContent")
        
        # 3. Content Layout with improved spacing
        content_layout = QVBoxLayout(self.main_content_widget)
        content_layout.setContentsMargins(16, 16, 16, 24)  # Increased margins for modern look
        content_layout.setSpacing(14)  # Increased spacing between groups
        
        # 4. Add Groups
        self._create_quick_toggle_group(content_layout)  # Add quick visibility toggles first
        self._create_file_info_group(content_layout)
        self._create_property_group(content_layout)
        self._create_visualization_group(content_layout)
        self._create_block_size_group(content_layout)
        
        # 5. Add Stretch at the bottom to compact items upwards
        content_layout.addStretch(1)
        
        # 6. Finalize Setup
        self.scroll_area.setWidget(self.main_content_widget)
        root_layout.addWidget(self.scroll_area)

    # --- UI Group Creation Helpers ---

    def _add_labeled_row(self, layout: QFormLayout, label_text: str, widget: QWidget, tooltip: str = ""):
        """Standardized row creation with modern styling."""
        label = QLabel(label_text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 500;
            }}
        """)
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        layout.addRow(label, widget)

    def _create_slider_row(self, layout: QVBoxLayout, label_text: str, min_val: int, max_val: int, 
                           init_val: int, callback, unit_scale: float = 1.0) -> Tuple[QSlider, QLabel]:
        """Creates a modern styled slider row with label and value display."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(12)
        
        # Label with modern styling (if provided)
        if label_text:
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(60)
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.TEXT_SECONDARY};
                    font-size: 11px;
                    font-weight: 500;
                }}
            """)
            h_layout.addWidget(lbl)
        
        # Slider with modern styling
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.setMinimumHeight(24)
        slider.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Value label with modern styling
        val_lbl = QLabel(f"{init_val * unit_scale:.2f}")
        val_lbl.setMinimumWidth(50)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_lbl.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-weight: 600;
                font-size: 13px;
                padding: 4px 8px;
                background-color: {ModernColors.ELEVATED_BG};
                border-radius: 4px;
            }}
        """)
        
        # Connection logic
        def on_change(val):
            real_val = val * unit_scale
            val_lbl.setText(f"{real_val:.2f}")
            callback(val)

        slider.valueChanged.connect(on_change)
        
        h_layout.addWidget(slider, stretch=1)
        h_layout.addWidget(val_lbl)
        
        layout.addWidget(container)
        return slider, val_lbl

    # --- Group Implementations ---

    def _create_quick_toggle_group(self, parent_layout: QVBoxLayout):
        """Create quick visibility toggle buttons for common layers with modern styling."""
        self.quick_toggle_group = CollapsibleGroup("⚡ Quick Layers", collapsed=False)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Description with modern styling
        hint = QLabel("Toggle layer visibility:")
        hint.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_HINT};
                font-size: 11px;
                font-style: italic;
            }}
        """)
        layout.addWidget(hint)
        
        # Button container with grid layout for better alignment
        btn_container = QHBoxLayout()
        btn_container.setSpacing(8)
        
        # Drillholes toggle with modern styling
        self.drillholes_toggle = QPushButton("🔷 Drillholes")
        self.drillholes_toggle.setCheckable(True)
        self.drillholes_toggle.setChecked(True)  # Default visible
        self.drillholes_toggle.setStyleSheet(get_button_stylesheet("toggle"))
        self.drillholes_toggle.setMinimumHeight(42)
        self.drillholes_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drillholes_toggle.clicked.connect(self._on_drillholes_toggle)
        btn_container.addWidget(self.drillholes_toggle)
        
        # Block Model toggle with modern styling
        self.block_model_toggle = QPushButton("🧊 Block Model")
        self.block_model_toggle.setCheckable(True)
        self.block_model_toggle.setChecked(True)  # Default visible
        self.block_model_toggle.setStyleSheet(get_button_stylesheet("toggle"))
        self.block_model_toggle.setMinimumHeight(42)
        self.block_model_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.block_model_toggle.clicked.connect(self._on_block_model_toggle)
        btn_container.addWidget(self.block_model_toggle)
        
        # Geology toggle with modern styling
        self.geology_toggle = QPushButton("🏔️ Geology")
        self.geology_toggle.setCheckable(True)
        self.geology_toggle.setChecked(True)  # Default visible
        self.geology_toggle.setStyleSheet(get_button_stylesheet("toggle"))
        self.geology_toggle.setMinimumHeight(42)
        self.geology_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.geology_toggle.clicked.connect(self._on_geology_toggle)
        btn_container.addWidget(self.geology_toggle)
        
        layout.addLayout(btn_container)
        self.quick_toggle_group.add_layout(layout)
        parent_layout.addWidget(self.quick_toggle_group)
        
        # Initially hide - will be shown when layers exist
        self.quick_toggle_group.setVisible(False)

    def _on_drillholes_toggle(self, checked: bool):
        """Toggle drillholes layer visibility."""
        if not self.renderer or not hasattr(self, 'drillholes_toggle'):
            return
        try:
            # Try common drillhole layer names
            layer_names = ["drillholes", "Drillholes", "Drillhole"]
            found = False
            for name in layer_names:
                if name in self.renderer.active_layers:
                    self.renderer.set_layer_visibility(name, checked)
                    found = True
                    logger.info(f"Toggled {name} visibility: {checked}")
                    break
            
            if not found:
                # Search for any layer with 'drillhole' in the name
                for layer_name in self.renderer.active_layers.keys():
                    if 'drillhole' in layer_name.lower():
                        self.renderer.set_layer_visibility(layer_name, checked)
                        found = True
                        logger.info(f"Toggled {layer_name} visibility: {checked}")
                        break
            
            if not found:
                logger.warning("No drillhole layer found to toggle")
                self.drillholes_toggle.setChecked(not checked)  # Revert
        except Exception as e:
            logger.error(f"Failed to toggle drillholes: {e}", exc_info=True)
            self.drillholes_toggle.setChecked(not checked)  # Revert on error

    def _on_block_model_toggle(self, checked: bool):
        """Toggle block model layer visibility.

        Toggles ALL block-type layers (Block Model, SGSIM, Kriging, Classification)
        together as a group, matching the behavior of drillholes and geology toggles.
        """
        if not self.renderer or not hasattr(self, 'block_model_toggle'):
            return
        try:
            # Collect ALL block-type layers to toggle together
            # This ensures Block Model + SGSIM layers toggle as a group
            block_layers = []
            for layer_name, layer_info in self.renderer.active_layers.items():
                if self._is_block_type_layer(layer_name, layer_info):
                    block_layers.append(layer_name)

            if not block_layers:
                logger.warning("No block model/sgsim layer found to toggle")
                with self._block_signal(self.block_model_toggle):
                    self.block_model_toggle.setChecked(not checked)  # Revert
                return

            # Toggle ALL block-type layers together
            for layer_name in block_layers:
                self.renderer.set_layer_visibility(layer_name, checked)

            logger.info(f"Toggled {len(block_layers)} block layers: visibility={checked}")

            # Force single render at the end (like geology toggle does)
            if hasattr(self.renderer, 'plotter') and self.renderer.plotter:
                self.renderer.plotter.render()

        except Exception as e:
            logger.error(f"Failed to toggle block model: {e}", exc_info=True)
            with self._block_signal(self.block_model_toggle):
                self.block_model_toggle.setChecked(not checked)  # Revert on error

    def _on_geology_toggle(self, checked: bool):
        """Toggle geology layers (surfaces, solids, contacts) visibility."""
        if not self.renderer or not hasattr(self, 'geology_toggle'):
            return
        try:
            # Collect all geology layers first
            geology_layers = []
            for layer_name in self.renderer.active_layers.keys():
                lname = layer_name.lower()
                layer_info = self.renderer.active_layers.get(layer_name, {})
                layer_type = layer_info.get('layer_type', layer_info.get('type', ''))
                
                is_geology = (
                    lname.startswith('geology_') or
                    lname.startswith('geosurface:') or
                    lname.startswith('geosolid:') or
                    layer_type in ('geology_surface', 'geology_solid', 'geology_contacts')
                )
                
                if is_geology:
                    geology_layers.append(layer_name)
            
            if not geology_layers:
                logger.debug("No geology layers found to toggle")
                with self._block_signal(self.geology_toggle):
                    self.geology_toggle.setChecked(not checked)  # Revert
                return
            
            # Batch toggle all geology layers (minimal logging)
            for layer_name in geology_layers:
                self.renderer.set_layer_visibility(layer_name, checked)
            
            # Single log message for all layers
            logger.info(f"Toggled {len(geology_layers)} geology layers: visibility={checked}")
            
            # Force single render at the end
            if hasattr(self.renderer, 'plotter') and self.renderer.plotter:
                self.renderer.plotter.render()
                
        except Exception as e:
            logger.error(f"Failed to toggle geology layers: {e}", exc_info=True)
            with self._block_signal(self.geology_toggle):
                self.geology_toggle.setChecked(not checked)  # Revert on error

    def _update_quick_toggle_states(self):
        """Update toggle button states based on actual layer visibility."""
        if not self.renderer or not hasattr(self, 'drillholes_toggle') or not hasattr(self, 'block_model_toggle'):
            return
        
        # Update drillholes toggle
        drillholes_visible = False
        drillholes_exists = False
        for layer_name in self.renderer.active_layers.keys():
            if 'drillhole' in layer_name.lower():
                drillholes_exists = True
                layer = self.renderer.active_layers[layer_name]
                if layer.get('visible', True):
                    drillholes_visible = True
                # Don't break, check all drillhole layers
        
        with self._block_signal(self.drillholes_toggle):
            self.drillholes_toggle.setChecked(drillholes_visible)
            self.drillholes_toggle.setEnabled(drillholes_exists)
        
        # Update block model toggle (includes SGSIM/Kriging/Classification)
        block_model_visible = False
        block_model_exists = False
        for layer_name, layer_info in self.renderer.active_layers.items():
            if self._is_block_type_layer(layer_name, layer_info):
                block_model_exists = True
                if layer_info.get('visible', True):
                    block_model_visible = True
                # Don't break
        
        with self._block_signal(self.block_model_toggle):
            self.block_model_toggle.setChecked(block_model_visible)
            self.block_model_toggle.setEnabled(block_model_exists)
        
        # Update geology toggle (surfaces, solids, contacts)
        geology_visible = False
        geology_exists = False
        for layer_name, layer_info in self.renderer.active_layers.items():
            lname = layer_name.lower()
            # FIX: Check both 'layer_type' and 'type' keys (renderer uses 'type')
            layer_type = layer_info.get('layer_type', layer_info.get('type', ''))
            
            is_geology = (
                lname.startswith('geology_') or
                layer_type in ('geology_surface', 'geology_solid', 'geology_contacts')
            )
            
            if is_geology:
                geology_exists = True
                if layer_info.get('visible', True):
                    geology_visible = True
        
        if hasattr(self, 'geology_toggle'):
            with self._block_signal(self.geology_toggle):
                self.geology_toggle.setChecked(geology_visible)
                self.geology_toggle.setEnabled(geology_exists)
        
        # Show/hide the quick toggle group based on whether layers exist
        any_layers_exist = drillholes_exists or block_model_exists or geology_exists
        if hasattr(self, 'quick_toggle_group'):
            self.quick_toggle_group.setVisible(any_layers_exist)

    def _create_file_info_group(self, parent_layout: QVBoxLayout):
        """Create file information group with modern card-style layout."""
        self.file_info_group = CollapsibleGroup("📁 File Information", collapsed=True)
        layout = QFormLayout()
        layout.setVerticalSpacing(12)
        layout.setHorizontalSpacing(12)
        
        # File name with prominent styling
        self.file_name_label = QLabel("No file loaded")
        self.file_name_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-weight: 600;
                font-size: 12px;
            }}
        """)
        self.file_name_label.setWordWrap(True)
        
        # Other info labels with secondary styling
        self.file_format_label = QLabel("-")
        self.file_format_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        
        self.block_count_label = QLabel("-")
        self.block_count_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        
        self.bounds_label = QLabel("-")
        self.bounds_label.setWordWrap(True)
        self.bounds_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_HINT};
                font-size: 11px;
                font-family: 'Consolas', 'Courier New', monospace;
            }}
        """)
        
        # Create styled labels for form field names
        def create_field_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.TEXT_SECONDARY};
                    font-size: 11px;
                    font-weight: 500;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
            """)
            return label
        
        layout.addRow(create_field_label("File:"), self.file_name_label)
        layout.addRow(create_field_label("Format:"), self.file_format_label)
        layout.addRow(create_field_label("Blocks:"), self.block_count_label)
        layout.addRow(create_field_label("Bounds:"), self.bounds_label)
        
        self.file_info_group.add_layout(layout)
        parent_layout.addWidget(self.file_info_group)

    def _create_property_group(self, parent_layout: QVBoxLayout):
        """Create property visualization group with modern styling."""
        self.property_group = CollapsibleGroup("🎨 Property Visualization", collapsed=False)
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Active Layer section
        layer_layout = QFormLayout()
        layer_layout.setVerticalSpacing(12)
        layer_layout.setHorizontalSpacing(12)
        
        self.active_layer_combo = QComboBox()
        self.active_layer_combo.currentTextChanged.connect(self._on_active_layer_changed)
        self.active_layer_combo.setMinimumHeight(32)
        self._add_labeled_row(layer_layout, "Active Layer:", self.active_layer_combo, "Select which layer to control")

        # Block Model selector (shown only for block model layers)
        self.block_model_combo = QComboBox()
        self.block_model_combo.setMinimumHeight(32)
        self.block_model_combo.currentIndexChanged.connect(self._on_block_model_selection_changed)
        self._add_labeled_row(layer_layout, "Block Model:", self.block_model_combo, "Select which block model to visualize")
        # Initially hidden - shown only when block model layer is active
        self.block_model_combo.hide()
        # Hide the label too
        for i in range(layer_layout.rowCount()):
            label_item = layer_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
            if label_item and hasattr(label_item.widget(), 'text') and label_item.widget().text() == "Block Model:":
                label_item.widget().hide()
                break

        # Property selector
        self.property_combo = QComboBox()
        self.property_combo.setMinimumHeight(32)
        self.property_combo.currentTextChanged.connect(self._on_property_changed)
        self._add_labeled_row(layer_layout, "Property:", self.property_combo, "Select data attribute to color by")
        
        layout.addLayout(layer_layout)
        
        # Apply Button with primary styling
        apply_btn = QPushButton("🔄 Update Visualization")
        apply_btn.setStyleSheet(get_button_stylesheet("primary"))
        apply_btn.setMinimumHeight(38)
        apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_btn.clicked.connect(lambda: self._on_property_changed(self.property_combo.currentText()))
        layout.addWidget(apply_btn)

        # Separator with modern styling
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {ModernColors.DIVIDER}; border: none;")
        layout.addWidget(separator)

        # Color settings section
        color_layout = QFormLayout()
        color_layout.setVerticalSpacing(10)
        color_layout.setHorizontalSpacing(12)

        # Color Mode
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(['Continuous', 'Discrete'])
        self.color_mode_combo.setMinimumHeight(32)
        self.color_mode_combo.currentTextChanged.connect(self._on_color_mode_changed)
        self._add_labeled_row(color_layout, "Color Mode:", self.color_mode_combo)

        # Colormap
        self.colormap_combo = QComboBox()
        self.colormap_combo.setMinimumHeight(32)
        # Populate maps
        categorical_maps = ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2']
        self.colormap_combo.addItem("--- Categorical ---")
        # Make separator non-selectable via keyboard (UX-008 fix)
        cat_separator_item = self.colormap_combo.model().item(0)
        cat_separator_item.setEnabled(False)
        cat_separator_item.setFlags(cat_separator_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.colormap_combo.addItems(categorical_maps)
        
        self.colormap_combo.addItem("--- Continuous ---")
        # Make separator non-selectable via keyboard (UX-008 fix)
        cont_separator_item = self.colormap_combo.model().item(len(categorical_maps) + 1)
        cont_separator_item.setEnabled(False)
        cont_separator_item.setFlags(cont_separator_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        continuous_maps = [cm for cm in self.color_mapper.get_available_colormaps() if cm not in categorical_maps]
        self.colormap_combo.addItems(continuous_maps)
        
        self.colormap_combo.setCurrentText('turbo')
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self._add_labeled_row(color_layout, "Colormap:", self.colormap_combo)
        
        layout.addLayout(color_layout)

        # Custom Colors Buttons with modern styling
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.custom_colors_btn = QPushButton("🎨 Define Colors")
        self.custom_colors_btn.setStyleSheet(get_button_stylesheet("secondary"))
        self.custom_colors_btn.setMinimumHeight(36)
        self.custom_colors_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.custom_colors_btn.clicked.connect(self._on_custom_colors_clicked)
        self.custom_colors_btn.setVisible(False)
        
        self.clear_custom_colors_btn = QPushButton("↺ Reset")
        self.clear_custom_colors_btn.setStyleSheet(get_button_stylesheet("secondary"))
        self.clear_custom_colors_btn.setMinimumHeight(36)
        self.clear_custom_colors_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_custom_colors_btn.clicked.connect(self._on_clear_custom_colors_clicked)
        self.clear_custom_colors_btn.setVisible(False)
        self.clear_custom_colors_btn.setEnabled(False)

        btn_layout.addWidget(self.custom_colors_btn)
        btn_layout.addWidget(self.clear_custom_colors_btn)
        layout.addLayout(btn_layout)

        self.property_group.add_layout(layout)
        parent_layout.addWidget(self.property_group)


    def _create_visualization_group(self, parent_layout: QVBoxLayout):
        """Create visualization settings group with modern styling."""
        self.visualization_group = CollapsibleGroup("⚙️ Display Settings", collapsed=True)
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Opacity slider with modern styling
        opacity_container = QWidget()
        opacity_layout = QVBoxLayout(opacity_container)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(6)
        
        opacity_label_header = QLabel("Opacity")
        opacity_label_header.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 500;
                text-transform: uppercase;
            }}
        """)
        opacity_layout.addWidget(opacity_label_header)
        
        self.opacity_slider, self.opacity_label = self._create_slider_row(
            opacity_layout, "", 0, 100, 80, self._on_opacity_changed, unit_scale=0.01)
        
        layout.addWidget(opacity_container)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {ModernColors.DIVIDER}; border: none;")
        layout.addWidget(separator)

        # Additional display controls
        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(12)

        # Mesh style
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Solid", "Wireframe", "Solid + Edges"])
        self.style_combo.setCurrentText("Solid")
        self.style_combo.setMinimumHeight(32)
        self.style_combo.setToolTip(
            "Render style for scene geometry:\n"
            "• Solid: filled surfaces\n"
            "• Wireframe: mesh edges only\n"
            "• Solid + Edges: solid with emphasized edges"
        )
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        form.addRow(self._create_field_label("Mesh Style:"), self.style_combo)

        # Shading
        self.shading_combo = QComboBox()
        self.shading_combo.addItems(["Flat", "Smooth"])
        self.shading_combo.setCurrentText("Flat")
        self.shading_combo.setMinimumHeight(32)
        self.shading_combo.setToolTip(
            "Shading mode for surfaces:\n"
            "• Flat: geological, faceted look (best for contacts)\n"
            "• Smooth: curved, polished look"
        )
        self.shading_combo.currentTextChanged.connect(self._on_shading_changed)
        form.addRow(self._create_field_label("Shading:"), self.shading_combo)

        # Line width
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 10)
        self.line_width_spin.setValue(2)
        self.line_width_spin.setMinimumHeight(32)
        self.line_width_spin.setToolTip(
            "Line width in pixels for wireframes, contact edges and polylines"
        )
        self.line_width_spin.valueChanged.connect(self._on_line_width_changed)
        form.addRow(self._create_field_label("Line Width:"), self.line_width_spin)

        # Point size
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(10)
        self.point_size_spin.setMinimumHeight(32)
        self.point_size_spin.setToolTip(
            "Point size in pixels for collars, scatter points and point clouds"
        )
        self.point_size_spin.valueChanged.connect(self._on_point_size_changed)
        form.addRow(self._create_field_label("Point Size:"), self.point_size_spin)

        layout.addLayout(form)

        self.visualization_group.add_layout(layout)
        parent_layout.addWidget(self.visualization_group)
    
    def _create_field_label(self, text: str) -> QLabel:
        """Create a styled field label for form layouts."""
        label = QLabel(text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 500;
            }}
        """)
        return label

    def _create_block_size_group(self, parent_layout: QVBoxLayout):
        """Create block geometry override group with modern styling."""
        self.block_size_group = CollapsibleGroup("📏 Block Geometry Override", collapsed=True)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Add hint text
        hint = QLabel("Override block dimensions for visualization:")
        hint.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_HINT};
                font-size: 11px;
                font-style: italic;
            }}
        """)
        layout.addWidget(hint)
        
        form = QFormLayout()
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(12)

        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.0001, 1e9)
        self.dx_spin.setValue(1.0)
        self.dx_spin.setMinimumHeight(32)
        self.dx_spin.setDecimals(4)
        form.addRow(self._create_field_label("DX (Width):"), self.dx_spin)

        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.0001, 1e9)
        self.dy_spin.setValue(1.0)
        self.dy_spin.setMinimumHeight(32)
        self.dy_spin.setDecimals(4)
        form.addRow(self._create_field_label("DY (Length):"), self.dy_spin)

        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.0001, 1e9)
        self.dz_spin.setValue(1.0)
        self.dz_spin.setMinimumHeight(32)
        self.dz_spin.setDecimals(4)
        form.addRow(self._create_field_label("DZ (Height):"), self.dz_spin)

        layout.addLayout(form)
        
        apply_btn = QPushButton("✓ Apply Resize")
        apply_btn.setStyleSheet(get_button_stylesheet("primary"))
        apply_btn.setMinimumHeight(38)
        apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_btn.clicked.connect(self._emit_block_size)
        layout.addWidget(apply_btn)

        self.block_size_group.add_layout(layout)
        parent_layout.addWidget(self.block_size_group)

    # --- Core Logic ---

    def set_renderer(self, renderer):
        self.renderer = renderer
        # NOTE: Do NOT call renderer.set_layer_change_callback here!
        # MainWindow sets a global callback that handles state updates.
        # PropertyPanel receives updates via MainWindow._on_renderer_layers_changed().
        
        # Show quick toggle group when renderer is available
        if hasattr(self, 'quick_toggle_group'):
            self.quick_toggle_group.setVisible(True)
        
        self.update_layer_controls()
        
        # Connect callbacks for syncing
        refresh = getattr(renderer, "_refresh_legend_from_active_layer", None)
        if callable(refresh):
            self.property_changed.connect(lambda *_: refresh())
            self.colormap_changed.connect(lambda *_: refresh())
            self.color_mode_changed.connect(lambda *_: refresh())

    def set_block_model(self, block_model: BlockModel):
        self.current_model = block_model
        self._update_file_info()
        self._update_property_lists()
        
        if block_model.bounds:
            # Reset slice sliders (if they exist)
            if hasattr(self, 'x_slice_slider') and hasattr(self, 'y_slice_slider') and hasattr(self, 'z_slice_slider'):
                for slider in [self.x_slice_slider, self.y_slice_slider, self.z_slice_slider]:
                    slider.setValue(0)
        
        # Update layer controls to refresh Active Layer combo box
        # Use QTimer to ensure renderer has finished registering the layer
        QTimer.singleShot(100, self.update_layer_controls)
    
    def get_registry(self):
        """
        Get DataRegistry instance via dependency injection.
        PropertyPanel doesn't inherit from BasePanel, so we need this method.
        """
        # Try to get registry from parent MainWindow
        parent = self.parent()
        while parent:
            if hasattr(parent, 'controller') and parent.controller:
                if hasattr(parent.controller, 'registry'):
                    return parent.controller.registry
            if hasattr(parent, '_registry') and parent._registry:
                return parent._registry
            parent = parent.parent()
        
        # Fallback to singleton (FIX CS-001: was calling self.get_registry() recursively)
        from ..core.data_registry import DataRegistry
        try:
            return DataRegistry.get_instance()
        except Exception:
            return None
    
    def set_drillhole_data(self, drillhole_data: Dict[str, Any]):
        """
        Legacy compatibility method - delegates to registry-based data loading.
        New code should use registry.drillholeDataLoaded signal.
        """
        # Store data and update UI
        self.drillhole_data = drillhole_data
        
        if 'composites_df' in drillhole_data:
            df = drillhole_data['composites_df']
            self.file_name_label.setText("Drillhole Data")
            self.file_format_label.setText("Composites")
            self.block_count_label.setText(f"{len(df):,} intervals")

            # Update Inputs
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            self.block_size_group.setEnabled(False)
            if hasattr(self, 'slice_group'):
                self.slice_group.setEnabled(False)
            
            with self._block_signal(self.property_combo):
                self.property_combo.clear()
                self.property_combo.addItems(numeric_cols)

            if hasattr(self, 'filter_property_combo') and self.filter_property_combo:
                with self._block_signal(self.filter_property_combo):
                    self.filter_property_combo.clear()
                    self.filter_property_combo.addItems(numeric_cols)
            
            # Set default
            current = drillhole_data.get('color_by', numeric_cols[0] if numeric_cols else None)
            if current in numeric_cols:
                self.property_combo.setCurrentText(current)
    
    def refresh_from_registry(self):
        """Refresh drillhole data from registry."""
        try:
            registry = self.get_registry()
            if registry:
                drillhole_data = registry.get_drillhole_data()
                if drillhole_data is not None:
                    # Convert registry format to panel format
                    if isinstance(drillhole_data, dict):
                        # Fix: Explicitly check for non-empty DataFrames to avoid ValueError
                        composites = drillhole_data.get('composites')
                        assays = drillhole_data.get('assays')
                        if isinstance(composites, pd.DataFrame) and not composites.empty:
                            composites_df = composites
                        elif isinstance(assays, pd.DataFrame) and not assays.empty:
                            composites_df = assays
                        else:
                            composites_df = None
                        
                        panel_data = {
                            'composites_df': composites_df,
                            'color_by': drillhole_data.get('color_by')
                        }
                        if panel_data['composites_df'] is not None:
                            self.set_drillhole_data(panel_data)
        except Exception as e:
            logger.error(f"Failed to refresh from registry: {e}", exc_info=True)

    def clear(self):
        """Clear panel state and reset to EMPTY state."""
        self.current_model = None
        self.drillhole_data = None
        self.block_size_group.setEnabled(True)
        self.file_name_label.setText("No file loaded")
        self.file_format_label.setText("-")
        self.block_count_label.setText("-")
        self.bounds_label.setText("-")
        self.property_combo.clear()
        if hasattr(self, 'filter_property_combo') and self.filter_property_combo:
            self.filter_property_combo.clear()
        
        # FIX CS-010: Clear custom color mappings on reset
        self._custom_discrete_colors.clear()
        
        # Clear active layer combo
        if hasattr(self, 'active_layer_combo'):
            with self._block_signal(self.active_layer_combo):
                self.active_layer_combo.clear()
                self.active_layer_combo.addItem(get_empty_state_message("active_layer_dropdown"))
        
        # Reset legend state
        self._legend_is_categorical = False
        self._legend_range = (None, None)

        # Apply EMPTY state
        self._apply_empty_state()

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        self.clear()
        super().clear_panel()
        logger.info("PropertyPanel: Panel fully cleared")

    def _update_file_info(self):
        if not self.current_model: return

        # Safely get block count - handle both BlockModel and DataFrame
        if hasattr(self.current_model, 'block_count'):
            block_count = self.current_model.block_count
            meta = self.current_model.metadata
            self.file_name_label.setText(Path(meta.source_file).name)
            self.file_format_label.setText(meta.file_format)
        elif isinstance(self.current_model, pd.DataFrame):
            block_count = len(self.current_model)
            self.file_name_label.setText("DataFrame")
            self.file_format_label.setText("DataFrame")
        else:
            block_count = 0
            self.file_name_label.setText("Unknown")
            self.file_format_label.setText("Unknown")

        self.block_count_label.setText(f"{block_count:,}")
        
        b = self.current_model.bounds
        if b:
            self.bounds_label.setText(f"X: {b[0]:.1f}-{b[1]:.1f}, Y: {b[2]:.1f}-{b[3]:.1f}, Z: {b[4]:.1f}-{b[5]:.1f}")

    def _update_property_lists(self):
        if not self.current_model: return
        props = self.current_model.get_property_names()
        self.property_combo.clear()
        self.property_combo.addItems(props)
        if hasattr(self, 'filter_property_combo') and self.filter_property_combo:
            self.filter_property_combo.clear()
            self.filter_property_combo.addItems(props)

    # --- Layer Management ---

    def update_layer_controls(self):
        """Populate active layer combobox based on renderer state.

        Block-type layers (SGSIM, Kriging, Classification, …) are **cached**
        so that they remain selectable even after the renderer replaces or
        removes them.  Selecting a cached layer will re-emit it to the viewer.
        """
        if not self.renderer:
            return

        active_layers = self.renderer.active_layers

        # ── 1. Cache every block-type grid currently in the renderer ──
        for name, info in active_layers.items():
            if self._is_block_type_layer(name, info):
                grid = info.get('data')
                if grid is not None and hasattr(grid, 'bounds'):
                    self._block_layer_cache[name] = grid

        with self._block_signal(self.active_layer_combo):
            current = self.active_layer_combo.currentText()
            self.active_layer_combo.clear()

            if active_layers:
                names = list(active_layers.keys())
                self.active_layer_combo.addItems(names)

                # ── 2. Add cached block layers that are no longer live ──
                for cached_name in sorted(self._block_layer_cache.keys()):
                    if cached_name not in names:
                        self.active_layer_combo.addItem(cached_name)
                        logger.debug(
                            f"[PROPERTY PANEL] Re-added cached layer to combo: {cached_name}"
                        )

                # Auto-switch to a newly added block layer
                all_names = [
                    self.active_layer_combo.itemText(i)
                    for i in range(self.active_layer_combo.count())
                ]
                block_volume_prefixes = [
                    "Block Model", "Kriging", "SGSIM", "Resource Classification"
                ]
                new_block_layer = None
                for name in all_names:
                    if any(name.startswith(p) for p in block_volume_prefixes):
                        if name != current and name in active_layers:
                            new_block_layer = name
                            break

                if new_block_layer:
                    self.active_layer_combo.setCurrentText(new_block_layer)
                    logger.info(
                        f"[PROPERTY PANEL] Auto-switched to new block layer: {new_block_layer}"
                    )
                elif current in all_names:
                    self.active_layer_combo.setCurrentText(current)
                elif all_names:
                    self.active_layer_combo.setCurrentIndex(0)

                self._apply_rendered_state()
            else:
                self.active_layer_combo.addItem("No layers active")

        # Manually trigger property update since signal was blocked
        current_layer = self.active_layer_combo.currentText()
        if current_layer and current_layer != "No layers active":
            self._on_active_layer_changed(current_layer)

        if hasattr(self, '_update_custom_colors_buttons_state'):
            self._update_custom_colors_buttons_state()

    def _on_active_layer_changed(self, layer_name: str):
        """
        Handle active layer selection changes.

        FIX CS-005: Validates layer existence before accessing properties.
        """
        logger.info(f"[PROPERTY PANEL] ===== _on_active_layer_changed CALLED with layer: '{layer_name}' =====")

        if not layer_name or layer_name == "No layers active":
            self.property_combo.clear()
            logger.debug(f"[PROPERTY PANEL] Cleared property combo (no active layer)")
            return
        
        if not self.renderer: 
            return
        
        # FIX CS-005: Validate layer still exists in renderer
        if layer_name not in self.renderer.active_layers:
            # ── Check the block-layer cache: re-emit grid if available ──
            cached_grid = self._block_layer_cache.get(layer_name)
            if cached_grid is not None:
                # ── Guard: skip re-emission if a new layer is being added ──
                # When _handle_classification_visualization (or similar) calls
                # add_layer(), mutual exclusivity removes the old layer, which
                # fires this callback.  Re-emitting the cached grid here would
                # overwrite the NEW mesh that add_layer is about to register.
                if self._suppress_cache_reemit:
                    logger.info(
                        f"[PROPERTY PANEL] Layer '{layer_name}' cache re-emission "
                        f"SUPPRESSED (new layer being added)"
                    )
                    return

                logger.info(
                    f"[PROPERTY PANEL] Layer '{layer_name}' not in renderer but "
                    f"found in cache — re-emitting to restore it"
                )
                self.request_visualization.emit(cached_grid, layer_name)

                # Populate properties from the cached grid RIGHT NOW so the
                # user doesn't see "No properties available" while the
                # renderer round-trip completes.
                with self._block_signal(self.property_combo):
                    self.property_combo.clear()
                    props = []
                    if hasattr(cached_grid, 'array_names'):
                        props = list(cached_grid.array_names)
                    elif hasattr(cached_grid, 'cell_data'):
                        props = list(cached_grid.cell_data.keys())
                    if props:
                        self.property_combo.addItems(props)
                        logger.info(
                            f"[PROPERTY PANEL] Populated {len(props)} properties "
                            f"from cached grid: {props}"
                        )
                    else:
                        self.property_combo.addItem("Restoring layer…")

                # Show block model selector for cached block layers
                self._show_block_model_selector(True)
                self._populate_block_model_list()
                return

            logger.warning(f"[PROPERTY PANEL] Layer '{layer_name}' no longer exists in renderer")
            with self._block_signal(self.property_combo):
                self.property_combo.clear()
                self.property_combo.addItem("Layer not found")
            return
        
        layer_info = self.renderer.active_layers.get(layer_name, {})
        if not layer_info:
            logger.warning(f"[PROPERTY PANEL] Layer '{layer_name}' has no info")
            return
        
        layer_data = layer_info.get('data')
        layer_type = layer_info.get('layer_type', layer_info.get('type', ''))
        
        # Debug logging
        logger.debug(f"[PROPERTY PANEL] Layer changed to: {layer_name}")
        logger.debug(f"[PROPERTY PANEL] layer_type: {layer_type}")
        logger.debug(f"[PROPERTY PANEL] layer_data type: {type(layer_data)}")
        if isinstance(layer_data, dict):
            logger.debug(f"[PROPERTY PANEL] layer_data keys: {list(layer_data.keys())[:10]}")

        with self._block_signal(self.property_combo):
            self.property_combo.clear()
            properties = []

            # Initialize flags before extraction logic
            is_drillhole = False
            is_geology_layer = False
            is_block_model = False

            # ================================================================
            # BLOCK MODEL / SGSIM LAYER HANDLING
            # ================================================================
            # Handle block models first (SGSIM, Kriging, Block Model layers)
            # These store PyVista grids directly in layer_data
            if self._is_block_type_layer(layer_name, layer_info):
                is_block_model = True
                logger.info(f"[PROPERTY PANEL] Detected block model layer: {layer_name}")

                # Show block model selector and populate with available models
                self._show_block_model_selector(True)
                self._populate_block_model_list()

                # Block models can store PyVista grids either directly or wrapped in {'mesh': mesh}
                # Try direct mesh first
                if hasattr(layer_data, 'array_names'):
                    properties = list(layer_data.array_names)
                    logger.debug(f"[PROPERTY PANEL] Extracted {len(properties)} properties from block model: {properties}")
                elif hasattr(layer_data, 'cell_data') and hasattr(layer_data, 'point_data'):
                    # Fallback: extract from cell_data and point_data
                    cell_props = list(layer_data.cell_data.keys()) if hasattr(layer_data, 'cell_data') else []
                    point_props = list(layer_data.point_data.keys()) if hasattr(layer_data, 'point_data') else []
                    properties = cell_props + point_props
                    logger.debug(f"[PROPERTY PANEL] Extracted properties from cell/point data: {properties}")
                # Handle wrapped mesh format: {'mesh': mesh}
                elif isinstance(layer_data, dict) and 'mesh' in layer_data:
                    mesh = layer_data['mesh']
                    if hasattr(mesh, 'array_names'):
                        mesh_props = list(mesh.array_names)
                        # For classification layers, show "Classification" as the primary property
                        if layer_type == 'classification':
                            properties = ["Classification"]
                            # Add other diagnostic properties but NOT redundant ones
                            excluded = {"Classification", "CLASS", "Category", "Classification_Categories"}
                            for prop in mesh_props:
                                if prop not in excluded:
                                    properties.append(prop)
                        else:
                            properties = mesh_props
                        logger.debug(f"[PROPERTY PANEL] Extracted {len(properties)} properties from wrapped mesh: {properties}")
                    else:
                        logger.warning(f"[PROPERTY PANEL] Wrapped mesh has no array_names attribute")
                else:
                    logger.warning(f"[PROPERTY PANEL] Block model layer has no recognizable property structure")

            # Extraction Logic
            elif isinstance(layer_data, dict):
                # Hide block model selector for non-block model layers
                self._show_block_model_selector(False)

                # Drillhole layer - extract lithology and assay properties
                is_drillhole = (layer_type == 'drillhole' or
                               'lith_to_index' in layer_data or
                               'hole_polys' in layer_data)
                logger.debug(f"[PROPERTY PANEL] is_drillhole: {is_drillhole}")

                if is_drillhole:
                    # Add lithology option if lithology data exists
                    has_lith = 'lith_to_index' in layer_data and layer_data['lith_to_index']
                    logger.debug(f"[PROPERTY PANEL] has_lith: {has_lith}")
                    if has_lith:
                        properties.append("Lithology")
                    
                    # Add assay fields - check hole_segment_assay for available fields
                    if 'hole_segment_assay' in layer_data:
                        # Get the current assay field
                        current_assay = layer_data.get('assay_field')
                        if current_assay:
                            properties.append(current_assay)
                            logger.debug(f"[PROPERTY PANEL] Added current assay field: {current_assay}")
                        
                        # Try to get all available assay fields from database
                        db = layer_data.get('database')
                        if db and hasattr(db, 'assays') and not db.assays.empty:
                            # database.assays is a DataFrame - extract columns directly
                            # Exclude non-assay columns (metadata, system columns)
                            exclude_cols = {
                                'hole_id', 'depth_from', 'depth_to', 'lithology', 'lith_code', 'lith',
                                'global_interval_id', 'GLOBAL_INTERVAL_ID',
                                'x', 'y', 'z', 'X', 'Y', 'Z',
                                'length', 'LENGTH', 'support', 'SUPPORT',
                                'sample_count', 'SAMPLE_COUNT', 'total_mass', 'TOTAL_MASS',
                                'method', 'METHOD', 'weighting', 'WEIGHTING',
                            }
                            assay_fields = set(db.assays.columns) - exclude_cols
                            for field in sorted(assay_fields):
                                if field not in properties and not field.lower().startswith('global_'):
                                    properties.append(field)
                            logger.debug(f"[PROPERTY PANEL] Found assay fields from db: {list(assay_fields)[:5]}")
                    
                    # Also check composite_df for additional properties
                    composite_df = layer_data.get('composite_df')
                    if composite_df is not None and hasattr(composite_df, 'columns'):
                        exclude = {
                            'HOLEID', 'HOLE_ID', 'FROM', 'TO', 'DEPTH_FROM', 'DEPTH_TO', 
                            'X', 'Y', 'Z', 'LENGTH', 'GLOBAL_INTERVAL_ID',
                            'SAMPLE_COUNT', 'TOTAL_MASS', 'SUPPORT', 'METHOD', 'WEIGHTING',
                            # FIX: Exclude interval measurement columns - these are not assay grades
                            'INTERVAL', 'INTERVAL_M', 'INTERVAL_LENGTH', 'INT_LENGTH',
                            'COMP_LENGTH', 'COMPOSITE_LENGTH', 'SAMPLE_LENGTH',
                        }
                        for col in composite_df.columns:
                            col_upper = col.upper()
                            if col_upper not in exclude and col not in properties:
                                # Skip system columns
                                if col_upper.startswith('GLOBAL_'):
                                    continue
                                # Skip interval-related columns (pattern match for variants)
                                if 'INTERVAL' in col_upper or col_upper.endswith('_LENGTH'):
                                    continue
                                # Only add numeric columns
                                if composite_df[col].dtype.kind in {'i', 'u', 'f'}:
                                    properties.append(col)
                        logger.debug(f"[PROPERTY PANEL] Found composite columns: {list(composite_df.columns)[:5]}")
                    
                    # Also check lith_colors for lithology codes (fallback)
                    if not has_lith and 'lith_colors' in layer_data and layer_data['lith_colors']:
                        properties.append("Lithology")
                        logger.debug(f"[PROPERTY PANEL] Added Lithology from lith_colors fallback")
                                    
                elif 'mesh' in layer_data and hasattr(layer_data['mesh'], 'array_names'):
                    mesh_props = list(layer_data['mesh'].array_names)
                    # For classification layers, show "Classification" as the primary property
                    if layer_type == 'classification':
                        properties = ["Classification"]
                        # Add other diagnostic properties but NOT redundant ones
                        excluded = {"Classification", "CLASS", "Category", "Classification_Categories"}
                        for prop in mesh_props:
                            if prop not in excluded:
                                properties.append(prop)
                    else:
                        properties = mesh_props
            elif hasattr(layer_data, 'array_names'):
                properties = list(layer_data.array_names)
            elif hasattr(layer_data, 'get_property_names'):
                properties = layer_data.get_property_names()

            # ================================================================
            # CLASSIFICATION LAYER HANDLING
            # ================================================================
            # When a classification layer is selected, automatically switch to discrete mode
            # Classification is inherently categorical (Measured, Indicated, Inferred, Unclassified)
            is_classification_layer = (layer_type == 'classification')

            if is_classification_layer:
                logger.info(f"[PROPERTY PANEL] Detected classification layer: {layer_name}")

                # CRITICAL FIX: Force RENDERED state for classification layers
                # This ensures controls are enabled for color editing
                self._apply_rendered_state()
                logger.debug("[PROPERTY PANEL] Applied RENDERED state for classification layer")

                # Auto-switch to Discrete mode for classification (categorical data)
                if hasattr(self, 'color_mode_combo'):
                    with self._block_signal(self.color_mode_combo):
                        self.color_mode_combo.setCurrentText("Discrete")
                    logger.debug("[PROPERTY PANEL] Auto-switched to Discrete mode for classification")

                # Show custom colors button for classification categories
                if hasattr(self, 'custom_colors_btn'):
                    self.custom_colors_btn.setVisible(True)
                if hasattr(self, 'clear_custom_colors_btn'):
                    self.clear_custom_colors_btn.setVisible(True)

                # Update UI to reflect classification layer state
                if hasattr(self, '_update_color_mode_visibility'):
                    self._update_color_mode_visibility()

            # ================================================================
            # GEOLOGY LAYER HANDLING (GeoSolid / GeoSurface)
            # ================================================================
            # When a geology layer is selected, automatically switch to discrete mode
            # and show "Formations" as the property (unit-based coloring)
            is_geology_layer = (
                "GeoSolid" in layer_name or 
                "GeoSurface" in layer_name or
                layer_type in ('geology_surface', 'geology_solid', 'geology_volume')
            )
            
            if is_geology_layer:
                logger.info(f"[PROPERTY PANEL] Detected geology layer: {layer_name}")
                
                # Clear and set to "Formations" (unit-based discrete coloring)
                properties = ["Formations"]
                
                # CRITICAL FIX: Force RENDERED state for geology layers
                # Geology layers mean the model IS visualized, so controls must be active
                # This overrides any DATA_LOADED state that may have kept controls disabled
                self._apply_rendered_state()
                logger.debug("[PROPERTY PANEL] Applied RENDERED state for geology layer")
                
                # Auto-switch to Discrete mode for geology
                if hasattr(self, 'color_mode_combo'):
                    with self._block_signal(self.color_mode_combo):
                        self.color_mode_combo.setCurrentText("Discrete")
                    logger.debug("[PROPERTY PANEL] Auto-switched to Discrete mode for geology")
                
                # Show custom colors button for geology units
                if hasattr(self, 'custom_colors_btn'):
                    self.custom_colors_btn.setVisible(True)
                
                # Update UI to reflect geology layer state
                if hasattr(self, '_update_color_mode_visibility'):
                    self._update_color_mode_visibility()
            
            logger.debug(f"[PROPERTY PANEL] Final properties list: {properties}")

            if properties:
                # Remove duplicates while preserving order
                seen = set()
                unique_props = []
                for p in properties:
                    if p not in seen:
                        seen.add(p)
                        unique_props.append(p)
                self.property_combo.addItems(unique_props)
                logger.info(f"[PROPERTY PANEL] Added {len(unique_props)} properties to combo")

                # CRITICAL FIX: Apply RENDERED state for ALL layers with properties
                # This ensures controls are enabled for block models, drillholes, AND geology
                # Note: Geology, classification, and block model layers need RENDERED state
                if is_block_model:
                    self._apply_rendered_state()
                    logger.debug(f"[PROPERTY PANEL] Applied RENDERED state for block model layer: {layer_name}")
                elif not is_geology_layer and not is_classification_layer:
                    self._apply_rendered_state()
                    logger.debug(f"[PROPERTY PANEL] Applied RENDERED state for layer: {layer_name}")
                
                # Set current selection to the active assay field (if drillhole layer)
                # Note: We're already inside _block_signal block, so no need to nest another one
                if is_drillhole and isinstance(layer_data, dict) and 'hole_segment_assay' in layer_data:
                    current_assay = layer_data.get('assay_field')
                    if current_assay and current_assay in unique_props:
                        self.property_combo.setCurrentText(current_assay)
                        logger.debug(f"[PROPERTY PANEL] Set current property to: {current_assay}")
                    elif current_assay:
                        logger.warning(f"[PROPERTY PANEL] Current assay field '{current_assay}' not found in properties list")

                # Set current selection for block model layers
                # Try to extract property name from layer name (e.g., "SGSIM: FE_PCT_SGSIM_MEAN" -> "FE_PCT_SGSIM_MEAN")
                elif is_block_model and unique_props:
                    # Try to find property name in layer name
                    property_from_layer = None
                    if ':' in layer_name:
                        # Extract property name after colon (e.g., "SGSIM: FE_PCT_SGSIM_MEAN")
                        property_from_layer = layer_name.split(':', 1)[1].strip()

                    if property_from_layer and property_from_layer in unique_props:
                        self.property_combo.setCurrentText(property_from_layer)
                        logger.debug(f"[PROPERTY PANEL] Set current property to: {property_from_layer}")
                    else:
                        # Fallback: select first property
                        self.property_combo.setCurrentIndex(0)
                        logger.debug(f"[PROPERTY PANEL] Set current property to first: {unique_props[0]}")
            else:
                # Add placeholder as disabled and non-selectable (UX-002 fix)
                self.property_combo.addItem("No properties available")
                placeholder_item = self.property_combo.model().item(0)
                if placeholder_item:
                    placeholder_item.setEnabled(False)
                    placeholder_item.setFlags(placeholder_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                logger.warning(f"[PROPERTY PANEL] No properties found for layer {layer_name}")

        # Final diagnostic: show what's actually in the property combo
        final_properties = [self.property_combo.itemText(i) for i in range(self.property_combo.count())]
        final_current = self.property_combo.currentText()
        logger.info(f"[PROPERTY PANEL] ===== FINAL STATE: property_combo has {len(final_properties)} items: {final_properties} =====")
        logger.info(f"[PROPERTY PANEL] ===== CURRENT SELECTION: '{final_current}' =====")

        # ================================================================
        # VISIBILITY TOGGLE: Hide other layers of the same type
        # ================================================================
        # When user selects a different block model, hide all other block models
        # This ensures only ONE block model is visible at a time (UX improvement)
        if is_block_model:
            logger.info(f"[PROPERTY PANEL] Toggling visibility for block model layers...")

            # Identify all block model layers in the scene
            for name, info in self.renderer.active_layers.items():
                if self._is_block_type_layer(name, info):
                    # Show the selected layer, hide all others
                    should_show = (name == layer_name)
                    self.renderer.set_layer_visibility(name, should_show)
                    logger.debug(f"[PROPERTY PANEL] Set visibility for '{name}': {should_show}")

            # Force immediate render update to show the visibility changes
            if self.renderer.plotter is not None:
                self.renderer.plotter.render()
                logger.debug(f"[PROPERTY PANEL] Triggered render after visibility toggle")

        if hasattr(self.renderer, 'set_active_layer_for_controls'):
            self.renderer.set_active_layer_for_controls(layer_name)

    def _is_block_type_layer(self, layer_name: str, layer_info: dict = None) -> bool:
        """
        Consistently determine if a layer is a block-model-type layer.

        This is the single source of truth used by the quick toggle, the
        active-layer visibility logic, and the block-model selector so that
        all three agree on what counts as a "block model".

        Parameters
        ----------
        layer_name : str
            Name of the layer in the renderer.
        layer_info : dict, optional
            Layer metadata dict from ``renderer.active_layers``.  When not
            supplied the method falls back to name-based heuristics only.
        """
        lname = layer_name.lower()
        layer_type = ''
        if layer_info:
            layer_type = layer_info.get('layer_type', layer_info.get('type', ''))

        return (
            layer_type in ('blocks', 'classification')
            or (
                ('block' in lname or 'sgsim' in lname or 'kriging' in lname
                 or 'classification' in lname)
                and 'drillhole' not in lname
            )
        )

    def _show_block_model_selector(self, visible: bool):
        """Show or hide the block model selector combo."""
        if not hasattr(self, 'block_model_combo'):
            return

        self.block_model_combo.setVisible(visible)

        # Also show/hide the label
        parent_layout = self.block_model_combo.parent().layout()
        if parent_layout and isinstance(parent_layout, QFormLayout):
            for i in range(parent_layout.rowCount()):
                label_item = parent_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
                if label_item and hasattr(label_item.widget(), 'text') and label_item.widget().text() == "Block Model:":
                    label_item.widget().setVisible(visible)
                    break

    def _populate_block_model_list(self):
        """Populate the block model selector with ALL available models.

        Sources (merged, de-duplicated):
        1. Registry block model list – includes models that have been created
           but may not yet be visualised (e.g. SGSIM stats the user hasn't
           clicked "Visualize" on).
        2. Renderer active_layers – includes every block-type layer currently
           in the 3-D viewer.

        Each item carries ``model_id`` as user-data so that
        ``_on_block_model_selection_changed`` can locate or build the grid.
        """
        if not hasattr(self, 'block_model_combo'):
            return

        self.block_model_combo.blockSignals(True)
        self.block_model_combo.clear()

        seen_ids: set = set()

        # ── 1. Registry models ────────────────────────────────────────
        try:
            if self.registry:
                models = self.registry.get_block_model_list() or []
                for model_info in models:
                    model_id = model_info['model_id']
                    is_current = model_info.get('is_current', False)
                    display = f"{model_id}{' (current)' if is_current else ''}"
                    self.block_model_combo.addItem(display, model_id)
                    seen_ids.add(model_id)
                    if is_current:
                        self.block_model_combo.setCurrentIndex(
                            self.block_model_combo.count() - 1
                        )
        except Exception as e:
            logger.warning(f"[PROPERTY PANEL] Failed to read registry block model list: {e}")

        # ── 2. Renderer layers not already covered by registry ────────
        try:
            if self.renderer and hasattr(self.renderer, 'active_layers'):
                for layer_name, layer_info in self.renderer.active_layers.items():
                    if not self._is_block_type_layer(layer_name, layer_info):
                        continue
                    # Avoid duplicates (registry model ids are often a substring
                    # of the renderer layer name, e.g. "SGSIM: FE_PCT_SGSIM_MEAN")
                    already_listed = any(
                        sid.lower() in layer_name.lower() or layer_name.lower() in sid.lower()
                        for sid in seen_ids
                    )
                    if already_listed or layer_name in seen_ids:
                        continue
                    self.block_model_combo.addItem(f"{layer_name} (layer)", layer_name)
                    seen_ids.add(layer_name)
        except Exception as e:
            logger.warning(f"[PROPERTY PANEL] Failed to read renderer layers: {e}")

        if self.block_model_combo.count() == 0:
            self.block_model_combo.addItem("No models available", None)

        self.block_model_combo.blockSignals(False)

    def _on_block_model_selection_changed(self, index: int):
        """Handle block model selection change.

        When the user picks a different block model from the combo, this
        method:
        1. Updates the registry so the rest of the app knows which model
           is "current".
        2. Finds the corresponding renderer layer and switches the
           ``active_layer_combo`` to it.  That triggers
           ``_on_active_layer_changed`` which hides every other
           block-type layer and shows only the selected one.
        3. **NEW** — If the model has no renderer layer yet (e.g. an
           SGSIM stat that was never "Visualize"d), builds the PyVista
           grid from registry data and emits ``request_visualization``
           so the main window adds it to the 3-D viewer.
        """
        if not hasattr(self, 'block_model_combo'):
            return

        model_id = self.block_model_combo.itemData(index)
        if not model_id:
            return

        logger.info(f"[PROPERTY PANEL] Block model selection changed to: {model_id}")

        # Set as current model in registry
        if self.registry:
            try:
                self.registry.set_current_block_model(model_id)
            except Exception:
                pass

        # ── Try to find an existing renderer layer that matches ──────
        if self.renderer and hasattr(self, 'active_layer_combo'):
            best_match = None
            model_id_lower = model_id.lower()

            for layer_name, layer_info in self.renderer.active_layers.items():
                if not self._is_block_type_layer(layer_name, layer_info):
                    continue
                lname = layer_name.lower()
                if model_id_lower in lname or lname in model_id_lower:
                    best_match = layer_name
                    break
                if ':' in layer_name:
                    suffix = layer_name.split(':', 1)[1].strip().lower()
                    if model_id_lower in suffix or suffix in model_id_lower:
                        best_match = layer_name
                        break

            if best_match:
                combo_idx = self.active_layer_combo.findText(best_match)
                if combo_idx >= 0:
                    self.active_layer_combo.setCurrentIndex(combo_idx)
                    logger.info(f"[PROPERTY PANEL] Switched active layer to: {best_match}")
                    return

        # ── No renderer layer found — build grid from registry ───────
        logger.info(f"[PROPERTY PANEL] Model '{model_id}' not in renderer — attempting to build grid")
        grid = self._build_grid_for_model(model_id)
        if grid is not None:
            layer_label = f"SGSIM: {model_id}"
            self.request_visualization.emit(grid, layer_label)
            logger.info(f"[PROPERTY PANEL] Emitted request_visualization for '{layer_label}'")
        else:
            # Final fallback: just refresh properties for whatever is selected
            current_layer = self.active_layer_combo.currentText()
            if current_layer:
                self._on_active_layer_changed(current_layer)

    # ------------------------------------------------------------------
    def _build_grid_for_model(self, model_id: str):
        """Build a PyVista grid from registry SGSIM data for *model_id*.

        Searches the SGSIM results stored in the registry for a summary
        statistic matching *model_id* (e.g. ``"FE_PCT_SGSIM_MEAN"``).
        If found, constructs a ``pv.ImageData`` grid and returns it.

        Returns ``None`` when the model cannot be materialised (missing
        registry, missing data, or PyVista not available).
        """
        if not self.registry:
            return None
        try:
            import pyvista as pv
            import numpy as np
        except ImportError:
            logger.warning("[PROPERTY PANEL] PyVista/numpy not available for grid building")
            return None

        # ── Get SGSIM results from registry ───────────────────────────
        try:
            results = self.registry.get_sgsim_results()
        except Exception:
            results = None

        if not results or not isinstance(results, dict):
            logger.debug("[PROPERTY PANEL] No SGSIM results in registry")
            return None

        params = results.get('params')
        summary = results.get('summary', {})
        variable = results.get('variable', results.get('property_name', 'VALUE'))

        if not params or not summary:
            logger.debug("[PROPERTY PANEL] SGSIM results missing params or summary")
            return None

        # ── Determine which statistic the user is asking for ──────────
        # model_id might be like "FE_PCT_SGSIM_MEAN" or just "MEAN"
        mid = model_id.upper()
        stat_key = None
        for candidate in ('MEAN', 'STD', 'P10', 'P50', 'P90'):
            if candidate in mid:
                stat_key = candidate.lower()
                break

        if stat_key is None:
            # Try to use the first available stat
            for k in ('mean', 'std', 'p10', 'p50', 'p90'):
                if k in summary:
                    stat_key = k
                    break

        if stat_key is None or stat_key not in summary:
            logger.debug(f"[PROPERTY PANEL] Could not resolve stat from model_id '{model_id}'")
            return None

        stat_data = summary[stat_key]
        if stat_data is None:
            return None

        stat_array = np.asarray(stat_data)

        # ── Build grid ────────────────────────────────────────────────
        try:
            nx, ny, nz = int(params.nx), int(params.ny), int(params.nz)
            dx, dy, dz = float(params.xinc), float(params.yinc), float(params.zinc)
            xmin, ymin, zmin = float(params.xmin), float(params.ymin), float(params.zmin)
        except Exception as e:
            logger.warning(f"[PROPERTY PANEL] Bad SGSIM params: {e}")
            return None

        expected = nx * ny * nz
        flat = stat_array.flatten(order='C')
        if len(flat) != expected:
            logger.warning(
                f"[PROPERTY PANEL] Stat array size mismatch: "
                f"got {len(flat)}, expected {expected} (nx={nx}, ny={ny}, nz={nz})"
            )
            return None

        try:
            grid = pv.ImageData(
                dimensions=(nx + 1, ny + 1, nz + 1),
                spacing=(dx, dy, dz),
                origin=(xmin, ymin, zmin),
            )
            property_name = f"{variable}_SGSIM_{stat_key.upper()}"
            grid.cell_data[property_name] = flat

            # Inherit the local-coordinate flag when origin is small
            if abs(xmin) < 1000 and abs(ymin) < 1000:
                grid._coordinate_shifted = True

            logger.info(
                f"[PROPERTY PANEL] Built grid for '{model_id}': "
                f"{grid.n_cells:,} cells, property={property_name}"
            )
            return grid

        except Exception as e:
            logger.error(f"[PROPERTY PANEL] Grid creation failed: {e}", exc_info=True)
            return None

    # --- Events ---

    def _on_property_changed(self, property_name: str):
        if not property_name or "available" in property_name: 
            self._update_custom_colors_buttons_state()
            return
        
        layer = self.active_layer_combo.currentText()
        if not layer or layer == "No layers active": 
            self._update_custom_colors_buttons_state()
            return
        
        if self.renderer:
            # Show loading cursor during property update (UX-005 fix)
            from PyQt6.QtWidgets import QApplication
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                cmap = self.colormap_combo.currentText()
                mode = self.color_mode_combo.currentText().lower()

                custom = None
                if mode == 'discrete':
                    custom = self._custom_discrete_colors.get((layer, property_name))

                self.renderer.update_layer_property(layer, property_name, cmap, mode, custom_colors=custom)
            except Exception as e:
                logger.error(f"Failed to update layer property '{property_name}' for layer '{layer}': {e}", exc_info=True)
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Visualization Update Failed",
                    f"Failed to update visualization:\n{str(e)}\n\nCheck the log for details."
                )
            finally:
                QApplication.restoreOverrideCursor()
            self._update_custom_colors_buttons_state()

    def _on_colormap_changed(self, colormap: str):
        if not colormap or "---" in colormap: return
        if self._updating_from_legend: return

        # Trigger update via property change logic - this updates the renderer
        self._on_property_changed(self.property_combo.currentText())

        # Emit signal for legend sync ONLY (not for rendering)
        # Use flag to prevent feedback loops and double processing
        self._updating_from_legend = True
        try:
            if self.signals:
                self.signals.colormapChanged.emit(colormap)
            else:
                self.colormap_changed.emit(colormap)
        finally:
            self._updating_from_legend = False

    def _on_color_mode_changed(self, mode: str):
        is_discrete = mode.lower() == 'discrete'
        
        # Show custom colors button for discrete mode, but only if we have a valid layer and property
        layer = self.active_layer_combo.currentText()
        prop = self.property_combo.currentText()
        can_define_colors = (
            is_discrete and 
            layer and 
            layer != "No layers active" and
            prop and 
            prop != "No properties available"
        )
        
        self.custom_colors_btn.setVisible(can_define_colors)
        self.clear_custom_colors_btn.setVisible(can_define_colors)
        
        if is_discrete and self.colormap_combo.currentText() == 'turbo':
            self.colormap_combo.setCurrentText('tab10')
            
        self.legend_style.mode = mode.lower()
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.colorModeChanged.emit(mode.lower())
        else:
            self.color_mode_changed.emit(mode.lower())
        
        # Trigger update
        self._on_property_changed(self.property_combo.currentText())

    def _update_color_mode_visibility(self) -> None:
        """
        Update visibility of color mode related controls based on current layer type.

        For geology layers:
        - Show custom colors button (discrete mode)
        - Hide continuous-only controls

        For other layers:
        - Standard behavior based on color mode selection
        """
        layer = self.active_layer_combo.currentText()
        if not layer or layer == "No layers active":
            return

        # Determine if this is a geology layer
        is_geology = False
        if self.renderer and layer in self.renderer.active_layers:
            layer_info = self.renderer.active_layers[layer]
            layer_type = layer_info.get('layer_type', layer_info.get('type', ''))
            is_geology = (
                "GeoSolid" in layer or
                "GeoSurface" in layer or
                layer_type in ('geology_surface', 'geology_solid', 'geology_volume', 'geology', 'geology_unified')
            )

        mode = self.color_mode_combo.currentText().lower()
        is_discrete = mode == 'discrete'

        # Custom colors button visibility
        if hasattr(self, 'custom_colors_btn'):
            # Always show for geology layers or discrete mode
            should_show = is_geology or is_discrete
            self.custom_colors_btn.setVisible(should_show)

        if hasattr(self, 'clear_custom_colors_btn'):
            # Show reset button only if custom colors exist
            prop = self.property_combo.currentText()
            has_custom = (layer, prop) in self._custom_discrete_colors
            self.clear_custom_colors_btn.setVisible(is_discrete or is_geology)
            self.clear_custom_colors_btn.setEnabled(has_custom)

        logger.debug(f"Updated color mode visibility for layer '{layer}': geology={is_geology}, discrete={is_discrete}")

    def _on_opacity_changed(self, value: int):
        # Emit both local and global-style opacity signals.
        alpha = value / 100.0
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.opacityChanged.emit(alpha)
            self.signals.transparencyChanged.emit(alpha)
        else:
            self.opacity_changed.emit(alpha)
            self.transparency_changed.emit(alpha)

    def _on_filter_changed(self):
        # Value filtering controls removed - method kept for compatibility
        if hasattr(self, 'filter_property_combo') and hasattr(self, 'min_value_spinbox') and hasattr(self, 'max_value_spinbox'):
            prop = self.filter_property_combo.currentText()
            if prop:
                min_val = self.min_value_spinbox.value()
                max_val = self.max_value_spinbox.value()
                # Emit via UISignals bus if available, otherwise fallback to direct signal
                if self.signals:
                    self.signals.filterChanged.emit(prop, min_val, max_val)
                else:
                    self.filter_changed.emit(prop, min_val, max_val)

    def _clear_filter(self):
        # Value filtering controls removed - method kept for compatibility
        if hasattr(self, 'min_value_spinbox') and hasattr(self, 'max_value_spinbox'):
            self.min_value_spinbox.setValue(-1e10)
            self.max_value_spinbox.setValue(1e10)

    # --- Display Settings callbacks ---

    def _on_style_changed(self, style_text: str):
        """Wireframe / Solid render style selector."""
        if not self.renderer:
            return
        style = (style_text or "").lower()
        if style.startswith("wireframe"):
            mode = "wireframe"
        elif "edges" in style:
            mode = "solid_edges"
        else:
            mode = "solid"
        try:
            if hasattr(self.renderer, "set_render_style"):
                self.renderer.set_render_style(mode)
        except Exception:
            logger.debug("PropertyPanel: set_render_style call failed", exc_info=True)

    def _on_shading_changed(self, text: str):
        """Smooth / Flat shading selector."""
        if not self.renderer:
            return
        mode = (text or "").lower()
        smooth = mode.startswith("smooth")
        try:
            if hasattr(self.renderer, "set_shading_mode"):
                self.renderer.set_shading_mode("smooth" if smooth else "flat")
        except Exception:
            logger.debug("PropertyPanel: set_shading_mode call failed", exc_info=True)

    def _on_line_width_changed(self, width: int):
        """Line width spinbox → renderer line width."""
        if not self.renderer:
            return
        try:
            if hasattr(self.renderer, "set_line_width"):
                self.renderer.set_line_width(int(width))
        except Exception:
            logger.debug("PropertyPanel: set_line_width call failed", exc_info=True)

    def _on_point_size_changed(self, size: int):
        """Point size spinbox → renderer point size."""
        if not self.renderer:
            return
        try:
            if hasattr(self.renderer, "set_point_size"):
                self.renderer.set_point_size(int(size))
        except Exception:
            logger.debug("PropertyPanel: set_point_size call failed", exc_info=True)

    # --- Slicing (Sliders 0-100 mapped to bounds) ---
    # Note: Spatial slicing controls removed - methods kept for compatibility
    def _on_x_slice_changed(self, value: int):
        if hasattr(self, 'x_slice_label'):
            self._handle_slice('x', value, 0, 1, self.x_slice_label)

    def _on_y_slice_changed(self, value: int):
        if hasattr(self, 'y_slice_label'):
            self._handle_slice('y', value, 2, 3, self.y_slice_label)

    def _on_z_slice_changed(self, value: int):
        if hasattr(self, 'z_slice_label'):
            self._handle_slice('z', value, 4, 5, self.z_slice_label)

    def _handle_slice(self, axis: str, val: int, b1_idx: int, b2_idx: int, label: QLabel):
        if self.current_model and self.current_model.bounds and label is not None:
            b = self.current_model.bounds
            pos = b[b1_idx] + (val / 100.0) * (b[b2_idx] - b[b1_idx])
            label.setText(f"{pos:.2f}")
            # Emit via UISignals bus if available, otherwise fallback to direct signal
            if self.signals:
                self.signals.sliceChanged.emit(axis, pos)
            else:
                self.slice_changed.emit(axis, pos)

    def _clear_slice(self):
        if hasattr(self, 'x_slice_slider') and hasattr(self, 'y_slice_slider') and hasattr(self, 'z_slice_slider'):
            for s in [self.x_slice_slider, self.y_slice_slider, self.z_slice_slider]:
                s.setValue(0)

    def _emit_block_size(self):
        dx = self.dx_spin.value()
        dy = self.dy_spin.value()
        dz = self.dz_spin.value()
        # Emit via UISignals bus if available, otherwise fallback to direct signal
        if self.signals:
            self.signals.blockSizeChanged.emit(dx, dy, dz)
        else:
            self.block_size_changed.emit(dx, dy, dz)

    def _on_change_font(self):
        font, ok = QFontDialog.getFont()
        if ok:
            # Emit via UISignals bus if available, otherwise fallback to direct signal
            if self.signals:
                self.signals.axisFontChanged.emit(font.family(), font.pointSize())
            else:
                self.axis_font_changed.emit(font.family(), font.pointSize())

    def _on_change_font_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            color_tuple = (c.redF(), c.greenF(), c.blueF())
            # Emit via UISignals bus if available, otherwise fallback to direct signal
            if self.signals:
                self.signals.axisColorChanged.emit(color_tuple)
            else:
                self.axis_color_changed.emit(color_tuple)

    # --- Custom Color Logic ---

    def _on_custom_colors_clicked(self):
        prop = self.property_combo.currentText()
        layer = self.active_layer_combo.currentText()

        # Get layer information to determine layer type
        if not self.renderer:
            return

        layer_info = self.renderer.active_layers.get(layer, {})
        layer_data = layer_info.get('data')
        layer_type = layer_info.get('layer_type', layer_info.get('type', ''))

        # Handle drillholes differently
        if layer == "drillholes" or "drillhole" in layer.lower() or layer_type == 'drillhole':
            # Get data from drillhole layer
            if not self.renderer or "drillholes" not in self.renderer.active_layers:
                QMessageBox.warning(self, "Error", "No drillhole layer active")
                return
            
            layer_data = self.renderer.active_layers["drillholes"].get("data", {})
            
            # Check if it's lithology property
            if prop.lower() in ["lithology", "lith_id", "lith_code"]:
                lith_to_index = layer_data.get("lith_to_index", {})
                
                if not lith_to_index:
                    QMessageBox.warning(self, "Error", "No lithology data available")
                    return
                
                uniques = sorted(lith_to_index.keys())
            else:
                # For assay properties, get unique values from the assay data
                # This allows users to define colors for specific assay values or ranges
                hole_segment_assay = layer_data.get("hole_segment_assay", {})
                
                if not hole_segment_assay:
                    QMessageBox.warning(self, "Error", f"No assay data available for property '{prop}'")
                    return
                
                # Collect all unique assay values for this property
                all_values = []
                for hole_id, assay_list in hole_segment_assay.items():
                    all_values.extend([v for v in assay_list if v is not None and not np.isnan(v)])
                
                if not all_values:
                    QMessageBox.warning(self, "Error", f"No valid assay values found for property '{prop}'")
                    return
                
                # Get unique values, rounded to reasonable precision for UI
                unique_vals = np.unique(np.round(all_values, decimals=2))
                uniques = sorted(unique_vals.tolist())
                
                # Limit to reasonable number for UI
                if len(uniques) > 50:
                    # For continuous data, show a dialog to let user specify value ranges or specific values
                    reply = QMessageBox.question(
                        self, 
                        "Many Values", 
                        f"Found {len(uniques)} unique values. Would you like to:\n\n"
                        "Yes: Define colors for value ranges (e.g., 0-10, 10-20, etc.)\n"
                        "No: Define colors for specific values (first 30)\n"
                        "Cancel: Abort",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                    )
                    
                    if reply == QMessageBox.StandardButton.Cancel:
                        return
                    elif reply == QMessageBox.StandardButton.Yes:
                        # Create value ranges dialog
                        self._create_value_ranges_dialog(prop, layer, uniques)
                        return
                    else:
                        # Use first 30 unique values
                        uniques = uniques[:30]
        elif layer_type == 'classification' and layer_data:
            # Special handling for classification layers
            # Handle both direct mesh and wrapped {'mesh': mesh} formats
            mesh = layer_data.get('mesh') if isinstance(layer_data, dict) else layer_data
            if not hasattr(mesh, 'array_names'):
                QMessageBox.warning(self, "Error", "Classification layer data is not a valid mesh")
                return
            if "Classification_Categories" in mesh.array_names:
                # Use the original categorical values stored in the mesh
                categories = mesh["Classification_Categories"]
                uniques = np.unique(categories)
                uniques = sorted(uniques)
            elif "Category" in mesh.array_names:
                # Use Category field if available (stored as strings)
                categories = mesh["Category"]
                uniques = np.unique(categories)
                # Use standard order: Measured, Indicated, Inferred, Unclassified
                standard_order = ["Measured", "Indicated", "Inferred", "Unclassified"]
                uniques = [c for c in standard_order if c in uniques]
            else:
                # Fallback: map numeric values back to categories
                # CRITICAL: Mapping must match JORC classification: 0=Measured, 1=Indicated, 2=Inferred, 3=Unclassified
                scalars = mesh["Classification"]
                reverse_mapping = {0: "Measured", 1: "Indicated", 2: "Inferred", 3: "Unclassified"}
                categories = np.array([reverse_mapping.get(int(s), "Unknown") for s in scalars])
                uniques = ["Measured", "Indicated", "Inferred", "Unclassified"]
        elif layer_type in ['geology', 'geology_unified'] and layer_data:
            # Special handling for geology layers (GeoUnified, GeoSurface, GeoSolid)
            # Handle both direct mesh and wrapped {'mesh': mesh} formats
            mesh = layer_data.get('mesh') if isinstance(layer_data, dict) else layer_data
            if not hasattr(mesh, 'array_names'):
                QMessageBox.warning(self, "Error", "Geology layer data is not a valid mesh")
                return

            if prop not in mesh.array_names:
                QMessageBox.warning(self, "Error", f"Property '{prop}' not found in geology layer")
                return
            
            # Get unique values from the mesh property
            scalars = mesh[prop]
            
            # Check if this is Formation_ID - use formation names if available
            if prop == "Formation_ID" and "formation_names" in layer_data:
                formation_names = layer_data["formation_names"]
                # Use formation names as categories
                uniques = sorted(formation_names)
            else:
                # For numeric properties, get unique values
                unique_vals = np.unique(scalars)
                # Filter out NaN and zeros
                unique_vals = unique_vals[~np.isnan(unique_vals)]
                if len(unique_vals) > 1:
                    unique_vals = unique_vals[unique_vals != 0]
                uniques = sorted(unique_vals.tolist())
        else:
            # Block model handling
            if not self.current_model:
                return
            raw = self.current_model.get_property(prop)
            if raw is None:
                return
            uniques = np.unique(raw[~np.isnan(raw)])
            uniques = sorted(uniques[uniques != 0])

        if len(uniques) > 50:
            if QMessageBox.question(self, "Warning", "More than 50 categories. Continue?") != QMessageBox.StandardButton.Yes:
                return

        # Dialog Setup
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Colors: {prop}")
        dlg.setLayout(QVBoxLayout())
        
        scroll = QScrollArea()
        container = QWidget()
        form = QFormLayout(container)
        
        inputs = {}
        for val in uniques[:30]: # Limit for UI
            row = QHBoxLayout()
            le = QLineEdit()
            # Pre-fill with existing custom color if available
            if (layer, prop) in self._custom_discrete_colors:
                existing = self._custom_discrete_colors[(layer, prop)].get(val)
                if existing:
                    le.setText(existing)
                    le.setStyleSheet(f"background-color: {existing}; color: {'white' if QColor(existing).lightness() < 128 else 'black'}")
            
            btn = QPushButton("...")
            btn.setFixedWidth(30)
            btn.clicked.connect(lambda _, x=le: self._pick_color_helper(x))
            row.addWidget(le)
            row.addWidget(btn)
            form.addRow(f"{val}:", row)
            inputs[val] = le
            
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        dlg.layout().addWidget(scroll)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        dlg.layout().addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            custom_map = {k: v.text() for k, v in inputs.items() if v.text()}
            if custom_map:
                self._custom_discrete_colors[(layer, prop)] = custom_map
                # For assay properties with custom colors, switch to discrete mode
                # This allows the renderer to use the custom color mapping
                self.color_mode_combo.setCurrentText('Discrete')
                self._on_property_changed(prop)

    def _create_value_ranges_dialog(self, prop: str, layer: str, uniques: list) -> None:
        """
        Create a dialog for defining color ranges for continuous data with many unique values.

        Args:
            prop: Property name
            layer: Layer name
            uniques: List of unique values in the property
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QHBoxLayout,
            QPushButton, QDoubleSpinBox, QScrollArea, QWidget,
            QDialogButtonBox, QLabel, QLineEdit
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Value Range Colors: {prop}")
        dlg.setMinimumWidth(450)
        dlg.setLayout(QVBoxLayout())

        # Calculate suggested ranges based on data distribution
        arr = np.array(uniques)
        min_val, max_val = float(arr.min()), float(arr.max())
        num_ranges = 5
        range_size = (max_val - min_val) / num_ranges

        # Instructions
        hint = QLabel("Define color ranges for value bins. Each range maps values to a color.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-style: italic; margin-bottom: 10px;")
        dlg.layout().addWidget(hint)

        # Scroll area for ranges
        scroll = QScrollArea()
        container = QWidget()
        form = QFormLayout(container)

        ranges = []
        default_colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']

        for i in range(num_ranges):
            row = QHBoxLayout()

            # Min value
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1e12, 1e12)
            min_spin.setDecimals(4)
            min_spin.setValue(min_val + i * range_size)
            min_spin.setFixedWidth(100)

            # Max value
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1e12, 1e12)
            max_spin.setDecimals(4)
            max_spin.setValue(min_val + (i + 1) * range_size)
            max_spin.setFixedWidth(100)

            # Color
            color_hex = default_colors[i % len(default_colors)]
            color_edit = QLineEdit()
            color_edit.setText(color_hex)
            color_edit.setFixedWidth(80)
            color_edit.setStyleSheet(f"background-color: {color_hex}; color: {'white' if QColor(color_hex).lightness() < 128 else 'black'}")

            color_btn = QPushButton("...")
            color_btn.setFixedWidth(30)
            color_btn.clicked.connect(lambda _, le=color_edit: self._pick_color_helper(le))

            row.addWidget(QLabel("From:"))
            row.addWidget(min_spin)
            row.addWidget(QLabel("To:"))
            row.addWidget(max_spin)
            row.addWidget(color_edit)
            row.addWidget(color_btn)
            row.addStretch()

            ranges.append((min_spin, max_spin, color_edit))
            form.addRow(f"Range {i+1}:", row)

        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        dlg.layout().addWidget(scroll)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        dlg.layout().addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Build custom color map based on ranges
            custom_map = {}
            for i, (min_spin, max_spin, color_edit) in enumerate(ranges):
                if color_edit.text():
                    # Store range as a string key for JSON compatibility
                    # Format: "min_value-max_value"
                    range_key = f"{min_spin.value():.4f}-{max_spin.value():.4f}"
                    custom_map[range_key] = color_edit.text()

            if custom_map:
                self._custom_discrete_colors[(layer, prop)] = custom_map
                self.color_mode_combo.setCurrentText('Discrete')
                self._on_property_changed(prop)
                logger.info(f"Applied custom color ranges for {layer}/{prop}: {len(custom_map)} ranges")

    def _pick_color_helper(self, line_edit):
        c = QColorDialog.getColor()
        if c.isValid():
            line_edit.setText(c.name())
            line_edit.setStyleSheet(f"background-color: {c.name()}; color: {'white' if c.lightness() < 128 else 'black'}")

    def _on_clear_custom_colors_clicked(self):
        layer = self.active_layer_combo.currentText()
        prop = self.property_combo.currentText()
        if (layer, prop) in self._custom_discrete_colors:
            del self._custom_discrete_colors[(layer, prop)]
            self._on_property_changed(prop)

    def _update_custom_colors_buttons_state(self):
        if not hasattr(self, 'clear_custom_colors_btn'): return
        layer = self.active_layer_combo.currentText()
        prop = self.property_combo.currentText()
        mode = self.color_mode_combo.currentText().lower()
        
        # Check if we can show/use custom colors
        can_use_custom = bool(
            mode == 'discrete' and 
            bool(layer) and 
            layer != "No layers active" and
            bool(prop) and 
            prop != "No properties available"
        )
        
        has_custom = (layer, prop) in self._custom_discrete_colors if can_use_custom else False
        
        # Update button visibility and state
        if hasattr(self, 'custom_colors_btn'):
            self.custom_colors_btn.setVisible(can_use_custom)
        self.clear_custom_colors_btn.setVisible(can_use_custom)
        self.clear_custom_colors_btn.setEnabled(can_use_custom and has_custom)

    @property
    def current_property(self) -> Optional[str]:
        return self.property_combo.currentText() if hasattr(self, 'property_combo') else None
    
    # --- External Setters (Safe) ---
    def set_colormap_from_external(self, colormap: str):
        if not colormap or "---" in colormap: return
        self._updating_from_legend = True
        with self._block_signal(self.colormap_combo):
            idx = self.colormap_combo.findText(colormap)
            if idx >= 0: self.colormap_combo.setCurrentIndex(idx)
        self._updating_from_legend = False

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors
        self.setStyleSheet(get_analysis_panel_stylesheet())
