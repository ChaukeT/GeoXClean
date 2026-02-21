"""
Block Info Panel for displaying selected block information.
"""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel,
    QGroupBox, QHBoxLayout, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea
from .modern_styles import get_theme_colors

try:
    from .base_analysis_panel import BaseAnalysisPanel
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

logger = logging.getLogger(__name__)


class BlockInfoPanel(BaseDockPanel if not BASE_AVAILABLE else BaseAnalysisPanel):
    """
    Panel for displaying information about selected blocks.
    
    Shows block ID, coordinates, and all property values when a block is clicked.
    """
    # PanelManager metadata
    PANEL_ID = "BlockInfoPanel"
    PANEL_NAME = "BlockInfo Panel"
    PANEL_CATEGORY = PanelCategory.INFO
    PANEL_DEFAULT_VISIBLE = True
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT


    
    # Signals
    copy_requested = pyqtSignal()
    clear_selection_requested = pyqtSignal()
    
    def __init__(self, parent=None, panel_id=None):
        # Initialize data attributes before calling super().__init__
        self.current_block_id = None
        self.current_block_data = None
        self.available_properties = []

        super().__init__(parent=parent, panel_id=panel_id)

        # Throttling: Debounce timer for rapid updates (100ms delay)
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._apply_pending_update)
        self._pending_update = None  # Stores (block_id, block_data, coordinates) tuple

        # Connect to DataRegistry if available
        if BASE_AVAILABLE:
            self._connect_registry()

        logger.info("Initialized block info panel")

    def _connect_registry(self):
        """Connect to DataRegistry for automatic block model updates."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                # Connect to block model signals
                self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
                self.registry.blockModelGenerated.connect(self._on_block_model_generated)
                self.registry.blockModelClassified.connect(self._on_block_model_classified)

                # Multi-model support - refresh when current model changes
                if hasattr(self.registry, 'currentBlockModelChanged'):
                    self.registry.currentBlockModelChanged.connect(self._refresh_available_block_models)

                # Load any existing block models from all sources
                self._refresh_available_block_models()

                logger.info("Block info panel connected to DataRegistry")
            else:
                logger.info("DataRegistry not available, block info panel running standalone")
                self.registry = None
        except Exception as e:
            logger.warning(f"Failed to connect block info panel to DataRegistry: {e}")
            self.registry = None
    
    def _refresh_available_block_models(self):
        """Refresh the list of available block models from all sources."""
        if not self.registry:
            return
        
        try:
            # Check regular block model
            block_model = self.registry.get_block_model()
            if block_model is not None:
                self._on_block_model_loaded(block_model)
            
            # Check classified block model
            try:
                classified_model = self.registry.get_classified_block_model()
                if classified_model is not None:
                    self._on_block_model_classified(classified_model)
            except Exception as e:
                logger.error(f"Failed to load classified block model: {e}", exc_info=True)
            
            # Also check renderer layers for block models that might be displayed
            self._check_renderer_layers_for_block_models()
            
        except Exception as e:
            logger.warning(f"Failed to refresh available block models: {e}")
    
    def _check_renderer_layers_for_block_models(self):
        """Check renderer layers for block models that might be displayed."""
        try:
            # Try to get renderer from parent/main window
            parent = self.parent()
            while parent:
                if hasattr(parent, 'viewer_widget') and hasattr(parent.viewer_widget, 'renderer'):
                    renderer = parent.viewer_widget.renderer
                    if hasattr(renderer, 'active_layers') and renderer.active_layers:
                        # Look for block model layers (SGSIM, kriging, etc.)
                        for layer_name, layer_info in renderer.active_layers.items():
                            layer_type = layer_info.get('type', '')
                            layer_data = layer_info.get('data', None)
                            
                            # Check if it's a block model layer
                            if layer_type in ('blocks', 'volume') and layer_data is not None:
                                # If we don't have a block model yet, try to use this one
                                if self.block_model is None:
                                    if hasattr(parent, '_extract_block_model_from_grid'):
                                        block_model = parent._extract_block_model_from_grid(layer_data, layer_name)
                                        if block_model is not None:
                                            self._on_block_model_generated(block_model)
                                            logger.info(f"Found block model in renderer layer: {layer_name}")
                    break
                parent = parent.parent() if parent else None
        except Exception as e:
            logger.debug(f"Could not check renderer layers: {e}")

    def _on_block_model_loaded(self, block_model):
        """Handle new block model loaded."""
        self._update_block_model(block_model)
        logger.info("Block info panel updated with new block model")

    def _on_block_model_generated(self, block_model):
        """Handle new block model generated."""
        self._update_block_model(block_model)
        logger.info("Block info panel updated with generated block model")

    def _on_block_model_classified(self, block_model):
        """Handle block model classification changes."""
        self._update_block_model(block_model)
        logger.info("Block info panel updated with classified block model")

    def _update_block_model(self, block_model):
        """Update internal block model reference and available properties."""
        # Use set_block_model() method instead of direct assignment
        # since block_model is a property with only a getter
        self.set_block_model(block_model)
        if block_model is not None:
            # Extract available properties
            if hasattr(block_model, 'columns'):
                # DataFrame-like block model
                self.available_properties = [col for col in block_model.columns
                                          if col not in ['X', 'Y', 'Z', 'DX', 'DY', 'DZ']]
            elif hasattr(block_model, 'properties'):
                # BlockModel API
                self.available_properties = list(block_model.properties.keys())
            else:
                self.available_properties = []

            logger.info(f"Block info panel: Block model updated with {len(self.available_properties)} properties")
        else:
            self.available_properties = []
            logger.info("Block info panel: Block model cleared")

    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("<b>Block Information</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Info text area
        info_group = QGroupBox("Selected Block")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Click on a block to see its information...")
        
        # Set monospace font for better alignment
        font = QFont("Courier New", 9)
        self.info_text.setFont(font)
        
        # Set minimum height
        self.info_text.setMinimumHeight(200)
        
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.copy_btn = QPushButton("Copy Info")
        self.copy_btn.setToolTip("Copy block information to clipboard")
        self.copy_btn.clicked.connect(self._on_copy_clicked)
        self.copy_btn.setEnabled(False)
        button_layout.addWidget(self.copy_btn)
        
        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.setToolTip("Clear the selected block highlight")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        # Statistics group
        stats_group = QGroupBox("Quick Stats")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No block selected")
        self.stats_label.setWordWrap(True)
        colors = get_theme_colors()
        self.stats_label.setStyleSheet(f"font-size: 9px; color: {colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Set max width for better layout
        self.setMaximumWidth(350)
        self.setMinimumWidth(280)
    
    def update_block_info(self, block_id: int, block_data: dict, coordinates: tuple = None):
        """
        Update the panel with information about a selected block.
        
        Uses throttling/debouncing to prevent UI blocking when mouse drags
        over many blocks quickly. Updates are delayed by 100ms and only the
        latest update is applied.
        
        Args:
            block_id: The ID/index of the selected block
            block_data: Dictionary of property values for the block
            coordinates: Optional tuple of (x, y, z) coordinates
        """
        # Store pending update data
        self._pending_update = (block_id, block_data, coordinates)
        
        # Cancel any pending timer and restart with 100ms delay
        self._update_timer.stop()
        self._update_timer.start(100)  # 100ms debounce delay
    
    def _apply_pending_update(self):
        """
        Apply the pending update to the UI.
        Called by the debounce timer after 100ms pause.
        """
        if self._pending_update is None:
            return
        
        block_id, block_data, coordinates = self._pending_update
        self._pending_update = None
        
        try:
            self.current_block_id = block_id
            self.current_block_data = block_data
            
            # Build info text
            info_lines = []
            info_lines.append("=" * 40)
            info_lines.append(f"BLOCK ID: {block_id}")
            info_lines.append("=" * 40)
            
            # Coordinates section
            if coordinates:
                try:
                    x, y, z = coordinates
                    info_lines.append("\nCOORDINATES:")
                    info_lines.append(f"  X (East):  {x:>12.2f} m")
                    info_lines.append(f"  Y (North): {y:>12.2f} m")
                    info_lines.append(f"  Z (Elev):  {z:>12.2f} m")
                except Exception as e:
                    logger.warning(f"Error formatting coordinates: {e}")
                    info_lines.append(f"\nCOORDINATES: {coordinates}")
            
            # Properties section
            if block_data:
                info_lines.append("\nPROPERTIES:")
                
                try:
                    # Sort properties for consistent display
                    sorted_props = sorted(block_data.items())
                    
                    for key, value in sorted_props:
                        # Skip coordinate columns if they're in the data
                        if key.upper() in ['X', 'Y', 'Z', 'XC', 'YC', 'ZC', 
                                            'XMORIG', 'YMORIG', 'ZMORIG']:
                            continue
                        
                        # Format based on data type
                        try:
                            if isinstance(value, (int, float)):
                                if isinstance(value, float):
                                    formatted_value = f"{value:>12.4f}"
                                else:
                                    formatted_value = f"{value:>12}"
                            else:
                                formatted_value = str(value)
                            
                            info_lines.append(f"  {key:<15} {formatted_value}")
                        except Exception as e:
                            logger.warning(f"Error formatting property {key}: {e}")
                            info_lines.append(f"  {key:<15} {str(value)}")
                            
                except Exception as e:
                    logger.error(f"Error processing block properties: {e}", exc_info=True)
                    info_lines.append("  Error displaying properties")
            
            info_lines.append("\n" + "=" * 40)
            
            # Update text
            info_text = "\n".join(info_lines)
            self.info_text.setPlainText(info_text)
            
            # Update stats
            self._update_stats(block_data)
            
            # Enable buttons
            self.copy_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            
            logger.debug(f"Applied throttled block info update for block ID: {block_id}")
            
        except Exception as e:
            logger.error(f"Error updating block info: {e}", exc_info=True)
            self.info_text.setPlainText(f"Error displaying block information:\n{str(e)}")
            self.copy_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
    
    def _update_stats(self, block_data: dict):
        """Update the quick stats section."""
        if not block_data:
            self.stats_label.setText("No properties available")
            return
        
        # Count numeric vs categorical properties
        numeric_props = []
        categorical_props = []
        
        for key, value in block_data.items():
            if key.upper() not in ['X', 'Y', 'Z', 'XC', 'YC', 'ZC', 
                                    'XMORIG', 'YMORIG', 'ZMORIG']:
                if isinstance(value, (int, float)):
                    numeric_props.append(key)
                else:
                    categorical_props.append(key)
        
        stats_text = f"Properties: {len(numeric_props)} numeric, {len(categorical_props)} categorical"
        self.stats_label.setText(stats_text)
    
    def clear_info(self):
        """Clear the block information display."""
        # Cancel any pending updates
        self._update_timer.stop()
        self._pending_update = None
        
        self.current_block_id = None
        self.current_block_data = None
        
        self.info_text.setPlainText("Click on a block to see its information...")
        self.stats_label.setText("No block selected")
        
        self.copy_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        logger.info("Cleared block info")

    def on_pick_event(self, pick_data: dict):
        """
        Handle global pick events (blocks, drillholes, geology).

        Args:
            pick_data: Dictionary with 'layer', 'block_id', 'properties', 'event_type', 'lod'
        """
        layer = pick_data.get('layer', 'Unknown')
        properties = pick_data.get('properties', {})
        event_type = pick_data.get('event_type', 'click')

        if event_type != 'click':
            return  # Only handle clicks, not hovers

        try:
            # Build display text based on layer type
            if layer == 'Block Model':
                block_id = pick_data.get('block_id', -1)
                self.update_block_info(block_id, properties)

            elif layer.startswith('Drillhole'):
                hole_id = pick_data.get('hole_id', 'Unknown')
                depth = properties.get('depth', 'N/A')
                lithology = properties.get('lithology', 'N/A')
                assay_value = properties.get('assay_value', 'N/A')

                # Build info lines
                info_lines = []
                info_lines.append("=" * 40)
                info_lines.append(f"DRILLHOLE: {hole_id}")
                info_lines.append("=" * 40)

                if 'collar_x' in properties:
                    info_lines.append("\nCOLLAR COORDINATES:")
                    info_lines.append(f"  X (East):  {properties['collar_x']:>12.2f} m")
                    info_lines.append(f"  Y (North): {properties['collar_y']:>12.2f} m")
                    info_lines.append(f"  Z (Elev):  {properties['collar_z']:>12.2f} m")

                info_lines.append("\nPROPERTIES:")
                if depth != 'N/A':
                    info_lines.append(f"  Depth:     {depth} m")
                if lithology != 'N/A':
                    info_lines.append(f"  Lithology: {lithology}")
                if assay_value != 'N/A':
                    info_lines.append(f"  Assay:     {assay_value}")

                # Add any other properties
                for key, value in properties.items():
                    if key not in ['hole_id', 'layer', 'collar_x', 'collar_y', 'collar_z', 'depth', 'lithology', 'assay_value']:
                        info_lines.append(f"  {key}: {value}")

                info_lines.append("\n" + "=" * 40)

                # Update display
                self.info_text.setPlainText("\n".join(info_lines))
                self.stats_label.setText(f"Drillhole: {hole_id}")
                self.copy_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)

            elif layer.startswith('Geology'):
                domain = pick_data.get('domain', 'Unknown')

                # Build info lines
                info_lines = []
                info_lines.append("=" * 40)
                info_lines.append(f"GEOLOGY SURFACE")
                info_lines.append("=" * 40)
                info_lines.append(f"\nDomain: {domain}")

                if properties:
                    info_lines.append("\nPROPERTIES:")
                    for key, value in properties.items():
                        if key != 'domain':
                            info_lines.append(f"  {key}: {value}")

                info_lines.append("\n" + "=" * 40)

                # Update display
                self.info_text.setPlainText("\n".join(info_lines))
                self.stats_label.setText(f"Geology: {domain}")
                self.copy_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)

            else:
                # Unknown layer type
                logger.warning(f"Unknown layer type in pick event: {layer}")

        except Exception as e:
            logger.error(f"Error handling pick event: {e}", exc_info=True)

    def _on_copy_clicked(self):
        """Handle copy button click."""
        if self.info_text.toPlainText():
            clipboard = QApplication.clipboard()
            clipboard.setText(self.info_text.toPlainText())
            logger.info("Copied block info to clipboard")
        
        self.copy_requested.emit()
    
    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.clear_info()
        self.clear_selection_requested.emit()
        logger.info("Clear selection requested")

    def refresh_theme(self):
        """Refresh styles when theme changes."""
        # Call parent refresh_theme if available
        if hasattr(super(), 'refresh_theme'):
            super().refresh_theme()

        # Update stats label color
        colors = get_theme_colors()
        if hasattr(self, 'stats_label'):
            self.stats_label.setStyleSheet(f"font-size: 9px; color: {colors.TEXT_SECONDARY};")

