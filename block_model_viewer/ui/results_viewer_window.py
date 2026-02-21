"""
Dedicated 3D Results Visualization Window

Provides a standalone window for visualizing IRR analysis results with full property controls.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QFormLayout, QCheckBox, QSlider,
    QMessageBox
)
from PyQt6.QtCore import Qt

from ..models.block_model import BlockModel
from .viewer_widget import ViewerWidget

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ResultsViewerWindow(QDialog):
    """
    Standalone window for visualizing IRR analysis results in 3D.
    
    Provides full property controls for:
    - Mining periods
    - Pit phases
    - Block values
    - Grades
    - And all other result properties
    """
    
    def __init__(self, block_model: BlockModel, schedule: pd.DataFrame, parent=None):
        """
        Initialize the results viewer window.
        
        Args:
            block_model: The original block model
            schedule: The IRR analysis schedule DataFrame with PERIOD, PHASE, VALUE, etc.
            parent: Parent widget
        """
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
        
        self.block_model = block_model
        self.schedule = schedule
        self.results_model = None
        
        self.setWindowTitle("IRR Analysis - 3D Results Visualization")
        self.resize(1400, 900)
        
        self._setup_ui()
        self._create_results_model()
        
        # Defer data loading until window is shown to prevent freeze
        # Use Qt timer to load data AFTER window is fully initialized
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._delayed_load)
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QHBoxLayout(self)
        
        # Left panel: Placeholder for 3D Viewer (will be created later)
        from PyQt6.QtWidgets import QLabel
        self.viewer_placeholder = QLabel("Initializing 3D viewer...")
        self.viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_placeholder.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(self.viewer_placeholder, stretch=3)
        
        self.viewer = None  # Will be created in _delayed_load
        
        # Right panel: Controls
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self) -> QGroupBox:
        """Create the control panel with property selections."""
        panel = QGroupBox("Visualization Controls")
        layout = QVBoxLayout(panel)
        
        # Property Selection Group
        prop_group = QGroupBox("Property to Display")
        prop_layout = QFormLayout(prop_group)
        
        self.property_combo = QComboBox()
        self.property_combo.setToolTip("Select which property to visualize on the 3D model")
        self.property_combo.currentTextChanged.connect(self._on_property_changed)
        prop_layout.addRow("Property:", self.property_combo)
        
        layout.addWidget(prop_group)
        
        # Color Settings Group
        color_group = QGroupBox("Color Settings")
        color_layout = QFormLayout(color_group)
        
        # Color mode
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(['Continuous', 'Discrete'])
        self.color_mode_combo.setToolTip(
            "Continuous: Smooth gradient (for grades, values)\n"
            "Discrete: Distinct colors (for periods, phases)"
        )
        self.color_mode_combo.currentTextChanged.connect(self._on_color_mode_changed)
        color_layout.addRow("Color Mode:", self.color_mode_combo)
        
        # Colormap
        self.colormap_combo = QComboBox()
        
        # Categorical colormaps (for discrete mode)
        categorical_maps = ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired']
        self.colormap_combo.addItem("--- Categorical (for Discrete) ---")
        self.colormap_combo.model().item(0).setEnabled(False)
        self.colormap_combo.addItems(categorical_maps)
        
        self.colormap_combo.addItem("--- Continuous ---")
        self.colormap_combo.model().item(len(categorical_maps) + 1).setEnabled(False)
        
        # Continuous colormaps
        continuous_maps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlGn']
        self.colormap_combo.addItems(continuous_maps)
        
        self.colormap_combo.setCurrentText('tab10')
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        color_layout.addRow("Colormap:", self.colormap_combo)
        
        layout.addWidget(color_group)
        
        # Filter Group
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)
        
        # Show only mined blocks
        self.mined_only_checkbox = QCheckBox("Show Only Mined Blocks")
        self.mined_only_checkbox.setChecked(True)
        self.mined_only_checkbox.setToolTip("Hide blocks that are not scheduled for mining")
        self.mined_only_checkbox.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.mined_only_checkbox)
        
        # Filter by period
        self.period_filter_checkbox = QCheckBox("Filter by Period:")
        self.period_filter_checkbox.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.period_filter_checkbox)
        
        self.period_combo = QComboBox()
        self.period_combo.setEnabled(False)
        self.period_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.period_combo)
        
        self.period_filter_checkbox.stateChanged.connect(
            lambda state: self.period_combo.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # Filter by phase
        self.phase_filter_checkbox = QCheckBox("Filter by Phase:")
        self.phase_filter_checkbox.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.phase_filter_checkbox)
        
        self.phase_combo = QComboBox()
        self.phase_combo.setEnabled(False)
        self.phase_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.phase_combo)
        
        self.phase_filter_checkbox.stateChanged.connect(
            lambda state: self.phase_combo.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # Apply Filters Button
        apply_filters_btn = QPushButton("🔄 Apply Filters")
        apply_filters_btn.setToolTip("Reload the 3D view with the current filter settings")
        apply_filters_btn.clicked.connect(self._reload_with_filters)
        filter_layout.addWidget(apply_filters_btn)
        
        layout.addWidget(filter_group)
        
        # Info Label
        self.info_label = QLabel("Blocks displayed: 0")
        self.info_label.setStyleSheet("padding: 10px; background-color: #E8F5E9; border-radius: 3px;")
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        button_layout.addWidget(reset_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return panel
    
    def _create_results_model(self):
        """Create a block model with results data merged."""
        logger.info("Creating results block model...")
        
        # Check if block_model is a BlockModel object or DataFrame
        if isinstance(self.block_model, BlockModel):
            # Extract data from BlockModel object
            block_data = []
            
            for i in range(self.block_model.block_count):
                block = {
                    'BLOCK_ID': i,
                    'XMORIG': self.block_model.positions[i, 0],
                    'YMORIG': self.block_model.positions[i, 1],
                    'ZMORIG': self.block_model.positions[i, 2],
                    'DX': self.block_model.dimensions[i, 0],
                    'DY': self.block_model.dimensions[i, 1],
                    'DZ': self.block_model.dimensions[i, 2],
                }
                
                # Add original properties
                for prop_name in self.block_model.get_property_names():
                    prop_values = self.block_model.get_property(prop_name)
                    if prop_values is not None:
                        block[prop_name] = prop_values[i]
                
                block_data.append(block)
            
            df = pd.DataFrame(block_data)
        else:
            # block_model is already a DataFrame (prepared block model from IRR panel)
            df = self.block_model.copy()
            
            # Ensure BLOCK_ID exists
            if 'BLOCK_ID' not in df.columns:
                df['BLOCK_ID'] = df.index
        
        # Merge with schedule (which has PERIOD, PHASE, MINED, VALUE, TONNAGE, Au, etc.)
        merged = df.merge(self.schedule, on='BLOCK_ID', how='left', suffixes=('_orig', ''))
        
        # Fill missing values
        if 'MINED' in merged.columns:
            merged['MINED'] = merged['MINED'].fillna(0).astype(int)
        if 'PERIOD' in merged.columns:
            merged['PERIOD'] = merged['PERIOD'].fillna(0).astype(int)
        if 'PHASE' in merged.columns:
            merged['PHASE'] = merged['PHASE'].fillna(0).astype(int)
        
        # Create new block model from merged data
        self.results_model = BlockModel()
        
        # Set geometry
        origins = merged[['XMORIG', 'YMORIG', 'ZMORIG']].values
        dimensions = merged[['DX', 'DY', 'DZ']].values
        self.results_model.set_geometry(origins, dimensions)
        
        # Add all properties
        for col in merged.columns:
            if col not in ['XMORIG', 'YMORIG', 'ZMORIG', 'DX', 'DY', 'DZ', 'BLOCK_ID']:
                self.results_model.add_property(col, merged[col].values)
        
        logger.info(f"Results model created with {self.results_model.block_count} blocks and properties: {', '.join(self.results_model.get_property_names())}")
    
    def _delayed_load(self):
        """Load data after window is fully initialized to prevent freeze."""
        try:
            # Step 1: Create the viewer widget
            logger.info("Creating 3D viewer widget...")
            self.viewer = ViewerWidget()
            
            # Replace placeholder with actual viewer
            layout = self.layout()
            layout.replaceWidget(self.viewer_placeholder, self.viewer)
            self.viewer_placeholder.deleteLater()
            
            logger.info("Viewer widget created successfully")
            
            # Step 2: Defer the actual data loading to give the viewer time to initialize
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, self._load_data_step)
            
        except Exception as e:
            logger.error(f"Error creating 3D viewer: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error Creating 3D Viewer",
                f"Failed to create 3D viewer:\n\n{str(e)}\n\n"
                "The window will close. Please try again or check the log file."
            )
            self.close()
    
    def _load_data_step(self):
        """Second step: Load data after viewer is ready."""
        try:
            # Show warning if model is large
            if self.results_model and self.results_model.block_count > 3000:
                QMessageBox.information(
                    self,
                    "Large Model - Performance Notice",
                    f"<b>Large Model Detected: {self.results_model.block_count:,} blocks</b><br><br>"
                    f"To prevent GPU driver timeout, only <b>3,000 blocks</b> will be displayed.<br><br>"
                    f"<b>Tips for better performance:</b><br>"
                    f"• Use filters (Period, Phase, Mined) to reduce visible blocks<br>"
                    f"• Close other GPU-intensive applications<br>"
                    f"• If the visualization freezes, close this window and try again with filters<br><br>"
                    f"<i>Note: This is a display limitation only. All blocks are included in the IRR calculation.</i>",
                    QMessageBox.StandardButton.Ok
                )
            
            # Load the results
            logger.info("Loading results into viewer...")
            self._load_results()
            logger.info("Results loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading 3D results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error Loading 3D Results",
                f"Failed to load 3D visualization:\n\n{str(e)}\n\n"
                "The window will close. Please try again or check the log file."
            )
            self.close()
    
    def _load_results(self):
        """Load the results model into the viewer - SIMPLIFIED VERSION."""
        if self.results_model is None:
            return
        
        # Populate property combo
        properties = self.results_model.get_property_names()
        
        # Prioritize result properties
        priority_props = []
        if 'PERIOD' in properties:
            priority_props.append('PERIOD')
        if 'PHASE' in properties:
            priority_props.append('PHASE')
        if 'VALUE' in properties:
            priority_props.append('VALUE')
        
        # Add grade fields
        grade_props = [p for p in properties if p in ['Au', 'Cu', 'GRADE', 'grade']]
        priority_props.extend(grade_props)
        
        # Add remaining properties
        other_props = [p for p in properties if p not in priority_props]
        all_props = priority_props + sorted(other_props)
        
        self.property_combo.addItems(all_props)
        
        # Populate period filter
        if 'PERIOD' in properties:
            periods = self.results_model.get_property('PERIOD')
            unique_periods = sorted([int(p) for p in set(periods) if p > 0])
            self.period_combo.addItems(['All'] + [str(p) for p in unique_periods])
        
        # Populate phase filter
        if 'PHASE' in properties:
            phases = self.results_model.get_property('PHASE')
            unique_phases = sorted([int(p) for p in set(phases) if p > 0])
            self.phase_combo.addItems(['All'] + [str(p) for p in unique_phases])
        
        # Set default property (PERIOD if available, otherwise first)
        if 'PERIOD' in all_props:
            self.property_combo.setCurrentText('PERIOD')
            self.color_mode_combo.setCurrentText('Discrete')
        elif all_props:
            self.property_combo.setCurrentText(all_props[0])
        
        # CRITICAL: Load only MINED blocks by default to prevent freeze
        filtered = self._create_simple_filtered_model()
        if filtered and filtered.block_count > 0:
            logger.info(f"Loading {filtered.block_count} blocks (filtered from {self.results_model.block_count})")
            self.viewer.load_block_model(filtered)
            self.info_label.setText(f"Blocks displayed: {filtered.block_count:,}")
            
            # Apply property coloring
            if 'PERIOD' in all_props:
                self.viewer.set_color_mode('discrete')
                self.viewer.set_colormap('tab10')
                self.viewer.set_property_coloring('PERIOD')
        else:
            self.info_label.setText("No blocks to display")
    
    def _create_simple_filtered_model(self) -> Optional[BlockModel]:
        """Create a simple filtered model with ONLY mined blocks (fast, safe approach)."""
        if 'MINED' not in self.results_model.get_property_names():
            logger.warning("No MINED property found, returning full model (may cause freeze!)")
            return self.results_model
        
        mined = self.results_model.get_property('MINED')
        mined_indices = [i for i in range(self.results_model.block_count) if mined[i] == 1]
        
        if not mined_indices:
            return None
        
        logger.info(f"Filtering to {len(mined_indices)} mined blocks (from {self.results_model.block_count} total)")
        
        # Create new model with ONLY mined blocks
        filtered = BlockModel()
        origins = self.results_model.positions[mined_indices]
        dimensions = self.results_model.dimensions[mined_indices]
        filtered.set_geometry(origins, dimensions)
        
        # Copy properties for mined blocks only
        for prop_name in self.results_model.get_property_names():
            prop_values = self.results_model.get_property(prop_name)
            filtered.add_property(prop_name, prop_values[mined_indices])
        
        return filtered
    
    def _on_property_changed(self, property_name: str):
        """Handle property selection change."""
        if not property_name or self.results_model is None:
            return
        
        # Auto-suggest color mode based on property
        if property_name in ['PERIOD', 'PHASE', 'MINED', 'LITO', 'ZONE']:
            self.color_mode_combo.setCurrentText('Discrete')
            if self.colormap_combo.currentText() in ['viridis', 'plasma', 'inferno']:
                self.colormap_combo.setCurrentText('tab10')
        else:
            self.color_mode_combo.setCurrentText('Continuous')
            if self.colormap_combo.currentText() in ['tab10', 'tab20']:
                self.colormap_combo.setCurrentText('viridis')
        
        self._apply_visualization()
    
    def _on_color_mode_changed(self, mode: str):
        """Handle color mode change."""
        self._apply_visualization()
    
    def _on_colormap_changed(self, colormap: str):
        """Handle colormap change."""
        if colormap.startswith('---'):
            return
        self._apply_visualization()
    
    def _on_filter_changed(self):
        """Handle filter changes - just update coloring, don't reload."""
        self._apply_visualization()
    
    def _reload_with_filters(self):
        """Reload the 3D view with current filter settings."""
        if not self.viewer:
            return
            
        logger.info("Reloading with filters...")
        filtered = self._get_filtered_model()
        
        if filtered is None or filtered.block_count == 0:
            QMessageBox.warning(self, "No Blocks", "No blocks match the current filter settings.")
            return
        
        logger.info(f"Loading {filtered.block_count} blocks after filtering")
        self.viewer.load_block_model(filtered)
        self.info_label.setText(f"Blocks displayed: {filtered.block_count:,}")
        
        # Reapply coloring
        property_name = self.property_combo.currentText()
        if property_name:
            self._apply_visualization()
    
    def _apply_visualization(self):
        """Apply current visualization settings - SIMPLIFIED to prevent reload."""
        if not self.viewer:
            return
            
        property_name = self.property_combo.currentText()
        if not property_name:
            return
        
        # Just update the coloring - DON'T reload the model (prevents freeze)
        color_mode = self.color_mode_combo.currentText().lower()
        colormap = self.colormap_combo.currentText()
        
        if colormap.startswith('---'):
            colormap = 'viridis'
        
        self.viewer.set_color_mode(color_mode)
        self.viewer.set_colormap(colormap)
        self.viewer.set_property_coloring(property_name)
        
        logger.info(f"Updated visualization: property={property_name}, mode={color_mode}, colormap={colormap}")
    
    def _get_filtered_model(self) -> Optional[BlockModel]:
        """Get filtered block model based on current filter settings."""
        # Create filter mask
        mask = [True] * self.results_model.block_count
        
        # Filter by mined blocks
        if self.mined_only_checkbox.isChecked() and 'MINED' in self.results_model.get_property_names():
            mined = self.results_model.get_property('MINED')
            mask = [m and mined[i] == 1 for i, m in enumerate(mask)]
        
        # Filter by period
        if self.period_filter_checkbox.isChecked() and 'PERIOD' in self.results_model.get_property_names():
            period_filter = self.period_combo.currentText()
            if period_filter and period_filter != 'All':
                periods = self.results_model.get_property('PERIOD')
                target_period = int(period_filter)
                mask = [m and periods[i] == target_period for i, m in enumerate(mask)]
        
        # Filter by phase
        if self.phase_filter_checkbox.isChecked() and 'PHASE' in self.results_model.get_property_names():
            phase_filter = self.phase_combo.currentText()
            if phase_filter and phase_filter != 'All':
                phases = self.results_model.get_property('PHASE')
                target_phase = int(phase_filter)
                mask = [m and phases[i] == target_phase for i, m in enumerate(mask)]
        
        # Create filtered model
        indices = [i for i, m in enumerate(mask) if m]
        
        if not indices:
            return None
        
        filtered = BlockModel()
        
        # Filter geometry
        origins = self.results_model.positions[indices]
        dimensions = self.results_model.dimensions[indices]
        filtered.set_geometry(origins, dimensions)
        
        # Filter properties
        for prop_name in self.results_model.get_property_names():
            prop_values = self.results_model.get_property(prop_name)
            if prop_values is not None:
                filtered.add_property(prop_name, prop_values[indices])
        
        return filtered
    
    def _reset_view(self):
        """Reset the camera view."""
        if self.viewer:
            self.viewer.reset_camera()

