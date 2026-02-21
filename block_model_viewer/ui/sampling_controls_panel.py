"""
Sampling Controls Panel for block model visualization.

Provides UI controls for adjusting block model sampling, LOD quality,
and performance settings for large datasets.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSlider, QSpinBox, QCheckBox, QPushButton, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
import logging

from .base_display_panel import BaseDisplayPanel

logger = logging.getLogger(__name__)


class SamplingControlsPanel(BaseDisplayPanel):
    """
    Panel for controlling block model sampling and LOD settings.
    
    Provides controls for:
    - Sampling factor (downsampling)
    - LOD quality
    - Maximum blocks to render
    - Performance optimization settings
    """
    
    # Signals
    sampling_changed = pyqtSignal(int, bool)  # factor, enabled
    lod_quality_changed = pyqtSignal(float)
    max_blocks_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        """Initialize the sampling controls panel."""
        super().__init__(parent=parent, panel_id="sampling_controls")
        
        # Current settings
        self.current_sampling_factor = 1
        self.current_sampling_enabled = False
        self.current_lod_quality = 0.7
        self.current_max_blocks = 500_000
        
        logger.info("Initialized sampling controls panel")
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout or QVBoxLayout(self)
        self.main_layout = layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("<b>Sampling & Performance Controls</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Sampling Controls Group
        sampling_group = self._create_sampling_group()
        layout.addWidget(sampling_group)
        
        # LOD Quality Group
        lod_group = self._create_lod_group()
        layout.addWidget(lod_group)
        
        # Performance Limits Group
        perf_group = self._create_performance_group()
        layout.addWidget(perf_group)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 10px; color: #666; padding: 5px;")
        layout.addWidget(self.info_label)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Set max width for better layout
        self.setMaximumWidth(350)
        self.setMinimumWidth(280)
    
    def _create_sampling_group(self) -> QGroupBox:
        """Create sampling controls group."""
        group = QGroupBox("Block Sampling")
        layout = QFormLayout(group)
        
        # Enable sampling checkbox
        self.sampling_enabled_check = QCheckBox("Enable Sampling")
        self.sampling_enabled_check.setChecked(self.current_sampling_enabled)
        self.sampling_enabled_check.setToolTip(
            "Enable block model sampling to reduce rendering load.\n"
            "When enabled, only every Nth block will be rendered."
        )
        self.sampling_enabled_check.stateChanged.connect(self._on_sampling_enabled_changed)
        layout.addRow(self.sampling_enabled_check)
        
        # Sampling factor
        factor_layout = QHBoxLayout()
        self.sampling_factor_spin = QSpinBox()
        self.sampling_factor_spin.setRange(1, 20)
        self.sampling_factor_spin.setValue(self.current_sampling_factor)
        self.sampling_factor_spin.setSuffix("x")
        self.sampling_factor_spin.setToolTip(
            "Sampling factor: 1 = no sampling, 2 = every 2nd block, 3 = every 3rd block, etc.\n"
            "Higher values reduce rendering load but may reduce detail."
        )
        self.sampling_factor_spin.valueChanged.connect(self._on_sampling_factor_changed)
        factor_layout.addWidget(self.sampling_factor_spin)
        
        # Info label
        self.sampling_info_label = QLabel("(1x = no sampling)")
        self.sampling_info_label.setStyleSheet("font-size: 10px; color: #666;")
        factor_layout.addWidget(self.sampling_info_label)
        factor_layout.addStretch()
        
        layout.addRow("Sampling Factor:", factor_layout)
        
        return group
    
    def _create_lod_group(self) -> QGroupBox:
        """Create LOD quality controls group."""
        group = QGroupBox("Level of Detail (LOD)")
        layout = QFormLayout(group)
        
        # LOD Quality slider
        lod_layout = QHBoxLayout()
        self.lod_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.lod_quality_slider.setRange(0, 100)
        self.lod_quality_slider.setValue(int(self.current_lod_quality * 100))
        self.lod_quality_slider.setToolTip(
            "LOD Quality (0=Low/Aggressive, 100=High/Minimal)\n"
            "Lower values use more aggressive LOD for better performance.\n"
            "Higher values preserve more detail but may impact performance."
        )
        self.lod_quality_slider.valueChanged.connect(self._on_lod_quality_changed)
        lod_layout.addWidget(self.lod_quality_slider)
        
        self.lod_quality_label = QLabel(f"{self.current_lod_quality:.1f}")
        self.lod_quality_label.setMinimumWidth(40)
        self.lod_quality_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        lod_layout.addWidget(self.lod_quality_label)
        
        layout.addRow("Quality:", lod_layout)
        
        # LOD info
        lod_info = QLabel(
            "LOD automatically adjusts detail based on camera distance\n"
            "and model complexity for optimal performance."
        )
        lod_info.setStyleSheet("font-size: 9px; color: #666; padding-top: 4px;")
        lod_info.setWordWrap(True)
        layout.addRow("", lod_info)
        
        return group
    
    def _create_performance_group(self) -> QGroupBox:
        """Create performance limits group."""
        group = QGroupBox("Performance Limits")
        layout = QFormLayout(group)
        
        # Max blocks to render
        max_blocks_layout = QHBoxLayout()
        self.max_blocks_spin = QSpinBox()
        self.max_blocks_spin.setRange(1000, 10_000_000)
        self.max_blocks_spin.setValue(self.current_max_blocks)
        self.max_blocks_spin.setSingleStep(50_000)
        self.max_blocks_spin.setToolTip(
            "Maximum number of blocks to render.\n"
            "Models exceeding this limit will be automatically sampled.\n"
            "Lower values improve performance but may reduce detail."
        )
        self.max_blocks_spin.valueChanged.connect(self._on_max_blocks_changed)
        max_blocks_layout.addWidget(self.max_blocks_spin)
        
        max_blocks_layout.addStretch()
        
        layout.addRow("Max Blocks:", max_blocks_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Reset all settings to default values")
        reset_btn.clicked.connect(self._reset_to_defaults)
        layout.addRow("", reset_btn)
        
        return group
    
    def _on_sampling_enabled_changed(self, state: int):
        """Handle sampling enabled checkbox change."""
        enabled = state == Qt.CheckState.Checked
        self.current_sampling_enabled = enabled
        self.sampling_changed.emit(self.current_sampling_factor, enabled)
        
        # Update UI
        self.sampling_factor_spin.setEnabled(enabled)
        self._update_info()
        
        # Apply to renderer if available
        try:
            if self.controller and hasattr(self.controller, 'renderer'):
                self.controller.renderer.set_sampling_factor(
                    self.current_sampling_factor, enabled
                )
        except Exception as e:
            logger.debug(f"Could not apply sampling to renderer: {e}")
    
    def _on_sampling_factor_changed(self, value: int):
        """Handle sampling factor change."""
        self.current_sampling_factor = value
        self.sampling_info_label.setText(f"({value}x = every {value}{'rd' if value == 3 else 'th' if value > 3 else 'nd' if value == 2 else ''} block)")
        self.sampling_changed.emit(value, self.current_sampling_enabled)
        
        # Apply to renderer if available
        try:
            if self.controller and hasattr(self.controller, 'renderer'):
                self.controller.renderer.set_sampling_factor(
                    value, self.current_sampling_enabled
                )
        except Exception as e:
            logger.debug(f"Could not apply sampling to renderer: {e}")
    
    def _on_lod_quality_changed(self, value: int):
        """Handle LOD quality slider change."""
        self.current_lod_quality = value / 100.0
        self.lod_quality_label.setText(f"{self.current_lod_quality:.2f}")
        self.lod_quality_changed.emit(self.current_lod_quality)
        
        # Apply to renderer if available
        try:
            if self.controller and hasattr(self.controller, 'renderer'):
                self.controller.renderer.set_lod_quality(self.current_lod_quality)
        except Exception as e:
            logger.debug(f"Could not apply LOD quality to renderer: {e}")
    
    def _on_max_blocks_changed(self, value: int):
        """Handle max blocks change."""
        self.current_max_blocks = value
        self.max_blocks_changed.emit(value)
        
        # Apply to renderer if available
        try:
            if self.controller and hasattr(self.controller, 'renderer'):
                self.controller.renderer.set_max_blocks_render(value)
        except Exception as e:
            logger.debug(f"Could not apply max blocks to renderer: {e}")
    
    def _reset_to_defaults(self):
        """Reset all settings to default values."""
        self.current_sampling_factor = 1
        self.current_sampling_enabled = False
        self.current_lod_quality = 0.7
        self.current_max_blocks = 500_000
        
        # Update UI
        self.sampling_enabled_check.setChecked(False)
        self.sampling_factor_spin.setValue(1)
        self.lod_quality_slider.setValue(70)
        self.max_blocks_spin.setValue(500_000)
        
        # Emit signals
        self.sampling_changed.emit(1, False)
        self.lod_quality_changed.emit(0.7)
        self.max_blocks_changed.emit(500_000)
        
        logger.info("Reset sampling controls to defaults")
    
    def _update_info(self):
        """Update info label with current settings."""
        if self.controller and hasattr(self.controller, 'current_model'):
            model = self.controller.current_model
            if model:
                total_blocks = model.block_count
                if self.current_sampling_enabled:
                    rendered_blocks = total_blocks // self.current_sampling_factor
                    info_text = (
                        f"Total blocks: {total_blocks:,}\n"
                        f"Rendered blocks: ~{rendered_blocks:,}\n"
                        f"Sampling: {self.current_sampling_factor}x"
                    )
                else:
                    rendered_blocks = min(total_blocks, self.current_max_blocks)
                    if total_blocks > self.current_max_blocks:
                        info_text = (
                            f"Total blocks: {total_blocks:,}\n"
                            f"Rendered blocks: ~{rendered_blocks:,}\n"
                            f"(Auto-sampled to fit limit)"
                        )
                    else:
                        info_text = f"Total blocks: {total_blocks:,}\nRendered blocks: {rendered_blocks:,}"
                self.info_label.setText(info_text)
            else:
                self.info_label.setText("No model loaded")
        else:
            self.info_label.setText("")
    
    def refresh(self):
        """Refresh the panel with current settings."""
        self._update_info()
        
        # Update from renderer if available
        try:
            if self.controller and hasattr(self.controller, 'renderer'):
                settings = self.controller.renderer.get_performance_settings()
                self.current_lod_quality = settings.get('lod_quality', 0.7)
                self.current_sampling_factor = settings.get('sampling_factor', 1)
                self.current_sampling_enabled = settings.get('sampling_enabled', False)
                self.current_max_blocks = settings.get('max_blocks_render', 500_000)
                
                # Update UI without triggering signals
                self.lod_quality_slider.blockSignals(True)
                self.lod_quality_slider.setValue(int(self.current_lod_quality * 100))
                self.lod_quality_label.setText(f"{self.current_lod_quality:.2f}")
                self.lod_quality_slider.blockSignals(False)
                
                self.sampling_factor_spin.blockSignals(True)
                self.sampling_factor_spin.setValue(self.current_sampling_factor)
                self.sampling_factor_spin.blockSignals(False)
                
                self.sampling_enabled_check.blockSignals(True)
                self.sampling_enabled_check.setChecked(self.current_sampling_enabled)
                self.sampling_enabled_check.blockSignals(False)
                
                self.max_blocks_spin.blockSignals(True)
                self.max_blocks_spin.setValue(self.current_max_blocks)
                self.max_blocks_spin.blockSignals(False)
        except Exception as e:
            logger.debug(f"Could not refresh from renderer: {e}")

