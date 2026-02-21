"""
Screenshot Export Dialog

UI for advanced screenshot export with branding and layout options.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt

from ..utils.screenshot_manager import ScreenshotManager

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ScreenshotExportDialog(QDialog):
    """
    Dialog for configuring and exporting advanced screenshots.
    
    Features:
    - Preset selection (presentation, publication, poster, etc.)
    - Custom dimensions and DPI
    - Title, subtitle, and watermark
    - Legend, scale bar, axes toggles
    - Company logo upload
    - Transparent background option
    """
    
    def __init__(self, screenshot_manager: ScreenshotManager, parent=None):
        super().__init__(parent)
        self.screenshot_manager = screenshot_manager
        self.company_logo_path: Optional[Path] = None
        
        self.setWindowTitle("Export Screenshot")
        self.setModal(True)
        self.resize(500, 700)
        
        self._build_ui()
        logger.info("Initialized ScreenshotExportDialog")
    


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
    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Advanced Screenshot Export")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #FF9800;")
        layout.addWidget(title)
        
        # === Preset Selection ===
        preset_group = self._create_preset_group()
        layout.addWidget(preset_group)
        
        # === Dimensions ===
        dimensions_group = self._create_dimensions_group()
        layout.addWidget(dimensions_group)
        
        # === Branding ===
        branding_group = self._create_branding_group()
        layout.addWidget(branding_group)
        
        # === Display Options ===
        display_group = self._create_display_group()
        layout.addWidget(display_group)
        
        # === Buttons ===
        button_row = self._create_buttons()
        layout.addLayout(button_row)
    
    def _create_preset_group(self) -> QGroupBox:
        """Create preset selection controls."""
        group = QGroupBox("Size Preset")
        layout = QVBoxLayout(group)
        
        # Preset combo
        self.preset_combo = QComboBox()
        presets = ScreenshotManager.list_presets()
        self.preset_combo.addItems(presets)
        self.preset_combo.setCurrentText('presentation')
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo)
        
        # Preset info label
        self.preset_info_label = QLabel()
        self.preset_info_label.setStyleSheet("color: #999; font-size: 9pt;")
        self.preset_info_label.setWordWrap(True)
        layout.addWidget(self.preset_info_label)
        
        # Update info
        self._on_preset_changed('presentation')
        
        return group
    
    def _create_dimensions_group(self) -> QGroupBox:
        """Create dimension controls."""
        group = QGroupBox("Dimensions (Custom)")
        form = QFormLayout(group)
        form.setSpacing(8)
        
        # Width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(640, 10000)
        self.width_spin.setValue(1920)
        self.width_spin.setSuffix(" px")
        form.addRow("Width:", self.width_spin)
        
        # Height
        self.height_spin = QSpinBox()
        self.height_spin.setRange(480, 10000)
        self.height_spin.setValue(1080)
        self.height_spin.setSuffix(" px")
        form.addRow("Height:", self.height_spin)
        
        # DPI
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setSuffix(" DPI")
        form.addRow("DPI:", self.dpi_spin)
        
        # Note
        note = QLabel("Note: Custom values only apply when 'custom' preset is selected")
        note.setStyleSheet("color: #888; font-size: 8pt; font-style: italic;")
        note.setWordWrap(True)
        form.addRow(note)
        
        return group
    
    def _create_branding_group(self) -> QGroupBox:
        """Create branding controls."""
        group = QGroupBox("Branding & Annotations")
        layout = QVBoxLayout(group)
        
        form = QFormLayout()
        form.setSpacing(8)
        
        # Title
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("e.g., Block Model Analysis")
        form.addRow("Title:", self.title_edit)
        
        # Subtitle
        self.subtitle_edit = QLineEdit()
        self.subtitle_edit.setPlaceholderText("e.g., Iron Ore Deposit - Grade Distribution")
        form.addRow("Subtitle:", self.subtitle_edit)
        
        # Watermark
        self.watermark_edit = QLineEdit()
        self.watermark_edit.setPlaceholderText("e.g., CONFIDENTIAL or company name")
        form.addRow("Watermark:", self.watermark_edit)
        
        layout.addLayout(form)
        
        # Company logo
        logo_row = QHBoxLayout()
        logo_row.addWidget(QLabel("Company Logo:"))
        
        self.logo_path_label = QLabel("No logo selected")
        self.logo_path_label.setStyleSheet("color: #888; font-style: italic;")
        logo_row.addWidget(self.logo_path_label, 1)
        
        self.btn_browse_logo = QPushButton("Browse...")
        self.btn_browse_logo.setMaximumWidth(100)
        self.btn_browse_logo.clicked.connect(self._on_browse_logo)
        logo_row.addWidget(self.btn_browse_logo)
        
        layout.addLayout(logo_row)
        
        return group
    
    def _create_display_group(self) -> QGroupBox:
        """Create display options."""
        group = QGroupBox("Display Options")
        layout = QVBoxLayout(group)
        
        # Checkboxes
        self.show_legend_check = QCheckBox("Show Color Legend")
        self.show_legend_check.setChecked(True)
        layout.addWidget(self.show_legend_check)
        
        self.show_scale_check = QCheckBox("Show Scale Bar")
        self.show_scale_check.setChecked(True)
        layout.addWidget(self.show_scale_check)
        
        self.show_axes_check = QCheckBox("Show Axes")
        self.show_axes_check.setChecked(True)
        layout.addWidget(self.show_axes_check)
        
        self.show_timestamp_check = QCheckBox("Show Timestamp")
        self.show_timestamp_check.setChecked(True)
        layout.addWidget(self.show_timestamp_check)
        
        self.transparent_check = QCheckBox("Transparent Background")
        self.transparent_check.setChecked(False)
        layout.addWidget(self.transparent_check)
        
        return group
    
    def _create_buttons(self) -> QHBoxLayout:
        """Create dialog buttons."""
        layout = QHBoxLayout()
        layout.addStretch()
        
        self.btn_export_simple = QPushButton("Quick Export")
        self.btn_export_simple.setToolTip("Simple screenshot without branding")
        self.btn_export_simple.clicked.connect(self._on_export_simple)
        layout.addWidget(self.btn_export_simple)
        
        self.btn_export_branded = QPushButton("Export Branded Layout")
        self.btn_export_branded.setToolTip("Full branded layout with title, legend, etc.")
        self.btn_export_branded.setDefault(True)
        self.btn_export_branded.clicked.connect(self._on_export_branded)
        layout.addWidget(self.btn_export_branded)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel)
        
        return layout
    
    def _on_preset_changed(self, preset: str):
        """Update info when preset changes."""
        info = ScreenshotManager.get_preset_info(preset)
        self.preset_info_label.setText(info['description'])
        
        # Update custom dimension fields
        if preset != 'custom':
            self.width_spin.setValue(info['width'])
            self.height_spin.setValue(info['height'])
            self.dpi_spin.setValue(info['dpi'])
    
    def _on_browse_logo(self):
        """Browse for company logo."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Company Logo",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if filepath:
            self.company_logo_path = Path(filepath)
            self.logo_path_label.setText(self.company_logo_path.name)
            self.logo_path_label.setStyleSheet("color: #4CAF50;")
    
    def _on_export_simple(self):
        """Export simple screenshot."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Simple Screenshot",
            "screenshot.png",
            "PNG Image (*.png);;All Files (*)"
        )
        
        if filepath:
            transparent = self.transparent_check.isChecked()
            
            success = self.screenshot_manager.export_simple_screenshot(
                Path(filepath),
                transparent=transparent,
                scale=2
            )
            
            if success:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Screenshot exported successfully to:\n{filepath}"
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Failed to export screenshot.\nCheck logs for details."
                )
    
    def _on_export_branded(self):
        """
        Export branded layout.
        
        NOTE: VTK render operations (render_window.GetImage()) MUST run on the main thread.
        Do NOT attempt to move this to a background thread - VTK will crash.
        """
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Branded Screenshot",
            "branded_screenshot.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;All Files (*)"
        )
        
        if filepath:
            preset = self.preset_combo.currentText()
            
            # Get dimensions from custom fields if custom preset
            if preset == 'custom':
                # Temporarily update the preset config
                ScreenshotManager.PRESETS['custom']['width'] = self.width_spin.value()
                ScreenshotManager.PRESETS['custom']['height'] = self.height_spin.value()
                ScreenshotManager.PRESETS['custom']['dpi'] = self.dpi_spin.value()
            
            success = self.screenshot_manager.export_branded_layout(
                filepath=Path(filepath),
                preset=preset,
                title=self.title_edit.text().strip(),
                subtitle=self.subtitle_edit.text().strip(),
                show_legend=self.show_legend_check.isChecked(),
                show_scale_bar=self.show_scale_check.isChecked(),
                show_axes=self.show_axes_check.isChecked(),
                show_timestamp=self.show_timestamp_check.isChecked(),
                watermark=self.watermark_edit.text().strip(),
                company_logo=self.company_logo_path,
                transparent=self.transparent_check.isChecked()
            )
            
            if success:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Branded screenshot exported successfully to:\n{filepath}\n\n"
                    f"Preset: {preset}\n"
                    f"Dimensions: {self.width_spin.value()}x{self.height_spin.value()}\n"
                    f"DPI: {self.dpi_spin.value()}"
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Failed to export branded screenshot.\nCheck logs for details."
                )
