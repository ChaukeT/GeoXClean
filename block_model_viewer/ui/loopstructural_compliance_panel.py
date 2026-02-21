"""
ComplianceValidationPanel - JORC/SAMREC Compliance Viewer.

The 'Auditor View' - visualizes where the geological model fails
to respect the drillhole data.

GeoX Panel Safety Rules:
- Panels initialize private state only (self._attr = None)
- Controllers bind data via explicit methods
- No assignments to @property without setter
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton, QGroupBox,
    QProgressBar, QComboBox, QDoubleSpinBox, QCheckBox, QFrame
)
from PyQt6.QtGui import QColor

if TYPE_CHECKING:
    from pyvistaqt import QtInteractor
    from ..geology.compliance_manager import AuditReport

logger = logging.getLogger(__name__)


class ComplianceValidationPanel(QWidget):
    """
    The 'Auditor View'.
    
    Visualizes where the geological model fails to respect the drillhole data
    by rendering error spheres in the 3D scene colored by misfit magnitude.
    
    This panel is a critical QC tool for JORC/SAMREC compliance verification.
    """
    
    # Signals
    compliance_checked = pyqtSignal(object)  # Emits AuditReport
    export_requested = pyqtSignal(str)  # Emits export path
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Private state (GeoX Panel Safety Rules)
        self._plotter: Optional[QtInteractor] = None
        self._current_report: Optional[AuditReport] = None
        self._misfit_actor_name: str = "misfit_glyphs"
        
        self._build_ui()
    


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
        """Build the modern panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 1. Compact Dark Compliance Header
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.CARD_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        header_frame_layout = QVBoxLayout(header_frame)
        header_frame_layout.setSpacing(12)
        
        header_layout = QHBoxLayout()
        
        self._status_label = QLabel("⏳ Compliance: PENDING")
        self._status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: 700;
                color: {ModernColors.WARNING};
                background: transparent;
            }}
        """)
        header_layout.addWidget(self._status_label)
        header_layout.addStretch()
        
        # Classification recommendation badge
        self._class_label = QLabel("")
        self._class_label.setStyleSheet(f"""
            QLabel {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.ACCENT_PRIMARY};
                padding: 4px 12px;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
            }}
        """)
        header_layout.addWidget(self._class_label)
        
        header_frame_layout.addLayout(header_layout)
        
        layout.addWidget(header_frame)
        
        # 2. Dark Statistical Table
        stats_group = QGroupBox("📊 Misfit Statistics")
        stats_group.setStyleSheet(f"""
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
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(16, 20, 16, 16)
        
        self._stats_table = QTableWidget(4, 2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.setMaximumHeight(150)
        self._stats_table.setStyleSheet(f"""
            QTableWidget {{
                background: {ModernColors.PANEL_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                gridline-color: {ModernColors.BORDER};
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 8px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QTableWidget::item:selected {{
                background: {ModernColors.CARD_HOVER};
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QHeaderView::section {{
                background: {ModernColors.CARD_BG};
                color: {ModernColors.TEXT_SECONDARY};
                padding: 10px;
                border: none;
                font-weight: 700;
                font-size: 11px;
            }}
        """)
        
        # Initialize with empty values
        self._set_cell(0, "Mean Residual", "—")
        self._set_cell(1, "P90 Error", "—")
        self._set_cell(2, "Total Contacts", "—")
        self._set_cell(3, "Classification", "—")
        
        stats_layout.addWidget(self._stats_table)
        layout.addWidget(stats_group)
        
        # 3. Dark JORC Compliance Info
        info_frame = QFrame()
        info_frame.setStyleSheet(f"""
            QFrame {{
                background: {ModernColors.PANEL_BG};
                border-left: 4px solid {ModernColors.ACCENT_PRIMARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        info_frame_layout = QVBoxLayout(info_frame)
        
        info_label = QLabel(
            "ℹ️ <b>JORC Standards:</b> Measured &lt;2m, Indicated &lt;5m"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
            }}
        """)
        info_frame_layout.addWidget(info_label)
        
        layout.addWidget(info_frame)
        
        # 4. Dark Visualization Controls
        viz_group = QGroupBox("🎨 3D Visualization")
        viz_group.setStyleSheet(f"""
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
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setSpacing(12)
        viz_layout.setContentsMargins(16, 20, 16, 16)
        
        # Scale factor for error spheres
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Sphere Scale:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.1, 10.0)
        self._scale_spin.setValue(2.0)
        self._scale_spin.setSingleStep(0.5)
        self._scale_spin.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self._scale_spin)
        scale_layout.addStretch()
        viz_layout.addLayout(scale_layout)
        
        # Modern Show/hide options
        self._show_glyphs = QCheckBox("Show Error Spheres")
        self._show_glyphs.setChecked(True)
        self._show_glyphs.setStyleSheet(f"""
            QCheckBox {{
                font-size: 12px;
                color: {ModernColors.TEXT_PRIMARY};
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {ModernColors.BORDER};
                border-radius: 4px;
                background: {ModernColors.PANEL_BG};
            }}
            QCheckBox::indicator:checked {{
                background: {ModernColors.ACCENT_PRIMARY};
                border-color: {ModernColors.ACCENT_PRIMARY};
                image: none;
            }}
            QCheckBox::indicator:hover {{
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        self._show_glyphs.stateChanged.connect(self._on_visibility_changed)
        viz_layout.addWidget(self._show_glyphs)
        
        # Modern Colormap selection
        cmap_layout = QHBoxLayout()
        cmap_layout.setSpacing(12)
        
        cmap_label = QLabel("Colormap:")
        cmap_label.setStyleSheet(f"font-size: 12px; color: {ModernColors.TEXT_SECONDARY}; font-weight: 500;")
        cmap_layout.addWidget(cmap_label)
        
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["Reds", "RdYlGn_r", "coolwarm", "plasma", "viridis"])
        self._cmap_combo.setStyleSheet(f"""
            QComboBox {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_PRIMARY};
                border: 2px solid {ModernColors.BORDER};
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 11px;
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {ModernColors.TEXT_SECONDARY};
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {ModernColors.CARD_BG};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 6px;
                selection-background-color: {ModernColors.CARD_HOVER};
                selection-color: {ModernColors.ACCENT_PRIMARY};
                color: {ModernColors.TEXT_PRIMARY};
                padding: 4px;
            }}
        """)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self._cmap_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        cmap_layout.addWidget(self._cmap_combo)
        cmap_layout.addStretch()
        viz_layout.addLayout(cmap_layout)
        
        layout.addWidget(viz_group)
        
        # 5. Progress Bar (for computing)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        # 6. Modern Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        modern_primary_btn_style = f"""
            QPushButton {{
                background: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background: {ModernColors.ACCENT_PRESSED};
            }}
        """
        
        modern_secondary_btn_style = f"""
            QPushButton {{
                background: {ModernColors.PANEL_BG};
                color: {ModernColors.ACCENT_PRIMARY};
                border: 2px solid {ModernColors.ACCENT_PRIMARY};
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {ModernColors.CARD_BG};
                border-color: {ModernColors.ACCENT_PRIMARY};
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
        
        self._refresh_btn = QPushButton("🔄 Refresh Compliance")
        self._refresh_btn.setStyleSheet(modern_primary_btn_style)
        self._refresh_btn.clicked.connect(self._on_refresh_clicked)
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._refresh_btn)
        
        self._export_btn = QPushButton("📥 Export Report")
        self._export_btn.setStyleSheet(modern_secondary_btn_style)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_btn.setEnabled(False)
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout.addWidget(self._export_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
    
    def bind_plotter(self, plotter: QtInteractor) -> None:
        """
        Bind the PyVista plotter for 3D visualization.
        
        Controller method - called by parent panel/controller.
        """
        self._plotter = plotter
    
    def update_viz(self, report: "AuditReport") -> None:
        """
        Update the panel with compliance report and render error visualization.
        
        Args:
            report: AuditReport from ComplianceManager
        """
        self._current_report = report
        
        # Update Table
        self._set_cell(0, "Mean Residual", f"{report.mean_residual:.2f} m")
        self._set_cell(1, "P90 Error", f"{report.p90_error:.2f} m")
        self._set_cell(2, "Total Contacts", str(report.total_contacts))
        self._set_cell(3, "Classification", report.classification_recommendation)
        
        # Update Status Style with dark design
        if report.status == "Acceptable":
            icon = "✅"
            color = f"{ModernColors.SUCCESS}"
        elif report.status == "Needs Review":
            icon = "⚠️"
            color = f"{ModernColors.WARNING}"
        else:
            icon = "❌"
            color = f"{ModernColors.ERROR}"
        
        self._status_label.setText(f"{icon} Compliance: {report.status}")
        self._status_label.setStyleSheet(
            f"QLabel {{ "
            f"font-size: 14px; font-weight: 700; color: {color}; "
            f"background: transparent; "
            f"}}"
        )
        
        self._class_label.setText(
            f"Recommended: {report.classification_recommendation} Resources"
        )
        
        # Enable export
        self._export_btn.setEnabled(True)
        
        # Render 3D visualization
        self._render_misfit_glyphs(report)
        
        # Emit signal
        self.compliance_checked.emit(report)
    
    def _render_misfit_glyphs(self, report: "AuditReport") -> None:
        """Render error spheres in the 3D scene."""
        if self._plotter is None:
            logger.warning("No plotter bound - cannot render misfit visualization")
            return
        
        if len(report.misfit_data) == 0:
            logger.warning("No misfit data to visualize")
            return
        
        try:
            import pyvista as pv
            
            # Remove existing glyphs
            try:
                self._plotter.remove_actor(self._misfit_actor_name)
            except Exception:
                pass
            
            if not self._show_glyphs.isChecked():
                return
            
            # Create point cloud from misfit data
            points = report.misfit_data[['X', 'Y', 'Z']].values
            cloud = pv.PolyData(points)
            cloud["Residuals (m)"] = report.misfit_data['residual_m'].values
            
            # Create spheres scaled by error magnitude
            geom = pv.Sphere(radius=1.0)
            scale_factor = self._scale_spin.value()
            glyphs = cloud.glyph(geom=geom, scale="Residuals (m)", factor=scale_factor)
            
            # Add to scene
            cmap = self._cmap_combo.currentText()
            self._plotter.add_mesh(
                glyphs,
                cmap=cmap,
                scalars="Residuals (m)",
                scalar_bar_args={'title': "Misfit Magnitude (m)"},
                name=self._misfit_actor_name,
                opacity=0.8,
                show_edges=False,
            )
            
            logger.info(f"Rendered {len(points)} misfit spheres")
            
        except Exception as e:
            logger.error(f"Failed to render misfit visualization: {e}")
    
    def _set_cell(self, row: int, label: str, val: str) -> None:
        """Set a cell in the statistics table."""
        self._stats_table.setItem(row, 0, QTableWidgetItem(label))
        self._stats_table.setItem(row, 1, QTableWidgetItem(val))
    
    def _on_scale_changed(self, value: float) -> None:
        """Handle scale factor change."""
        if self._current_report:
            self._render_misfit_glyphs(self._current_report)
    
    def _on_visibility_changed(self, state: int) -> None:
        """Handle visibility toggle."""
        if self._current_report:
            self._render_misfit_glyphs(self._current_report)
    
    def _on_cmap_changed(self, cmap: str) -> None:
        """Handle colormap change."""
        if self._current_report:
            self._render_misfit_glyphs(self._current_report)
    
    def _on_refresh_clicked(self) -> None:
        """Handle refresh button click."""
        # Emit signal - the parent panel/controller should handle the actual refresh
        logger.info("Compliance refresh requested")
        # This would typically be connected to the main panel's refresh logic
    
    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        from PyQt6.QtWidgets import QFileDialog
        
        if self._current_report is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Compliance Report",
            "compliance_report",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            from ..geology.compliance_manager import ComplianceManager
            # Remove extension if present
            if filepath.endswith('.json'):
                filepath = filepath[:-5]
            
            ComplianceManager.export_audit_report(
                self._current_report,
                filepath,
                include_spatial_data=True
            )
            
            self.export_requested.emit(filepath)
            logger.info(f"Exported compliance report to {filepath}")
    
    def show_progress(self, value: int, message: str = "") -> None:
        """Show progress bar with value and optional message."""
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(value)
        if message:
            self._progress_bar.setFormat(f"{message} %p%")
    
    def hide_progress(self) -> None:
        """Hide progress bar."""
        self._progress_bar.setVisible(False)
    
    def clear(self) -> None:
        """Clear the panel state."""
        self._current_report = None
        self._status_label.setText("Compliance Status: PENDING")
        self._status_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {ModernColors.WARNING};"
        )
        self._class_label.setText("")
        self._set_cell(0, "Mean Residual", "—")
        self._set_cell(1, "P90 Error", "—")
        self._set_cell(2, "Total Contacts", "—")
        self._set_cell(3, "Classification", "—")
        self._export_btn.setEnabled(False)
        
        # Remove visualization
        if self._plotter:
            try:
                self._plotter.remove_actor(self._misfit_actor_name)
            except Exception:
                pass

