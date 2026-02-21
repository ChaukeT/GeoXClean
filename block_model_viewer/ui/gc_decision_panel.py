"""
GC Decision Panel - Displays Grade Control Decision Engine results

Shows spider chart (radar chart) visualization of block decision DNA
when a block is selected, providing explainability for the MCDA classification.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QPushButton, QTextEdit, QMessageBox, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)

# Matplotlib imports (backend is set in main.py)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        FigureCanvas = None
        Figure = None

# Import visualization
try:
    from ..visualization.gc_spider_chart import plot_block_dna
    SPIDER_CHART_AVAILABLE = True
except ImportError as e:
    SPIDER_CHART_AVAILABLE = False
    logger.warning(f"GC spider chart visualization not available: {e}")


class GCDecisionPanel(BaseDockPanel):
    """
    Panel for displaying GC Decision Engine results with spider chart visualization.
    
    Shows decision DNA (spider chart) when a block with GC scores is selected.
    """
    # PanelManager metadata
    PANEL_ID = "GCDecisionPanel"
    PANEL_NAME = "GCDecision Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT


    
    # Signals
    export_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_block_id = None
        self.current_block_data = None
        self.current_figure = None
        self.current_canvas = None
        
        # Subscribe to block model updates
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_updated)
            self.registry.blockModelLoaded.connect(self._on_block_model_updated)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized GC Decision Panel")
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout  # Use inherited layout from BaseDockPanel
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("<b>GC Decision Engine - Block DNA</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Info label
        info_label = QLabel(
            "Select a block to view its decision DNA (spider chart) showing "
            "all component scores that contribute to the GC classification."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 9px; color: #666; padding: 5px;")
        layout.addWidget(info_label)
        
        # Chart display group
        chart_group = QGroupBox("Decision DNA Visualization")
        chart_layout = QVBoxLayout(chart_group)
        
        # Canvas placeholder
        if MATPLOTLIB_AVAILABLE and SPIDER_CHART_AVAILABLE:
            self.chart_canvas = FigureCanvas(Figure(figsize=(6, 6)))
            self.chart_canvas.setMinimumSize(400, 400)
            chart_layout.addWidget(self.chart_canvas)
            
            # Initial empty chart
            self._show_empty_chart()
        else:
            no_chart_label = QLabel(
                "Chart visualization unavailable.\n"
                "Install matplotlib to enable spider charts."
            )
            no_chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_chart_label.setStyleSheet("color: #999; padding: 20px;")
            chart_layout.addWidget(no_chart_label)
            self.chart_canvas = None
        
        layout.addWidget(chart_group)
        
        # Block information group
        info_group = QGroupBox("Block GC Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Click on a block with GC scores to see its decision breakdown...")
        self.info_text.setMaximumHeight(150)
        
        # Set monospace font
        font = QFont("Courier New", 9)
        self.info_text.setFont(font)
        
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Chart")
        self.export_btn.setToolTip("Export spider chart to image file")
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Clear the current selection")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        # Add stretch
        layout.addStretch()
        
        # Set max width
        self.setMaximumWidth(500)
        self.setMinimumWidth(400)
    
    def _show_empty_chart(self):
        """Display an empty chart placeholder."""
        if not self.chart_canvas:
            return
        
        fig = self.chart_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Select a block\nto view Decision DNA', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='#999')
        ax.axis('off')
        self.chart_canvas.draw()
    
    def _on_block_model_updated(self, block_model):
        """Handle block model updates from registry."""
        # Could refresh chart if needed
        pass
    
    def update_block_info(self, block_id: int, block_data: dict, coordinates: tuple = None):
        """
        Update the panel with GC decision information for a selected block.
        
        Args:
            block_id: The ID/index of the selected block
            block_data: Dictionary of property values for the block
            coordinates: Optional tuple of (x, y, z) coordinates
        """
        try:
            self.current_block_id = block_id
            self.current_block_data = block_data
            
            # Check if block has GC scores
            gc_score_keys = [
                'SCORE_GRADE', 'SCORE_PENALTY', 'SCORE_UNCERT',
                'SCORE_GEO', 'SCORE_ECO', 'SCORE_CONTEXT'
            ]
            
            has_gc_scores = any(key in block_data for key in gc_score_keys)
            
            if not has_gc_scores:
                # Block doesn't have GC scores
                self._show_no_gc_data()
                self.export_btn.setEnabled(False)
                self.clear_btn.setEnabled(True)
                return
            
            # Update spider chart
            if self.chart_canvas and SPIDER_CHART_AVAILABLE:
                self._update_spider_chart(block_data, block_id)
            
            # Update info text
            self._update_info_text(block_data, block_id, coordinates)
            
            # Enable buttons
            self.export_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            
            logger.info(f"Updated GC Decision panel for block ID: {block_id}")
            
        except Exception as e:
            logger.error(f"Error updating GC Decision panel: {e}", exc_info=True)
            self._show_error(str(e))
            self.export_btn.setEnabled(False)
            self.clear_btn.setEnabled(True)
    
    def _update_spider_chart(self, block_data: dict, block_id: int):
        """Update the spider chart with block data."""
        if not self.chart_canvas or not SPIDER_CHART_AVAILABLE:
            return
        
        try:
            # Create spider chart
            fig = plot_block_dna(block_data, block_id=block_id, backend='matplotlib')
            
            if fig is None:
                self._show_empty_chart()
                return
            
            # Replace canvas figure
            self.chart_canvas.figure = fig
            self.current_figure = fig
            self.chart_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating spider chart: {e}", exc_info=True)
            self._show_empty_chart()
    
    def _update_info_text(self, block_data: dict, block_id: int, coordinates: tuple = None):
        """Update the information text area."""
        info_lines = []
        info_lines.append("=" * 50)
        info_lines.append(f"BLOCK ID: {block_id}")
        info_lines.append("=" * 50)
        
        # Coordinates
        if coordinates:
            try:
                x, y, z = coordinates
                info_lines.append("\nCOORDINATES:")
                info_lines.append(f"  X (East):  {x:>12.2f} m")
                info_lines.append(f"  Y (North): {y:>12.2f} m")
                info_lines.append(f"  Z (Elev):  {z:>12.2f} m")
            except Exception:
                pass
        
        # GC Scores
        info_lines.append("\nGC COMPONENT SCORES:")
        score_keys = [
            ('SCORE_GRADE', 'Grade Score'),
            ('SCORE_PENALTY', 'Penalty Score'),
            ('SCORE_UNCERT', 'Uncertainty Score'),
            ('SCORE_GEO', 'Geology Score'),
            ('SCORE_ECO', 'Economic Score'),
            ('SCORE_CONTEXT', 'Context Score')
        ]
        
        for key, label in score_keys:
            if key in block_data:
                val = block_data[key]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    info_lines.append(f"  {label:<20} {val:>8.4f}")
        
        # Final GC Score and Route
        info_lines.append("\nGC DECISION:")
        if 'GC_SCORE' in block_data:
            gc_score = block_data['GC_SCORE']
            if isinstance(gc_score, (int, float)) and not np.isnan(gc_score):
                info_lines.append(f"  Final GC Score: {gc_score:>8.4f}")
        
        if 'GC_ROUTE' in block_data:
            route = block_data['GC_ROUTE']
            info_lines.append(f"  Classification: {route}")
        
        info_lines.append("\n" + "=" * 50)
        
        self.info_text.setPlainText("\n".join(info_lines))
    
    def _show_no_gc_data(self):
        """Show message when block has no GC data."""
        if self.chart_canvas:
            self._show_empty_chart()
        
        self.info_text.setPlainText(
            "This block does not have GC Decision Engine scores.\n\n"
            "Run the GC Decision Engine classification on your block model "
            "to generate scores and routes."
        )
    
    def _show_error(self, error_msg: str):
        """Show error message."""
        if self.chart_canvas:
            self._show_empty_chart()
        
        self.info_text.setPlainText(f"Error displaying GC Decision data:\n{error_msg}")
    
    def clear_info(self):
        """Clear the panel display."""
        self.current_block_id = None
        self.current_block_data = None
        self.current_figure = None
        
        if self.chart_canvas:
            self._show_empty_chart()
        
        self.info_text.setPlainText("Click on a block with GC scores to see its decision breakdown...")
        
        self.export_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        
        logger.info("Cleared GC Decision panel")
    
    def _on_export_clicked(self):
        """Handle export button click."""
        if not self.current_figure:
            QMessageBox.warning(self, "Export Error", "No chart to export.")
            return
        
        try:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export Spider Chart",
                f"gc_block_{self.current_block_id}_dna.png",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )
            
            if file_path:
                # Determine format from extension
                if file_path.endswith('.pdf'):
                    fmt = 'pdf'
                elif file_path.endswith('.svg'):
                    fmt = 'svg'
                else:
                    fmt = 'png'
                
                self.current_figure.savefig(file_path, format=fmt, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "Export Success", f"Chart exported to:\n{file_path}")
                logger.info(f"Exported GC spider chart to {file_path}")
        
        except Exception as e:
            logger.error(f"Error exporting chart: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export chart:\n{str(e)}")
    
    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.clear_info()
        self.export_requested.emit()

