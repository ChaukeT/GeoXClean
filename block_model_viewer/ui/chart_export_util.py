"""
Chart Export Utility

Provides functionality for exporting individual charts from the IRR Analysis
to various formats (PNG, PDF, SVG) with customizable options.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)


class ChartExportDialog(QDialog):
    """Dialog for configuring chart export options."""
    
    def __init__(self, chart_name: str, parent=None):
        super().__init__(parent)
        self.chart_name = chart_name
        self.setWindowTitle(f"Export Chart: {chart_name}")
        self.resize(400, 250)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the export dialog UI."""
        layout = QVBoxLayout(self)
        
        # Format selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG (Recommended)", "PDF (Vector)", "SVG (Vector)", "EPS (Publication)"])
        format_layout.addWidget(QLabel("Select format:"))
        format_layout.addWidget(self.format_combo)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # Options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()
        
        self.include_title_check = QCheckBox("Include chart title")
        self.include_title_check.setChecked(True)
        options_layout.addWidget(self.include_title_check)
        
        self.include_legend_check = QCheckBox("Include legend")
        self.include_legend_check.setChecked(True)
        options_layout.addWidget(self.include_legend_check)
        
        self.transparent_bg_check = QCheckBox("Transparent background")
        self.transparent_bg_check.setChecked(False)
        options_layout.addWidget(self.transparent_bg_check)
        
        self.high_dpi_check = QCheckBox("High DPI (300 dpi)")
        self.high_dpi_check.setChecked(True)
        options_layout.addWidget(self.high_dpi_check)
        
        self.timestamp_check = QCheckBox("Add timestamp to filename")
        self.timestamp_check.setChecked(False)
        options_layout.addWidget(self.timestamp_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        export_btn.setDefault(True)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        
        layout.addLayout(button_layout)
    
    def get_format(self) -> str:
        """Get the selected format extension."""
        format_map = {
            "PNG (Recommended)": "png",
            "PDF (Vector)": "pdf",
            "SVG (Vector)": "svg",
            "EPS (Publication)": "eps"
        }
        return format_map[self.format_combo.currentText()]
    
    def get_options(self) -> Dict:
        """Get the export options as a dictionary."""
        return {
            'include_title': self.include_title_check.isChecked(),
            'include_legend': self.include_legend_check.isChecked(),
            'transparent': self.transparent_bg_check.isChecked(),
            'dpi': 300 if self.high_dpi_check.isChecked() else 150,
            'add_timestamp': self.timestamp_check.isChecked()
        }


def save_figure(fig: Figure, path: str, dpi: int = 300, **kwargs):
    """
    Unified figure saving function with error handling and directory creation.
    
    This is the standard way to save all plots/screenshots.
    
    Args:
        fig: Matplotlib Figure object
        path: Full file path (will create directories if needed)
        dpi: DPI for raster formats (default: 300)
        **kwargs: Additional matplotlib savefig options
    
    Returns:
        True if successful, False otherwise
    """
    from pathlib import Path
    
    path = Path(path)
    
    # Create parent directories if they don't exist
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory for {path}: {e}")
        return False
    
    # Determine format from extension
    format_map = {
        '.png': 'png',
        '.pdf': 'pdf',
        '.svg': 'svg',
        '.eps': 'eps',
        '.jpg': 'jpg',
        '.jpeg': 'jpeg'
    }
    
    suffix = path.suffix.lower()
    format = format_map.get(suffix, 'png')
    
    # Default savefig options
    savefig_kwargs = {
        'format': format,
        'dpi': dpi,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white',
        'edgecolor': 'none',
        **kwargs
    }
    
    try:
        fig.savefig(str(path), **savefig_kwargs)
        logger.info(f"Saved figure to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save figure to {path}: {e}", exc_info=True)
        return False


def export_chart(fig: Figure, filename: str, format: str = 'png', **kwargs):
    """
    Export a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure to export
        filename: Target filename (without extension)
        format: Export format ('png', 'pdf', 'svg', 'eps')
        **kwargs: Additional options:
            - dpi: DPI for raster formats (default: 300)
            - transparent: Transparent background (default: False)
            - bbox_inches: Bounding box setting (default: 'tight')
            - pad_inches: Padding (default: 0.1)
    """
    dpi = kwargs.get('dpi', 300)
    transparent = kwargs.get('transparent', False)
    bbox_inches = kwargs.get('bbox_inches', 'tight')
    pad_inches = kwargs.get('pad_inches', 0.1)
    
    full_filename = f"{filename}.{format}"
    
    try:
        fig.savefig(
            full_filename,
            format=format,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor='white' if not transparent else 'none',
            edgecolor='none'
        )
        logger.info(f"Exported chart to {full_filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to export chart: {e}", exc_info=True)
        return False


class ChartWidget(QWidget):
    """
    Widget containing a single chart with an export button.
    """
    
    def __init__(self, figure: Figure, chart_name: str, chart_id: str, parent=None):
        super().__init__(parent)
        self.figure = figure
        self.chart_name = chart_name
        self.chart_id = chart_id
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the chart widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Canvas
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Export button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.export_btn = QPushButton(f"📊 Export Chart")
        self.export_btn.clicked.connect(self._export_chart)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
    
    def _export_chart(self):
        """Handle chart export."""
        # Show export dialog
        dialog = ChartExportDialog(self.chart_name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # Get format and options
        format = dialog.get_format()
        options = dialog.get_options()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if options['add_timestamp'] else ""
        base_filename = f"{self.chart_id}"
        if timestamp:
            base_filename += f"_{timestamp}"
        
        # File dialog
        file_filter = f"{format.upper()} Files (*.{format})"
        default_filename = f"{base_filename}.{format}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {self.chart_name}",
            default_filename,
            file_filter
        )
        
        if not filename:
            return
        
        # Remove extension if present
        filename = str(Path(filename).with_suffix(''))
        
        # Apply options to figure if needed
        if not options['include_title']:
            # Temporarily remove title
            original_titles = []
            for ax in self.figure.axes:
                original_titles.append(ax.get_title())
                ax.set_title('')
        
        if not options['include_legend']:
            # Temporarily remove legends
            original_legends = []
            for ax in self.figure.axes:
                legend = ax.get_legend()
                original_legends.append(legend)
                if legend:
                    legend.remove()
        
        # Export
        success = export_chart(
            self.figure,
            filename,
            format,
            dpi=options['dpi'],
            transparent=options['transparent']
        )
        
        # Restore title and legend
        if not options['include_title']:
            for ax, title in zip(self.figure.axes, original_titles):
                ax.set_title(title)
        
        if not options['include_legend']:
            for ax, legend in zip(self.figure.axes, original_legends):
                if legend:
                    ax.legend()
        
        self.canvas.draw()
        
        # Show result
        if success:
            QMessageBox.information(
                self,
                "Export Successful",
                f"Chart exported successfully to:\n{filename}.{format}"
            )
        else:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export chart to:\n{filename}.{format}"
            )


def create_individual_chart(data, chart_type: str, chart_id: str, title: str, **kwargs):
    """
    Create an individual chart figure from data.
    
    Args:
        data: Data to plot
        chart_type: Type of chart ('line', 'bar', 'hist', etc.)
        chart_id: Unique identifier for the chart
        title: Chart title
        **kwargs: Additional plotting parameters
    
    Returns:
        Figure object
    """
    from matplotlib.figure import Figure
    
    fig = Figure(figsize=kwargs.get('figsize', (8, 6)))
    ax = fig.add_subplot(111)
    
    if chart_type == 'line':
        x = kwargs.get('x', range(len(data)))
        ax.plot(x, data, **kwargs.get('plot_kwargs', {}))
    elif chart_type == 'bar':
        x = kwargs.get('x', range(len(data)))
        ax.bar(x, data, **kwargs.get('bar_kwargs', {}))
    elif chart_type == 'hist':
        ax.hist(data, **kwargs.get('hist_kwargs', {}))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig


def batch_export_charts(figures: Dict[str, Figure], output_dir: str, format: str = 'png', **kwargs):
    """
    Export multiple charts to a directory.
    
    Args:
        figures: Dictionary of {chart_id: figure}
        output_dir: Output directory path
        format: Export format
        **kwargs: Additional export options
    
    Returns:
        List of successfully exported filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    failed_files = []
    
    for chart_id, fig in figures.items():
        filename = output_path / chart_id
        success = export_chart(fig, str(filename), format, **kwargs)
        
        if success:
            exported_files.append(f"{filename}.{format}")
        else:
            failed_files.append(chart_id)
    
    logger.info(f"Batch export: {len(exported_files)} successful, {len(failed_files)} failed")
    
    return exported_files, failed_files

