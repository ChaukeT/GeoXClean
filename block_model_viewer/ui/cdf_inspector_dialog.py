"""
CDF Inspector Dialog

UI component for inspecting Indicator Kriging CDF at individual blocks.
Shows probabilities, computed statistics, and CDF plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QTextEdit, QPushButton, QSplitter, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..geostats.indicator_kriging import _correct_single_row


class CDFInspectorDialog(QDialog):
    """
    Dialog for inspecting IK CDF at a specific block location.
    """

    def __init__(self, ik_result: dict, block_coords: tuple, parent=None):
        """
        Args:
            ik_result: Indicator Kriging result dict
            block_coords: (i, j, k) block indices
            parent: Parent widget
        """
        super().__init__(parent)
        self.ik_result = ik_result
        self.block_coords = block_coords
        self.thresholds = ik_result['thresholds']
        self.probabilities = ik_result['probabilities']

        self.setWindowTitle(f"IK CDF Inspector - Block {block_coords}")
        self.resize(800, 600)

        self._extract_block_data()
        self._setup_ui()

    def _extract_block_data(self):
        """Extract data for the specific block."""
        i, j, k = self.block_coords
        self.block_probs = self.probabilities[i, j, k, :]

        # Get grid coordinates
        grid_x, grid_y, grid_z = self.ik_result['grid_x'], self.ik_result['grid_y'], self.ik_result['grid_z']
        self.coords = (grid_x[i,j,k], grid_y[i,j,k], grid_z[i,j,k])

        # Compute corrected CDF and statistics
        corrected_probs = self.block_probs.copy()
        mean_val, median_val = _correct_single_row(corrected_probs, self.thresholds, len(self.thresholds))

        self.corrected_probs = corrected_probs
        self.mean_val = mean_val
        self.median_val = median_val

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        # Header info
        header_group = QGroupBox("Block Information")
        header_layout = QFormLayout(header_group)

        coord_text = ".2f"
        header_layout.addRow("Coordinates (X,Y,Z):", QLabel(coord_text))
        header_layout.addRow("Grid Indices (I,J,K):", QLabel(f"{self.block_coords}"))
        header_layout.addRow("Number of Thresholds:", QLabel(str(len(self.thresholds))))

        layout.addWidget(header_group)

        # Splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - CDF plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.addWidget(QLabel("CDF Plot:"))
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self._create_cdf_plot()

        # Right side - Data table and statistics
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        # Statistics summary
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)

        stats_layout.addRow("E-Type Mean:", QLabel(f"{self.mean_val:.4f}" if not np.isnan(self.mean_val) else "N/A"))
        stats_layout.addRow("Median (P50):", QLabel(f"{self.median_val:.4f}" if not np.isnan(self.median_val) else "N/A"))

        # Add percentile information
        if len(self.thresholds) > 0:
            stats_layout.addRow("Min Threshold:", QLabel(f"{self.thresholds.min():.3f}"))
            stats_layout.addRow("Max Threshold:", QLabel(f"{self.thresholds.max():.3f}"))
            stats_layout.addRow("Threshold Range:", QLabel(f"{self.thresholds.max() - self.thresholds.min():.3f}"))

        data_layout.addWidget(stats_group)

        # Probabilities table
        table_group = QGroupBox("Probabilities P(Z ≤ t)")
        table_layout = QVBoxLayout(table_group)

        self.prob_table = QTableWidget()
        self.prob_table.setColumnCount(4)
        self.prob_table.setHorizontalHeaderLabels(["Threshold", "Raw Prob", "Corrected Prob", "Difference"])
        self.prob_table.setRowCount(len(self.thresholds))

        for row, thresh in enumerate(self.thresholds):
            # Threshold
            self.prob_table.setItem(row, 0, QTableWidgetItem(f"{thresh:.4f}"))

            # Raw probability
            raw_prob = self.block_probs[row]
            self.prob_table.setItem(row, 1, QTableWidgetItem(f"{raw_prob:.4f}" if not np.isnan(raw_prob) else "NaN"))

            # Corrected probability
            corr_prob = self.corrected_probs[row]
            self.prob_table.setItem(row, 2, QTableWidgetItem(f"{corr_prob:.4f}" if not np.isnan(corr_prob) else "NaN"))

            # Difference
            if not np.isnan(raw_prob) and not np.isnan(corr_prob):
                diff = corr_prob - raw_prob
                self.prob_table.setItem(row, 3, QTableWidgetItem(f"{diff:.4f}"))
            else:
                self.prob_table.setItem(row, 3, QTableWidgetItem("N/A"))

        self.prob_table.resizeColumnsToContents()
        table_layout.addWidget(self.prob_table)

        data_layout.addWidget(table_group)

        # CDF consistency check
        consistency_group = QGroupBox("CDF Consistency Check")
        consistency_layout = QVBoxLayout(consistency_group)

        # Check monotonicity
        is_monotonic = all(self.corrected_probs[i] <= self.corrected_probs[i+1]
                          for i in range(len(self.corrected_probs)-1))

        # Check bounds
        in_bounds = all(0.0 <= p <= 1.0 for p in self.corrected_probs if not np.isnan(p))

        consistency_text = []
        if is_monotonic:
            consistency_text.append("✓ Monotonic (probabilities increase with threshold)")
        else:
            consistency_text.append("✗ Non-monotonic - order relation violation")

        if in_bounds:
            consistency_text.append("✓ All probabilities in [0,1] range")
        else:
            consistency_text.append("✗ Probabilities outside [0,1] range")

        # Check for extreme values
        extreme_count = sum(1 for p in self.corrected_probs if not np.isnan(p) and (p < 0.01 or p > 0.99))
        if extreme_count > 0:
            consistency_text.append(f"⚠ {extreme_count} extreme probability values (<0.01 or >0.99)")

        consistency_label = QLabel("\\n".join(consistency_text))
        consistency_label.setWordWrap(True)
        consistency_layout.addWidget(consistency_label)

        data_layout.addWidget(consistency_group)

        data_layout.addStretch()

        # Add widgets to splitter
        splitter.addWidget(plot_widget)
        splitter.addWidget(data_widget)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_btn = QPushButton("Export CDF Data")
        export_btn.clicked.connect(self._export_cdf_data)
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _create_cdf_plot(self):
        """Create the CDF plot."""
        self.figure.clear()

        # Create subplots
        ax1 = self.figure.add_subplot(111)

        # Plot CDF
        valid_mask = ~np.isnan(self.corrected_probs)
        if np.any(valid_mask):
            valid_thresholds = self.thresholds[valid_mask]
            valid_probs = self.corrected_probs[valid_mask]

            # Main CDF line
            ax1.plot(valid_thresholds, valid_probs, 'b-o', linewidth=2, markersize=4, label='Corrected CDF')

            # Add raw probabilities if different
            if not np.array_equal(self.block_probs, self.corrected_probs):
                raw_valid = self.block_probs[valid_mask]
                ax1.plot(valid_thresholds, raw_valid, 'r--s', linewidth=1, markersize=3, label='Raw Probabilities', alpha=0.7)

            # Add median line
            if not np.isnan(self.median_val):
                ax1.axvline(x=self.median_val, color='green', linestyle='--', alpha=0.7,
                           label=f'Median = {self.median_val:.3f}')

            # Add mean line
            if not np.isnan(self.mean_val):
                ax1.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='P=0.5')

        ax1.set_xlabel('Grade Threshold')
        ax1.set_ylabel('Probability P(Z ≤ t)')
        ax1.set_title('Indicator Kriging CDF')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Set reasonable axis limits
        if len(valid_thresholds) > 0:
            x_margin = (valid_thresholds[-1] - valid_thresholds[0]) * 0.1
            ax1.set_xlim(valid_thresholds[0] - x_margin, valid_thresholds[-1] + x_margin)
            ax1.set_ylim(-0.05, 1.05)

        self.figure.tight_layout()
        self.canvas.draw()

    def _export_cdf_data(self):
        """Export CDF data to clipboard or file."""
        try:
            # Create export data
            export_lines = [
                f"IK CDF Inspector - Block {self.block_coords}",
                f"Coordinates: {self.coords}",
                "",
                "Threshold\tRaw_Prob\tCorrected_Prob\tDifference",
            ]

            for i, thresh in enumerate(self.thresholds):
                raw_prob = self.block_probs[i]
                corr_prob = self.corrected_probs[i]

                raw_str = ".4f" if not np.isnan(raw_prob) else "NaN"
                corr_str = ".4f" if not np.isnan(corr_prob) else "NaN"

                if not np.isnan(raw_prob) and not np.isnan(corr_prob):
                    diff_str = ".4f"
                else:
                    diff_str = "N/A"

                export_lines.append(f"{thresh:.4f}\t{raw_str}\t{corr_str}\t{diff_str}")

            export_lines.extend([
                "",
                f"E-Type Mean: {self.mean_val:.4f}" if not np.isnan(self.mean_val) else "E-Type Mean: N/A",
                f"Median (P50): {self.median_val:.4f}" if not np.isnan(self.median_val) else "Median (P50): N/A"
            ])

            export_text = "\\n".join(export_lines)

            # Copy to clipboard (would need clipboard access)
            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(export_text)

            # Show success message
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Complete",
                                  "CDF data copied to clipboard!")

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", f"Failed to export data: {str(e)}")


def show_cdf_inspector(ik_result: dict, block_coords: tuple, parent=None):
    """
    Convenience function to show CDF inspector dialog.

    Args:
        ik_result: IK result dict
        block_coords: (i, j, k) block coordinates
        parent: Parent widget
    """
    dialog = CDFInspectorDialog(ik_result, block_coords, parent)
    dialog.exec()
