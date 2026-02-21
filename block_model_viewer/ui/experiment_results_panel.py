"""
Experiment Results Panel

Display and analyze experiment results.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QMessageBox,
    QFileDialog, QTextEdit, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
import numpy as np

from .base_analysis_panel import BaseAnalysisPanel
from ..research.reporting import (
    experiment_to_dataframe, to_excel, to_csv, to_latex_table
)

logger = logging.getLogger(__name__)


class ExperimentResultsPanel(BaseAnalysisPanel):
    """Panel for displaying experiment results."""
    
    task_name = "experiment_results"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="experiment_results")
        self.setWindowTitle("Experiment Results")
        self.setup_ui()
        self.current_results: Optional[Dict[str, Any]] = None
        logger.info("Initialized Experiment Results panel")
    
    def setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Summary
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(100)
        summary_layout.addWidget(self.summary_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Results table
        table_group = QGroupBox("Results")
        table_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        table_layout.addWidget(self.results_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Export actions
        export_layout = QHBoxLayout()
        
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(lambda: self._export_results('csv'))
        export_layout.addWidget(export_csv_btn)
        
        export_excel_btn = QPushButton("Export Excel")
        export_excel_btn.clicked.connect(lambda: self._export_results('excel'))
        export_layout.addWidget(export_excel_btn)
        
        export_latex_btn = QPushButton("Copy LaTeX Table")
        export_latex_btn.clicked.connect(self._export_latex)
        export_layout.addWidget(export_latex_btn)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
    
    def load_results(self, results_dict: Dict[str, Any]):
        """Load experiment results."""
        self.current_results = results_dict
        
        # Update summary
        metadata = results_dict.get('metadata', {})
        n_instances = metadata.get('n_instances', 0)
        n_successful = metadata.get('n_successful', 0)
        definition_name = metadata.get('definition_name', 'Unknown')
        
        self.summary_text.clear()
        self.summary_text.append(f"Experiment: {definition_name}\n")
        self.summary_text.append(f"Total Instances: {n_instances}\n")
        self.summary_text.append(f"Successful: {n_successful}\n")
        self.summary_text.append(f"Metrics: {', '.join(results_dict.get('metrics', []))}")
        
        # Convert to DataFrame and display
        try:
            from ..research.runner import ExperimentRunResult
            
            run_result = ExperimentRunResult(
                definition_id=results_dict.get('definition_id', ''),
                results=results_dict.get('results', []),
                metrics=results_dict.get('metrics', []),
                metadata=metadata
            )
            
            df = experiment_to_dataframe(run_result)
            self._populate_table(df)
        except Exception as e:
            logger.error(f"Failed to load results: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load results:\n{e}")
    
    def _populate_table(self, df):
        """Populate results table from DataFrame."""
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                value = row[col]
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        item = QTableWidgetItem("---")
                    else:
                        item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()
    
    def _export_results(self, format: str):
        """Export results to CSV or Excel."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No results to export")
            return
        
        if format == 'csv':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "experiment_results.csv", "CSV Files (*.csv)"
            )
            if file_path:
                try:
                    from ..research.runner import ExperimentRunResult
                    from pathlib import Path
                    
                    run_result = ExperimentRunResult(
                        definition_id=self.current_results.get('definition_id', ''),
                        results=self.current_results.get('results', []),
                        metrics=self.current_results.get('metrics', []),
                        metadata=self.current_results.get('metadata', {})
                    )
                    
                    to_csv(run_result, Path(file_path))
                    QMessageBox.information(self, "Success", f"Exported to {file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to export:\n{e}")
        
        elif format == 'excel':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Excel", "experiment_results.xlsx", "Excel Files (*.xlsx)"
            )
            if file_path:
                try:
                    from ..research.runner import ExperimentRunResult
                    from pathlib import Path
                    
                    run_result = ExperimentRunResult(
                        definition_id=self.current_results.get('definition_id', ''),
                        results=self.current_results.get('results', []),
                        metrics=self.current_results.get('metrics', []),
                        metadata=self.current_results.get('metadata', {})
                    )
                    
                    to_excel(run_result, Path(file_path))
                    QMessageBox.information(self, "Success", f"Exported to {file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to export:\n{e}")
    
    def _export_latex(self):
        """Export LaTeX table to clipboard."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No results to export")
            return
        
        try:
            from ..research.runner import ExperimentRunResult
            from PyQt6.QtWidgets import QApplication
            
            run_result = ExperimentRunResult(
                definition_id=self.current_results.get('definition_id', ''),
                results=self.current_results.get('results', []),
                metrics=self.current_results.get('metrics', []),
                metadata=self.current_results.get('metadata', {})
            )
            
            latex_str = to_latex_table(run_result)
            
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(latex_str)
            
            QMessageBox.information(self, "Success", "LaTeX table copied to clipboard")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export LaTeX:\n{e}")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

