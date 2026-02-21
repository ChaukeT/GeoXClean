"""
Process History Panel.

Displays the history of processes/tasks that have been executed during the current session.
Automatically shown when loading a saved project to show what analyses have been performed.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit,
    QScrollArea, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont

from .panel_manager import PanelCategory, DockArea
from .base_analysis_panel import BaseAnalysisPanel
from ..core.process_history_tracker import get_process_history_tracker, ProcessExecution

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ProcessHistoryPanel(BaseAnalysisPanel):
    """
    Panel for displaying process execution history.

    Shows:
    - Chronological list of executed processes
    - Process status (success/failed/running)
    - Execution time and parameters
    - Result summaries
    """

    # PanelManager metadata
    PANEL_ID = "ProcessHistoryPanel"
    PANEL_NAME = "Process History"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    task_name = "process_history"

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="process_history")
        self.setWindowTitle("Process History")

        # Process tracker
        self.process_tracker = get_process_history_tracker()

        # Auto-refresh timer for running processes
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_history)
        self.refresh_timer.setInterval(1000)  # Refresh every second

        self._setup_ui()
        self.refresh_history()
        logger.info("Initialized Process History panel")



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
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header with summary
        header_group = QGroupBox("Session Summary")
        header_layout = QVBoxLayout(header_group)

        self.summary_label = QLabel("No processes executed yet")
        self.summary_label.setWordWrap(True)
        header_layout.addWidget(self.summary_label)

        layout.addWidget(header_group)

        # Process history table
        table_group = QGroupBox("Process History")
        table_layout = QVBoxLayout(table_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Task", "Status", "Time", "Duration", "Summary"
        ])

        # Configure table
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Task
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Status
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Time
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Duration
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # Summary

        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        table_layout.addWidget(self.history_table)

        # Details area
        details_group = QGroupBox("Process Details")
        details_layout = QVBoxLayout(details_group)

        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)

        # Connect table selection to details display
        self.history_table.itemSelectionChanged.connect(self._show_selected_details)

        layout.addWidget(table_group)
        layout.addWidget(details_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_history)
        button_layout.addWidget(self.refresh_button)

        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self._clear_history)
        button_layout.addWidget(self.clear_button)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._export_history)
        button_layout.addWidget(self.export_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def refresh_history(self):
        """Refresh the process history display."""
        try:
            history = self.process_tracker.get_history(limit=100)  # Show last 100 processes

            # Update summary
            stats = self.process_tracker.get_summary_stats()
            if stats['total_processes'] > 0:
                summary_text = (
                    f"Total processes: {stats['total_processes']} "
                    f"(✓ {stats['successful']}, ✗ {stats['failed']}, ⟳ {stats['running']})\n"
                    f"Success rate: {stats['success_rate']:.1f}%\n"
                    f"Total runtime: {stats['total_runtime_seconds']:.1f} seconds"
                )
                if stats['most_recent']:
                    summary_text += f"\nMost recent: {stats['most_recent']}"
            else:
                summary_text = "No processes executed yet"

            self.summary_label.setText(summary_text)

            # Update table
            self.history_table.setRowCount(len(history))

            for row, execution in enumerate(reversed(history)):  # Most recent first
                # Task name
                task_item = QTableWidgetItem(execution.task_name.replace('_', ' ').title())
                self.history_table.setItem(row, 0, task_item)

                # Status with color
                status_item = QTableWidgetItem()
                if execution.status == "success":
                    status_item.setText("✓ Success")
                    status_item.setBackground(QColor("#d4edda"))  # Light green
                elif execution.status == "failed":
                    status_item.setText("✗ Failed")
                    status_item.setBackground(QColor("#f8d7da"))  # Light red
                else:  # running
                    status_item.setText("⟳ Running")
                    status_item.setBackground(QColor("#fff3cd"))  # Light yellow

                self.history_table.setItem(row, 1, status_item)

                # Time
                time_str = execution.timestamp.strftime("%H:%M:%S")
                time_item = QTableWidgetItem(time_str)
                self.history_table.setItem(row, 2, time_item)

                # Duration
                if execution.duration_seconds is not None:
                    duration_str = f"{execution.duration_seconds:.1f}s"
                else:
                    duration_str = "-"
                duration_item = QTableWidgetItem(duration_str)
                self.history_table.setItem(row, 3, duration_item)

                # Summary
                summary_text = execution.result_summary or "No summary"
                summary_item = QTableWidgetItem(summary_text)
                summary_item.setToolTip(summary_text)  # Show full text on hover
                self.history_table.setItem(row, 4, summary_item)

            # Auto-scroll to top (most recent)
            if len(history) > 0:
                self.history_table.scrollToTop()

        except Exception as e:
            logger.error(f"Error refreshing process history: {e}")
            self.summary_label.setText(f"Error loading history: {e}")

    def _show_selected_details(self):
        """Show detailed information for selected process."""
        try:
            selected_rows = set()
            for item in self.history_table.selectedItems():
                selected_rows.add(item.row())

            if not selected_rows:
                self.details_text.clear()
                return

            # Get the first selected row
            row = next(iter(selected_rows))

            # Convert back to history index (table is reversed)
            history = self.process_tracker.get_history(limit=100)
            if row < len(history):
                execution = history[-(row + 1)]  # Reverse the reversal

                details = []
                details.append(f"Task: {execution.task_name}")
                details.append(f"Status: {execution.status}")
                details.append(f"Started: {execution.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                if execution.duration_seconds is not None:
                    details.append(f"Duration: {execution.duration_seconds:.2f} seconds")

                if execution.result_summary:
                    details.append(f"Result: {execution.result_summary}")

                if execution.error_message:
                    details.append(f"Error: {execution.error_message}")

                if execution.parameters:
                    details.append("Parameters:")
                    for key, value in execution.parameters.items():
                        details.append(f"  {key}: {value}")

                self.details_text.setPlainText("\n".join(details))
            else:
                self.details_text.setPlainText("No details available")

        except Exception as e:
            logger.error(f"Error showing process details: {e}")
            self.details_text.setPlainText(f"Error loading details: {e}")

    def _clear_history(self):
        """Clear the process history."""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear the process history?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.process_tracker.clear_history()
                self.refresh_history()
                logger.info("Process history cleared by user")
            except Exception as e:
                logger.error(f"Error clearing history: {e}")
                QMessageBox.warning(self, "Error", f"Failed to clear history:\n{e}")

    def _export_history(self):
        """Export process history to a file."""
        try:
            from pathlib import Path
            from PyQt6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Process History", "",
                "JSON files (*.json);;Text files (*.txt)"
            )

            if not filename:
                return

            history = self.process_tracker.get_history()
            stats = self.process_tracker.get_summary_stats()

            if filename.endswith('.json'):
                import json
                export_data = {
                    "summary": stats,
                    "processes": [proc.to_dict() for proc in history]
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                # Text format
                lines = []
                lines.append("PROCESS HISTORY EXPORT")
                lines.append("=" * 50)
                lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("")

                lines.append("SUMMARY:")
                lines.append(f"  Total processes: {stats['total_processes']}")
                lines.append(f"  Successful: {stats['successful']}")
                lines.append(f"  Failed: {stats['failed']}")
                lines.append(f"  Running: {stats['running']}")
                lines.append(".1f")
                lines.append(".1f")
                lines.append("")

                lines.append("PROCESS DETAILS:")
                lines.append("-" * 50)

                for proc in reversed(history):  # Most recent first
                    lines.append(f"Task: {proc.task_name}")
                    lines.append(f"Status: {proc.status}")
                    lines.append(f"Time: {proc.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    if proc.duration_seconds:
                        lines.append(".2f")
                    if proc.result_summary:
                        lines.append(f"Result: {proc.result_summary}")
                    if proc.error_message:
                        lines.append(f"Error: {proc.error_message}")
                    lines.append("")

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

            QMessageBox.information(self, "Export Complete", f"History exported to:\n{filename}")

        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            QMessageBox.warning(self, "Export Error", f"Failed to export history:\n{e}")

    def showEvent(self, event):
        """Start refresh timer when panel becomes visible."""
        super().showEvent(event)
        self.refresh_timer.start()

    def hideEvent(self, event):
        """Stop refresh timer when panel becomes hidden."""
        super().hideEvent(event)
        self.refresh_timer.stop()
