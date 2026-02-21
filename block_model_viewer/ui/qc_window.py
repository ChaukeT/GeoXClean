"""
GeoX QC Window - Professional Drillhole Data Validation Interface

Leapfrog-style QC window with:
- Left: Issue tree (table → hole → issues)
- Right: Manual editor (assays / lith / survey tabs)
- Top: Toolbar for Ignore, Auto-Fix, Re-QC, Undo/Redo, Save
"""

from __future__ import annotations

import sys
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)
from PyQt6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSize,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QFont,
    QBrush,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QTableView,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QToolBar,
    QToolButton,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QLineEdit,
    QHeaderView,
    QMenu,
    QStatusBar,
    QStyle,
    QLabel,
    QSizePolicy,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QCheckBox,
    QInputDialog,
    QTextEdit,
    QButtonGroup,
    QRadioButton,
    QGroupBox,
    QPushButton,
)

# Import backend engines
from ..drillholes.drillhole_validation import (
    ValidationConfig,
    ValidationViolation,
    run_drillhole_validation,
)
from ..drillholes.drillhole_autofix import run_drillhole_autofix, AutoFixResult
from ..drillholes.drillhole_ignore import apply_ignore_rules, IgnoreRule, IgnoreResult
from ..drillholes.drillhole_manual_edit import ManualEditEngine
from ..drillholes.drillhole_audit_trail import AuditTrail
from ..drillholes.audit_export import export_to_excel, export_to_pdf, export_to_csv, OPENPYXL_AVAILABLE, REPORTLAB_AVAILABLE
from ..drillholes.database import DrillholeDatabaseManager
from ..drillholes.datamodel import DrillholeDatabase

# Import modern status bar
from .drillhole_status_bar import DrillholeProcessStatusBar, StatusLevel, ProcessStage, create_progress_callback
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

# --- EXTERNAL PANELS (Preserved) ---
# Geology decision panel removed


def _get_qc_window_stylesheet() -> str:
    """Get the stylesheet for the QC Window (includes extra QMainWindow styling)."""
    colors = get_theme_colors()
    base_style = get_analysis_panel_stylesheet()
    # Add QMainWindow-specific and QC Window-specific styles
    extra_style = f"""
        QMainWindow {{
            background-color: {colors.PANEL_BG};
        }}
        QToolBar {{
            background-color: {colors.CARD_BG};
            border-bottom: 1px solid {colors.BORDER};
            spacing: 5px;
            padding: 5px;
        }}
        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 4px;
        }}
        QToolButton:hover {{
            background-color: {colors.CARD_HOVER};
            border: 1px solid {colors.BORDER_LIGHT};
        }}
        QToolButton:pressed {{
            background-color: {colors.ELEVATED_BG};
        }}
        QSplitter::handle {{
            background-color: {colors.BORDER};
            width: 2px;
        }}
        QTreeWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {colors.DIVIDER};
        }}
        QTreeWidget::item:hover {{
            background-color: {colors.CARD_HOVER};
        }}
        QTableView {{
            background-color: {colors.ELEVATED_BG};
            gridline-color: {colors.DIVIDER};
            border: 1px solid {colors.BORDER};
            selection-background-color: {colors.ACCENT_PRIMARY};
            selection-color: white;
        }}
    """
    return base_style + extra_style


# =========================================================
# Pandas Table Model
# =========================================================

class PandasTableModel(QAbstractTableModel):
    """Simple pandas-backed model. All edits go via ManualEditEngine."""

    def __init__(self, df: pd.DataFrame, table_name: str, editor_engine: ManualEditEngine, audit_trail=None):
        super().__init__()
        self._df_full = df  # Keep full data
        self._df = df  # Current view (may be filtered)
        self._table_name = table_name
        self._editor = editor_engine
        self._audit_trail = audit_trail
        self._show_problems_only = False
        self._problem_indices = set()

    def update_df(self, df: pd.DataFrame):
        """Update the underlying dataframe and refresh the model."""
        self.beginResetModel()
        self._df_full = df.copy()  # Bug #1 fix: use copy to avoid shared reference
        if self._show_problems_only:
            self._apply_problem_filter()
        else:
            self._df = self._df_full  # Reference to our copy is safe
        self.endResetModel()

    def set_problem_indices(self, indices: set):
        """Set the indices of rows that have problems."""
        self._problem_indices = set(indices) if indices else set()  # Ensure it's a set copy

    def set_show_problems_only(self, show: bool):
        """Toggle showing only problem rows."""
        self.beginResetModel()
        self._show_problems_only = show
        if show:
            self._apply_problem_filter()
        else:
            self._df = self._df_full  # Reference to full data
        self.endResetModel()

    def _apply_problem_filter(self):
        """Filter to show only problem rows."""
        if self._df_full is None or self._df_full.empty:
            self._df = pd.DataFrame()  # Bug #2 fix: handle empty dataframe
            return
        if self._problem_indices:
            # Filter to only rows in problem_indices
            # Bug #19 fix: ensure indices are compatible types
            valid_indices = {idx for idx in self._problem_indices if idx in self._df_full.index}
            if valid_indices:
                mask = self._df_full.index.isin(valid_indices)
                self._df = self._df_full[mask]
            else:
                self._df = self._df_full  # No valid problem indices, show all
        else:
            # If no problem indices defined, show rows with any missing values
            filtered = self._df_full[self._df_full.isna().any(axis=1)]
            self._df = filtered if not filtered.empty else self._df_full  # Show all if no missing

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        value = self._df.iat[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            return "" if pd.isna(value) else str(value)

        # Highlight missing values with orange/red background
        if role == Qt.ItemDataRole.BackgroundRole:
            if pd.isna(value) or value == "":
                return QBrush(QColor("#ff6b6b"))  # Red for missing
            # Check if this row has any validation error
            row_idx = self._df.index[index.row()]
            if row_idx in self._problem_indices:
                return QBrush(QColor("#5a4a32"))  # Subtle tan/brown for problem row

        # White text on colored cells for readability
        if role == Qt.ItemDataRole.ForegroundRole:
            if pd.isna(value) or value == "":
                return QBrush(QColor(f"{ModernColors.TEXT_PRIMARY}"))

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.ItemIsEnabled
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False

        row = self._df.index[index.row()]
        col_name = self._df.columns[index.column()]

        # Optimistic update check
        current_val = self._df.iat[index.row(), index.column()]
        if str(current_val) == str(value):
            return False

        # Route the change through the manual edit engine
        self._editor.edit_cell(
            table=self._table_name,
            row_index=row,
            column=col_name,
            new_value=value,
            reason="manual-edit-ui",
        )
        
        # Track in audit trail if available
        if self._audit_trail:
            df_table = getattr(self._editor, self._table_name, None)
            # Bug #7 fix: check df_table is not None and has hole_id column
            hole_id = ""
            if df_table is not None and not df_table.empty:
                if "hole_id" in df_table.columns and row in df_table.index:
                    hole_id = str(df_table.at[row, "hole_id"])
            self._audit_trail.add_manual_edit(
                table=self._table_name,
                hole_id=hole_id,
                row_index=row,
                column=col_name,
                old_value=current_val,
                new_value=value,
                reason="manual-edit-ui",
                user=self._editor.user,
            )

        # Pull updated DF from engine
        tables = self._editor.get_tables()
        self._df = tables[self._table_name]

        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
        return True


# =========================================================
# Expert Decision Panel
# =========================================================

class ExpertDecisionPanel(QWidget):
    """
    Panel for handling violations that require expert geological judgment.

    Supports different fix options based on violation type:
    - GAP violations: extend_previous, pull_next, split_difference, ignore
    - OVERLAP violations: truncate_previous, delay_next, ignore
    - MISSING_FIELDS: manual_entry, ignore
    """

    # Signal emitted when a fix is applied
    fix_applied = pyqtSignal(dict)  # Emits fix_details dict
    # Signal emitted when user cancels
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._violation = None
        self._editor_engine = None
        self._cfg = None
        self._gap_info = None  # Store gap/overlap details
        self._setup_ui()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        self.title_label = QLabel("Expert Decision Required")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #e74c3c;")
        layout.addWidget(self.title_label)

        # Violation details group
        details_group = QGroupBox("Violation Details")
        details_layout = QFormLayout(details_group)

        self.lbl_table = QLabel("-")
        self.lbl_hole = QLabel("-")
        self.lbl_rule = QLabel("-")
        self.lbl_severity = QLabel("-")
        self.lbl_message = QLabel("-")
        self.lbl_message.setWordWrap(True)

        details_layout.addRow("Table:", self.lbl_table)
        details_layout.addRow("Hole ID:", self.lbl_hole)
        details_layout.addRow("Rule:", self.lbl_rule)
        details_layout.addRow("Severity:", self.lbl_severity)
        details_layout.addRow("Message:", self.lbl_message)

        layout.addWidget(details_group)

        # Context info group (shows interval details for gaps/overlaps)
        self.context_group = QGroupBox("Context")
        context_layout = QFormLayout(self.context_group)

        self.lbl_prev_interval = QLabel("-")
        self.lbl_curr_interval = QLabel("-")
        self.lbl_gap_size = QLabel("-")

        context_layout.addRow("Previous Interval:", self.lbl_prev_interval)
        context_layout.addRow("Current Interval:", self.lbl_curr_interval)
        context_layout.addRow("Gap/Overlap Size:", self.lbl_gap_size)

        layout.addWidget(self.context_group)

        # Fix options group
        self.options_group = QGroupBox("Fix Options")
        self.options_layout = QVBoxLayout(self.options_group)

        # Radio buttons for fix options (will be populated dynamically)
        self.option_buttons = QButtonGroup(self)
        self.rb_extend_prev = QRadioButton("Extend previous interval (set to_depth to close gap)")
        self.rb_pull_next = QRadioButton("Pull next interval (set from_depth to close gap)")
        self.rb_split_diff = QRadioButton("Split the difference (adjust both intervals)")
        self.rb_truncate_prev = QRadioButton("Truncate previous interval (fix overlap)")
        self.rb_delay_next = QRadioButton("Delay next interval (fix overlap)")
        self.rb_manual_entry = QRadioButton("Manual entry (edit cells directly)")
        self.rb_ignore = QRadioButton("Ignore this issue (requires justification)")

        self.option_buttons.addButton(self.rb_extend_prev, 1)
        self.option_buttons.addButton(self.rb_pull_next, 2)
        self.option_buttons.addButton(self.rb_split_diff, 3)
        self.option_buttons.addButton(self.rb_truncate_prev, 4)
        self.option_buttons.addButton(self.rb_delay_next, 5)
        self.option_buttons.addButton(self.rb_manual_entry, 6)
        self.option_buttons.addButton(self.rb_ignore, 7)

        self.options_layout.addWidget(self.rb_extend_prev)
        self.options_layout.addWidget(self.rb_pull_next)
        self.options_layout.addWidget(self.rb_split_diff)
        self.options_layout.addWidget(self.rb_truncate_prev)
        self.options_layout.addWidget(self.rb_delay_next)
        self.options_layout.addWidget(self.rb_manual_entry)
        self.options_layout.addWidget(self.rb_ignore)

        layout.addWidget(self.options_group)

        # Justification field (for ignore option)
        self.justification_group = QGroupBox("Justification (required for ignore)")
        just_layout = QVBoxLayout(self.justification_group)
        self.justification_edit = QTextEdit()
        self.justification_edit.setPlaceholderText("Enter geological/technical justification for ignoring this issue...")
        self.justification_edit.setMaximumHeight(80)
        just_layout.addWidget(self.justification_edit)
        layout.addWidget(self.justification_group)

        # Connect ignore radio to enable/disable justification
        self.rb_ignore.toggled.connect(self._on_ignore_toggled)
        self.justification_group.setEnabled(False)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_apply = QPushButton("Apply Fix")
        self.btn_apply.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px 16px;")
        self.btn_apply.clicked.connect(self._on_apply_clicked)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("padding: 8px 16px;")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        layout.addStretch()

    def _on_ignore_toggled(self, checked: bool):
        """Enable/disable justification field based on ignore selection."""
        self.justification_group.setEnabled(checked)
        if checked:
            self.justification_edit.setFocus()

    def load_violation(self, violation: ValidationViolation, editor_engine, cfg):
        """Load a violation into the panel for expert decision."""
        self._violation = violation
        self._editor_engine = editor_engine
        self._cfg = cfg

        # Update violation details
        self.lbl_table.setText(violation.table)
        self.lbl_hole.setText(violation.hole_id)
        self.lbl_rule.setText(violation.rule_code)
        self.lbl_severity.setText(violation.severity)
        self.lbl_message.setText(violation.message)

        # Set severity color
        if violation.severity == "ERROR":
            self.lbl_severity.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #e74c3c;")
        else:
            self.lbl_severity.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #e67e22;")

        # Configure options based on violation type
        self._configure_options_for_violation(violation)

        # Load context info
        self._load_context_info(violation)

    def _configure_options_for_violation(self, v: ValidationViolation):
        """Show/hide options based on violation type."""
        rule = v.rule_code.upper()

        # Hide all first
        self.rb_extend_prev.setVisible(False)
        self.rb_pull_next.setVisible(False)
        self.rb_split_diff.setVisible(False)
        self.rb_truncate_prev.setVisible(False)
        self.rb_delay_next.setVisible(False)
        self.rb_manual_entry.setVisible(False)
        self.rb_ignore.setVisible(True)  # Always available

        if "GAP" in rule:
            # Gap violations - show gap fix options
            self.rb_extend_prev.setVisible(True)
            self.rb_pull_next.setVisible(True)
            self.rb_split_diff.setVisible(True)
            self.rb_extend_prev.setChecked(True)
            self.context_group.setVisible(True)
        elif "OVERLAP" in rule:
            # Overlap violations
            self.rb_truncate_prev.setVisible(True)
            self.rb_delay_next.setVisible(True)
            self.rb_truncate_prev.setChecked(True)
            self.context_group.setVisible(True)
        elif "MISSING" in rule:
            # Missing field violations
            self.rb_manual_entry.setVisible(True)
            self.rb_manual_entry.setChecked(True)
            self.context_group.setVisible(False)
        else:
            # Other violations - just ignore option
            self.rb_ignore.setChecked(True)
            self.context_group.setVisible(False)

    def _load_context_info(self, v: ValidationViolation):
        """Load context information for the violation."""
        self._gap_info = None

        if not self._editor_engine:
            return

        tables = self._editor_engine.get_tables()
        df = tables.get(v.table)

        if df is None or df.empty:
            return

        # Find column names
        hole_col = None
        for col in ["hole_id", "holeid", "HOLE_ID", "HoleID"]:
            if col in df.columns:
                hole_col = col
                break

        from_col = None
        for col in ["from_depth", "depth_from", "from", "FROM"]:
            if col in df.columns:
                from_col = col
                break

        to_col = None
        for col in ["to_depth", "depth_to", "to", "TO"]:
            if col in df.columns:
                to_col = col
                break

        if not hole_col or not from_col or not to_col:
            return

        # Get hole data sorted by depth
        hole_data = df[df[hole_col] == v.hole_id].sort_values(from_col)

        if v.row_index not in hole_data.index:
            return

        # Get current row position
        idx_list = list(hole_data.index)
        try:
            pos = idx_list.index(v.row_index)
        except ValueError:
            return

        curr_row = hole_data.loc[v.row_index]
        curr_from = curr_row.get(from_col)
        curr_to = curr_row.get(to_col)

        # Handle NaN values for display
        curr_from_str = f"{curr_from:.2f}" if pd.notna(curr_from) else "N/A"
        curr_to_str = f"{curr_to:.2f}" if pd.notna(curr_to) else "N/A"
        self.lbl_curr_interval.setText(f"{curr_from_str} - {curr_to_str}")

        # Get previous interval if exists
        if pos > 0:
            prev_idx = idx_list[pos - 1]
            prev_row = hole_data.loc[prev_idx]
            prev_from = prev_row.get(from_col)
            prev_to = prev_row.get(to_col)
            # Handle NaN values for display
            prev_from_str = f"{prev_from:.2f}" if pd.notna(prev_from) else "N/A"
            prev_to_str = f"{prev_to:.2f}" if pd.notna(prev_to) else "N/A"
            self.lbl_prev_interval.setText(f"{prev_from_str} - {prev_to_str}")

            # Calculate gap/overlap - check for NaN values first
            if pd.isna(curr_from) or pd.isna(prev_to):
                self.lbl_gap_size.setText("Cannot calculate (missing depth)")
                self.lbl_gap_size.setStyleSheet("color: #888;")
            else:
                try:
                    gap = float(curr_from) - float(prev_to)
                    if gap > 0:
                        self.lbl_gap_size.setText(f"{gap:.3f}m GAP")
                        self.lbl_gap_size.setStyleSheet("color: #e67e22;")
                    elif gap < 0:
                        self.lbl_gap_size.setText(f"{abs(gap):.3f}m OVERLAP")
                        self.lbl_gap_size.setStyleSheet("color: #e74c3c;")
                    else:
                        self.lbl_gap_size.setText("0m (continuous)")
                        self.lbl_gap_size.setStyleSheet("color: #27ae60;")

                    # Store gap info for fixes
                    self._gap_info = {
                        "prev_idx": prev_idx,
                        "curr_idx": v.row_index,
                        "prev_to": float(prev_to),
                        "curr_from": float(curr_from),
                        "gap": gap,
                        "from_col": from_col,
                        "to_col": to_col,
                    }
                except (ValueError, TypeError):
                    self.lbl_gap_size.setText("Unable to calculate")
        else:
            self.lbl_prev_interval.setText("(first interval)")
            self.lbl_gap_size.setText("N/A")

    def _on_apply_clicked(self):
        """Apply the selected fix."""
        if not self._violation:
            return

        # Get selected option
        selected = self.option_buttons.checkedButton()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a fix option.")
            return

        # Map radio buttons to option IDs
        option_map = {
            self.rb_extend_prev: "extend_previous",
            self.rb_pull_next: "pull_next",
            self.rb_split_diff: "split_difference",
            self.rb_truncate_prev: "truncate_previous",
            self.rb_delay_next: "delay_next",
            self.rb_manual_entry: "manual_entry",
            self.rb_ignore: "ignore",
        }

        option_id = option_map.get(selected, "ignore")

        # Validate justification for ignore
        if option_id == "ignore":
            justification = self.justification_edit.toPlainText().strip()
            if not justification:
                QMessageBox.warning(
                    self, "Justification Required",
                    "Please provide a justification for ignoring this issue.\n\n"
                    "This is required for audit compliance (JORC/SAMREC)."
                )
                return

        # Build fix details
        fix_details = {
            "violation": self._violation,
            "option_id": option_id,
            "justification": self.justification_edit.toPlainText().strip() if option_id == "ignore" else "",
        }

        # Add gap/overlap specific info
        if self._gap_info:
            fix_details["gap_info"] = self._gap_info
            fix_details["gap_start"] = self._gap_info.get("prev_to")
            fix_details["gap_end"] = self._gap_info.get("curr_from")

        # Emit signal
        self.fix_applied.emit(fix_details)

    def _on_cancel_clicked(self):
        """Cancel and hide the panel."""
        # Clear the panel
        self._violation = None
        self._gap_info = None
        self.justification_edit.clear()
        # Emit cancelled signal so parent can switch tabs
        self.cancelled.emit()


# =========================================================
# Find & Replace Dialog
# =========================================================

class FindReplaceDialog(QDialog):
    """Dialog for find and replace operations."""
    
    def __init__(self, parent=None, find_only: bool = True):
        super().__init__(parent)
        self.find_only = find_only
        self.setWindowTitle("Find" if find_only else "Find & Replace")
        self.setModal(True)
        self.resize(400, 150 if find_only else 200)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        # Find text
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Enter text to find...")
        form.addRow("Find:", self.find_edit)
        
        # Replace text (only if not find-only)
        if not find_only:
            self.replace_edit = QLineEdit()
            self.replace_edit.setPlaceholderText("Enter replacement text...")
            form.addRow("Replace with:", self.replace_edit)
        
        layout.addLayout(form)
        
        # Options
        options_layout = QVBoxLayout()
        self.case_sensitive_check = QCheckBox("Case sensitive")
        self.case_sensitive_check.setChecked(False)
        options_layout.addWidget(self.case_sensitive_check)
        
        if not find_only:
            self.match_whole_cell_check = QCheckBox("Match whole cell only")
            self.match_whole_cell_check.setChecked(False)
            options_layout.addWidget(self.match_whole_cell_check)
        
        layout.addLayout(options_layout)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Set focus to find field
        self.find_edit.setFocus()
        
    @property
    def find_text(self) -> str:
        return self.find_edit.text()
    
    @property
    def replace_text(self) -> str:
        # Bug #5 fix: defensive check for replace_edit existence
        if hasattr(self, 'replace_edit') and not self.find_only:
            return self.replace_edit.text()
        return ""

    @property
    def case_sensitive(self) -> bool:
        return self.case_sensitive_check.isChecked()

    @property
    def match_whole_cell(self) -> bool:
        # Bug #5 fix: defensive check for match_whole_cell_check existence
        if hasattr(self, 'match_whole_cell_check') and not self.find_only:
            return self.match_whole_cell_check.isChecked()
        return False
    
    def accept(self):
        if not self.find_text:
            QMessageBox.warning(self, "Invalid Input", "Please enter text to find.")
            return
        super().accept()


# =========================================================
# QC Window
# =========================================================

@dataclass
class QCPipelineState:
    """State container for QC pipeline configuration."""
    cfg: ValidationConfig
    ignore_rules: List[IgnoreRule]
    ignore_all_minor: bool = False
    ignore_all_warnings: bool = False


class QCWindow(QMainWindow):
    """
    GeoX QCWindow – Leapfrog style:
    - left: tree of issues (table → hole → issues)
    - right: manual editor (assays / lith / survey tabs)
    - top: toolbar for Ignore, Auto-Fix, Re-QC, Undo/Redo, Save
    """

    def __init__(
        self,
        collars: pd.DataFrame,
        surveys: pd.DataFrame,
        assays: pd.DataFrame,
        lithology: pd.DataFrame,
        cfg: Optional[ValidationConfig] = None,
        user: str = "GEOLOGIST",
        parent=None,
        controller=None,
    ):
        super().__init__(parent)

        # Setup main window props
        self.setWindowTitle("GeoX Drillhole QC")
        self.resize(1600, 900)
        self.setStyleSheet(_get_qc_window_stylesheet())

        # Data & Logic
        self.cfg = cfg or ValidationConfig()
        self.pipeline_state = QCPipelineState(cfg=self.cfg, ignore_rules=[])
        self.user = user
        self.controller = controller
        self.editor_engine = ManualEditEngine(collars, surveys, assays, lithology, user=self.user)
        
        # Audit Trail for SAMREC/JORC compliance
        self.audit_trail = AuditTrail(project_name="Drillhole QC", user=self.user)

        # State
        self.violations_all: List[ValidationViolation] = []
        self.violations_visible: List[ValidationViolation] = []
        self.violations_ignored: List[ValidationViolation] = []
        
        # Grouping mode: "table", "type", "hole", "severity"
        self.grouping_mode = "table"

        # Icons (Standard Qt icons for portability)
        style = self.style()
        self.icon_error = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        self.icon_warning = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        self.icon_refresh = style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        self.icon_save = style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        self.icon_undo = style.standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        self.icon_redo = style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        self.icon_magic = style.standardIcon(QStyle.StandardPixmap.SP_DialogYesButton)  # Fallback for autofix
        self.icon_table = style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.icon_hole = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)

        self._build_ui()
        self._run_full_qc(initial=True)

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self.setStyleSheet(_get_qc_window_stylesheet())

    # -----------------------------------------------------
    # UI Construction
    # -----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(central)

        # --- Toolbar ---
        self._build_toolbar()

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        main_layout.addWidget(splitter)

        # --- Left Panel: Issues ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 5, 10)

        # Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter issues (e.g. hole ID, rule)...")
        self.search_bar.textChanged.connect(self._filter_issue_tree)
        left_layout.addWidget(self.search_bar)

        # Tree Widget
        self.issue_tree = QTreeWidget()
        self.issue_tree.setHeaderHidden(True)
        self.issue_tree.itemClicked.connect(self.on_issue_clicked)
        self.issue_tree.itemDoubleClicked.connect(self._on_issue_double_clicked)
        self.issue_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.issue_tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        left_layout.addWidget(self.issue_tree)

        splitter.addWidget(left_panel)

        # --- Right Panel: Editor Tabs ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 10, 10, 10)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        tables = self.editor_engine.get_tables()

        # Init Views and Models
        self.assays_view, self.assays_model = self._create_table_view(tables["assays"], "assays")
        self.lith_view, self.lith_model = self._create_table_view(tables["lithology"], "lithology")
        self.survey_view, self.survey_model = self._create_table_view(tables["surveys"], "surveys")
        self.collars_view, self.collars_model = self._create_table_view(tables["collars"], "collars")

        # Add Tabs with Icons
        self.tab_widget.addTab(self.collars_view, self.icon_table, "Collars")
        self.tab_widget.addTab(self.survey_view, self.icon_table, "Surveys")
        self.tab_widget.addTab(self.lith_view, self.icon_table, "Lithology")
        self.tab_widget.addTab(self.assays_view, self.icon_table, "Assays")
        
        # Geologist Decision Panel (for expert fixes)
        self.decision_panel_container = QWidget()
        self.decision_panel_layout = QVBoxLayout(self.decision_panel_container)
        self.decision_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.decision_panel = None  # Will be created when needed
        self.tab_widget.addTab(self.decision_panel_container, self.icon_error, "Expert Decision")

        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)

        # Set Initial Splitter sizes (approx 1/3 left, 2/3 right)
        splitter.setSizes([400, 900])

        # --- Modern Status Bar ---
        self.modern_status_bar = DrillholeProcessStatusBar.create_for_qc(self)
        self.modern_status_bar.cancel_requested.connect(self._on_cancel_operation)
        right_layout.addWidget(self.modern_status_bar)
        
        # Keep basic status bar for compatibility
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)

    def _create_table_view(self, df, name):
        """Create a table view with model for a given dataframe."""
        model = PandasTableModel(df, name, self.editor_engine, audit_trail=self.audit_trail)
        view = QTableView()
        view.setModel(model)

        # UX Settings
        view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        view.setAlternatingRowColors(True)
        view.verticalHeader().setDefaultSectionSize(24)
        view.horizontalHeader().setHighlightSections(False)
        view.horizontalHeader().setStretchLastSection(True)
        
        # Enable editing
        view.setEditTriggers(
            QTableView.EditTrigger.DoubleClicked | 
            QTableView.EditTrigger.SelectedClicked |
            QTableView.EditTrigger.AnyKeyPressed
        )

        # Optimization
        view.setCornerButtonEnabled(False)

        return view, model

    def _build_toolbar(self):
        """Build the main toolbar with dropdown menus for compact layout."""
        tb = QToolBar("QC Tools")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # ===== ALWAYS VISIBLE - Core Actions =====

        # Main Actions
        act_rerun = QAction(self.icon_refresh, "Re-Run QC", self)
        act_rerun.setShortcut("F5")
        act_rerun.setToolTip("Re-run QC validation (F5)")
        act_rerun.triggered.connect(lambda: self._run_full_qc())
        tb.addAction(act_rerun)

        act_autofix = QAction(self.icon_magic, "Auto-Fix", self)
        act_autofix.setToolTip("Apply automatic fixes for common errors")
        act_autofix.triggered.connect(self._run_autofix_safe)
        tb.addAction(act_autofix)

        self.act_apply_registry = QAction(self.icon_save, "Apply to Registry", self)
        self.act_apply_registry.setToolTip("Apply cleaned data to DataRegistry for Compositing, Kriging, etc.")
        self.act_apply_registry.triggered.connect(self._apply_to_registry)
        tb.addAction(self.act_apply_registry)

        tb.addSeparator()

        # Undo/Redo - Always visible
        act_undo = QAction(self.icon_undo, "Undo", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.setToolTip("Undo last edit (Ctrl+Z)")
        act_undo.triggered.connect(self._undo_edit)
        tb.addAction(act_undo)

        act_redo = QAction(self.icon_redo, "Redo", self)
        act_redo.setShortcut("Ctrl+Y")
        act_redo.setToolTip("Redo last undone edit (Ctrl+Y)")
        act_redo.triggered.connect(self._redo_edit)
        tb.addAction(act_redo)

        tb.addSeparator()

        # ===== DROPDOWN MENUS - Grouped Actions =====

        # EDIT MENU (Find, Replace, Delete)
        edit_menu = QMenu(self)

        act_find = QAction("Find...", self)
        act_find.setShortcut("Ctrl+F")
        act_find.triggered.connect(self._show_find_dialog)
        edit_menu.addAction(act_find)

        act_replace = QAction("Find && Replace...", self)
        act_replace.setShortcut("Ctrl+H")
        act_replace.triggered.connect(self._show_replace_dialog)
        edit_menu.addAction(act_replace)

        edit_menu.addSeparator()

        act_delete_selected = QAction("Delete Selected Rows", self)
        act_delete_selected.setShortcut("Ctrl+Delete")
        act_delete_selected.setToolTip("Delete selected rows from current table (Ctrl+Del)")
        act_delete_selected.triggered.connect(self._delete_selected_rows)
        edit_menu.addAction(act_delete_selected)

        edit_btn = QToolButton()
        edit_btn.setText("Edit")
        edit_btn.setIcon(self.icon_table)
        edit_btn.setMenu(edit_menu)
        edit_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        edit_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(edit_btn)

        # BATCH OPERATIONS MENU
        batch_menu = QMenu(self)

        act_delete_orphans = QAction("Delete Orphans", self)
        act_delete_orphans.setToolTip("Delete survey/assay/lithology records with no matching collar")
        act_delete_orphans.triggered.connect(self._delete_orphan_records)
        batch_menu.addAction(act_delete_orphans)

        act_close_gaps = QAction("Close All Gaps", self)
        act_close_gaps.setToolTip("Automatically close all interval gaps by adjusting from_depth values")
        act_close_gaps.triggered.connect(self._close_all_gaps)
        batch_menu.addAction(act_close_gaps)

        batch_btn = QToolButton()
        batch_btn.setText("Batch")
        batch_btn.setMenu(batch_menu)
        batch_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        batch_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(batch_btn)

        # FILTER MENU (Toggles)
        filter_menu = QMenu(self)

        self.act_ignore_warn = QAction("Ignore All Warnings", self)
        self.act_ignore_warn.setCheckable(True)
        self.act_ignore_warn.setToolTip("Ignore all WARNING-level issues (won't affect validation status)")
        self.act_ignore_warn.toggled.connect(self._toggle_ignore_warnings)
        filter_menu.addAction(self.act_ignore_warn)

        self.act_ignore_minor = QAction("Ignore All Minor", self)
        self.act_ignore_minor.setCheckable(True)
        self.act_ignore_minor.setToolTip("Ignore all INFO and WARNING level issues (won't affect validation status)")
        self.act_ignore_minor.toggled.connect(self._toggle_ignore_minor)
        filter_menu.addAction(self.act_ignore_minor)

        filter_menu.addSeparator()

        self.act_show_problems = QAction("Show Problem Rows Only", self)
        self.act_show_problems.setCheckable(True)
        self.act_show_problems.setToolTip("Filter tables to show only rows with missing/invalid data")
        self.act_show_problems.toggled.connect(self._toggle_show_problem_rows)
        filter_menu.addAction(self.act_show_problems)

        filter_btn = QToolButton()
        filter_btn.setText("Filter")
        filter_btn.setMenu(filter_menu)
        filter_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        filter_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(filter_btn)

        tb.addSeparator()

        # VIEW - Group by (kept visible as it's frequently used)
        tb.addWidget(QLabel("Group by:"))
        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems([
            "Table",
            "Type",
            "Hole",
            "Severity"
        ])
        self.grouping_combo.setCurrentIndex(0)
        self.grouping_combo.currentIndexChanged.connect(self._on_grouping_changed)
        self.grouping_combo.setToolTip("Change how issues are grouped in the tree")
        tb.addWidget(self.grouping_combo)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tb.addWidget(spacer)

        # EXPORT MENU
        export_menu = QMenu(self)

        act_save_db = QAction(self.icon_save, "Save to Database", self)
        act_save_db.setToolTip("Save cleaned drillhole data to database")
        act_save_db.triggered.connect(self._save_to_database)
        export_menu.addAction(act_save_db)

        act_export_audit = QAction(self.icon_save, "Export Audit Trail", self)
        act_export_audit.setToolTip("Export audit trail for SAMREC/JORC compliance (PDF/Excel/CSV)")
        act_export_audit.triggered.connect(self._export_audit_trail)
        export_menu.addAction(act_export_audit)

        act_save = QAction(self.icon_save, "Export Clean Data", self)
        act_save.setToolTip("Export cleaned drillhole data to CSV")
        act_save.triggered.connect(self._export_clean)
        export_menu.addAction(act_save)

        export_btn = QToolButton()
        export_btn.setText("Export")
        export_btn.setIcon(self.icon_save)
        export_btn.setMenu(export_menu)
        export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addWidget(export_btn)

    # -----------------------------------------------------
    # QC Pipeline Logic
    # -----------------------------------------------------
    
    def _on_cancel_operation(self):
        """Handle cancel request from the modern status bar."""
        # For now, just update the status to show cancellation was requested
        # In the future, this could be enhanced to actually interrupt long operations
        self.modern_status_bar.update_status("Cancellation requested", StatusLevel.WARNING)
        self.status_label.setText("Operation cancelled by user")
        logger.info("User requested operation cancellation")

    def _run_full_qc(self, initial: bool = False):
        """
        Run validation on current tables + apply ignore filters.
        
        SAFETY: This method is wrapped in try-except to ensure validation 
        errors never crash the UI. All errors are captured and displayed 
        as validation violations or error messages.
        
        AUDIT: Stores validation state in DataRegistry for downstream engines
        (compositing, geostatistics) to query before processing.
        """
        # Start modern status bar progress
        self.modern_status_bar.start_process(cancellable=False)
        self.modern_status_bar.update_status("Starting validation process", StatusLevel.INFO)
        self.status_label.setText("Running validation...")
        QApplication.processEvents()  # Force UI update

        try:
            # Step 1: Data Check
            self.modern_status_bar.advance_step()
            self.modern_status_bar.update_status("Validating input data structure")
            QApplication.processEvents()
            
            tables = self.editor_engine.get_tables()

            # Step 2-5: Run validation with progress callback
            self.modern_status_bar.advance_step()
            self.modern_status_bar.update_status("Running comprehensive validation")
            QApplication.processEvents()
            
            # Create progress callback for the validation engine
            def validation_progress(percent: int, message: str):
                self.modern_status_bar.update_status(message)
                QApplication.processEvents()
            
            val_result = run_drillhole_validation(
                collars=tables["collars"],
                surveys=tables["surveys"],
                assays=tables["assays"],
                lithology=tables["lithology"],
                cfg=self.cfg,
                progress_callback=validation_progress,
            )
            
            # Advance through validation steps (the callback handled detailed progress)
            for _ in range(4):  # collar, survey, assay, lithology validation
                self.modern_status_bar.advance_step()
                QApplication.processEvents()

            self.violations_all = val_result.violations
            
            # Log any schema errors for debugging
            if hasattr(val_result, 'schema_errors') and val_result.schema_errors:
                for err in val_result.schema_errors:
                    logger.warning(f"Schema error during validation: {err}")
            
            # Step 6: Cross Validation
            self.modern_status_bar.advance_step()
            self.modern_status_bar.update_status("Checking consistency across tables")
            QApplication.processEvents()
            
            # Store validation state in registry for downstream engines
            self._store_validation_state_in_registry(val_result)

            ignore_result: IgnoreResult = apply_ignore_rules(
                violations=self.violations_all,
                ignore_rules=self.pipeline_state.ignore_rules,
                ignore_all_minor=self.pipeline_state.ignore_all_minor,
                ignore_all_warnings=self.pipeline_state.ignore_all_warnings,
            )

            self.violations_visible = ignore_result.visible
            self.violations_ignored = ignore_result.ignored

            # Log filtering results for debugging
            logger.info(f"After ignore filtering: {len(self.violations_visible)} visible, {len(self.violations_ignored)} ignored")
            logger.info(f"ignore_all_warnings={self.pipeline_state.ignore_all_warnings}, ignore_all_minor={self.pipeline_state.ignore_all_minor}")

            # Step 7: Report Generation
            self.modern_status_bar.advance_step()
            self.modern_status_bar.update_status("Generating validation report")
            QApplication.processEvents()
            
            self._refresh_issue_tree()
            self._refresh_models()

            msg = f"Validation Complete: {len(self.violations_visible)} Issues Found"
            if len(self.violations_ignored) > 0:
                msg += f" ({len(self.violations_ignored)} ignored)"

            # Calculate effective status based on VISIBLE violations (not ignored ones)
            # If all issues are ignored, status should be PASS
            visible_errors = sum(1 for v in self.violations_visible if v.severity == "ERROR")
            visible_warnings = sum(1 for v in self.violations_visible if v.severity == "WARNING")

            if visible_errors > 0:
                effective_status = "FAIL"
            elif visible_warnings > 0:
                effective_status = "WARN"
            else:
                effective_status = "PASS"

            # Add status indicator to message
            msg += f" [Status: {effective_status}]"

            # Finish the modern status bar
            success = effective_status != 'FAIL'
            status_level = StatusLevel.SUCCESS if success else StatusLevel.ERROR
            self.modern_status_bar.finish_process(success=True, message=msg)  # Always show as complete since process finished
            self.modern_status_bar.update_status(msg, status_level)
            self.status_label.setText(msg)

            if not initial and not self.violations_visible:
                QMessageBox.information(self, "QC Passed", "Great job! No visible issues found.")
                
        except Exception as e:
            # CRITICAL: Validation must NEVER crash the UI
            logger.error(f"Critical error during validation: {e}", exc_info=True)
            
            # Create a synthetic violation to show the error
            from ..drillholes.drillhole_validation import ValidationViolation
            error_violation = ValidationViolation(
                table="system",
                rule_code="VALIDATION_CRASH",
                severity="ERROR",
                hole_id="",
                row_index=-1,
                message=f"Validation engine error: {str(e)}. Check logs for details."
            )
            
            self.violations_all = [error_violation]
            self.violations_visible = [error_violation]
            self.violations_ignored = []
            
            self._refresh_issue_tree()
            self._refresh_models()
            
            # Update both status displays for error
            error_msg = f"Validation Error: {str(e)[:50]}..."
            self.modern_status_bar.update_status(error_msg, StatusLevel.ERROR)
            self.modern_status_bar.set_step_error(str(e))
            self.status_label.setText(error_msg)
            
            if not initial:
                QMessageBox.critical(
                    self, 
                    "Validation Error",
                    f"An error occurred during validation:\n\n{str(e)}\n\n"
                    "The validation engine has recovered. Please check your data "
                    "and try again. See logs for full error details."
                )

    def _run_autofix_safe(self):
        """Run the safe auto-fix engine and rebuild state."""
        # Create a simple status bar for autofix operations
        self.modern_status_bar.update_status("Running automatic fixes", StatusLevel.INFO)
        self.status_label.setText("Running auto-fix...")
        QApplication.processEvents()

        try:
            tables = self.editor_engine.get_tables()

            # Log violations before autofix
            logger.info(f"Auto-fix starting with {len(self.violations_all)} violations")

            af: AutoFixResult = run_drillhole_autofix(
                collars=tables["collars"],
                surveys=tables["surveys"],
                assays=tables["assays"],
                lithology=tables["lithology"],
                cfg=self.cfg,
            )

            # Log what was fixed
            logger.info(f"Auto-fix completed: {len(af.fixes)} fixes applied")
            logger.info(f"Violations before: {len(af.violations_before)}, after: {len(af.violations_after)}")

            # Record all auto-fixes in audit trail
            for fix in af.fixes:
                for col_name, col_data in fix.columns.items():
                    self.audit_trail.add_autofix(
                        table=fix.table,
                        rule_code=fix.rule_code,
                        hole_id=fix.hole_id,
                        row_index=fix.row_index,
                        column=col_name,
                        old_value=col_data["old"],
                        new_value=col_data["new"],
                        reason=fix.reason,
                        confidence=fix.confidence,
                    )

            self.editor_engine = ManualEditEngine(
                collars=af.collars,
                surveys=af.surveys,
                assays=af.assays,
                lithology=af.lithology,
                user=self.user,
            )

            # Re-hook editors
            self.assays_model._editor = self.editor_engine
            self.lith_model._editor = self.editor_engine
            self.survey_model._editor = self.editor_engine
            self.collars_model._editor = self.editor_engine

            self._run_full_qc()

            # Build informative message
            if len(af.fixes) > 0:
                fix_msg = f"Auto-Fix applied {len(af.fixes)} fixes."
                remaining = len(self.violations_visible)
                if remaining > 0:
                    fix_msg += f" {remaining} issues remain (require manual edit)."
                self.modern_status_bar.update_status(fix_msg, StatusLevel.SUCCESS)
                self.status_label.setText(fix_msg)
            else:
                # No fixes were applied - explain why
                remaining = len(self.violations_visible)
                if remaining > 0:
                    # Count issue types that autofix cannot handle
                    missing_count = sum(1 for v in self.violations_visible if "MISSING" in v.rule_code)
                    fix_msg = f"Auto-Fix: No automatic fixes available. {remaining} issues require manual editing."
                    if missing_count > 0:
                        fix_msg = f"Auto-Fix: {missing_count} missing value issues require manual editing."
                    self.modern_status_bar.update_status(fix_msg, StatusLevel.WARNING)
                    self.status_label.setText(fix_msg)
                else:
                    fix_msg = "Auto-Fix: No issues to fix."
                    self.modern_status_bar.update_status(fix_msg, StatusLevel.SUCCESS)
                    self.status_label.setText(fix_msg)

        except Exception as e:
            error_msg = f"Auto-fix failed: {str(e)}"
            self.modern_status_bar.update_status(error_msg, StatusLevel.ERROR)
            self.status_label.setText(error_msg)
            logger.error(f"Auto-fix operation failed: {e}", exc_info=True)

    def _refresh_models(self):
        """Updates all table models with latest data from engine."""
        tables = self.editor_engine.get_tables()
        self.assays_model.update_df(tables["assays"])
        self.lith_model.update_df(tables["lithology"])
        self.survey_model.update_df(tables["surveys"])
        self.collars_model.update_df(tables["collars"])

    # -----------------------------------------------------
    # Batch Operations
    # -----------------------------------------------------

    def _delete_orphan_records(self):
        """Delete records from surveys/assays/lithology that have no matching collar."""
        tables = self.editor_engine.get_tables()
        collars = tables["collars"]

        # Get valid hole_ids from collars
        collar_hole_col = None
        for col in ["hole_id", "holeid", "HOLE_ID", "HoleID"]:
            if col in collars.columns:
                collar_hole_col = col
                break

        if not collar_hole_col:
            QMessageBox.warning(self, "Error", "Cannot find hole_id column in collars table.")
            return

        valid_hole_ids = set(collars[collar_hole_col].dropna().astype(str).str.strip().str.upper())

        deleted_counts = {"surveys": 0, "assays": 0, "lithology": 0}

        for table_name in ["surveys", "assays", "lithology"]:
            df = tables[table_name]
            if df is None or df.empty:
                continue

            # Find hole_id column
            hole_col = None
            for col in ["hole_id", "holeid", "HOLE_ID", "HoleID"]:
                if col in df.columns:
                    hole_col = col
                    break

            if not hole_col:
                continue

            # Find orphan rows (hole_id not in valid_hole_ids or is empty)
            df_hole_ids = df[hole_col].fillna("").astype(str).str.strip().str.upper()
            orphan_mask = ~df_hole_ids.isin(valid_hole_ids) | (df_hole_ids == "")
            orphan_count = orphan_mask.sum()

            if orphan_count > 0:
                # Remove orphan rows
                tables[table_name] = df[~orphan_mask].reset_index(drop=True)
                deleted_counts[table_name] = orphan_count

        total_deleted = sum(deleted_counts.values())

        if total_deleted == 0:
            QMessageBox.information(self, "No Orphans", "No orphan records found. All records have matching collars.")
            return

        # Confirm deletion
        msg = f"Found {total_deleted} orphan records:\n"
        for table, count in deleted_counts.items():
            if count > 0:
                msg += f"  - {table}: {count} rows\n"
        msg += "\nDelete these records?"

        reply = QMessageBox.question(self, "Delete Orphans?", msg,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # Update editor engine with cleaned data
            self.editor_engine = ManualEditEngine(
                collars=tables["collars"],
                surveys=tables["surveys"],
                assays=tables["assays"],
                lithology=tables["lithology"],
                user=self.user,
            )

            # Re-hook models
            self.assays_model._editor = self.editor_engine
            self.lith_model._editor = self.editor_engine
            self.survey_model._editor = self.editor_engine
            self.collars_model._editor = self.editor_engine

            self._refresh_models()
            self._run_full_qc()

            self.status_label.setText(f"Deleted {total_deleted} orphan records.")
            logger.info(f"Deleted orphan records: {deleted_counts}")

    def _close_all_gaps(self):
        """Close all interval gaps by snapping from_depth to previous to_depth."""
        tables = self.editor_engine.get_tables()
        fixes_applied = 0

        for table_name in ["assays", "lithology"]:
            df = tables[table_name]
            if df is None or df.empty:
                continue

            # Find column names
            hole_col = None
            for col in ["hole_id", "holeid", "HOLE_ID", "HoleID"]:
                if col in df.columns:
                    hole_col = col
                    break

            from_col = None
            for col in ["from_depth", "depth_from", "from", "FROM", "MFROM"]:
                if col in df.columns:
                    from_col = col
                    break

            to_col = None
            for col in ["to_depth", "depth_to", "to", "TO", "MTO"]:
                if col in df.columns:
                    to_col = col
                    break

            if not hole_col or not from_col or not to_col:
                continue

            # Process each hole
            df = df.sort_values([hole_col, from_col]).copy()

            for hole_id, group in df.groupby(hole_col, sort=False):
                indices = list(group.index)
                prev_to = None

                for idx in indices:
                    from_val = df.at[idx, from_col]
                    to_val = df.at[idx, to_col]

                    # Bug #9 fix: skip NaN values without updating prev_to
                    if pd.isna(from_val) or pd.isna(to_val):
                        continue  # Don't update prev_to with NaN

                    try:
                        from_val = float(from_val)
                        to_val = float(to_val)
                    except (ValueError, TypeError):
                        continue  # Bug #9 fix: Don't update prev_to with invalid values

                    # Check for gap
                    if prev_to is not None:
                        try:
                            prev_to_float = float(prev_to)
                            gap = from_val - prev_to_float

                            # Close any gap (not just small ones)
                            if gap > 0:
                                df.at[idx, from_col] = prev_to_float
                                fixes_applied += 1
                        except (ValueError, TypeError):
                            pass

                    prev_to = to_val

            tables[table_name] = df

        if fixes_applied == 0:
            QMessageBox.information(self, "No Gaps", "No interval gaps found to close.")
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Close Gaps?",
            f"Found {fixes_applied} gaps to close.\n\n"
            "This will adjust from_depth values to match the previous to_depth.\n\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.editor_engine = ManualEditEngine(
                collars=tables["collars"],
                surveys=tables["surveys"],
                assays=tables["assays"],
                lithology=tables["lithology"],
                user=self.user,
            )

            self.assays_model._editor = self.editor_engine
            self.lith_model._editor = self.editor_engine
            self.survey_model._editor = self.editor_engine
            self.collars_model._editor = self.editor_engine

            self._refresh_models()
            self._run_full_qc()

            self.status_label.setText(f"Closed {fixes_applied} interval gaps.")
            logger.info(f"Closed {fixes_applied} interval gaps")

    def _delete_selected_rows(self):
        """Delete selected rows from the current table."""
        # Get current table view
        current_widget = self.tab_widget.currentWidget()

        view_map = {
            self.collars_view: ("collars", self.collars_model),
            self.survey_view: ("surveys", self.survey_model),
            self.assays_view: ("assays", self.assays_model),
            self.lith_view: ("lithology", self.lith_model),
        }

        table_name = None
        model = None
        view = None

        for v, (name, m) in view_map.items():
            if v == current_widget:
                table_name = name
                model = m
                view = v
                break

        if not table_name or not view:
            QMessageBox.warning(self, "Error", "Please select a table tab first.")
            return

        # Get selected rows
        selection = view.selectionModel().selectedRows()
        if not selection:
            QMessageBox.warning(self, "No Selection", "Please select rows to delete first.")
            return

        # Get the actual dataframe indices
        # Bug #10 fix: add bounds checking for filtered dataframes
        row_indices = []
        for idx in selection:
            view_row = idx.row()
            if view_row < len(model._df.index):
                df_row = model._df.index[view_row]
                row_indices.append(df_row)
            else:
                logger.warning(f"Selection row {view_row} out of bounds for filtered dataframe")

        # Confirm
        reply = QMessageBox.question(
            self, "Delete Rows?",
            f"Delete {len(row_indices)} selected rows from {table_name}?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            tables = self.editor_engine.get_tables()
            df = tables[table_name]

            # Delete the rows
            df = df.drop(index=row_indices).reset_index(drop=True)
            tables[table_name] = df

            # Update engine
            self.editor_engine = ManualEditEngine(
                collars=tables["collars"],
                surveys=tables["surveys"],
                assays=tables["assays"],
                lithology=tables["lithology"],
                user=self.user,
            )

            self.assays_model._editor = self.editor_engine
            self.lith_model._editor = self.editor_engine
            self.survey_model._editor = self.editor_engine
            self.collars_model._editor = self.editor_engine

            self._refresh_models()
            self._run_full_qc()

            self.status_label.setText(f"Deleted {len(row_indices)} rows from {table_name}.")
            logger.info(f"Deleted {len(row_indices)} rows from {table_name}")

    # -----------------------------------------------------
    # Tree & Search Logic
    # -----------------------------------------------------

    def _on_grouping_changed(self, index: int):
        """Handle grouping mode change."""
        modes = ["table", "type", "hole", "severity"]
        self.grouping_mode = modes[index]
        self._refresh_issue_tree()

    def _filter_issue_tree(self, text: str):
        """Simple text filtering for the tree."""
        self._refresh_issue_tree(filter_text=text)

    def _refresh_issue_tree(self, filter_text: str = ""):
        """Refresh the issue tree with current violations."""
        # Save expanded state before clearing
        expanded_items = set()
        def save_expanded(item):
            if item.isExpanded():
                # Use a unique identifier for all item types
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if data:
                    if isinstance(data, tuple) and len(data) == 2:
                        item_type, payload = data
                        if item_type == "table":
                            expanded_items.add(("table", payload))
                        elif item_type == "hole":
                            # Handle both tuple (table_name, hole_id) and string (hole_id only)
                            if isinstance(payload, tuple):
                                table_name, hole_id = payload
                                expanded_items.add(("hole", table_name, hole_id))
                            else:
                                expanded_items.add(("hole", payload))
                        elif item_type == "type":
                            expanded_items.add(("type", payload))
                        elif item_type == "severity":
                            expanded_items.add(("severity", payload))
            for i in range(item.childCount()):
                save_expanded(item.child(i))
        
        for i in range(self.issue_tree.topLevelItemCount()):
            save_expanded(self.issue_tree.topLevelItem(i))
        
        self.issue_tree.clear()
        filter_text = filter_text.lower()

        # Filter violations first
        filtered_violations = []
        for v in self.violations_visible:
            text_repr = f"{v.table} {v.hole_id} {v.rule_code} {v.message}".lower()
            if filter_text and filter_text not in text_repr:
                continue
            filtered_violations.append(v)

        # Build Tree based on grouping mode
        font_bold = QFont()
        font_bold.setBold(True)

        if self.grouping_mode == "table":
            # Group: Table -> Hole -> Issue (default)
            by_table: Dict[str, Dict[str, List[ValidationViolation]]] = {}
            for v in filtered_violations:
                t = v.table
                h = v.hole_id or "Unknown Hole"
                by_table.setdefault(t, {}).setdefault(h, []).append(v)

            for table_name, holes in by_table.items():
                count_issues = sum(len(x) for x in holes.values())
                table_item = QTreeWidgetItem([f"{table_name.upper()}  [{count_issues}]"])
                table_item.setIcon(0, self.icon_table)
                table_item.setFont(0, font_bold)
                table_item.setData(0, Qt.ItemDataRole.UserRole, ("table", table_name))
                should_expand_table = ("table", table_name) in expanded_items
                table_item.setExpanded(should_expand_table)
                self.issue_tree.addTopLevelItem(table_item)

                for hole_id, vlist in holes.items():
                    hole_item = QTreeWidgetItem([f"{hole_id}  ({len(vlist)})"])
                    hole_item.setIcon(0, self.icon_hole)
                    hole_item.setData(0, Qt.ItemDataRole.UserRole, ("hole", (table_name, hole_id)))
                    should_expand_hole = ("hole", table_name, hole_id) in expanded_items
                    hole_item.setExpanded(should_expand_hole)
                    table_item.addChild(hole_item)

                    for v in vlist:
                        self._add_issue_item(hole_item, v)

                if ("table", table_name) not in expanded_items:
                    table_item.setExpanded(True)

        elif self.grouping_mode == "type":
            # Group: Type (Rule Code) -> Table -> Hole
            by_type: Dict[str, Dict[str, Dict[str, List[ValidationViolation]]]] = {}
            for v in filtered_violations:
                rule = v.rule_code
                t = v.table
                h = v.hole_id or "Unknown Hole"
                by_type.setdefault(rule, {}).setdefault(t, {}).setdefault(h, []).append(v)

            for rule_code, by_table in by_type.items():
                count_issues = sum(sum(len(vlist) for vlist in holes.values()) for holes in by_table.values())
                type_item = QTreeWidgetItem([f"{rule_code}  [{count_issues}]"])
                # Determine icon based on severity of violations under this type
                all_violations = [v for holes in by_table.values() for vlist in holes.values() for v in vlist]
                has_errors = any(v.severity == "ERROR" for v in all_violations)
                type_item.setIcon(0, self.icon_error if has_errors else self.icon_warning)
                type_item.setFont(0, font_bold)
                type_item.setData(0, Qt.ItemDataRole.UserRole, ("type", rule_code))
                should_expand_type = ("type", rule_code) in expanded_items
                type_item.setExpanded(should_expand_type)
                self.issue_tree.addTopLevelItem(type_item)

                for table_name, holes in by_table.items():
                    table_item = QTreeWidgetItem([f"{table_name.upper()}  [{sum(len(x) for x in holes.values())}]"])
                    table_item.setIcon(0, self.icon_table)
                    table_item.setData(0, Qt.ItemDataRole.UserRole, ("table", table_name))
                    should_expand_table = ("table", table_name) in expanded_items
                    table_item.setExpanded(should_expand_table)
                    type_item.addChild(table_item)

                    for hole_id, vlist in holes.items():
                        hole_item = QTreeWidgetItem([f"{hole_id}  ({len(vlist)})"])
                        hole_item.setIcon(0, self.icon_hole)
                        hole_item.setData(0, Qt.ItemDataRole.UserRole, ("hole", (table_name, hole_id)))
                        should_expand_hole = ("hole", table_name, hole_id) in expanded_items
                        hole_item.setExpanded(should_expand_hole)
                        table_item.addChild(hole_item)

                        for v in vlist:
                            self._add_issue_item(hole_item, v)

        elif self.grouping_mode == "hole":
            # Group: Hole -> Table -> Issue
            by_hole: Dict[str, Dict[str, List[ValidationViolation]]] = {}
            for v in filtered_violations:
                h = v.hole_id or "Unknown Hole"
                t = v.table
                by_hole.setdefault(h, {}).setdefault(t, []).append(v)

            for hole_id, by_table in by_hole.items():
                count_issues = sum(len(x) for x in by_table.values())
                hole_item = QTreeWidgetItem([f"{hole_id}  [{count_issues}]"])
                hole_item.setIcon(0, self.icon_hole)
                hole_item.setFont(0, font_bold)
                hole_item.setData(0, Qt.ItemDataRole.UserRole, ("hole", hole_id))
                should_expand_hole = ("hole", hole_id) in expanded_items
                hole_item.setExpanded(should_expand_hole)
                self.issue_tree.addTopLevelItem(hole_item)

                for table_name, vlist in by_table.items():
                    table_item = QTreeWidgetItem([f"{table_name.upper()}  [{len(vlist)}]"])
                    table_item.setIcon(0, self.icon_table)
                    table_item.setData(0, Qt.ItemDataRole.UserRole, ("table", table_name))
                    should_expand_table = ("table", table_name) in expanded_items
                    table_item.setExpanded(should_expand_table)
                    hole_item.addChild(table_item)

                    for v in vlist:
                        self._add_issue_item(table_item, v)

        elif self.grouping_mode == "severity":
            # Group: Severity -> Table -> Hole
            by_severity: Dict[str, Dict[str, Dict[str, List[ValidationViolation]]]] = {}
            for v in filtered_violations:
                sev = v.severity
                t = v.table
                h = v.hole_id or "Unknown Hole"
                by_severity.setdefault(sev, {}).setdefault(t, {}).setdefault(h, []).append(v)

            for severity, by_table in by_severity.items():
                count_issues = sum(sum(len(x) for x in holes.values()) for holes in by_table.values())
                sev_item = QTreeWidgetItem([f"{severity}  [{count_issues}]"])
                sev_item.setIcon(0, self.icon_error if severity == "ERROR" else self.icon_warning)
                sev_item.setFont(0, font_bold)
                sev_item.setData(0, Qt.ItemDataRole.UserRole, ("severity", severity))
                should_expand_severity = ("severity", severity) in expanded_items
                sev_item.setExpanded(should_expand_severity)
                self.issue_tree.addTopLevelItem(sev_item)

                for table_name, holes in by_table.items():
                    table_item = QTreeWidgetItem([f"{table_name.upper()}  [{sum(len(x) for x in holes.values())}]"])
                    table_item.setIcon(0, self.icon_table)
                    table_item.setData(0, Qt.ItemDataRole.UserRole, ("table", table_name))
                    should_expand_table = ("table", table_name) in expanded_items
                    table_item.setExpanded(should_expand_table)
                    sev_item.addChild(table_item)

                    for hole_id, vlist in holes.items():
                        hole_item = QTreeWidgetItem([f"{hole_id}  ({len(vlist)})"])
                        hole_item.setIcon(0, self.icon_hole)
                        hole_item.setData(0, Qt.ItemDataRole.UserRole, ("hole", (table_name, hole_id)))
                        should_expand_hole = ("hole", table_name, hole_id) in expanded_items
                        hole_item.setExpanded(should_expand_hole)
                        table_item.addChild(hole_item)

                        for v in vlist:
                            self._add_issue_item(hole_item, v)

        # Only expand all by default if no previous state was saved (first load)
        if filter_text:
            self.issue_tree.expandAll()
        elif not expanded_items:
            # First load - expand all
            self.issue_tree.expandAll()

    def _add_issue_item(self, parent: QTreeWidgetItem, v: ValidationViolation):
        """Helper to add an issue item to the tree."""
        icon = self.icon_error if v.severity == "ERROR" else self.icon_warning
        msg_clean = f"[{v.rule_code}] {v.message}"
        issue_item = QTreeWidgetItem([msg_clean])
        issue_item.setIcon(0, icon)

        if v.severity == "ERROR":
            issue_item.setForeground(0, QBrush(QColor("#ff6b6b")))
        else:
            issue_item.setForeground(0, QBrush(QColor("#e6a84d")))  # Warm amber for warnings

        issue_item.setData(0, Qt.ItemDataRole.UserRole, ("violation", v))
        parent.addChild(issue_item)

    def on_issue_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle clicking on an issue in the tree."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, payload = data

        if kind != "violation":
            return

        v: ValidationViolation = payload

        # Geology decision panel removed - just jump to violation
        self._jump_to_violation(v)

    def _on_issue_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-clicking on an issue to directly edit the cell."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, payload = data

        if kind != "violation":
            return

        v: ValidationViolation = payload
        self._edit_violation_cell(v)

    def _open_expert_decision(self, v: ValidationViolation):
        """Open the Expert Decision panel for a violation."""
        # Create a new ExpertDecisionPanel
        panel = ExpertDecisionPanel(self)

        # Use _show_decision_panel to set it up properly
        self._show_decision_panel(v, panel)

        self.status_label.setText(f"Expert Decision panel opened for: {v.rule_code}")
        logger.info(f"Expert Decision opened for violation: {v.rule_code} on {v.hole_id}")

    def _show_decision_panel(self, v: ValidationViolation, decision_panel):
        """Show the expert decision panel for violations requiring geological judgment."""
        # Clear existing panel
        if self.decision_panel:
            # Disconnect signals before deleting to prevent memory leak
            try:
                self.decision_panel.fix_applied.disconnect(self._on_expert_fix_applied)
            except (TypeError, RuntimeError):
                pass  # Signal was not connected or already disconnected
            try:
                self.decision_panel.cancelled.disconnect(self._on_expert_decision_cancelled)
            except (TypeError, RuntimeError):
                pass
            self.decision_panel.setParent(None)
            self.decision_panel.deleteLater()

        # Create and setup new panel
        self.decision_panel = decision_panel
        # Check if panel has required methods/signals before using
        if hasattr(self.decision_panel, 'load_violation'):
            self.decision_panel.load_violation(v, self.editor_engine, self.cfg)
        if hasattr(self.decision_panel, 'fix_applied'):
            self.decision_panel.fix_applied.connect(self._on_expert_fix_applied)
        if hasattr(self.decision_panel, 'cancelled'):
            self.decision_panel.cancelled.connect(self._on_expert_decision_cancelled)

        # Add to container
        self.decision_panel_layout.addWidget(self.decision_panel)

        # Switch to Expert Decision tab
        self.tab_widget.setCurrentWidget(self.decision_panel_container)

        # Also jump to the violation in the table for context
        self._jump_to_violation(v)

    def _on_expert_decision_cancelled(self):
        """Handle expert decision being cancelled."""
        # Switch back to first tab (Collars)
        self.tab_widget.setCurrentIndex(0)
        self.status_label.setText("Expert decision cancelled")
    
    def _on_expert_fix_applied(self, fix_details: Dict[str, Any]):
        """Handle expert fix being applied."""
        violation = fix_details.get("violation")
        option_id = fix_details.get("option_id")
        
        if not violation or not option_id:
            return
        
        try:
            # Apply the fix based on option
            if option_id == "ignore":
                # Add ignore rule
                # Note: justification is logged but not stored in IgnoreRule (dataclass doesn't support it)
                justification = fix_details.get("justification", "")
                if justification:
                    logger.info(f"Expert ignore with justification: {justification}")
                self.pipeline_state.ignore_rules.append(
                    IgnoreRule(
                        rule_code=violation.rule_code,
                        hole_id=violation.hole_id,
                        row_index=violation.row_index,
                    )
                )
                self.status_label.setText(f"Issue ignored: {violation.rule_code}")
            else:
                # Apply the specific fix (this stores old values in fix_details)
                self._apply_expert_fix(fix_details)
                
                # Track in audit trail (must be after _apply_expert_fix to get old values)
                if self.audit_trail:
                    self._track_expert_fix_in_audit(fix_details)
            
            # Re-run QC to update violations
            self._run_full_qc()
            
            QMessageBox.information(
                self, "Fix Applied",
                f"Expert fix applied: {option_id}\n\nRe-running validation..."
            )
        except Exception as e:
            logger.error(f"Error applying expert fix: {e}", exc_info=True)
            QMessageBox.critical(self, "Fix Failed", f"Failed to apply fix:\n{str(e)}")
    
    def _apply_expert_fix(self, fix_details: Dict[str, Any]):
        """Apply the expert fix based on option."""
        option_id = fix_details.get("option_id")
        violation = fix_details.get("violation")
        gap_info = fix_details.get("gap_info", {})
        tables = self.editor_engine.get_tables()

        # Store old values for audit trail before making changes
        old_values = {}

        df = tables.get(violation.table)
        if df is None or df.empty:
            logger.warning(f"Table {violation.table} not found or empty")
            return

        # Get column names from gap_info or use defaults
        from_col = gap_info.get("from_col", "from_depth")
        to_col = gap_info.get("to_col", "to_depth")

        if option_id == "extend_previous":
            # Extend previous interval's to_depth to close gap
            prev_idx = gap_info.get("prev_idx")
            gap_end = fix_details.get("gap_end") or gap_info.get("curr_from")
            if prev_idx is not None and prev_idx in df.index and gap_end is not None:
                old_values[prev_idx] = {
                    "column": to_col,
                    "old_value": df.at[prev_idx, to_col] if to_col in df.columns else None,
                    "new_value": gap_end,
                }
                self.editor_engine.edit_cell(
                    violation.table, prev_idx, to_col, gap_end,
                    reason="expert-fix-extend-previous"
                )
                logger.info(f"Extended previous interval to_depth to {gap_end}")

        elif option_id == "pull_next":
            # Pull current interval's from_depth back to close gap
            gap_start = fix_details.get("gap_start") or gap_info.get("prev_to")
            if violation.row_index in df.index and gap_start is not None:
                old_values[violation.row_index] = {
                    "column": from_col,
                    "old_value": df.at[violation.row_index, from_col] if from_col in df.columns else None,
                    "new_value": gap_start,
                }
                self.editor_engine.edit_cell(
                    violation.table, violation.row_index, from_col, gap_start,
                    reason="expert-fix-pull-next"
                )
                logger.info(f"Pulled next interval from_depth to {gap_start}")

        elif option_id == "split_difference":
            # Split the difference - adjust both intervals to meet in the middle
            prev_idx = gap_info.get("prev_idx")
            prev_to = gap_info.get("prev_to")
            curr_from = gap_info.get("curr_from")
            if prev_idx is not None and prev_to is not None and curr_from is not None:
                midpoint = (prev_to + curr_from) / 2.0
                # Extend previous
                if prev_idx in df.index:
                    old_values[prev_idx] = {
                        "column": to_col,
                        "old_value": df.at[prev_idx, to_col] if to_col in df.columns else None,
                        "new_value": midpoint,
                    }
                    self.editor_engine.edit_cell(
                        violation.table, prev_idx, to_col, midpoint,
                        reason="expert-fix-split-extend"
                    )
                # Pull next
                if violation.row_index in df.index:
                    old_values[violation.row_index] = {
                        "column": from_col,
                        "old_value": df.at[violation.row_index, from_col] if from_col in df.columns else None,
                        "new_value": midpoint,
                    }
                    self.editor_engine.edit_cell(
                        violation.table, violation.row_index, from_col, midpoint,
                        reason="expert-fix-split-pull"
                    )
                logger.info(f"Split difference at midpoint {midpoint}")

        elif option_id in ("truncate_previous", "trim_previous"):
            # Truncate previous interval to fix overlap
            prev_idx = gap_info.get("prev_idx")
            curr_from = gap_info.get("curr_from")
            if prev_idx is not None and prev_idx in df.index and curr_from is not None:
                old_values[prev_idx] = {
                    "column": to_col,
                    "old_value": df.at[prev_idx, to_col] if to_col in df.columns else None,
                    "new_value": curr_from,
                }
                self.editor_engine.edit_cell(
                    violation.table, prev_idx, to_col, curr_from,
                    reason="expert-fix-truncate-previous"
                )
                logger.info(f"Truncated previous interval to {curr_from}")

        elif option_id in ("delay_next", "shift_next"):
            # Delay/shift next interval to fix overlap
            prev_to = gap_info.get("prev_to")
            if violation.row_index in df.index and prev_to is not None:
                old_values[violation.row_index] = {
                    "column": from_col,
                    "old_value": df.at[violation.row_index, from_col] if from_col in df.columns else None,
                    "new_value": prev_to,
                }
                self.editor_engine.edit_cell(
                    violation.table, violation.row_index, from_col, prev_to,
                    reason="expert-fix-delay-next"
                )
                logger.info(f"Delayed next interval to {prev_to}")

        elif option_id == "manual_entry":
            # Manual entry - just jump to the cell for editing
            self._jump_to_violation(violation)
            self.status_label.setText("Edit the cell manually, then re-run QC")
            logger.info("Manual entry requested - jumping to violation cell")

        elif option_id == "insert_placeholder":
            # Insert placeholder interval (would need to add row - complex)
            logger.warning("Insert placeholder not yet implemented")
            QMessageBox.information(
                self, "Not Implemented",
                "Insert placeholder is not yet implemented.\n\n"
                "Please use 'Extend Previous' or 'Pull Next' to close the gap."
            )

        # Store old values in fix_details for audit tracking
        fix_details["_old_values"] = old_values
    
    def _track_expert_fix_in_audit(self, fix_details: Dict[str, Any]):
        """Track expert fix in audit trail."""
        violation = fix_details.get("violation")
        option_id = fix_details.get("option_id")
        
        if not violation or not option_id or option_id == "ignore":
            return
        
        # Get old values stored during fix application
        old_values = fix_details.get("_old_values", {})
        tables = self.editor_engine.get_tables()
        df = tables.get(violation.table) if violation.table in tables else None
        
        # Track all changes made by the expert fix
        for row_index, change_info in old_values.items():
            column = change_info["column"]
            old_value = change_info["old_value"]
            new_value = change_info["new_value"]
            
            # Get hole_id for this row
            hole_id = violation.hole_id
            if df is not None and row_index in df.index and "hole_id" in df.columns:
                hole_id = df.at[row_index, "hole_id"]
            
            self.audit_trail.add_manual_edit(
                table=violation.table,
                hole_id=str(hole_id),
                row_index=row_index,
                column=column,
                old_value=old_value,
                new_value=new_value,
                reason=f"Expert fix: {option_id}",
                user=self.user,
            )

    def _jump_to_violation(self, v: ValidationViolation):
        """Navigates the correct table to the row causing the issue."""
        table = v.table
        row_index = v.row_index

        if row_index < 0:
            return

        # Map table name to view/model
        mapping = {
            "assays": (self.assays_view, self.assays_model),
            "lithology": (self.lith_view, self.lith_model),
            "surveys": (self.survey_view, self.survey_model),
            "collars": (self.collars_view, self.collars_model),
        }

        if table not in mapping:
            return

        view, model = mapping[table]

        # Switch tab
        self.tab_widget.setCurrentWidget(view)

        # Reset editor selection
        self.editor_engine.clear_selection(table)
        self.editor_engine.select_row(table, row_index)

        # Find row in visual model
        try:
            df = model._df
            row_pos = list(df.index).index(row_index)

            # Determine column (default to 0 if unknown)
            col_pos = 0
            # Heuristic: try to find a column related to the error (e.g. from_depth)
            if "depth" in v.message.lower():
                for i, col in enumerate(df.columns):
                    if "depth" in col.lower():
                        col_pos = i
                        break

            qindex = model.index(row_pos, col_pos)

            # Scroll and Select
            view.scrollTo(qindex, QTableView.ScrollHint.PositionAtCenter)
            view.selectRow(row_pos)
        except ValueError:
            self.status_label.setText("Error: Row index not found in current view.")

    def _edit_violation_cell(self, v: ValidationViolation):
        """Navigate to violation cell and enter edit mode."""
        table = v.table
        row_index = v.row_index

        if row_index < 0:
            return

        # Map table name to view/model
        mapping = {
            "assays": (self.assays_view, self.assays_model),
            "lithology": (self.lith_view, self.lith_model),
            "surveys": (self.survey_view, self.survey_model),
            "collars": (self.collars_view, self.collars_model),
        }

        if table not in mapping:
            return

        view, model = mapping[table]

        # Switch tab
        self.tab_widget.setCurrentWidget(view)

        try:
            df = model._df
            row_pos = list(df.index).index(row_index)

            # Determine column to edit
            col_pos = 0
            col_name = None

            # Parse message to find missing field columns
            msg = v.message.lower()
            if "missing" in msg:
                # Try to extract field names from message like "missing required fields: x, y, z"
                match = re.search(r'missing[^:]*:\s*(.+?)\.?\s*$', msg)
                if match:
                    fields = [f.strip() for f in match.group(1).split(',')]
                    # Find first matching column
                    for i, col in enumerate(df.columns):
                        if col.lower() in [f.lower() for f in fields]:
                            col_pos = i
                            col_name = col
                            break
                        # Also check partial matches
                        for f in fields:
                            if f.lower() in col.lower() or col.lower() in f.lower():
                                col_pos = i
                                col_name = col
                                break
                        if col_name:
                            break

            # Fallback: find column by heuristics
            if col_name is None:
                if "depth" in msg:
                    for i, col in enumerate(df.columns):
                        if "depth" in col.lower():
                            col_pos = i
                            break
                elif "azimuth" in msg:
                    for i, col in enumerate(df.columns):
                        if "azimuth" in col.lower():
                            col_pos = i
                            break
                elif "dip" in msg:
                    for i, col in enumerate(df.columns):
                        if "dip" in col.lower():
                            col_pos = i
                            break

            qindex = model.index(row_pos, col_pos)

            # Scroll to cell and enter edit mode
            view.scrollTo(qindex, QTableView.ScrollHint.PositionAtCenter)
            view.setCurrentIndex(qindex)
            view.edit(qindex)

            self.status_label.setText(f"Editing cell at row {row_pos + 1}, column '{df.columns[col_pos]}'")
        except ValueError:
            self.status_label.setText("Error: Row index not found in current view.")

    def _show_tree_context_menu(self, position):
        """Show context menu for tree items."""
        item = self.issue_tree.itemAt(position)
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, payload = data

        menu = QMenu()

        if kind == "violation":
            v = payload

            # Edit Cell action - allows editing the flagged cell directly
            action_edit = QAction("Edit Cell", self)
            action_edit.triggered.connect(lambda checked=False, viol=v: self._edit_violation_cell(viol))
            menu.addAction(action_edit)

            # Expert Decision - for violations that need geological judgment
            rule_upper = v.rule_code.upper()
            if "GAP" in rule_upper or "OVERLAP" in rule_upper or "MISSING" in rule_upper:
                action_expert = QAction("Expert Decision...", self)
                action_expert.setToolTip("Open Expert Decision panel for detailed fix options")
                action_expert.triggered.connect(lambda checked=False, viol=v: self._open_expert_decision(viol))
                menu.addAction(action_expert)

            menu.addSeparator()

            action_ignore = QAction(f"Ignore Issue", self)
            # Bug fix: capture v values explicitly to avoid closure issues
            action_ignore.triggered.connect(lambda checked=False, rc=v.rule_code, hid=v.hole_id, ri=v.row_index:
                self._add_ignore_rule(IgnoreRule(rule_code=rc, hole_id=hid, row_index=ri)))
            menu.addAction(action_ignore)

            action_ignore_rule = QAction(f"Ignore Rule '{v.rule_code}'", self)
            action_ignore_rule.triggered.connect(lambda checked=False, rc=v.rule_code:
                self._add_ignore_rule(IgnoreRule(rule_code=rc)))
            menu.addAction(action_ignore_rule)

        elif kind == "hole":
            # Bug fix: handle both tuple (table, hole_id) and string (hole_id) payloads
            # depending on grouping mode
            if isinstance(payload, tuple):
                table, hole_id = payload
            else:
                # In "hole" grouping mode, payload is just the hole_id string
                hole_id = payload
            action_ignore_hole = QAction(f"Ignore Hole '{hole_id}'", self)
            action_ignore_hole.triggered.connect(lambda checked=False, hid=hole_id: self._add_ignore_rule(
                IgnoreRule(hole_id=hid)
            ))
            menu.addAction(action_ignore_hole)

        menu.exec(self.issue_tree.viewport().mapToGlobal(position))

    # -----------------------------------------------------
    # Ignore Helpers
    # -----------------------------------------------------

    def _add_ignore_rule(self, rule: IgnoreRule):
        """Add an ignore rule and re-run QC."""
        self.pipeline_state.ignore_rules.append(rule)
        self.status_label.setText("Rule ignored. Re-running QC...")
        self._run_full_qc()

    def _toggle_ignore_warnings(self, checked: bool):
        """Toggle ignoring all warnings (won't affect validation status)."""
        self.pipeline_state.ignore_all_warnings = checked
        logger.info(f"Ignore All Warnings toggled: {checked}")

        # Count warnings before filtering
        warning_count = sum(1 for v in self.violations_all if v.severity == "WARNING")

        if checked:
            self.status_label.setText(f"Ignoring {warning_count} warnings...")
        else:
            self.status_label.setText("Showing all warnings...")

        self._run_full_qc()

    def _toggle_ignore_minor(self, checked: bool):
        """Toggle ignoring all minor issues (INFO + WARNING, won't affect validation status)."""
        self.pipeline_state.ignore_all_minor = checked

        # Count minor issues
        minor_count = sum(1 for v in self.violations_all if v.severity in ("WARNING", "INFO"))
        if checked:
            self.status_label.setText(f"Ignoring {minor_count} minor issues...")
        else:
            self.status_label.setText("Showing all issues...")

        self._run_full_qc()

    def _toggle_show_problem_rows(self, checked: bool):
        """Toggle filtering tables to show only rows with problems."""
        # Collect problem row indices from violations
        problem_indices = {
            "collars": set(),
            "surveys": set(),
            "assays": set(),
            "lithology": set(),
        }

        for v in self.violations_all:
            if v.row_index >= 0 and v.table in problem_indices:
                problem_indices[v.table].add(v.row_index)

        # Also find rows with any missing values in required columns
        tables = self.editor_engine.get_tables()
        for table_name, df in tables.items():
            if df is None or df.empty:
                continue
            # Check for missing hole_id
            hole_col = None
            for col in ["hole_id", "holeid", "HOLE_ID", "HoleID"]:
                if col in df.columns:
                    hole_col = col
                    break
            if hole_col:
                missing_hole_rows = df[df[hole_col].isna() | (df[hole_col] == "")].index
                problem_indices[table_name].update(missing_hole_rows)

        # Update each model
        self.collars_model.set_problem_indices(problem_indices["collars"])
        self.survey_model.set_problem_indices(problem_indices["surveys"])
        self.assays_model.set_problem_indices(problem_indices["assays"])
        self.lith_model.set_problem_indices(problem_indices["lithology"])

        # Toggle filter
        self.collars_model.set_show_problems_only(checked)
        self.survey_model.set_show_problems_only(checked)
        self.assays_model.set_show_problems_only(checked)
        self.lith_model.set_show_problems_only(checked)

        # Count total problem rows
        total_problems = sum(len(indices) for indices in problem_indices.values())

        if checked:
            self.status_label.setText(f"Showing {total_problems} problem rows only")
            logger.info(f"Problem row filter ON: {total_problems} rows across all tables")
        else:
            self.status_label.setText("Showing all rows")
            logger.info("Problem row filter OFF")

    # -----------------------------------------------------
    # Undo / Redo / IO
    # -----------------------------------------------------

    def _undo_edit(self):
        """Undo last edit."""
        self.editor_engine.undo()
        self._refresh_models()
        self._run_full_qc()
        self.status_label.setText("Undone last edit.")

    def _redo_edit(self):
        """Redo last undone edit."""
        self.editor_engine.redo()
        self._refresh_models()
        self._run_full_qc()
        self.status_label.setText("Redone last edit.")

    def _show_find_dialog(self):
        """Show find dialog."""
        dialog = FindReplaceDialog(self, find_only=True)
        if dialog.exec():
            find_text = dialog.find_text
            case_sensitive = dialog.case_sensitive
            self._find_in_table(find_text, case_sensitive)

    def _show_replace_dialog(self):
        """Show find and replace dialog."""
        dialog = FindReplaceDialog(self, find_only=False)
        if dialog.exec():
            find_text = dialog.find_text
            replace_text = dialog.replace_text
            case_sensitive = dialog.case_sensitive
            match_whole_cell = dialog.match_whole_cell
            self._replace_in_table(find_text, replace_text, case_sensitive, match_whole_cell)

    def _get_current_table_view(self):
        """Get the currently visible table view."""
        current_index = self.tab_widget.currentIndex()
        if current_index == 0:
            return self.collars_view, self.collars_model
        elif current_index == 1:
            return self.survey_view, self.survey_model
        elif current_index == 2:
            return self.lith_view, self.lith_model
        elif current_index == 3:
            return self.assays_view, self.assays_model
        return None, None

    def _find_in_table(self, find_text: str, case_sensitive: bool = False):
        """Find text in the current table."""
        view, model = self._get_current_table_view()
        if not view or not model:
            QMessageBox.warning(self, "No Table", "Please select a table tab first.")
            return

        df = model._df
        if df.empty:
            return

        # Search through all cells
        matches = []
        for row_idx in range(len(df)):
            for col_idx in range(len(df.columns)):
                value = str(df.iat[row_idx, col_idx])
                if not case_sensitive:
                    value = value.lower()
                    find_text_lower = find_text.lower()
                    if find_text_lower in value:
                        matches.append((row_idx, col_idx))
                else:
                    if find_text in value:
                        matches.append((row_idx, col_idx))

        if not matches:
            QMessageBox.information(self, "Find", f"No matches found for '{find_text}'")
            return

        # Select and scroll to first match
        first_match = matches[0]
        index = model.index(first_match[0], first_match[1])
        view.setCurrentIndex(index)
        view.scrollTo(index, QTableView.ScrollHint.EnsureVisible)
        view.selectRow(first_match[0])
        
        self.status_label.setText(f"Found {len(matches)} match(es). Showing first match.")

    def _replace_in_table(self, find_text: str, replace_text: str, case_sensitive: bool = False, match_whole_cell: bool = False):
        """Replace text in the current table."""
        view, model = self._get_current_table_view()
        if not view or not model:
            QMessageBox.warning(self, "No Table", "Please select a table tab first.")
            return

        df = model._df.copy()
        if df.empty:
            return

        # Find all matches
        matches = []
        for row_idx in range(len(df)):
            for col_idx in range(len(df.columns)):
                value = str(df.iat[row_idx, col_idx])
                original_value = value
                
                if case_sensitive:
                    search_value = value
                    search_find = find_text
                else:
                    search_value = value.lower()
                    search_find = find_text.lower()

                if match_whole_cell:
                    if search_value == search_find:
                        matches.append((row_idx, col_idx, original_value))
                else:
                    if search_find in search_value:
                        matches.append((row_idx, col_idx, original_value))

        if not matches:
            QMessageBox.information(self, "Replace", f"No matches found for '{find_text}'")
            return

        # Confirm replacement
        reply = QMessageBox.question(
            self, "Confirm Replace",
            f"Found {len(matches)} match(es). Replace all?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Perform replacements
        table_name = model._table_name
        replaced_count = 0
        find_replace_edits = []

        for row_idx, col_idx, original_value in matches:
            row = df.index[row_idx]
            col_name = df.columns[col_idx]
            value = str(original_value)

            if match_whole_cell:
                new_value = replace_text
            else:
                if case_sensitive:
                    new_value = value.replace(find_text, replace_text)
                else:
                    # Case-insensitive replace
                    new_value = re.sub(re.escape(find_text), replace_text, value, flags=re.IGNORECASE)

            # Try to convert to original type if numeric
            try:
                original_val = df.iat[row_idx, col_idx]
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    new_value = float(new_value) if '.' in str(new_value) else int(new_value)
            except (ValueError, TypeError):
                pass  # Keep as string

            # Apply edit through editor engine
            try:
                self.editor_engine.edit_cell(
                    table=table_name,
                    row_index=row,
                    column=col_name,
                    new_value=new_value,
                    reason="find-replace",
                )
                
                # Track in audit trail
                hole_id = df.at[row, "hole_id"] if "hole_id" in df.columns else ""
                find_replace_edits.append({
                    "hole_id": hole_id,
                    "row_index": row,
                    "column": col_name,
                    "old_value": original_val,
                    "new_value": new_value,
                })
                
                replaced_count += 1
            except Exception as e:
                logger.warning(f"Failed to replace at row {row_idx}, col {col_idx}: {e}")

        # Record find & replace in audit trail
        if find_replace_edits and self.audit_trail:
            self.audit_trail.add_find_replace(
                table=table_name,
                edits=find_replace_edits,
                find_text=find_text,
                replace_text=replace_text,
                user=self.user,
            )

        # Refresh the model
        tables = self.editor_engine.get_tables()
        model.update_df(tables[table_name])

        # Re-run QC to update violations
        self._run_full_qc()

        QMessageBox.information(
            self, "Replace Complete",
            f"Replaced {replaced_count} of {len(matches)} match(es)."
        )

    def _export_clean(self):
        """Export cleaned tables to CSV."""
        tables = self.editor_engine.get_tables()
        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not out_dir:
            return

        try:
            tables["collars"].to_csv(f"{out_dir}/collars_clean.csv", index=False)
            tables["surveys"].to_csv(f"{out_dir}/surveys_clean.csv", index=False)
            tables["assays"].to_csv(f"{out_dir}/assays_clean.csv", index=False)
            tables["lithology"].to_csv(f"{out_dir}/lithology_clean.csv", index=False)
            QMessageBox.information(self, "Export Success", f"Files saved to:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))
    
    def _save_to_database(self):
        """Save cleaned drillhole data to database."""
        tables = self.editor_engine.get_tables()
        
        # Check if we have any data
        has_data = False
        for table_name, df in tables.items():
            if df is not None and not df.empty:
                has_data = True
                break
        
        if not has_data:
            QMessageBox.warning(
                self, "No Data",
                "No drillhole data available to save.\n\n"
                "Please load and clean data first."
            )
            return
        
        # Show dialog to get/select project name
        projects = []
        try:
            db_manager = DrillholeDatabaseManager()
            projects = db_manager.list_projects()
        except Exception as e:
            logger.warning(f"Could not list existing projects: {e}")
        
        project_names = [p['name'] for p in projects]
        
        # Show input dialog
        if project_names:
            project_name, ok = QInputDialog.getItem(
                self,
                "Save to Database",
                "Select or enter project name:",
                project_names + ["[New Project...]"],
                editable=True
            )
            if not ok:
                return
            
            if project_name == "[New Project...]":
                project_name, ok = QInputDialog.getText(
                    self,
                    "New Project",
                    "Enter new project name:",
                    text="Drillhole_QC_Cleaned"
                )
                if not ok or not project_name:
                    return
        else:
            project_name, ok = QInputDialog.getText(
                self,
                "Save to Database",
                "Enter project name:",
                text="Drillhole_QC_Cleaned"
            )
            if not ok or not project_name:
                return
        
        try:
            db_manager = DrillholeDatabaseManager()
            
            # Convert DataFrames to DrillholeDatabase format
            new_db = DrillholeDatabase(metadata={
                "source": "QC Window",
                "user": self.user,
                "saved_at": pd.Timestamp.now().isoformat(),
                "audit_trail_entries": len(self.audit_trail.entries),
            })
            
            def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
                """Find a column by common aliases (case-insensitive)."""
                if df is None or df.empty:
                    return None
                lowered = {c.lower(): c for c in df.columns}
                for cand in candidates:
                    if cand in df.columns:
                        return cand
                    if cand.lower() in lowered:
                        return lowered[cand.lower()]
                return None
            
            saved_components = []
            
            # Collars
            collars_df = tables.get("collars")
            if collars_df is not None and not collars_df.empty:
                hole_col = _find_column(collars_df, ["hole_id", "hole", "holeid", "HOLEID"])
                x_col = _find_column(collars_df, ["x", "easting", "east", "X", "EASTING"])
                y_col = _find_column(collars_df, ["y", "northing", "north", "Y", "NORTHING"])
                z_col = _find_column(collars_df, ["z", "elevation", "rl", "Z", "ELEVATION"])
                az_col = _find_column(collars_df, ["azimuth", "azi", "AZIMUTH"])
                dip_col = _find_column(collars_df, ["dip", "DIP"])
                len_col = _find_column(collars_df, ["length", "total_length", "len", "total_depth", "depth", "LENGTH", "DEPTH"])
                
                if hole_col and x_col and y_col and z_col:
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in collars_df.iterrows():
                        if pd.isna(row[hole_col]) or pd.isna(row[x_col]) or pd.isna(row[y_col]) or pd.isna(row[z_col]):
                            continue
                        rows.append({
                            'hole_id': str(row[hole_col]),
                            'x': float(row[x_col]),
                            'y': float(row[y_col]),
                            'z': float(row[z_col]),
                            'azimuth': float(row[az_col]) if az_col and pd.notna(row.get(az_col)) else None,
                            'dip': float(row[dip_col]) if dip_col and pd.notna(row.get(dip_col)) else None,
                            'length': float(row[len_col]) if len_col and pd.notna(row.get(len_col)) else None,
                        })
                    
                    if rows:
                        collar_df_new = pd.DataFrame(rows)
                        if new_db.collars.empty:
                            new_db.collars = collar_df_new
                        else:
                            new_db.collars = pd.concat([new_db.collars, collar_df_new], ignore_index=True)
                    saved_components.append(f"Collars: {len(new_db.collars)} records")
            
            # Surveys
            surveys_df = tables.get("surveys")
            if surveys_df is not None and not surveys_df.empty:
                hole_col = _find_column(surveys_df, ["hole_id", "holeid", "hole", "HOLEID"])
                from_col = _find_column(surveys_df, ["depth_from", "from", "mfrom", "depth", "FROM", "DEPTH_FROM"])
                to_col = _find_column(surveys_df, ["depth_to", "to", "mto", "TO", "DEPTH_TO"])
                az_col = _find_column(surveys_df, ["azimuth", "azi", "az", "AZIMUTH"])
                dip_col = _find_column(surveys_df, ["dip", "inclination", "incl", "inc", "DIP"])
                
                if hole_col and from_col and az_col and dip_col:
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in surveys_df.iterrows():
                        if pd.isna(row[hole_col]) or pd.isna(row[from_col]) or pd.isna(row[az_col]) or pd.isna(row[dip_col]):
                            continue
                        depth_to_val = row[to_col] if to_col and pd.notna(row.get(to_col)) else row[from_col]
                        rows.append({
                            'hole_id': str(row[hole_col]),
                            'depth_from': float(row[from_col]),
                            'depth_to': float(depth_to_val),
                            'azimuth': float(row[az_col]),
                            'dip': float(row[dip_col]),
                        })
                    
                    if rows:
                        survey_df_new = pd.DataFrame(rows)
                        if new_db.surveys.empty:
                            new_db.surveys = survey_df_new
                        else:
                            new_db.surveys = pd.concat([new_db.surveys, survey_df_new], ignore_index=True)
                    saved_components.append(f"Surveys: {len(new_db.surveys)} records")
            
            # Assays
            assays_df = tables.get("assays")
            if assays_df is not None and not assays_df.empty:
                hole_col = _find_column(assays_df, ["hole_id", "holeid", "hole", "HOLEID"])
                from_col = _find_column(assays_df, ["depth_from", "from", "mfrom", "FROM", "DEPTH_FROM"])
                to_col = _find_column(assays_df, ["depth_to", "to", "mto", "TO", "DEPTH_TO"])
                
                if hole_col and from_col and to_col:
                    meta_exclude_upper = {
                        "HOLEID", "HOLE_ID", "HOLE", "SAMPLEID", "SAMPLE_ID",
                        "FROM", "TO", "DEPTH", "DEPTH_FROM", "DEPTH_TO", "LENGTH",
                        "X", "Y", "Z", "EAST", "EASTING", "NORTH", "NORTHING",
                        "RL", "ELEV", "ELEVATION"
                    }
                    exclude_cols = {hole_col, from_col, to_col}
                    
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in assays_df.iterrows():
                        if pd.isna(row[hole_col]) or pd.isna(row[from_col]) or pd.isna(row[to_col]):
                            continue
                        
                        row_dict = {
                            'hole_id': str(row[hole_col]),
                            'depth_from': float(row[from_col]),
                            'depth_to': float(row[to_col]),
                        }
                        
                        # Add assay value columns
                        for col in assays_df.columns:
                            if col in exclude_cols or col.upper() in meta_exclude_upper:
                                continue
                            if pd.notna(row[col]):
                                try:
                                    row_dict[col] = float(row[col])
                                except Exception:
                                    continue
                        
                        rows.append(row_dict)
                    
                    if rows:
                        assay_df_new = pd.DataFrame(rows)
                        if new_db.assays.empty:
                            new_db.assays = assay_df_new
                        else:
                            new_db.assays = pd.concat([new_db.assays, assay_df_new], ignore_index=True)
                    saved_components.append(f"Assays: {len(new_db.assays)} records")
            
            # Lithology
            lith_df = tables.get("lithology")
            if lith_df is not None and not lith_df.empty:
                hole_col = _find_column(lith_df, ["hole_id", "holeid", "hole", "HOLEID"])
                from_col = _find_column(lith_df, ["depth_from", "from", "mfrom", "FROM", "DEPTH_FROM"])
                to_col = _find_column(lith_df, ["depth_to", "to", "mto", "TO", "DEPTH_TO"])
                code_col = _find_column(lith_df, ["lith_code", "lithology", "code", "lith", "LITH_CODE", "LITHOLOGY"])
                
                if hole_col and from_col and to_col and code_col:
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in lith_df.iterrows():
                        if pd.isna(row[hole_col]) or pd.isna(row[from_col]) or pd.isna(row[to_col]) or pd.isna(row[code_col]):
                            continue
                        rows.append({
                            'hole_id': str(row[hole_col]),
                            'depth_from': float(row[from_col]),
                            'depth_to': float(row[to_col]),
                            'lith_code': str(row[code_col]),
                        })
                    
                    if rows:
                        lith_df_new = pd.DataFrame(rows)
                        if new_db.lithology.empty:
                            new_db.lithology = lith_df_new
                        else:
                            new_db.lithology = pd.concat([new_db.lithology, lith_df_new], ignore_index=True)
                    saved_components.append(f"Lithology: {len(new_db.lithology)} records")
            
            # Save to database
            if saved_components:
                db_manager.save_database(new_db, project_name)
                message = f"Successfully saved cleaned data to project '{project_name}':\n\n" + "\n".join(saved_components)
                if self.audit_trail.entries:
                    message += f"\n\nAudit trail: {len(self.audit_trail.entries)} changes recorded"
                QMessageBox.information(self, "Save Success", message)
                logger.info(f"Saved cleaned drillhole data to project '{project_name}': {saved_components}")
            else:
                QMessageBox.warning(
                    self, "No Data Saved",
                    "No valid data could be saved.\n\n"
                    "Please ensure your data has the required columns:\n"
                    "- Collars: hole_id, x, y, z\n"
                    "- Surveys: hole_id, depth_from, azimuth, dip\n"
                    "- Assays: hole_id, depth_from, depth_to\n"
                    "- Lithology: hole_id, depth_from, depth_to, lith_code"
                )
        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Failed", f"Failed to save to database:\n{str(e)}")
    
    def _apply_to_registry(self):
        """
        Apply cleaned drillhole data to the DataRegistry.
        
        This makes the cleaned data available to:
        - Compositing Window
        - Variogram Panel
        - Kriging Panels
        - JORC Classification
        - All other analysis tools
        
        IMPORTANT: After running QC validation and fixing issues, use this
        to ensure downstream tools use the cleaned data, not raw data.
        """
        tables = self.editor_engine.get_tables()
        
        # Check if we have any data
        has_data = False
        for table_name, df in tables.items():
            if df is not None and not df.empty:
                has_data = True
                break
        
        if not has_data:
            QMessageBox.warning(
                self, "No Data",
                "No drillhole data available to apply.\n\n"
                "Please load and clean data first."
            )
            return
        
        # Get registry
        registry = None
        if hasattr(self, 'controller') and self.controller and hasattr(self.controller, 'registry'):
            registry = self.controller.registry
        
        if registry is None:
            # Try singleton as fallback
            try:
                from ..core.data_registry import DataRegistry
                registry = DataRegistry.instance()
            except Exception:
                pass
        
        if registry is None:
            QMessageBox.warning(
                self, "No Registry",
                "DataRegistry not available.\n\n"
                "Cannot apply cleaned data to registry."
            )
            return
        
        # Build data dict for registry
        data_dict = {}
        summary_parts = []
        
        collars_df = tables.get("collars")
        if collars_df is not None and not collars_df.empty:
            data_dict['collars'] = collars_df.copy()
            summary_parts.append(f"Collars: {len(collars_df)} records")
        
        surveys_df = tables.get("surveys")
        if surveys_df is not None and not surveys_df.empty:
            data_dict['surveys'] = surveys_df.copy()
            summary_parts.append(f"Surveys: {len(surveys_df)} records")
        
        assays_df = tables.get("assays")
        if assays_df is not None and not assays_df.empty:
            data_dict['assays'] = assays_df.copy()
            summary_parts.append(f"Assays: {len(assays_df)} records")
        
        lithology_df = tables.get("lithology")
        if lithology_df is not None and not lithology_df.empty:
            data_dict['lithology'] = lithology_df.copy()
            summary_parts.append(f"Lithology: {len(lithology_df)} records")
        
        if not data_dict:
            QMessageBox.warning(
                self, "No Data",
                "No valid data tables to apply."
            )
            return
        
        try:
            # CRITICAL FIX: Re-run validation on the CLEANED data being applied
            # This ensures validation state (excluded_rows) matches the actual data in registry
            # Without this, compositing could receive raw data with stale excluded_rows
            self.status_label.setText("Validating cleaned data before applying...")
            QApplication.processEvents()

            val_result = run_drillhole_validation(
                collars=data_dict.get('collars', pd.DataFrame()),
                surveys=data_dict.get('surveys', pd.DataFrame()),
                assays=data_dict.get('assays', pd.DataFrame()),
                lithology=data_dict.get('lithology', pd.DataFrame()),
                cfg=self.cfg,
            )

            # Calculate excluded_rows for the data being applied
            excluded_rows: Dict[str, List[int]] = {}
            for v in val_result.violations:
                if v.severity == "ERROR":
                    table = v.table.lower()
                    if table not in excluded_rows:
                        excluded_rows[table] = []
                    if v.row_index not in excluded_rows[table]:
                        excluded_rows[table].append(v.row_index)

            # Warn if there are still errors in the cleaned data
            if excluded_rows:
                total_excluded = sum(len(rows) for rows in excluded_rows.values())
                reply = QMessageBox.warning(
                    self, "Data Has Errors",
                    f"The cleaned data still has {val_result.fatal_count} ERROR(s).\n\n"
                    f"{total_excluded} rows will be EXCLUDED from compositing/kriging.\n\n"
                    "Do you want to apply anyway?\n\n"
                    "TIP: Fix or delete error rows before applying for best results.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.status_label.setText("Apply cancelled - fix errors first")
                    return

            # Build metadata
            metadata = {
                "source": "QC Window - Cleaned Data",
                "user": self.user,
                "applied_at": pd.Timestamp.now().isoformat(),
                "audit_entries": len(self.audit_trail.entries) if self.audit_trail else 0,
                "qc_validated": True,
                "validation_status": val_result.status,
                "error_count": val_result.fatal_count,
                "warning_count": val_result.warn_count,
            }

            # Add QC summary to metadata
            if self.audit_trail and self.audit_trail.entries:
                summary = self.audit_trail.get_summary()
                metadata["qc_summary"] = summary

            # Register to DataRegistry
            success = registry.register_drillhole_data(
                data=data_dict,
                source_panel="QC Window",
                metadata=metadata,
            )

            if success:
                # CRITICAL: Store validation state IN SYNC with the data just registered
                # This ensures excluded_rows matches the data compositing will receive
                registry.set_drillholes_validation_state(
                    status=val_result.status,
                    timestamp=val_result.timestamp or "",
                    config_hash=val_result.config_hash or "",
                    fatal_count=val_result.fatal_count,
                    warn_count=val_result.warn_count,
                    info_count=val_result.info_count,
                    violations_summary=val_result.get_violations_summary(),
                    tables_validated=val_result.tables_validated,
                    schema_errors=val_result.schema_errors,
                    excluded_rows=excluded_rows,
                )

                logger.info(
                    f"Applied cleaned data AND validation state to registry: "
                    f"status={val_result.status}, excluded_rows={len(excluded_rows)} tables"
                )

                message = (
                    "Successfully applied cleaned data to DataRegistry!\n\n"
                    + "\n".join(summary_parts) + "\n\n"
                    f"Validation Status: {val_result.status}\n"
                )
                if excluded_rows:
                    total_excluded = sum(len(rows) for rows in excluded_rows.values())
                    message += f"Rows excluded from downstream: {total_excluded}\n"
                message += (
                    "\nThe cleaned data will now be used by:\n"
                    "• Compositing Window\n"
                    "• Variogram Panel\n"
                    "• Kriging Panels\n"
                    "• JORC Classification\n"
                    "• All analysis tools"
                )
                if self.audit_trail and self.audit_trail.entries:
                    message += f"\n\nQC Changes Applied: {len(self.audit_trail.entries)} modifications"

                QMessageBox.information(self, "Applied to Registry", message)
                logger.info(f"Applied cleaned drillhole data to registry: {summary_parts}")
            else:
                QMessageBox.warning(
                    self, "Apply Failed",
                    "Failed to apply data to registry.\n\n"
                    "Please check the logs for details."
                )
        except Exception as e:
            logger.error(f"Error applying to registry: {e}", exc_info=True)
            QMessageBox.critical(self, "Apply Failed", f"Failed to apply to registry:\n{str(e)}")
    
    def closeEvent(self, event):
        """
        Override close event to prevent accidental data loss.
        Shows confirmation dialog if there are unsaved changes.
        """
        # Check if there are unsaved changes
        has_changes = False
        change_summary = []
        
        if self.audit_trail and self.audit_trail.entries:
            summary = self.audit_trail.get_summary()
            if summary['total_changes'] > 0:
                has_changes = True
                change_summary.append(f"• {summary['total_changes']} total changes made")
                if summary['auto_fixes'] > 0:
                    change_summary.append(f"  - {summary['auto_fixes']} auto-fixes")
                if summary['manual_edits'] > 0:
                    change_summary.append(f"  - {summary['manual_edits']} manual edits")
                if summary['find_replace'] > 0:
                    change_summary.append(f"  - {summary['find_replace']} find & replace operations")
        
        if has_changes:
            # Show confirmation dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Close QC Window?")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("You have unsaved changes in the QC window.")
            msg.setInformativeText(
                "Changes made:\n" + "\n".join(change_summary) + "\n\n" +
                "What would you like to do?\n\n"
                "TIP: Use 'Apply to Registry' to make cleaned data available\n"
                "to Compositing, Variogram, Kriging, and other analysis tools."
            )
            
            # Add buttons
            apply_btn = msg.addButton("Apply to Registry", QMessageBox.ButtonRole.AcceptRole)
            save_btn = msg.addButton("Save to Database", QMessageBox.ButtonRole.AcceptRole)
            hide_btn = msg.addButton("Hide Window", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = msg.addButton("Discard Changes", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            msg.setDefaultButton(cancel_btn)
            msg.exec()
            
            clicked = msg.clickedButton()
            
            if clicked == cancel_btn:
                # Cancel close
                event.ignore()
                return
            elif clicked == apply_btn:
                # Apply to registry first
                try:
                    self._apply_to_registry()
                    # After applying, hide the window
                    self.hide()
                    event.ignore()
                except Exception as e:
                    retry = QMessageBox.question(
                        self, "Apply Failed",
                        f"Failed to apply to registry:\n{str(e)}\n\n"
                        "Do you want to close anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if retry == QMessageBox.StandardButton.Yes:
                        event.accept()
                    else:
                        event.ignore()
                return
            elif clicked == save_btn:
                # Save to database first
                try:
                    self._save_to_database()
                    # After saving, hide the window (don't destroy it)
                    self.hide()
                    event.ignore()
                except Exception as e:
                    # If save fails, ask again
                    retry = QMessageBox.question(
                        self, "Save Failed",
                        f"Failed to save to database:\n{str(e)}\n\n"
                        "Do you want to close anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if retry == QMessageBox.StandardButton.Yes:
                        self.hide()
                        event.ignore()
                    else:
                        event.ignore()
                return
            elif clicked == hide_btn:
                # Hide window but keep state
                self.hide()
                event.ignore()
                return
            elif clicked == discard_btn:
                # User confirmed they want to discard changes
                # Hide window (state is already lost if they discard)
                self.hide()
                event.ignore()
                return
        else:
            # No changes - just hide the window
            self.hide()
            event.ignore()
    
    def _export_audit_trail(self):
        """Export audit trail for SAMREC/JORC compliance."""
        if not self.audit_trail.entries:
            QMessageBox.information(
                self, "No Audit Trail",
                "No changes have been recorded in this session.\n\n"
                "The audit trail will be populated when you:\n"
                "- Run Auto-Fix\n"
                "- Make manual edits\n"
                "- Use Find & Replace"
            )
            return
        
        # Show file dialog with format options
        from datetime import datetime
        default_name = f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build filter string based on available libraries
        filters = []
        if OPENPYXL_AVAILABLE:
            filters.append("Excel Files (*.xlsx)")
        if REPORTLAB_AVAILABLE:
            filters.append("PDF Files (*.pdf)")
        filters.append("CSV Files (*.csv)")
        filters.append("All Files (*.*)")
        filter_string = ";;".join(filters)
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Audit Trail",
            default_name,
            filter_string
        )
        
        if not file_path:
            return
        
        try:
            # Determine format from file extension or filter
            if file_path.endswith('.xlsx') or 'Excel' in selected_filter:
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                export_to_excel(self.audit_trail, file_path)
                msg = "Excel"
            elif file_path.endswith('.pdf') or 'PDF' in selected_filter:
                if not file_path.endswith('.pdf'):
                    file_path += '.pdf'
                export_to_pdf(self.audit_trail, file_path)
                msg = "PDF"
            elif file_path.endswith('.csv') or 'CSV' in selected_filter:
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                export_to_csv(self.audit_trail, file_path)
                msg = "CSV"
            else:
                # Default to CSV (always available)
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                export_to_csv(self.audit_trail, file_path)
                msg = "CSV"
            
            summary = self.audit_trail.get_summary()
            QMessageBox.information(
                self, "Export Success",
                f"Audit trail exported successfully to {msg} format.\n\n"
                f"File: {file_path}\n\n"
                f"Summary:\n"
                f"• Total Changes: {summary['total_changes']}\n"
                f"• Auto-Fixes: {summary['auto_fixes']}\n"
                f"• Manual Edits: {summary['manual_edits']}\n"
                f"• Batch Edits: {summary['batch_edits']}\n"
                f"• Find & Replace: {summary['find_replace']}"
            )
        except ImportError as e:
            error_msg = f"Required library not installed:\n{str(e)}\n\n"
            if "openpyxl" in str(e).lower():
                error_msg += "To enable Excel export, install openpyxl:\n"
                error_msg += "  pip install openpyxl\n\n"
            if "reportlab" in str(e).lower():
                error_msg += "To enable PDF export, install reportlab:\n"
                error_msg += "  pip install reportlab\n\n"
            error_msg += "CSV export is always available."
            QMessageBox.critical(self, "Export Failed", error_msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export audit trail:\n{str(e)}")
    
    def get_registry(self):
        """
        Get DataRegistry instance via dependency injection.
        
        Tries to get registry from:
        1. Controller (if bound) -> controller.registry
        2. MainWindow parent -> main_window.controller.registry
        3. Falls back to singleton for backward compatibility
        
        Returns:
            DataRegistry instance
        """
        # Try controller first (preferred - dependency injection)
        if self.controller and hasattr(self.controller, 'registry'):
            return self.controller.registry
        
        # Try MainWindow parent
        parent = self.parent()
        while parent:
            if hasattr(parent, 'controller') and parent.controller:
                if hasattr(parent.controller, 'registry'):
                    return parent.controller.registry
            # Check if parent is MainWindow
            if hasattr(parent, '_registry') and parent._registry:
                return parent._registry
            parent = parent.parent()
        
        # Fallback to singleton (backward compatibility)
        from ..core.data_registry import DataRegistry
        return DataRegistry.instance()
    
    def _store_validation_state_in_registry(self, val_result) -> None:
        """
        Store validation state in DataRegistry for downstream engines.

        This enables compositing and other engines to check validation status
        before processing data. Ensures JORC/SAMREC compliance by maintaining
        data lineage.

        IMPORTANT: Rows with ERROR-level violations are tracked as 'excluded_rows'
        and will be filtered out of downstream processing (compositing, kriging, etc.)
        even if the user chooses to "ignore" them in the UI. Ignoring hides from
        display but does NOT make invalid data valid for analysis.

        Args:
            val_result: ValidationResult from run_drillhole_validation()
        """
        try:
            registry = self.get_registry()
            if registry is None:
                logger.warning("Cannot store validation state: no registry available")
                return

            # Calculate excluded rows - rows with ERROR violations should be excluded
            # from downstream processing regardless of whether they're "ignored" in UI
            excluded_rows: Dict[str, List[int]] = {}
            for v in val_result.violations:
                if v.severity == "ERROR":
                    table = v.table.lower()  # normalize table name
                    if table not in excluded_rows:
                        excluded_rows[table] = []
                    if v.row_index not in excluded_rows[table]:
                        excluded_rows[table].append(v.row_index)

            # Log excluded rows for audit trail
            if excluded_rows:
                total_excluded = sum(len(rows) for rows in excluded_rows.values())
                logger.warning(
                    f"Validation identified {total_excluded} rows with ERROR violations "
                    f"that will be excluded from downstream processing: {excluded_rows}"
                )

            # Store validation state
            registry.set_drillholes_validation_state(
                status=val_result.status,
                timestamp=val_result.timestamp or "",
                config_hash=val_result.config_hash or "",
                fatal_count=val_result.fatal_count,
                warn_count=val_result.warn_count,
                info_count=val_result.info_count,
                violations_summary=val_result.get_violations_summary(),
                tables_validated=val_result.tables_validated,
                schema_errors=val_result.schema_errors,
                excluded_rows=excluded_rows,
            )
            
            logger.info(
                f"Validation state stored in registry: status={val_result.status}, "
                f"fatal={val_result.fatal_count}, warn={val_result.warn_count}"
            )
            
            # Audit logging
            try:
                from ..core.audit_manager import AuditManager
                audit = AuditManager()
                audit.log_event(
                    module="drillhole_validation",
                    action="validation_run",
                    parameters={
                        "config_hash": val_result.config_hash,
                        "tables_validated": val_result.tables_validated,
                    },
                    result_summary={
                        "status": val_result.status,
                        "fatal_count": val_result.fatal_count,
                        "warn_count": val_result.warn_count,
                        "info_count": val_result.info_count,
                        "total_violations": len(val_result.violations),
                    }
                )
            except Exception as e:
                logger.debug(f"Audit logging failed (non-critical): {e}")
                
        except Exception as e:
            logger.warning(f"Failed to store validation state in registry: {e}")


# =========================================================
# Launcher (for testing)
# =========================================================

def main():
    """Simple launcher for testing."""
    # Dummy Data Generation for Testing
    collars = pd.DataFrame({
        "hole_id": ["DDH-001", "DDH-002", "DDH-003"],
        "easting": [1000.0, 1010.0, 1050.0],
        "northing": [2000.0, 2010.0, 2050.0],
        "elevation": [100.0, 101.0, 99.0],
        "total_depth": [100.0, 120.0, 150.0],
    })

    surveys = pd.DataFrame({
        "hole_id": ["DDH-001", "DDH-001", "DDH-002"],
        "depth": [0.0, 50.0, 0.0],
        "dip": [0.0, -60.0, -70.0],
        "azimuth": [0.0, 90.0, 450.0],  # Error: 450
    })

    assays = pd.DataFrame({
        "hole_id": ["DDH-001", "DDH-001", "DDH-001", "DDH-003"],
        "from_depth": [0.0, 10.1, 20.0, 0.0],  # Error: gap 10.0-10.1
        "to_depth": [10.0, 20.0, 30.0, 50.0],
        "sample_id": ["S1", "S2", "S3", "S4"],
        "au_ppm": [0.5, 1.2, 0.1, -5.0],  # Error: negative assay
    })

    lithology = pd.DataFrame({
        "hole_id": ["DDH-001", "DDH-001"],
        "from_depth": [0.0, 15.0],  # Error: Overlap (Assay 10-20 vs Lith 15)
        "to_depth": [15.0, 35.0],
        "lith_code": ["BASALT", "SHALE"],
    })

    app = QApplication(sys.argv)

    # Configure validation
    cfg = ValidationConfig(
        max_interval_gap=0.10,
        max_small_overlap=0.02,
        standard_sample_length=1.0,
    )

    win = QCWindow(collars, surveys, assays, lithology, cfg=cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

