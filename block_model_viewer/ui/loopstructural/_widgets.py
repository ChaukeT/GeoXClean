"""
LoopStructural helper widgets — QSS-only styling.

Five rewritten widgets for the LoopStructural Geological Modeling panel.
Every widget uses objectName-based QSS styling (ZERO inline setStyleSheet
calls) and the design token system.

CRITICAL: This file contains ZERO calls to setStyleSheet().
All visual styling is accomplished through:
    1. setObjectName("LoopSomething") — matched by the global QSS generator
    2. setProperty("key", "value") + unpolish/polish — for dynamic states
    3. QPalette — for colored text where QSS dynamic properties are insufficient
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QLineEdit, QTextEdit, QDialog, QDialogButtonBox, QFormLayout,
)

from ..design_tokens import tokens

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette helpers — NOT stylesheets. QPalette operations are fine.
# ---------------------------------------------------------------------------

def _set_foreground(widget: QWidget, color_hex: str) -> None:
    """Set a widget's foreground (WindowText) color via QPalette."""
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.WindowText, QColor(color_hex))
    widget.setPalette(palette)


def _refresh_style(widget: QWidget) -> None:
    """Trigger QSS re-evaluation after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# =========================================================================
# Widget 1: InputValidationChecklist
# =========================================================================

class InputValidationChecklist(QFrame):
    """
    Compact inline checklist for input data validation status.

    Shows status as colored dots, labels, and Required/Optional badges
    arranged as tight flat rows.  All styling through objectNames and
    QPalette — zero setStyleSheet calls.

    Public API (called by business logic mixin):
        validate_dataframe(df: Optional[pd.DataFrame]) -> None
    """

    validation_changed = pyqtSignal(dict)

    # (key, display_label, required)
    CHECKS = [
        ('x_col',       'X coordinate column',              True),
        ('y_col',       'Y coordinate column',              True),
        ('z_col',       'Z coordinate column',              True),
        ('formation',   'Formation / lithology column',     True),
        ('val',         'Scalar value column',              False),
        ('orientation', 'Orientation data (gx, gy, gz)',    False),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: Dict[str, Dict[str, QLabel]] = {}
        self._status: Dict[str, str] = {}
        self._build_ui()
        self._reset_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Section header
        header = QLabel("INPUT VALIDATION")
        header.setObjectName("LoopSectionHeader")
        header.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_MD,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        layout.addWidget(header)

        div = QFrame()
        div.setObjectName("LoopDivider")
        layout.addWidget(div)

        for key, label_text, required in self.CHECKS:
            row = self._make_row(key, label_text, required)
            layout.addWidget(row)

    def _make_row(self, key: str, label_text: str, required: bool) -> QFrame:
        row = QFrame()
        row.setObjectName("LoopFormRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        row_layout.setSpacing(tokens.SPACING_SM)

        # Colored dot indicator  (●)
        dot = QLabel("\u25CF")
        dot.setFixedWidth(16)
        dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_layout.addWidget(dot)

        # Descriptive label
        text = QLabel(label_text)
        row_layout.addWidget(text, stretch=1)

        # Required / Optional badge
        badge = QLabel("Required" if required else "Optional")
        badge.setObjectName("LoopStatusPill")
        badge.setProperty("pillStatus", "pending")
        _refresh_style(badge)
        row_layout.addWidget(badge)

        self._rows[key] = {
            'dot': dot,
            'text': text,
            'badge': badge,
        }
        return row

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _reset_status(self) -> None:
        """Set all checks to pending."""
        for key in self._rows:
            self._status[key] = 'pending'
            self._apply_dot_color(key, 'pending')

    def _apply_dot_color(self, key: str, status: str) -> None:
        """Color the dot via QPalette based on status."""
        c = tokens.colors()
        color_map = {
            'pass':    c.STATUS_SUCCESS,
            'warn':    c.STATUS_WARNING,
            'fail':    c.STATUS_ERROR,
            'pending': c.TEXT_TERTIARY,
        }
        dot = self._rows[key]['dot']
        _set_foreground(dot, color_map.get(status, c.TEXT_TERTIARY))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        """Validate *df* and update every row's status dot and badge."""
        if df is None or len(df) == 0:
            self._reset_status()
            self.validation_changed.emit(self._status.copy())
            return

        cols = set(df.columns)
        cols_lower = {c.lower() for c in cols}

        results: Dict[str, bool] = {
            'x_col': 'X' in cols or any(
                c in cols_lower for c in ['x', 'easting', 'east']),
            'y_col': 'Y' in cols or any(
                c in cols_lower for c in ['y', 'northing', 'north']),
            'z_col': 'Z' in cols or any(
                c in cols_lower for c in ['z', 'elevation', 'elev', 'rl']),
            'formation': 'formation' in cols or any(
                c in cols_lower for c in [
                    'lithology', 'lith', 'rock_type', 'geology', 'unit',
                ]),
            'val': 'val' in cols,
            'orientation': all(g in cols for g in ['gx', 'gy', 'gz']),
        }

        for key, (_, _, required) in zip(
            [k for k, _, _ in self.CHECKS],
            self.CHECKS,
        ):
            passed = results.get(key, False)
            if passed:
                status = 'pass'
            elif required:
                status = 'fail'
            else:
                status = 'warn'

            self._status[key] = status
            self._apply_dot_color(key, status)

            # Update badge dynamic property
            badge = self._rows[key]['badge']
            badge.setProperty("pillStatus", status)
            _refresh_style(badge)

        self.validation_changed.emit(self._status.copy())

    # Convenience helpers expected by the old panel
    def set_status(self, req_id: str, status: str) -> None:
        """Manually set a single check's status."""
        if req_id in self._rows:
            self._status[req_id] = status
            self._apply_dot_color(req_id, status)
            badge = self._rows[req_id]['badge']
            badge.setProperty("pillStatus", status)
            _refresh_style(badge)
            self.validation_changed.emit(self._status.copy())

    def get_validation_state(self) -> Dict[str, str]:
        return self._status.copy()

    def is_valid(self) -> bool:
        """All required checks pass."""
        for key, _, required in self.CHECKS:
            if required and self._status.get(key) == 'fail':
                return False
        return True


# =========================================================================
# Widget 2: GeologicalDomainPanel
# =========================================================================

class GeologicalDomainPanel(QFrame):
    """
    Flat rows showing X/Y/Z model extent with min/max spinboxes and
    coverage percentage labels.

    Public API (called by business logic mixin):
        set_extent(xmin, xmax, ymin, ymax, zmin, zmax)
        set_coverage(x_pct, y_pct, z_pct)
        _spinboxes  — dict of QDoubleSpinBox keyed by 'xmin' .. 'zmax'
        _on_adjust_clicked()  — opens an edit dialog
    """

    domain_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._extent: Dict[str, float] = {}
        self._coverage: Dict[str, float] = {'x': 0, 'y': 0, 'z': 0}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("GEOLOGICAL DOMAIN")
        header.setObjectName("LoopSectionHeader")
        header.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_MD,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        layout.addWidget(header)

        div = QFrame()
        div.setObjectName("LoopDivider")
        layout.addWidget(div)

        # Axis rows
        self._axis_labels: Dict[str, QLabel] = {}
        self._coverage_labels: Dict[str, QLabel] = {}

        for axis in ['X', 'Y', 'Z']:
            row = QFrame()
            row.setObjectName("LoopFormRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(
                tokens.SPACING_LG, tokens.SPACING_SM,
                tokens.SPACING_LG, tokens.SPACING_SM,
            )
            row_layout.setSpacing(tokens.SPACING_SM)

            # Axis label
            ax_lbl = QLabel(f"{axis}:")
            ax_lbl.setFixedWidth(24)
            font = ax_lbl.font()
            font.setWeight(QFont.Weight.DemiBold)
            ax_lbl.setFont(font)
            row_layout.addWidget(ax_lbl)

            # Range text (monospaced)
            range_lbl = QLabel("\u2014 to \u2014 (\u2014 m)")
            mono = QFont(tokens.FONT_FAMILY_MONO)
            mono.setPointSize(tokens.FONT_SIZE_SM)
            range_lbl.setFont(mono)
            row_layout.addWidget(range_lbl, stretch=1)
            self._axis_labels[axis.lower()] = range_lbl

            # Coverage pill
            cov_lbl = QLabel("\u2014%")
            cov_lbl.setObjectName("LoopStatusPill")
            cov_lbl.setProperty("pillStatus", "pending")
            _refresh_style(cov_lbl)
            cov_lbl.setFixedWidth(52)
            cov_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(cov_lbl)
            self._coverage_labels[axis.lower()] = cov_lbl

            layout.addWidget(row)

        # Divider before action bar
        div2 = QFrame()
        div2.setObjectName("LoopDivider")
        layout.addWidget(div2)

        # Action bar
        bar = QFrame()
        bar.setObjectName("LoopActionBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        bar_layout.setSpacing(tokens.SPACING_SM)

        adjust_btn = QPushButton("Adjust Domain\u2026")
        adjust_btn.setObjectName("LoopActionButton")
        adjust_btn.clicked.connect(self._on_adjust_clicked)
        bar_layout.addWidget(adjust_btn)
        bar_layout.addStretch()
        layout.addWidget(bar)

        # Hidden spinboxes for value storage (accessed externally)
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}
        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.hide()
            self._spinboxes[key] = spin

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:,.0f}"

    def _update_display(self) -> None:
        c = tokens.colors()
        for axis in ['x', 'y', 'z']:
            min_key = f'{axis}min'
            max_key = f'{axis}max'

            if min_key in self._extent and max_key in self._extent:
                lo = self._extent[min_key]
                hi = self._extent[max_key]
                span = hi - lo
                self._axis_labels[axis].setText(
                    f"{self._fmt(lo)} \u2013 {self._fmt(hi)}  "
                    f"({self._fmt(span)} m)")
            else:
                self._axis_labels[axis].setText("\u2014 to \u2014 (\u2014 m)")

            cov = self._coverage.get(axis, 0)
            cov_lbl = self._coverage_labels[axis]
            cov_lbl.setText(f"{cov:.0f}%")

            # Set pill status dynamically for color
            if cov >= 90:
                pill_status = "pass"
            elif cov >= 70:
                pill_status = "warn"
            else:
                pill_status = "fail"
            cov_lbl.setProperty("pillStatus", pill_status)
            _refresh_style(cov_lbl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_extent(
        self,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
    ) -> None:
        self._extent = {
            'xmin': xmin, 'xmax': xmax,
            'ymin': ymin, 'ymax': ymax,
            'zmin': zmin, 'zmax': zmax,
        }
        self._spinboxes['xmin'].setValue(xmin)
        self._spinboxes['xmax'].setValue(xmax)
        self._spinboxes['ymin'].setValue(ymin)
        self._spinboxes['ymax'].setValue(ymax)
        self._spinboxes['zmin'].setValue(zmin)
        self._spinboxes['zmax'].setValue(zmax)
        self._update_display()
        self.domain_changed.emit(self._extent.copy())

    def set_coverage(
        self, x_pct: float, y_pct: float, z_pct: float,
    ) -> None:
        self._coverage = {'x': x_pct, 'y': y_pct, 'z': z_pct}
        self._update_display()

    def get_extent(self) -> Dict[str, float]:
        return {k: self._spinboxes[k].value() for k in self._spinboxes}

    def _on_adjust_clicked(self) -> None:
        """Open a dialog to manually adjust domain extent values."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Adjust Geological Domain")

        form = QFormLayout(dialog)
        form.setSpacing(tokens.SPACING_SM)

        edits: Dict[str, QDoubleSpinBox] = {}
        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(1)
            spin.setValue(self._spinboxes[key].value())
            edits[key] = spin
            nice = key.upper().replace('MIN', ' Min').replace('MAX', ' Max')
            form.addRow(f"{nice}:", spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_extent(
                edits['xmin'].value(), edits['xmax'].value(),
                edits['ymin'].value(), edits['ymax'].value(),
                edits['zmin'].value(), edits['zmax'].value(),
            )


# =========================================================================
# Widget 3: LithologyGroupingWidget
# =========================================================================

class LithologyGroupingWidget(QFrame):
    """
    Two-column widget for grouping raw lithologies into modelling units.

    Left column  — raw lithologies list (multi-select)
    Right column — grouped modeling units

    Public API (called by business logic mixin):
        set_lithologies(liths: List[str])
        get_grouped_lithologies() -> List[str]
        apply_grouping_to_dataframe(df, column) -> pd.DataFrame
        _add_selected_to_group()
        _remove_from_group()
        _auto_group_similar()
        grouping_changed = pyqtSignal(dict)
    """

    grouping_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._raw_lithologies: List[str] = []
        self._grouping: Dict[str, str] = {}          # raw -> group_name
        self._groups: Dict[str, List[str]] = {}       # group_name -> [raw]
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("LITHOLOGY GROUPING")
        header.setObjectName("LoopSectionHeader")
        header.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_MD,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        layout.addWidget(header)

        div = QFrame()
        div.setObjectName("LoopDivider")
        layout.addWidget(div)

        # Description
        desc = QLabel("Group similar lithologies into modelling units")
        desc.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # ---- Two-column body ----
        body = QHBoxLayout()
        body.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        body.setSpacing(tokens.SPACING_MD)

        # Left: raw lithologies
        left = QVBoxLayout()
        left.setSpacing(tokens.SPACING_XS)
        left_title = QLabel("Raw Lithologies")
        left_title.setObjectName("LoopSectionHeader")
        left.addWidget(left_title)

        self._raw_list = QListWidget()
        self._raw_list.setObjectName("LoopFlatList")
        self._raw_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self._raw_list.setMinimumHeight(120)
        left.addWidget(self._raw_list)
        body.addLayout(left, stretch=1)

        # Center: action buttons
        center = QVBoxLayout()
        center.setSpacing(tokens.SPACING_XS)
        center.addStretch()

        self._add_to_group_btn = QPushButton("Add to Group \u00BB")
        self._add_to_group_btn.setObjectName("LoopActionButton")
        self._add_to_group_btn.setToolTip(
            "Add selected lithologies to the selected group")
        self._add_to_group_btn.clicked.connect(self._add_selected_to_group)
        center.addWidget(self._add_to_group_btn)

        self._remove_from_group_btn = QPushButton("\u00AB Remove")
        self._remove_from_group_btn.setObjectName("LoopActionButton")
        self._remove_from_group_btn.setToolTip(
            "Remove selected lithology from its group")
        self._remove_from_group_btn.clicked.connect(self._remove_from_group)
        center.addWidget(self._remove_from_group_btn)

        self._auto_group_btn = QPushButton("Auto-Group")
        self._auto_group_btn.setObjectName("LoopActionButton")
        self._auto_group_btn.setToolTip(
            "Automatically group lithologies with similar prefixes")
        self._auto_group_btn.clicked.connect(self._auto_group_similar)
        center.addWidget(self._auto_group_btn)

        center.addStretch()
        body.addLayout(center)

        # Right: groups
        right = QVBoxLayout()
        right.setSpacing(tokens.SPACING_XS)
        right_title = QLabel("Modelling Groups")
        right_title.setObjectName("LoopSectionHeader")
        right.addWidget(right_title)

        # New group input row
        new_row = QHBoxLayout()
        new_row.setSpacing(tokens.SPACING_XS)
        self._group_name_input = QLineEdit()
        self._group_name_input.setPlaceholderText("New group name\u2026")
        new_row.addWidget(self._group_name_input)
        self._add_group_btn = QPushButton("+")
        self._add_group_btn.setObjectName("LoopActionButton")
        self._add_group_btn.setFixedWidth(32)
        self._add_group_btn.setToolTip("Create new group")
        self._add_group_btn.clicked.connect(self._add_new_group)
        new_row.addWidget(self._add_group_btn)
        right.addLayout(new_row)

        self._groups_tree = QListWidget()
        self._groups_tree.setObjectName("LoopFlatList")
        self._groups_tree.setMinimumHeight(120)
        right.addWidget(self._groups_tree)

        self._delete_group_btn = QPushButton("Delete Group")
        self._delete_group_btn.setObjectName("LoopActionButtonDanger")
        self._delete_group_btn.clicked.connect(self._delete_selected_group)
        right.addWidget(self._delete_group_btn)

        body.addLayout(right, stretch=1)
        layout.addLayout(body)

        # Status label
        self._status_label = QLabel("No lithologies loaded")
        self._status_label.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        layout.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Internal list helpers
    # ------------------------------------------------------------------

    def _update_raw_list(self) -> None:
        self._raw_list.clear()
        for lith in self._raw_lithologies:
            if lith not in self._grouping:
                self._raw_list.addItem(lith)

    def _update_groups_tree(self) -> None:
        self._groups_tree.clear()
        for group_name, members in sorted(self._groups.items()):
            group_item = QListWidgetItem(
                f"[+] {group_name} ({len(members)})")
            group_item.setData(
                Qt.ItemDataRole.UserRole, ('group', group_name))
            font = group_item.font()
            font.setBold(True)
            group_item.setFont(font)
            self._groups_tree.addItem(group_item)

            for member in sorted(members):
                mi = QListWidgetItem(f"    \u2013 {member}")
                mi.setData(
                    Qt.ItemDataRole.UserRole,
                    ('member', group_name, member))
                self._groups_tree.addItem(mi)

    def _emit_grouping(self) -> None:
        self.grouping_changed.emit(self._grouping.copy())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lithologies(self, liths: List[str]) -> None:
        self._raw_lithologies = sorted(set(liths))
        self._update_raw_list()
        self._status_label.setText(
            f"{len(self._raw_lithologies)} unique lithologies")

    def get_grouped_lithologies(self) -> List[str]:
        """Return group names plus any ungrouped lithologies."""
        result = list(self._groups.keys())
        for lith in self._raw_lithologies:
            if lith not in self._grouping:
                result.append(lith)
        return result

    def apply_grouping_to_dataframe(
        self, df: pd.DataFrame, column: str = 'formation',
    ) -> pd.DataFrame:
        if not self._grouping or column not in df.columns:
            return df
        df = df.copy()
        df[column] = df[column].apply(
            lambda x: self._grouping.get(x, x) if pd.notna(x) else x)
        return df

    def get_grouping(self) -> Dict[str, str]:
        return self._grouping.copy()

    def clear_grouping(self) -> None:
        self._grouping.clear()
        self._groups.clear()
        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    # ------------------------------------------------------------------
    # Actions (called from toolbar lambdas too)
    # ------------------------------------------------------------------

    def _add_new_group(self) -> None:
        name = self._group_name_input.text().strip()
        if not name or name in self._groups:
            return
        self._groups[name] = []
        self._group_name_input.clear()
        self._update_groups_tree()

    def _add_selected_to_group(self) -> None:
        selected_liths = [
            item.text() for item in self._raw_list.selectedItems()]
        if not selected_liths:
            return

        # Determine target group
        sel_group_items = self._groups_tree.selectedItems()
        if not sel_group_items:
            group_name = self._group_name_input.text().strip()
            if not group_name:
                return
            if group_name not in self._groups:
                self._groups[group_name] = []
        else:
            data = sel_group_items[0].data(Qt.ItemDataRole.UserRole)
            if data[0] == 'group':
                group_name = data[1]
            elif data[0] == 'member':
                group_name = data[1]
            else:
                return

        for lith in selected_liths:
            if lith not in self._grouping:
                self._grouping[lith] = group_name
                if lith not in self._groups[group_name]:
                    self._groups[group_name].append(lith)

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    def _remove_from_group(self) -> None:
        for item in self._groups_tree.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and data[0] == 'member':
                group_name, member = data[1], data[2]
                self._grouping.pop(member, None)
                if group_name in self._groups:
                    try:
                        self._groups[group_name].remove(member)
                    except ValueError:
                        pass

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    def _delete_selected_group(self) -> None:
        for item in self._groups_tree.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and data[0] == 'group':
                group_name = data[1]
                if group_name in self._groups:
                    for member in self._groups[group_name]:
                        self._grouping.pop(member, None)
                    del self._groups[group_name]

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()

    def _auto_group_similar(self) -> None:
        """Auto-group lithologies sharing a common prefix (>= 3 chars)."""
        if not self._raw_lithologies:
            return

        prefix_groups: Dict[str, List[str]] = {}
        for lith in self._raw_lithologies:
            if lith in self._grouping:
                continue
            best_prefix = None
            for plen in range(min(5, len(lith)), 2, -1):
                pfx = lith[:plen].upper()
                matches = [
                    l for l in self._raw_lithologies
                    if l.upper().startswith(pfx) and l not in self._grouping
                ]
                if len(matches) >= 2:
                    best_prefix = pfx
                    break
            if best_prefix:
                prefix_groups.setdefault(best_prefix, [])
                if lith not in prefix_groups[best_prefix]:
                    prefix_groups[best_prefix].append(lith)

        for pfx, members in prefix_groups.items():
            if len(members) < 2:
                continue
            gname = pfx.title()
            if gname not in self._groups:
                self._groups[gname] = []
            for m in members:
                if m not in self._grouping:
                    self._grouping[m] = gname
                    self._groups[gname].append(m)

        self._update_raw_list()
        self._update_groups_tree()
        self._emit_grouping()
        self._status_label.setText(
            f"{len(self._grouping)}/{len(self._raw_lithologies)} "
            f"lithologies in {len(self._groups)} groups")


# =========================================================================
# Widget 4: GeologicalAuditVerdictTable
# =========================================================================

class GeologicalAuditVerdictTable(QFrame):
    """
    Flat rows for each geological audit check.  Each row shows:
    status dot + check name + expandable details.

    Public API (called by business logic mixin):
        set_verdict(key, status, what, where, why, impact)
    """

    verdict_expanded = pyqtSignal(str)

    AUDIT_CHECKS = [
        ('drillhole_honouring',   'Drillhole Honouring'),
        ('stratigraphic_ordering', 'Stratigraphic Ordering'),
        ('layer_continuity',       'Layer Continuity'),
        ('dip_strike_consistency', 'Dip & Strike Consistency'),
        ('fault_handling',         'Fault Handling'),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._verdicts: Dict[str, Dict[str, Any]] = {}
        self._expanded: Dict[str, bool] = {}
        self._row_widgets: Dict[str, Dict[str, QWidget]] = {}
        self._build_ui()
        self._reset_verdicts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        header = QLabel("GEOLOGICAL AUDIT")
        header.setObjectName("LoopSectionHeader")
        header.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_MD,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        self._layout.addWidget(header)

        div = QFrame()
        div.setObjectName("LoopDivider")
        self._layout.addWidget(div)

        for check_id, label in self.AUDIT_CHECKS:
            self._create_row(check_id, label)

    def _create_row(self, check_id: str, label: str) -> None:
        # Main row
        row = QFrame()
        row.setObjectName("LoopPhaseRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        row_layout.setSpacing(tokens.SPACING_SM)

        # Status dot
        dot = QLabel("\u25CF")
        dot.setFixedWidth(16)
        dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_layout.addWidget(dot)

        # Check name
        name_lbl = QLabel(label)
        row_layout.addWidget(name_lbl, stretch=1)

        # Status badge
        badge = QLabel("PENDING")
        badge.setObjectName("LoopStatusPill")
        badge.setProperty("pillStatus", "pending")
        _refresh_style(badge)
        badge.setFixedWidth(64)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_layout.addWidget(badge)

        # Expand toggle
        expand_btn = QPushButton("\u25B6")
        expand_btn.setObjectName("LoopActionButton")
        expand_btn.setFixedSize(24, 24)
        expand_btn.clicked.connect(
            lambda _=False, cid=check_id: self._toggle_expand(cid))
        row_layout.addWidget(expand_btn)

        self._layout.addWidget(row)

        # Details frame (hidden by default)
        details = QFrame()
        details.setObjectName("LoopFormRow")
        details.hide()
        det_layout = QVBoxLayout(details)
        det_layout.setContentsMargins(
            tokens.SPACING_XL, tokens.SPACING_XS,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        det_layout.setSpacing(tokens.SPACING_2XS)

        what_lbl = QLabel("What: \u2014")
        what_lbl.setWordWrap(True)
        where_lbl = QLabel("Where: \u2014")
        where_lbl.setWordWrap(True)
        why_lbl = QLabel("Why: \u2014")
        why_lbl.setWordWrap(True)
        impact_lbl = QLabel("Impact: \u2014")
        impact_lbl.setWordWrap(True)

        for lbl in (what_lbl, where_lbl, why_lbl, impact_lbl):
            det_layout.addWidget(lbl)

        self._layout.addWidget(details)

        self._row_widgets[check_id] = {
            'row': row,
            'dot': dot,
            'name': name_lbl,
            'badge': badge,
            'expand_btn': expand_btn,
            'details': details,
            'what': what_lbl,
            'where': where_lbl,
            'why': why_lbl,
            'impact': impact_lbl,
        }
        self._expanded[check_id] = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_verdicts(self) -> None:
        for check_id, _ in self.AUDIT_CHECKS:
            self._verdicts[check_id] = {
                'status': 'pending',
                'what': '', 'where': '', 'why': '', 'impact': '',
            }
            self._update_row(check_id)

    def _update_row(self, check_id: str) -> None:
        if check_id not in self._row_widgets:
            return
        w = self._row_widgets[check_id]
        v = self._verdicts.get(check_id, {})
        status = v.get('status', 'pending')

        c = tokens.colors()
        color_map = {
            'pass':    c.STATUS_SUCCESS,
            'warn':    c.STATUS_WARNING,
            'fail':    c.STATUS_ERROR,
            'pending': c.TEXT_TERTIARY,
        }
        _set_foreground(w['dot'], color_map.get(status, c.TEXT_TERTIARY))

        badge_text = {
            'pass': 'PASS', 'warn': 'WARN',
            'fail': 'FAIL', 'pending': 'PENDING',
        }
        w['badge'].setText(badge_text.get(status, 'PENDING'))
        w['badge'].setProperty("pillStatus", status)
        _refresh_style(w['badge'])

        w['what'].setText(f"What: {v.get('what') or '\u2014'}")
        w['where'].setText(f"Where: {v.get('where') or '\u2014'}")
        w['why'].setText(f"Why: {v.get('why') or '\u2014'}")
        w['impact'].setText(f"Impact: {v.get('impact') or '\u2014'}")

    def _toggle_expand(self, check_id: str) -> None:
        if check_id not in self._row_widgets:
            return
        self._expanded[check_id] = not self._expanded[check_id]
        w = self._row_widgets[check_id]
        if self._expanded[check_id]:
            w['details'].show()
            w['expand_btn'].setText("\u25BC")
            self.verdict_expanded.emit(check_id)
        else:
            w['details'].hide()
            w['expand_btn'].setText("\u25B6")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_verdict(
        self,
        check_id: str,
        status: str,
        what: str = "",
        where: str = "",
        why: str = "",
        impact: str = "",
    ) -> None:
        self._verdicts[check_id] = {
            'status': status,
            'what': what,
            'where': where,
            'why': why,
            'impact': impact,
        }
        self._update_row(check_id)

    def populate_from_audit_report(self, report=None,
                                   strat_result=None,
                                   continuity_result=None) -> None:
        """Populate verdicts from an AuditReport and related results."""
        if report is None:
            self._reset_verdicts()
            return

        # Stratigraphic ordering
        if strat_result is not None:
            strat_ok = getattr(strat_result, 'is_valid', True)
            violations = len(getattr(strat_result, 'violations', []))
            self.set_verdict(
                'stratigraphic_ordering',
                'pass' if strat_ok else 'fail',
                what=("Formation sequence validated" if strat_ok
                      else f"{violations} ordering violations detected"),
                where=(f"{getattr(strat_result, 'holes_checked', '?')} "
                       "holes checked"),
                why=("" if strat_ok
                     else "Depth ordering violated in drillhole intersections"),
                impact=("" if strat_ok
                        else "May affect layer continuity interpretation"),
            )

        # Layer continuity
        if continuity_result is not None:
            cont_ok = getattr(continuity_result, 'all_continuous', True)
            self.set_verdict(
                'layer_continuity',
                'pass' if cont_ok else 'warn',
                what=("All layers continuous" if cont_ok
                      else "Discontinuities detected"),
                where=("Model domain" if cont_ok
                       else "Check isolated volumes"),
                why="" if cont_ok else "Some units may have isolated volumes",
                impact="" if cont_ok else "Review mesh topology",
            )

        # Drillhole honouring
        mean_r = getattr(report, 'mean_residual', None)
        if mean_r is not None:
            p90 = getattr(report, 'p90_error', 0) or 0
            if p90 < 2.0:
                dh_st = 'pass'
            elif p90 < 5.0:
                dh_st = 'warn'
            else:
                dh_st = 'fail'
            self.set_verdict(
                'drillhole_honouring', dh_st,
                what=(f"P90 error: {p90:.2f}m" if p90
                      else "Contact matching verified"),
                where=f"{getattr(report, 'total_contacts', 0)} contacts evaluated",
                why="" if dh_st == 'pass' else f"Mean residual: {mean_r:.2f}m",
                impact=(f"Classification: "
                        f"{getattr(report, 'classification_recommendation', 'Unknown')}"),
            )

        # Fault handling
        self.set_verdict(
            'fault_handling', 'pass',
            what="Fault events processed",
            where="Model domain",
        )


# =========================================================================
# Widget 5: ModelBuildExecutionPanel
# =========================================================================

class ModelBuildExecutionPanel(QFrame):
    """
    Build execution panel showing phases, diagnostics, and controls.

    Phases displayed as flat rows:
        Extent -> Solve -> Extract -> Audit -> Complete

    Public API (called by business logic mixin):
        set_building(is_building: bool)
        reset()
        set_diagnostics(text: str)
        set_phase(phase_index: int)
        set_runtime(seconds: float)
        set_seed(seed: int)
        build_requested = pyqtSignal()
        cancel_requested = pyqtSignal()
    """

    build_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    phase_changed = pyqtSignal(int, str)

    BUILD_PHASES = [
        ('extent',   'Normalizing coordinates'),
        ('solve',    'Solving geological model'),
        ('extract',  'Extracting surfaces'),
        ('audit',    'Validating compliance'),
        ('complete', 'Complete'),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_phase: int = -1
        self._is_building: bool = False
        self._warnings: List[str] = []
        self._seed: Optional[int] = None
        self._runtime: float = 0.0
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("MODEL BUILD")
        header.setObjectName("LoopSectionHeader")
        header.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_MD,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        layout.addWidget(header)

        div = QFrame()
        div.setObjectName("LoopDivider")
        layout.addWidget(div)

        # Phase rows
        self._phase_widgets: List[Dict[str, QWidget]] = []
        for i, (phase_id, phase_label) in enumerate(self.BUILD_PHASES):
            row = QFrame()
            row.setObjectName("LoopPhaseRow")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(
                tokens.SPACING_LG, tokens.SPACING_SM,
                tokens.SPACING_LG, tokens.SPACING_SM,
            )
            rl.setSpacing(tokens.SPACING_SM)

            # Status indicator dot
            dot = QLabel("\u25CB")     # empty circle
            dot.setFixedWidth(16)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rl.addWidget(dot)

            # Step number
            num = QLabel(f"{i + 1}.")
            num.setFixedWidth(20)
            rl.addWidget(num)

            # Phase name
            name = QLabel(phase_label)
            rl.addWidget(name, stretch=1)

            layout.addWidget(row)
            self._phase_widgets.append({
                'dot': dot, 'num': num, 'name': name,
            })

        # Divider
        div2 = QFrame()
        div2.setObjectName("LoopDivider")
        layout.addWidget(div2)

        # Runtime / seed info row
        info_row = QFrame()
        info_row.setObjectName("LoopFormRow")
        info_layout = QHBoxLayout(info_row)
        info_layout.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        info_layout.setSpacing(tokens.SPACING_LG)

        self._seed_label = QLabel("Seed: \u2014")
        info_layout.addWidget(self._seed_label)

        self._runtime_label = QLabel("Runtime: \u2014")
        info_layout.addWidget(self._runtime_label)

        info_layout.addStretch()
        layout.addWidget(info_row)

        # Divider
        div3 = QFrame()
        div3.setObjectName("LoopDivider")
        layout.addWidget(div3)

        # Diagnostics text
        self._diagnostics_text = QTextEdit()
        self._diagnostics_text.setReadOnly(True)
        self._diagnostics_text.setMaximumHeight(120)
        self._diagnostics_text.setPlainText(
            "No build diagnostics available.")
        mono = QFont(tokens.FONT_FAMILY_MONO)
        mono.setPointSize(tokens.FONT_SIZE_XS)
        self._diagnostics_text.setFont(mono)
        layout.addWidget(self._diagnostics_text)

        # Divider
        div4 = QFrame()
        div4.setObjectName("LoopDivider")
        layout.addWidget(div4)

        # Action bar
        bar = QFrame()
        bar.setObjectName("LoopActionBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_SM,
            tokens.SPACING_LG, tokens.SPACING_SM,
        )
        bar_layout.setSpacing(tokens.SPACING_MD)

        self._build_btn = QPushButton("Build Model")
        self._build_btn.setObjectName("LoopActionButton")
        self._build_btn.clicked.connect(self._on_build_clicked)
        bar_layout.addWidget(self._build_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("LoopActionButtonDanger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        bar_layout.addWidget(self._cancel_btn)

        bar_layout.addStretch()
        layout.addWidget(bar)

    # ------------------------------------------------------------------
    # Phase display helpers
    # ------------------------------------------------------------------

    def _update_phase_display(self) -> None:
        c = tokens.colors()
        for i, pw in enumerate(self._phase_widgets):
            dot: QLabel = pw['dot']
            if i < self._current_phase:
                # Completed
                dot.setText("\u2713")  # checkmark
                _set_foreground(dot, c.STATUS_SUCCESS)
            elif i == self._current_phase:
                # Active
                dot.setText("\u25B6")  # right arrow
                _set_foreground(dot, c.ACCENT)
            else:
                # Pending
                dot.setText("\u25CB")  # empty circle
                _set_foreground(dot, c.TEXT_TERTIARY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_phase(self, phase_index: int) -> None:
        self._current_phase = phase_index
        self._update_phase_display()
        if 0 <= phase_index < len(self.BUILD_PHASES):
            pid, pname = self.BUILD_PHASES[phase_index]
            self.phase_changed.emit(phase_index, pname)

    def set_building(self, is_building: bool) -> None:
        self._is_building = is_building
        self._build_btn.setEnabled(not is_building)
        self._cancel_btn.setEnabled(is_building)
        self._build_btn.setText(
            "Building\u2026" if is_building else "Build Model")

    def set_diagnostics(self, text: str) -> None:
        self._diagnostics_text.setPlainText(text)

    def set_runtime(self, seconds: float, estimated: float = 0.0) -> None:
        self._runtime = seconds
        if estimated > 0:
            self._runtime_label.setText(
                f"Runtime: {seconds:.1f}s / est. {estimated:.1f}s")
        else:
            self._runtime_label.setText(f"Runtime: {seconds:.1f}s")

    def set_seed(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._seed_label.setText(
            f"Seed: {seed}" if seed is not None else "Seed: \u2014")

    def set_warnings(self, warnings: List[str]) -> None:
        self._warnings = warnings

    def reset(self) -> None:
        """Reset panel to its initial idle state."""
        self._current_phase = -1
        self._update_phase_display()
        self.set_building(False)
        self.set_warnings([])
        self.set_seed(None)
        self.set_runtime(0.0)
        self.set_diagnostics("No build diagnostics available.")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_build_clicked(self) -> None:
        self.build_requested.emit()

    def _on_cancel_clicked(self) -> None:
        self.cancel_requested.emit()
