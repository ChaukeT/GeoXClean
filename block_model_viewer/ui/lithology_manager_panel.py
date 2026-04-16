"""
Lithology Manager Panel -- Lithology grouping, stratigraphic column, domain codes.
===================================================================================

3-tab panel:
  Tab 1 -- Raw Lithology Codes: view unique codes with statistics
  Tab 2 -- Grouping Editor: merge codes into modelling units
  Tab 3 -- Stratigraphic Column: order units, define contacts, assign domains

Provides domain codes to ARBF for domain-constrained estimation.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QCheckBox, QWidget, QFrame,
    QTabWidget, QFormLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem, QScrollArea,
    QSizePolicy, QColorDialog, QMessageBox,
    QFileDialog, QAbstractItemView, QInputDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QBrush

from .base_analysis_panel import BaseAnalysisPanel
from .collapsible_group import CollapsibleGroup
from .design_tokens import tokens
from .modern_styles import ModernColors, get_theme_colors
from .panel_toolkit import (
    section, make_form, form_row, make_combo,
    action_button, hint_label, separator,
    PANEL_MARGINS, PANEL_SPACING,
)
from .panel_manager import PanelCategory, DockArea

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Tab scroll helper (same pattern as ARBF)
# ═══════════════════════════════════════════════════════════════════

def _make_tab_scroll(tab_widget: QTabWidget, name: str) -> Tuple[QVBoxLayout, QWidget]:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    content = QWidget()
    lay = QVBoxLayout(content)
    lay.setContentsMargins(*PANEL_MARGINS)
    lay.setSpacing(PANEL_SPACING)
    scroll.setWidget(content)
    tab_widget.addTab(scroll, name)
    return lay, content


# ═══════════════════════════════════════════════════════════════════
# Default colours for modelling units
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_UNIT_COLORS = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


class LithologyManagerPanel(BaseAnalysisPanel):
    """Lithology manager for grouping codes into modelling units."""

    PANEL_ID = "LithologyManagerPanel"
    PANEL_NAME = "Lithology Manager"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    task_name = "lithology_manager"

    # Signals
    domainCodesAssigned = pyqtSignal(object)  # Emits domain codes dict
    contactsExtracted = pyqtSignal(object)    # Emits contacts DataFrame

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._ui_ready = False
        self.registry = None
        self.main_window = None

        # State
        self._drillhole_data: Optional[Dict] = None
        self._lithology_df: Optional[pd.DataFrame] = None
        self._unique_codes: List[str] = []
        self._code_stats: Dict[str, Dict] = {}

        # Grouping: {unit_name: [raw_codes]}
        self._grouping: Dict[str, List[str]] = {}
        # Unit metadata
        self._unit_colors: Dict[str, str] = {}
        self._unit_types: Dict[str, str] = {}

        # Stratigraphic column (ordered list of unit names)
        self._strat_order: List[str] = []
        self._contact_types: Dict[str, str] = {}  # "UNIT1_UNIT2" -> type

        super().__init__(parent=parent, panel_id="lithology_manager")

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_base_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        root.addWidget(self.tab_widget, stretch=1)

        self._build_tab1_raw_codes()
        self._build_tab2_grouping()
        self._build_tab3_stratigraphy()

        self._ui_ready = True
        self._init_registry_connections()
        self._is_initialized = True

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Card")
        frame.setStyleSheet(
            f"QFrame#Card {{"
            f"  background-color: {ModernColors.ELEVATED_BG};"
            f"  border-bottom: 1px solid {ModernColors.DIVIDER};"
            f"}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)

        title = QLabel("Lithology Manager")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        lay.addWidget(title)
        lay.addStretch()

        sub = QLabel("Group | Order | Assign Domains")
        sub.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        lay.addWidget(sub)
        lay.addStretch()

        self.status_label = QLabel("No data loaded")
        self.status_label.setObjectName("statusLabel")
        lay.addWidget(self.status_label)
        return frame

    def _init_registry_connections(self) -> None:
        """Connect to data registry signals."""
        try:
            from ..core.data_registry import DataRegistry
            reg = DataRegistry.instance()
            self.registry = reg
            if hasattr(reg, 'signals'):
                reg.signals.drillholeDataLoaded.connect(self._on_drillhole_loaded)
        except Exception as e:
            logger.debug("Registry connection: %s", e)

    def _on_drillhole_loaded(self, data: Any) -> None:
        if self._ui_ready:
            self.status_label.setText("Drillhole data available")

    # ══════════════════════════════════════════════════════════════
    # TAB 1 -- RAW LITHOLOGY CODES
    # ══════════════════════════════════════════════════════════════

    def _build_tab1_raw_codes(self) -> None:
        lay, content = _make_tab_scroll(self.tab_widget, "Raw Codes")

        # Data source
        grp = section("Data Source")
        form = make_form()

        self.dataset_combo = make_combo(tooltip="Select drillhole dataset")
        form_row(form, "Dataset:", self.dataset_combo)

        self.lith_col_combo = make_combo(tooltip="Column containing lithology codes")
        form_row(form, "Lithology Column:", self.lith_col_combo)

        btn_load = action_button("Load Lithology Data", style="primary",
                                 tooltip="Load unique lithology codes from selected dataset")
        btn_load.clicked.connect(self._on_load_lithology)
        form.addRow("", btn_load)

        grp.add_layout(form)
        lay.addWidget(grp)

        # Statistics table
        grp_table = section("Lithology Codes")

        self.code_table = QTableWidget()
        self.code_table.setColumnCount(6)
        self.code_table.setHorizontalHeaderLabels([
            "Code", "Count", "%", "Mean Depth", "Depth Range", "Status",
        ])
        self.code_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch,
        )
        self.code_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows,
        )
        self.code_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self.code_table.setSortingEnabled(True)
        self.code_table.setMinimumHeight(200)
        grp_table.add_widget(self.code_table)

        # Auto-detect buttons
        btn_row = QHBoxLayout()
        btn_prefix = action_button("Auto-Group by Prefix", style="secondary",
                                   tooltip="Group codes sharing common prefixes")
        btn_prefix.clicked.connect(self._on_auto_group_prefix)
        btn_row.addWidget(btn_prefix)

        btn_merge = action_button("Suggest Merges", style="secondary",
                                  tooltip="Find codes likely to be typos (Levenshtein distance)")
        btn_merge.clicked.connect(self._on_suggest_merges)
        btn_row.addWidget(btn_merge)

        grp_table.add_layout(btn_row)
        lay.addWidget(grp_table)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 2 -- GROUPING EDITOR
    # ══════════════════════════════════════════════════════════════

    def _build_tab2_grouping(self) -> None:
        lay, content = _make_tab_scroll(self.tab_widget, "Grouping")

        # Two-panel layout
        split = QHBoxLayout()
        split.setSpacing(12)

        # Left: ungrouped codes
        left_frame = QFrame()
        left_frame.setObjectName("Card")
        left_lay = QVBoxLayout(left_frame)
        left_lay.setContentsMargins(8, 8, 8, 8)

        left_lay.addWidget(QLabel("Ungrouped Codes"))

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter codes...")
        self.filter_edit.textChanged.connect(self._on_filter_changed)
        left_lay.addWidget(self.filter_edit)

        self.ungrouped_list = QListWidget()
        self.ungrouped_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        left_lay.addWidget(self.ungrouped_list)

        split.addWidget(left_frame, stretch=1)

        # Right: modelling units
        right_frame = QFrame()
        right_frame.setObjectName("Card")
        right_lay = QVBoxLayout(right_frame)
        right_lay.setContentsMargins(8, 8, 8, 8)

        right_lay.addWidget(QLabel("Modelling Units"))

        # Assignment controls
        assign_row = QHBoxLayout()
        self.target_unit_combo = make_combo(tooltip="Target modelling unit")
        assign_row.addWidget(self.target_unit_combo)

        btn_assign = action_button("Assign", style="primary",
                                   tooltip="Assign selected codes to target unit")
        btn_assign.clicked.connect(self._on_assign_codes)
        assign_row.addWidget(btn_assign)
        right_lay.addLayout(assign_row)

        # Unit list (scrollable area for unit cards)
        self.units_scroll = QScrollArea()
        self.units_scroll.setWidgetResizable(True)
        self.units_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.units_container = QWidget()
        self.units_layout = QVBoxLayout(self.units_container)
        self.units_layout.setSpacing(8)
        self.units_layout.addStretch()
        self.units_scroll.setWidget(self.units_container)
        right_lay.addWidget(self.units_scroll, stretch=1)

        # Add/remove unit buttons
        unit_btn_row = QHBoxLayout()
        btn_add = action_button("Add Unit", style="secondary")
        btn_add.clicked.connect(self._on_add_unit)
        unit_btn_row.addWidget(btn_add)

        btn_remove = action_button("Remove Unit", style="danger")
        btn_remove.clicked.connect(self._on_remove_unit)
        unit_btn_row.addWidget(btn_remove)
        right_lay.addLayout(unit_btn_row)

        split.addWidget(right_frame, stretch=2)
        lay.addLayout(split)

        # Save/load
        io_row = QHBoxLayout()
        btn_save = action_button("Save Grouping", style="secondary")
        btn_save.clicked.connect(self._on_save_grouping)
        io_row.addWidget(btn_save)

        btn_load = action_button("Load Grouping", style="secondary")
        btn_load.clicked.connect(self._on_load_grouping)
        io_row.addWidget(btn_load)
        lay.addLayout(io_row)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 -- STRATIGRAPHIC COLUMN
    # ══════════════════════════════════════════════════════════════

    def _build_tab3_stratigraphy(self) -> None:
        lay, content = _make_tab_scroll(self.tab_widget, "Stratigraphy")

        # Column builder
        grp_col = section("Stratigraphic Column")
        col_lay = QVBoxLayout()

        col_lay.addWidget(QLabel("Units (top = youngest, bottom = oldest):"))

        self.strat_list = QListWidget()
        self.strat_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.strat_list.setMinimumHeight(150)
        col_lay.addWidget(self.strat_list)

        move_row = QHBoxLayout()
        btn_up = action_button("Move Up", style="secondary")
        btn_up.clicked.connect(self._on_move_up)
        move_row.addWidget(btn_up)

        btn_down = action_button("Move Down", style="secondary")
        btn_down.clicked.connect(self._on_move_down)
        move_row.addWidget(btn_down)

        btn_refresh = action_button("Refresh from Grouping", style="secondary")
        btn_refresh.clicked.connect(self._on_refresh_strat)
        move_row.addWidget(btn_refresh)
        col_lay.addLayout(move_row)

        grp_col.add_layout(col_lay)
        lay.addWidget(grp_col)

        # Contact types
        grp_contacts = section("Contact Types")
        self.contact_table = QTableWidget()
        self.contact_table.setColumnCount(3)
        self.contact_table.setHorizontalHeaderLabels([
            "Surface", "Contact Type", "N Contacts",
        ])
        self.contact_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch,
        )
        self.contact_table.setMinimumHeight(120)
        grp_contacts.add_widget(self.contact_table)
        lay.addWidget(grp_contacts)

        # Actions
        grp_actions = section("Actions")
        act_lay = QVBoxLayout()

        btn_extract = action_button("Extract Contacts", style="primary",
                                    tooltip="Extract contacts from drillhole lithology using grouping")
        btn_extract.clicked.connect(self._on_extract_contacts)
        act_lay.addWidget(btn_extract)

        btn_assign = action_button("Assign Domain Codes", style="primary",
                                   tooltip="Assign integer domain codes to composites")
        btn_assign.clicked.connect(self._on_assign_domains)
        act_lay.addWidget(btn_assign)

        btn_send_arbf = action_button("Send to ARBF", style="primary",
                                      tooltip="Make domain codes available to ARBF panel")
        btn_send_arbf.clicked.connect(self._on_send_to_arbf)
        act_lay.addWidget(btn_send_arbf)

        self.action_status = QLabel("")
        self.action_status.setObjectName("statusLabel")
        act_lay.addWidget(self.action_status)

        grp_actions.add_layout(act_lay)
        lay.addWidget(grp_actions)

        lay.addStretch()

    # ══════════════════════════════════════════════════════════════
    # TAB 1 LOGIC
    # ══════════════════════════════════════════════════════════════

    def _on_load_lithology(self) -> None:
        """Load lithology data from registry and populate table."""
        if self.registry is None:
            self.status_label.setText("No registry available")
            return

        try:
            dh_data = self.registry.get_drillhole_data()
        except Exception:
            dh_data = None

        if dh_data is None:
            self.status_label.setText("No drillhole data loaded")
            return

        self._drillhole_data = dh_data

        # Find lithology dataframe
        lith_df = None
        if isinstance(dh_data, dict):
            lith_df = dh_data.get("lithology")
            if lith_df is None:
                # Try assays with a lith_code column
                assays = dh_data.get("assays")
                if assays is not None and "lith_code" in assays.columns:
                    lith_df = assays

        if lith_df is None or lith_df.empty:
            self.status_label.setText("No lithology data found")
            return

        self._lithology_df = lith_df

        # Auto-detect lithology column
        self.lith_col_combo.clear()
        string_cols = []
        for col in lith_df.columns:
            if lith_df[col].dtype == object or str(lith_df[col].dtype) == "category":
                string_cols.append(col)
        if not string_cols:
            string_cols = list(lith_df.columns)
        self.lith_col_combo.addItems(string_cols)

        # Prefer lith_code if present
        if "lith_code" in string_cols:
            self.lith_col_combo.setCurrentText("lith_code")

        # Populate table
        self._refresh_code_table()
        self.status_label.setText(f"Loaded {len(lith_df)} records")

    def _refresh_code_table(self) -> None:
        """Populate the code statistics table."""
        if self._lithology_df is None:
            return

        lith_col = self.lith_col_combo.currentText()
        if not lith_col or lith_col not in self._lithology_df.columns:
            return

        df = self._lithology_df
        codes = df[lith_col].astype(str).str.strip()
        total = len(codes)

        # Compute statistics per code
        stats = {}
        for code, group in df.groupby(codes):
            code = str(code).strip()
            count = len(group)

            depth_col = None
            for dc in ["depth_from", "from", "FROM"]:
                if dc in group.columns:
                    depth_col = dc
                    break

            if depth_col:
                depths = group[depth_col].astype(float)
                mean_depth = float(depths.mean())
                depth_range = f"{depths.min():.1f}-{depths.max():.1f}"
            else:
                mean_depth = 0.0
                depth_range = "N/A"

            # Check if grouped
            status = "Ungrouped"
            for unit, unit_codes in self._grouping.items():
                if code in unit_codes:
                    status = f"Grouped ({unit})"
                    break

            stats[code] = {
                "count": count,
                "pct": 100.0 * count / max(total, 1),
                "mean_depth": mean_depth,
                "depth_range": depth_range,
                "status": status,
            }

        self._code_stats = stats
        self._unique_codes = sorted(stats.keys())

        # Populate table
        self.code_table.setSortingEnabled(False)
        self.code_table.setRowCount(len(self._unique_codes))

        for i, code in enumerate(self._unique_codes):
            s = stats[code]
            self.code_table.setItem(i, 0, QTableWidgetItem(code))
            self.code_table.setItem(i, 1, QTableWidgetItem(str(s["count"])))
            self.code_table.setItem(i, 2, QTableWidgetItem(f"{s['pct']:.1f}"))
            self.code_table.setItem(i, 3, QTableWidgetItem(f"{s['mean_depth']:.1f}"))
            self.code_table.setItem(i, 4, QTableWidgetItem(s["depth_range"]))

            status_item = QTableWidgetItem(s["status"])
            if "Grouped" in s["status"]:
                status_item.setForeground(QBrush(QColor("#3cb44b")))
            elif s["status"] == "Excluded":
                status_item.setForeground(QBrush(QColor("#e6194B")))
            self.code_table.setItem(i, 5, status_item)

        self.code_table.setSortingEnabled(True)

        # Update ungrouped list in Tab 2
        self._refresh_ungrouped_list()

    def _on_auto_group_prefix(self) -> None:
        """Auto-group codes by common prefix."""
        if not self._unique_codes:
            return

        # Find common prefixes (min 2 chars)
        prefix_groups: Dict[str, List[str]] = defaultdict(list)
        for code in self._unique_codes:
            # Try prefixes of length 2, 3, 4
            for plen in [3, 2]:
                if len(code) >= plen:
                    prefix = code[:plen].upper().rstrip("_- ")
                    prefix_groups[prefix].append(code)
                    break

        # Only suggest groups with 2+ codes
        suggested = {p: codes for p, codes in prefix_groups.items() if len(codes) >= 2}

        if not suggested:
            self.status_label.setText("No prefix groups found")
            return

        # Create units from prefixes
        for prefix, codes in suggested.items():
            unit_name = prefix.upper()
            if unit_name not in self._grouping:
                self._grouping[unit_name] = []
            for code in codes:
                if code not in self._grouping[unit_name]:
                    # Check not already in another group
                    already = False
                    for existing_codes in self._grouping.values():
                        if code in existing_codes:
                            already = True
                            break
                    if not already:
                        self._grouping[unit_name].append(code)

            if unit_name not in self._unit_colors:
                idx = len(self._unit_colors) % len(_DEFAULT_UNIT_COLORS)
                self._unit_colors[unit_name] = _DEFAULT_UNIT_COLORS[idx]
            if unit_name not in self._unit_types:
                self._unit_types[unit_name] = "stratigraphic"

        self._refresh_grouping_ui()
        self._refresh_code_table()
        self.status_label.setText(f"Created {len(suggested)} prefix groups")

    def _on_suggest_merges(self) -> None:
        """Suggest merges using Levenshtein distance."""
        if len(self._unique_codes) < 2:
            return

        suggestions = []
        codes = self._unique_codes

        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                dist = _levenshtein(codes[i].upper(), codes[j].upper())
                if 0 < dist <= 2:
                    suggestions.append((codes[i], codes[j], dist))

        if not suggestions:
            QMessageBox.information(self, "Merge Suggestions", "No similar codes found.")
            return

        msg = "Possible typos/synonyms:\n\n"
        for a, b, d in suggestions[:20]:
            msg += f"  {a}  <->  {b}  (distance={d})\n"

        QMessageBox.information(self, "Merge Suggestions", msg)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 LOGIC
    # ══════════════════════════════════════════════════════════════

    def _refresh_ungrouped_list(self) -> None:
        """Refresh the ungrouped codes list."""
        self.ungrouped_list.clear()
        grouped_codes = set()
        for codes in self._grouping.values():
            grouped_codes.update(codes)

        for code in self._unique_codes:
            if code not in grouped_codes:
                self.ungrouped_list.addItem(code)

        self._refresh_target_combo()

    def _refresh_target_combo(self) -> None:
        """Refresh the target unit combo."""
        current = self.target_unit_combo.currentText()
        self.target_unit_combo.clear()
        self.target_unit_combo.addItems(sorted(self._grouping.keys()))
        if current and self.target_unit_combo.findText(current) >= 0:
            self.target_unit_combo.setCurrentText(current)

    def _refresh_grouping_ui(self) -> None:
        """Rebuild the unit cards in the grouping tab."""
        # Clear existing cards
        while self.units_layout.count() > 1:  # keep the stretch
            item = self.units_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for unit_name in sorted(self._grouping.keys()):
            card = self._make_unit_card(unit_name)
            self.units_layout.insertWidget(self.units_layout.count() - 1, card)

        self._refresh_ungrouped_list()
        self._refresh_target_combo()

    def _make_unit_card(self, unit_name: str) -> QFrame:
        """Create a unit card widget."""
        card = QFrame()
        card.setObjectName("Card")
        card.setStyleSheet(
            f"QFrame#Card {{ border: 1px solid {ModernColors.DIVIDER}; "
            f"border-radius: 4px; padding: 4px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setSpacing(4)
        lay.setContentsMargins(8, 6, 8, 6)

        # Header row: color swatch + name + type
        header = QHBoxLayout()

        color = self._unit_colors.get(unit_name, "#808080")
        swatch = QPushButton("")
        swatch.setFixedSize(20, 20)
        swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #555;")
        swatch.clicked.connect(lambda _, u=unit_name: self._on_pick_color(u))
        header.addWidget(swatch)

        name_label = QLabel(f"<b>{unit_name}</b>")
        name_label.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")
        header.addWidget(name_label)
        header.addStretch()

        utype = self._unit_types.get(unit_name, "stratigraphic")
        type_label = QLabel(utype)
        type_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        header.addWidget(type_label)

        lay.addLayout(header)

        # Code list
        codes = self._grouping.get(unit_name, [])
        n_samples = sum(self._code_stats.get(c, {}).get("count", 0) for c in codes)
        codes_text = ", ".join(codes) if codes else "(empty)"
        codes_label = QLabel(f"Codes: {codes_text}")
        codes_label.setWordWrap(True)
        codes_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10px;")
        lay.addWidget(codes_label)

        count_label = QLabel(f"Samples: {n_samples}")
        count_label.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        lay.addWidget(count_label)

        return card

    def _on_filter_changed(self, text: str) -> None:
        """Filter ungrouped codes list."""
        text = text.strip().upper()
        for i in range(self.ungrouped_list.count()):
            item = self.ungrouped_list.item(i)
            item.setHidden(text != "" and text not in item.text().upper())

    def _on_assign_codes(self) -> None:
        """Assign selected ungrouped codes to the target unit."""
        target = self.target_unit_combo.currentText()
        if not target:
            return

        selected = [item.text() for item in self.ungrouped_list.selectedItems()]
        if not selected:
            return

        if target not in self._grouping:
            self._grouping[target] = []

        for code in selected:
            if code not in self._grouping[target]:
                self._grouping[target].append(code)

        self._refresh_grouping_ui()
        self._refresh_code_table()

    def _on_add_unit(self) -> None:
        """Add a new modelling unit."""
        name, ok = QInputDialog.getText(self, "New Unit", "Unit name:")
        if not ok or not name.strip():
            return
        name = name.strip().upper()
        if name in self._grouping:
            QMessageBox.warning(self, "Duplicate", f"Unit '{name}' already exists.")
            return

        self._grouping[name] = []
        idx = len(self._unit_colors) % len(_DEFAULT_UNIT_COLORS)
        self._unit_colors[name] = _DEFAULT_UNIT_COLORS[idx]
        self._unit_types[name] = "stratigraphic"
        self._refresh_grouping_ui()

    def _on_remove_unit(self) -> None:
        """Remove the selected unit (codes go back to ungrouped)."""
        target = self.target_unit_combo.currentText()
        if not target or target not in self._grouping:
            return

        del self._grouping[target]
        self._unit_colors.pop(target, None)
        self._unit_types.pop(target, None)
        self._refresh_grouping_ui()
        self._refresh_code_table()

    def _on_pick_color(self, unit_name: str) -> None:
        """Open color picker for a unit."""
        current = QColor(self._unit_colors.get(unit_name, "#808080"))
        color = QColorDialog.getColor(current, self, f"Color for {unit_name}")
        if color.isValid():
            self._unit_colors[unit_name] = color.name()
            self._refresh_grouping_ui()

    def _on_save_grouping(self) -> None:
        """Save grouping to JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Grouping", "lithology_grouping.json", "JSON Files (*.json)",
        )
        if not path:
            return
        data = {
            "grouping": self._grouping,
            "unit_colors": self._unit_colors,
            "unit_types": self._unit_types,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.status_label.setText(f"Saved to {os.path.basename(path)}")

    def _on_load_grouping(self) -> None:
        """Load grouping from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Grouping", "", "JSON Files (*.json)",
        )
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self._grouping = data.get("grouping", {})
        self._unit_colors = data.get("unit_colors", {})
        self._unit_types = data.get("unit_types", {})
        self._refresh_grouping_ui()
        self._refresh_code_table()
        self.status_label.setText(f"Loaded from {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════
    # TAB 3 LOGIC
    # ══════════════════════════════════════════════════════════════

    def _on_refresh_strat(self) -> None:
        """Refresh stratigraphic column from grouping."""
        strat_units = [
            name for name in self._grouping
            if self._unit_types.get(name, "stratigraphic") == "stratigraphic"
        ]
        self._strat_order = sorted(strat_units)

        self.strat_list.clear()
        for unit in self._strat_order:
            color = self._unit_colors.get(unit, "#808080")
            n = sum(self._code_stats.get(c, {}).get("count", 0)
                    for c in self._grouping.get(unit, []))
            item = QListWidgetItem(f"{unit}  ({n} samples)")
            item.setData(Qt.ItemDataRole.UserRole, unit)
            item.setForeground(QBrush(QColor(color)))
            self.strat_list.addItem(item)

        self._refresh_contact_table()

    def _on_move_up(self) -> None:
        row = self.strat_list.currentRow()
        if row > 0:
            item = self.strat_list.takeItem(row)
            self.strat_list.insertItem(row - 1, item)
            self.strat_list.setCurrentRow(row - 1)
            self._sync_strat_order()
            self._refresh_contact_table()

    def _on_move_down(self) -> None:
        row = self.strat_list.currentRow()
        if row < self.strat_list.count() - 1:
            item = self.strat_list.takeItem(row)
            self.strat_list.insertItem(row + 1, item)
            self.strat_list.setCurrentRow(row + 1)
            self._sync_strat_order()
            self._refresh_contact_table()

    def _sync_strat_order(self) -> None:
        """Sync strat_order from the list widget."""
        self._strat_order = []
        for i in range(self.strat_list.count()):
            item = self.strat_list.item(i)
            unit = item.data(Qt.ItemDataRole.UserRole)
            if unit:
                self._strat_order.append(unit)

    def _refresh_contact_table(self) -> None:
        """Populate contact type table from stratigraphic column."""
        n_surfaces = max(0, len(self._strat_order) - 1)
        self.contact_table.setRowCount(n_surfaces)

        for i in range(n_surfaces):
            above = self._strat_order[i]
            below = self._strat_order[i + 1]
            surface_name = f"{above}_{below}"

            self.contact_table.setItem(i, 0, QTableWidgetItem(surface_name))

            # Contact type combo
            ctype = self._contact_types.get(surface_name, "Conformable")
            type_combo = QComboBox()
            type_combo.addItems(["Conformable", "Unconformable (erosion)",
                                 "Unconformable (onlap)", "Gradational"])
            type_combo.setCurrentText(ctype)
            type_combo.currentTextChanged.connect(
                lambda t, s=surface_name: self._contact_types.__setitem__(s, t)
            )
            self.contact_table.setCellWidget(i, 1, type_combo)

            # N contacts (will be filled after extraction)
            self.contact_table.setItem(i, 2, QTableWidgetItem("--"))

    def _on_extract_contacts(self) -> None:
        """Extract contacts from drillhole lithology data."""
        if self._lithology_df is None:
            self.action_status.setText("No lithology data loaded")
            return

        if not self._grouping:
            self.action_status.setText("No grouping defined")
            return

        lith_col = self.lith_col_combo.currentText()
        if not lith_col:
            self.action_status.setText("No lithology column selected")
            return

        try:
            from geology.implicit.signed_distance import extract_contacts_from_lithology

            contacts_df = extract_contacts_from_lithology(
                self._lithology_df,
                lithology_column=lith_col,
                grouping=self._grouping,
            )

            # Update contact count in table
            for i in range(self.contact_table.rowCount()):
                sname = self.contact_table.item(i, 0).text()
                n = len(contacts_df[contacts_df["surface_name"] == sname])
                self.contact_table.setItem(i, 2, QTableWidgetItem(str(n)))
                if n < 5:
                    self.contact_table.item(i, 2).setForeground(
                        QBrush(QColor("#e6194B"))
                    )

            # Register in data registry
            if self.registry:
                try:
                    self.registry.store("contact_set", contacts_df)
                except Exception:
                    pass

            self.contactsExtracted.emit(contacts_df)
            self.action_status.setText(f"Extracted {len(contacts_df)} contacts")

        except Exception as e:
            logger.error("Contact extraction failed: %s", e)
            self.action_status.setText(f"Error: {e}")

    def _on_assign_domains(self) -> None:
        """Assign integer domain codes to composites."""
        if not self._grouping:
            self.action_status.setText("No grouping defined")
            return

        if self.registry is None:
            self.action_status.setText("No registry")
            return

        # Get composites
        composites = None
        try:
            dh_data = self.registry.get_drillhole_data()
            if isinstance(dh_data, dict):
                composites = dh_data.get("composites")
                if composites is None:
                    composites = dh_data.get("assays")
        except Exception:
            pass

        if composites is None or composites.empty:
            self.action_status.setText("No composites/assays in registry")
            return

        lith_col = self.lith_col_combo.currentText()
        if lith_col not in composites.columns:
            self.action_status.setText(f"Column '{lith_col}' not in composites")
            return

        try:
            from geology.implicit.domain_model import assign_domains_from_lithology

            domain_codes = assign_domains_from_lithology(
                composites, self._grouping, lithology_column=lith_col,
            )

            # Add domain column to composites
            composites["domain_code"] = domain_codes

            # Build unit name mapping
            unit_names = sorted(self._grouping.keys())
            unit_map = {i: name for i, name in enumerate(unit_names)}
            composites["domain_name"] = [
                unit_map.get(c, "Unassigned") for c in domain_codes
            ]

            # Store domain info in registry
            domain_info = {
                "domain_codes": domain_codes,
                "grouping": self._grouping,
                "unit_names": unit_names,
                "unit_colors": self._unit_colors,
                "composites_with_domains": composites,
            }

            try:
                self.registry.store("domain_model", domain_info)
            except Exception:
                pass

            self.domainCodesAssigned.emit(domain_info)

            n_unassigned = int(np.sum(domain_codes < 0))
            self.action_status.setText(
                f"Assigned {len(domain_codes)} samples to "
                f"{len(unit_names)} domains "
                f"({n_unassigned} unassigned)"
            )

        except Exception as e:
            logger.error("Domain assignment failed: %s", e)
            self.action_status.setText(f"Error: {e}")

    def _on_send_to_arbf(self) -> None:
        """Make domain codes available to ARBF panel via registry."""
        if self.registry is None:
            self.action_status.setText("No registry")
            return

        try:
            domain_model = self.registry.get("domain_model")
        except Exception:
            domain_model = None

        if domain_model is None:
            self.action_status.setText("Assign domain codes first")
            return

        # Emit signal that ARBF panel can listen to
        try:
            if hasattr(self.registry, 'signals'):
                self.registry.signals.domainModelLoaded.emit(domain_model)
        except Exception:
            pass

        self.action_status.setText("Domain codes sent to ARBF")


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr

    return prev[-1]
