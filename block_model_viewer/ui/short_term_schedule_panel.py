"""
Short-Term Schedule Panel (STEP 30)

PyQt6 panel for block-model-driven short-term scheduling.

Features:
    - Block model import (CSV) or load from DataRegistry
    - Resource classification filtering (Measured/Indicated/Inferred)
    - Configurable block grouping
    - Grade spec editing
    - Schedule generation with blend compliance
    - Shift plan generation
    - Results table with export to CSV

Author: BlockModelViewer Team
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Dict, Any, List

import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
    QMessageBox, QFileDialog, QCheckBox, QTabWidget, QProgressBar,
    QSplitter, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
    QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from .base_analysis_panel import BaseAnalysisPanel

from block_model_viewer.mine_planning.scheduling.short_term.block_model_scheduler import (
    ResourceClass, GroupByField, PeriodType, GradeSpec, ColumnMapping,
    ShortTermScheduleConfig, ShortTermScheduleResult,
    build_short_term_schedule,
)
from block_model_viewer.mine_planning.scheduling.short_term.short_term_blend import (
    BlendSource, BlendSpec, ShortTermBlendConfig, BlendResult,
    optimise_short_term_blend, build_blend_sources_from_units,
)
from block_model_viewer.mine_planning.scheduling.short_term.shift_plan import (
    ShiftConfig, EquipmentUnit, ShiftPlanResult, generate_shift_plan,
)

logger = logging.getLogger(__name__)


def _extract_dataframe(model) -> "Optional[pd.DataFrame]":
    """Extract a pandas DataFrame from whatever the registry hands back."""
    if model is None:
        return None
    if isinstance(model, pd.DataFrame):
        return model if not model.empty else None
    # BlockModel or similar object with to_dataframe()
    if hasattr(model, "to_dataframe"):
        df = model.to_dataframe()
        return df if (isinstance(df, pd.DataFrame) and not df.empty) else None
    # Object that wraps a DataFrame in .data
    if hasattr(model, "data") and isinstance(model.data, pd.DataFrame):
        return model.data if not model.data.empty else None
    return None


# ─── Worker Thread ────────────────────────────────────────────────────────────

class SchedulerWorker(QThread):
    """Run scheduler in background thread."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, block_model: pd.DataFrame, config: ShortTermScheduleConfig):
        super().__init__()
        self.block_model = block_model
        self.config = config

    def run(self):
        try:
            self.progress.emit("Filtering blocks...")
            result = build_short_term_schedule(self.block_model, self.config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ─── Main Panel ──────────────────────────────────────────────────────────────

class ShortTermSchedulePanel(BaseAnalysisPanel):
    """
    Panel for short-term scheduling (weekly/daily).

    Workflow:
        1. Load block model (CSV import or from DataRegistry)
        2. Configure column mapping, classification, grouping
        3. Set grade specs, plant target, period type
        4. Run scheduler → view results
        5. Optionally run blend optimisation / shift plan
    """

    task_name = "short_term_block_schedule"

    def __init__(self, parent: Optional[QWidget] = None):
        # Pre-init attrs referenced by setup_ui() (called from super().__init__)
        self.registry = None
        self._sched_df: Optional[pd.DataFrame] = None
        self.schedule_result: Optional[ShortTermScheduleResult] = None
        self.shift_plan_result: Optional[ShiftPlanResult] = None
        self._worker: Optional[SchedulerWorker] = None
        self._classified_available: bool = False

        super().__init__(parent=parent, panel_id="short_term_schedule")

        # Subscribe to DataRegistry
        try:
            self.registry = self.get_registry()
            # Prefer classified model; fall back to generated/loaded
            self.registry.blockModelClassified.connect(self._on_classified_model_received)
            self.registry.blockModelGenerated.connect(self._on_generated_model_received)
            self.registry.blockModelLoaded.connect(self._on_generated_model_received)

            # Enable registry buttons now that registry is connected
            if hasattr(self, "btn_load_classified"):
                self.btn_load_classified.setEnabled(True)
            if hasattr(self, "btn_load_generated"):
                self.btn_load_generated.setEnabled(True)

            # Auto-load on startup — classified preferred
            classified = self.registry.get_classified_block_model()
            df = _extract_dataframe(classified)
            if df is not None:
                self._load_block_model(df, "Classified Block Model")
                self._classified_available = True
            else:
                generated = self.registry.get_block_model()
                df = _extract_dataframe(generated)
                if df is not None:
                    self._load_block_model(df, "Generated Block Model")
        except Exception as e:
            logger.warning(f"Short-Term Panel: DataRegistry connection failed: {e}")
            self.registry = None

        logger.info("Initialised Short-Term Schedule Panel")

    # ── UI Setup ──────────────────────────────────────────────────────────

    def setup_ui(self):
        layout = self.main_layout

        # ── Header ────────────────────────────────────────────────────────
        header = QLabel(
            "Short-Term Block Model Scheduler — "
            "Import → Classify → Group → Schedule → Shift Plan"
        )
        header.setStyleSheet(
            "background-color: #e3f2fd; padding: 10px; border-radius: 5px;"
            "font-weight: bold;"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Import & Configure
        self._build_config_tab()

        # Tab 2: Results
        self._build_results_tab()

        # Tab 3: Shift Plan
        self._build_shift_tab()

        # ── Status bar ────────────────────────────────────────────────────
        self.status_label = QLabel("No block model loaded")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 4px;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def _build_config_tab(self):
        tab = QWidget()
        main = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Import & Column Mapping ─────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Import buttons
        import_group = QGroupBox("Block Model Import")
        import_lay = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.btn_import_csv = QPushButton("Import CSV...")
        self.btn_import_csv.clicked.connect(self._on_import_csv)
        btn_row.addWidget(self.btn_import_csv)
        import_lay.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.btn_load_classified = QPushButton("Load Classified Model")
        self.btn_load_classified.setToolTip(
            "Load the resource-classified block model from the JORC/Classification panel"
        )
        self.btn_load_classified.clicked.connect(self._on_load_classified)
        self.btn_load_classified.setEnabled(self.registry is not None)
        self.btn_load_classified.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 5px; }"
            "QPushButton:hover { background-color: #1b5e20; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        btn_row2.addWidget(self.btn_load_classified)

        self.btn_load_generated = QPushButton("Load Generated Model")
        self.btn_load_generated.setToolTip("Load the raw generated/imported block model")
        self.btn_load_generated.clicked.connect(self._on_load_generated)
        self.btn_load_generated.setEnabled(self.registry is not None)
        btn_row2.addWidget(self.btn_load_generated)
        import_lay.addLayout(btn_row2)

        self.lbl_import_status = QLabel("No data loaded")
        import_lay.addWidget(self.lbl_import_status)

        self.lbl_classified_badge = QLabel("")
        self.lbl_classified_badge.setStyleSheet(
            "color: #2e7d32; font-size: 10px; font-weight: bold;"
        )
        import_lay.addWidget(self.lbl_classified_badge)

        import_group.setLayout(import_lay)
        left_layout.addWidget(import_group)

        # Column mapping
        mapping_group = QGroupBox("Column Mapping")
        mapping_form = QFormLayout()
        self._col_combos: Dict[str, QComboBox] = {}
        for field_name, label in [
            ("block_id", "Block ID"),
            ("x", "X / Easting"),
            ("y", "Y / Northing"),
            ("z", "Z / RL"),
            ("pit", "Pit / Zone"),
            ("bench", "Bench / RL"),
            ("domain", "Domain"),
            ("resource_class", "Resource Classification"),
            ("material", "Material Type"),
            ("tonnes", "Tonnes *"),
            ("density", "Density / SG"),
        ]:
            combo = QComboBox()
            combo.setMinimumWidth(150)
            combo.addItem("— select —")
            self._col_combos[field_name] = combo
            mapping_form.addRow(f"{label}:", combo)

        mapping_group.setLayout(mapping_form)
        left_layout.addWidget(mapping_group)
        left_layout.addStretch()

        # ── Right: Filters & Targets ──────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # Resource classification
        class_group = QGroupBox("Resource Classification Filter")
        class_lay = QVBoxLayout()
        self._class_checks: Dict[str, QCheckBox] = {}
        for cls in ResourceClass:
            cb = QCheckBox(cls.value)
            cb.setChecked(cls in ResourceClass.schedulable_default())
            self._class_checks[cls.value] = cb
            class_lay.addWidget(cb)
        class_group.setLayout(class_lay)
        right_layout.addWidget(class_group)

        # Grouping
        group_group = QGroupBox("Block Grouping")
        group_form = QFormLayout()
        self.combo_group_by = QComboBox()
        for g in GroupByField:
            self.combo_group_by.addItem(g.value.replace("_", " ").title(), g)
        group_form.addRow("Group By:", self.combo_group_by)
        group_group.setLayout(group_form)
        right_layout.addWidget(group_group)

        # Plant target & period
        target_group = QGroupBox("Schedule Targets")
        target_form = QFormLayout()

        self.spin_plant_target = QDoubleSpinBox()
        self.spin_plant_target.setRange(0, 10_000_000)
        self.spin_plant_target.setValue(25_000)
        self.spin_plant_target.setSuffix(" t/period")
        self.spin_plant_target.setDecimals(0)
        target_form.addRow("Plant Target:", self.spin_plant_target)

        self.combo_period = QComboBox()
        for pt in PeriodType:
            label = pt.value.replace("_", " ").title()
            self.combo_period.addItem(label, pt)
        target_form.addRow("Period Type:", self.combo_period)

        self.spin_shifts = QSpinBox()
        self.spin_shifts.setRange(1, 4)
        self.spin_shifts.setValue(2)
        target_form.addRow("Shifts/Day:", self.spin_shifts)

        target_group.setLayout(target_form)
        right_layout.addWidget(target_group)

        # Grade specs
        grade_group = QGroupBox("Grade Specifications")
        grade_lay = QVBoxLayout()
        self.grade_table = QTableWidget(0, 3)
        self.grade_table.setHorizontalHeaderLabels(["Element", "Min", "Max"])
        self.grade_table.horizontalHeader().setStretchLastSection(True)
        grade_lay.addWidget(self.grade_table)

        grade_btn_row = QHBoxLayout()
        btn_add_grade = QPushButton("+ Add Element")
        btn_add_grade.clicked.connect(self._on_add_grade)
        grade_btn_row.addWidget(btn_add_grade)

        self.combo_preset = QComboBox()
        self.combo_preset.addItems([
            "— Preset —", "Iron Ore", "Gold", "Copper",
            "Coal", "Manganese", "Platinum (PGM)"
        ])
        self.combo_preset.currentTextChanged.connect(self._on_preset_selected)
        grade_btn_row.addWidget(self.combo_preset)
        grade_lay.addLayout(grade_btn_row)
        grade_group.setLayout(grade_lay)
        right_layout.addWidget(grade_group)

        right_layout.addStretch()

        # ── Run button ────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Generate Schedule")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; padding: 10px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_schedule)
        right_layout.addWidget(self.btn_run)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 400])
        main.addWidget(splitter)

        self.tabs.addTab(tab, "Configure")

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # KPI row
        self.kpi_layout = QHBoxLayout()
        layout.addLayout(self.kpi_layout)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(QLabel("Schedule Results:"))
        layout.addWidget(self.results_table)

        # Export button
        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("Export to CSV...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_csv)
        btn_row.addWidget(self.btn_export)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Results")

    def _build_shift_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        btn_row = QHBoxLayout()
        self.btn_gen_shift = QPushButton("Generate Shift Plan")
        self.btn_gen_shift.setEnabled(False)
        self.btn_gen_shift.clicked.connect(self._on_gen_shift_plan)
        btn_row.addWidget(self.btn_gen_shift)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.shift_table = QTableWidget()
        self.shift_table.setAlternatingRowColors(True)
        layout.addWidget(QLabel("Shift Plan:"))
        layout.addWidget(self.shift_table)

        self.tabs.addTab(tab, "Shift Plan")

    # ── Data Import ───────────────────────────────────────────────────────

    def _on_import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Block Model CSV",
            "", "CSV files (*.csv *.tsv *.txt);;All files (*)"
        )
        if not path:
            return

        try:
            sep = "\t" if path.endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep)
            self._load_block_model(df, os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to load CSV:\n{e}")

    def _on_load_classified(self):
        """Load the resource-classified block model from the registry."""
        if not self.registry:
            return
        try:
            model = self.registry.get_classified_block_model()
            df = _extract_dataframe(model)
            if df is not None:
                self._load_block_model(df, "Classified Block Model")
                self._classified_available = True
            else:
                QMessageBox.warning(
                    self, "No Classified Model",
                    "No classified block model found.\n\n"
                    "Run the Resource Classification panel first, then return here."
                )
        except Exception as e:
            QMessageBox.critical(self, "Registry Error", str(e))

    def _on_load_generated(self):
        """Load the raw generated/loaded block model from the registry."""
        if not self.registry:
            return
        try:
            model = self.registry.get_block_model()
            df = _extract_dataframe(model)
            if df is not None:
                self._load_block_model(df, "Generated Block Model")
            else:
                QMessageBox.warning(self, "No Data", "No block model found in registry.")
        except Exception as e:
            QMessageBox.critical(self, "Registry Error", str(e))

    def _on_classified_model_received(self, model):
        """Signal handler: classified block model was updated in the registry."""
        df = _extract_dataframe(model)
        if df is not None:
            self._classified_available = True
            # Update badge without auto-loading — user may have manually loaded CSV
            if hasattr(self, "lbl_classified_badge"):
                self.lbl_classified_badge.setText(
                    f"✓ Classified model available ({len(df):,} blocks) — click 'Load Classified Model' to use"
                )
            logger.info(
                f"Short-Term Panel: Classified model updated ({len(df):,} blocks)"
            )

    def _on_generated_model_received(self, model):
        """Signal handler: generated/loaded block model was updated in the registry."""
        # Only notify; don't auto-load (user controls what they want to schedule)
        df = _extract_dataframe(model)
        if df is not None and not self._classified_available:
            if hasattr(self, "lbl_classified_badge"):
                self.lbl_classified_badge.setText(
                    f"Block model available ({len(df):,} blocks) — click 'Load Generated Model' to use"
                )

    def _load_block_model(self, df: pd.DataFrame, source: str):
        self._sched_df = df
        cols = list(df.columns)

        # Populate column combos
        for key, combo in self._col_combos.items():
            combo.clear()
            combo.addItem("— select —")
            combo.addItems(cols)
            # Auto-detect
            self._auto_map_column(combo, key, cols)

        self.lbl_import_status.setText(
            f"✓ Loaded {len(df):,} blocks · {len(cols)} columns · Source: {source}"
        )
        self.lbl_import_status.setStyleSheet("color: green; font-weight: bold;")
        if hasattr(self, "lbl_classified_badge"):
            self.lbl_classified_badge.setText("")  # clear badge after loading
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"Block model loaded: {len(df):,} blocks from {source}")

    def _auto_map_column(self, combo: QComboBox, field: str, cols: List[str]):
        """Try to auto-detect column mapping."""
        keywords = {
            "block_id": ["id", "block_id", "blk_id", "blockid"],
            "x": ["x", "east", "easting", "xc", "xcentre"],
            "y": ["y", "north", "northing", "yc", "ycentre"],
            "z": ["z", "rl", "elev", "zc", "zcentre", "bench"],
            "pit": ["pit", "zone", "area", "region"],
            "bench": ["rl", "bench", "level"],
            "domain": ["domain", "rock", "rocktype", "lith", "geol"],
            "resource_class": ["class", "resource", "resclass", "category", "jorc", "ni43"],
            "material": ["material", "mat", "ore_waste", "orewaste", "type"],
            "tonnes": ["tonnes", "tons", "tonnage", "mass"],
            "density": ["density", "sg", "bulk_density", "bd"],
        }
        for kw in keywords.get(field, []):
            for i, col in enumerate(cols):
                if kw in col.lower():
                    combo.setCurrentIndex(i + 1)  # +1 for "— select —"
                    return

    # ── Grade Specs ───────────────────────────────────────────────────────

    def _on_add_grade(self):
        row = self.grade_table.rowCount()
        self.grade_table.insertRow(row)
        self.grade_table.setItem(row, 0, QTableWidgetItem(""))
        self.grade_table.setItem(row, 1, QTableWidgetItem("0"))
        self.grade_table.setItem(row, 2, QTableWidgetItem("100"))

    def _on_preset_selected(self, text: str):
        from block_model_viewer.mine_planning.scheduling.short_term.block_model_scheduler import GradeSpec
        presets = {
            "Iron Ore": [("Fe",58,65),("SiO2",0,6),("Al2O3",0,3.5),("P",0,0.07),("LOI",0,5)],
            "Gold": [("Au",0.5,100),("Ag",0,50),("S",0,5),("As",0,1)],
            "Copper": [("Cu",0.3,5),("Mo",0,0.1),("Au",0,2),("S",0,10)],
            "Coal": [("CV",5500,7000),("Ash",0,15),("VM",0,30),("Moisture",0,12),("S",0,1)],
            "Manganese": [("Mn",36,52),("Fe",0,8),("SiO2",0,8),("P",0,0.05)],
            "Platinum (PGM)": [("Pt",1,10),("Pd",0.5,8),("Rh",0,1),("Cr2O3",0,5)],
        }
        if text in presets:
            self.grade_table.setRowCount(0)
            for el, mn, mx in presets[text]:
                row = self.grade_table.rowCount()
                self.grade_table.insertRow(row)
                self.grade_table.setItem(row, 0, QTableWidgetItem(el))
                self.grade_table.setItem(row, 1, QTableWidgetItem(str(mn)))
                self.grade_table.setItem(row, 2, QTableWidgetItem(str(mx)))

    # ── Build Config ──────────────────────────────────────────────────────

    def _build_config(self) -> ShortTermScheduleConfig:
        # Column mapping
        cm = ColumnMapping()
        for field_name, combo in self._col_combos.items():
            val = combo.currentText()
            if val and val != "— select —":
                setattr(cm, field_name, val)

        # Resource classes
        allowed = [
            ResourceClass(name)
            for name, cb in self._class_checks.items()
            if cb.isChecked()
        ]

        # Grade specs
        grade_specs = []
        for row in range(self.grade_table.rowCount()):
            el_item = self.grade_table.item(row, 0)
            min_item = self.grade_table.item(row, 1)
            max_item = self.grade_table.item(row, 2)
            if el_item and el_item.text():
                grade_specs.append(GradeSpec(
                    element=el_item.text(),
                    min_grade=float(min_item.text()) if min_item else 0,
                    max_grade=float(max_item.text()) if max_item else 100,
                ))

        # Period type
        period_type = self.combo_period.currentData() or PeriodType.DAILY_7

        # Group by
        group_by = self.combo_group_by.currentData() or GroupByField.BENCH

        return ShortTermScheduleConfig(
            column_mapping=cm,
            grade_specs=grade_specs,
            allowed_classes=allowed,
            allowed_materials=["Ore"],
            group_by=group_by,
            plant_target_per_period=self.spin_plant_target.value(),
            period_type=period_type,
        )

    # ── Run Schedule ──────────────────────────────────────────────────────

    def _on_run_schedule(self):
        if self._sched_df is None:
            QMessageBox.warning(self, "No Data", "Load a block model first.")
            return

        config = self._build_config()

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running scheduler...")

        self._worker = SchedulerWorker(self._sched_df, config)
        self._worker.finished.connect(self._on_schedule_complete)
        self._worker.error.connect(self._on_schedule_error)
        self._worker.start()

    def _on_schedule_complete(self, result: ShortTermScheduleResult):
        self.schedule_result = result
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_gen_shift.setEnabled(True)

        self.status_label.setText(
            f"Schedule complete: {result.total_ore / 1000:.0f}k ore, "
            f"{result.total_waste / 1000:.0f}k waste, "
            f"{result.compliance_rate * 100:.0f}% compliance"
        )

        self._populate_results_table(result)
        self._populate_kpis(result)
        self.tabs.setCurrentIndex(1)  # Switch to Results tab

    def _on_schedule_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Scheduler Error", msg)

    # ── Results Display ───────────────────────────────────────────────────

    def _populate_kpis(self, result: ShortTermScheduleResult):
        # Clear existing KPI widgets
        while self.kpi_layout.count():
            item = self.kpi_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        kpis = [
            ("Total Ore", f"{result.total_ore / 1000:.0f}k t"),
            ("Total Waste", f"{result.total_waste / 1000:.0f}k t"),
            ("Compliance", f"{result.compliance_rate * 100:.0f}%"),
            ("Units", str(len(result.units))),
            ("Periods", str(len(result.periods))),
        ]
        for el, grade in result.avg_blended_grades.items():
            kpis.append((f"Avg {el}", f"{grade:.2f}"))
            if len(kpis) >= 8:
                break

        for label, value in kpis:
            frame = QFrame()
            frame.setFrameStyle(QFrame.Shape.Box)
            frame.setStyleSheet(
                "QFrame { background: #f5f5f5; border: 1px solid #ddd; "
                "border-radius: 4px; padding: 8px; }"
            )
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 4, 8, 4)
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 10px; color: #666; font-weight: bold;")
            val = QLabel(value)
            val.setStyleSheet("font-size: 16px; font-weight: bold;")
            fl.addWidget(lbl)
            fl.addWidget(val)
            self.kpi_layout.addWidget(frame)

    def _populate_results_table(self, result: ShortTermScheduleResult):
        df = result.to_dataframe()
        if df.empty:
            return

        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(list(df.columns))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col in enumerate(df.columns):
                val = row[col]
                if isinstance(val, float):
                    text = f"{val:.2f}" if val < 1000 else f"{val:,.0f}"
                else:
                    text = str(val)
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(text))

        self.results_table.resizeColumnsToContents()

    # ── Shift Plan ────────────────────────────────────────────────────────

    def _on_gen_shift_plan(self):
        if not self.schedule_result:
            return

        n_shifts = self.spin_shifts.value()
        if n_shifts == 1:
            shifts = [ShiftConfig("Full Day", 24.0)]
        elif n_shifts == 2:
            shifts = [ShiftConfig("Day", 12.0), ShiftConfig("Night", 12.0)]
        elif n_shifts == 3:
            shifts = [
                ShiftConfig("Morning", 8.0),
                ShiftConfig("Afternoon", 8.0),
                ShiftConfig("Night", 8.0),
            ]
        else:
            shifts = [ShiftConfig(f"Shift {i+1}", 24.0 / n_shifts) for i in range(n_shifts)]

        self.shift_plan_result = generate_shift_plan(self.schedule_result, shifts)

        # Populate shift table
        entries = self.shift_plan_result.entries
        cols = ["Period", "Shift", "Hours", "Source", "Tonnes", "Destination"]
        rows = []
        for entry in entries:
            for a in entry.assignments:
                rows.append([
                    entry.period_label, entry.shift_name, f"{entry.shift_hours:.0f}h",
                    a.unit_name, f"{a.tonnes:,.0f}", a.destination,
                ])

        self.shift_table.setRowCount(len(rows))
        self.shift_table.setColumnCount(len(cols))
        self.shift_table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.shift_table.setItem(r, c, QTableWidgetItem(val))
        self.shift_table.resizeColumnsToContents()

        self.tabs.setCurrentIndex(2)

    # ── Export ────────────────────────────────────────────────────────────

    def _on_export_csv(self):
        if not self.schedule_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Schedule CSV", "short_term_schedule.csv",
            "CSV files (*.csv)"
        )
        if path:
            df = self.schedule_result.to_dataframe()
            df.to_csv(path, index=False)
            self.status_label.setText(f"Exported to {path}")

    # ── Required overrides ────────────────────────────────────────────────

    def gather_parameters(self) -> Dict[str, Any]:
        return {"plant_target": self.spin_plant_target.value()}

    def validate_inputs(self) -> bool:
        return self._sched_df is not None

    def on_results(self, payload: Dict[str, Any]) -> None:
        pass
