"""
Pit Optimisation Panel

GUI panel for open-pit optimization using Lerchs-Grossmann maximum closure algorithm.
Works with loaded or estimated block models.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import time
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QLineEdit, QComboBox, QPushButton, QMessageBox,
    QProgressBar, QApplication
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from .base_analysis_panel import BaseAnalysisPanel

# PyVista removed - all rendering via Renderer

from ..models.pit_optimizer import (
    PitParams, optimise_pit, normalize_coordinate_columns, ColumnMapping,
    ElementSpec, UnitType, CostStructure, GeoTechSector
)

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


def _has_model_data(model: Any) -> bool:
    """Safe model presence check that handles pandas DataFrames."""
    if model is None:
        return False
    try:
        return not bool(getattr(model, "empty"))
    except Exception:
        return True


# PitOptimisationWorker and NestedShellsWorker removed - logic moved to controller


class PitOptimisationPanel(BaseAnalysisPanel):
    """
    Panel for open-pit optimization using Lerchs-Grossmann algorithm.
    
    Works with block models loaded in the viewer. Computes ultimate pit
    and optionally nested pushback shells.
    """
    # PanelManager metadata
    PANEL_ID = "PitOptimisationPanel"
    PANEL_NAME = "PitOptimisation Panel"
    PANEL_CATEGORY = PanelCategory.PLANNING
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "pit_opt"  # Use existing pit_opt task
    
    # Signal to request visualization in main viewer
    request_visualization = pyqtSignal(object, str)  # mesh_or_grid, layer_name
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="pit_optimisation")
        self.block_df: Optional[pd.DataFrame] = None
        self.grid_spec: Optional[Dict] = None
        self.current_operation: Optional[str] = None  # 'single_pit' or 'nested_shells'
        
        # Store results for table viewing
        self.ultimate_pit_df: Optional[pd.DataFrame] = None
        self.nested_shells_summary: Optional[pd.DataFrame] = None
        
        # Column mapping configuration
        self.column_mapping: Optional[ColumnMapping] = None
        self.constant_densities: Dict[str, float] = {}  # For constant density values
        
        # Cancellation flag
        self._cancel_requested = False
        
        # Subscribe to block model updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_loaded)
            
            # Prefer classified block model when available.
            existing_block_model = self.registry.get_classified_block_model()
            if not _has_model_data(existing_block_model):
                existing_block_model = self.registry.get_block_model()
            if _has_model_data(existing_block_model):
                self._on_block_model_loaded(existing_block_model)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
        logger.info("Initialized Pit Optimisation panel")
    


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
        """Build the UI layout."""
        layout = self.main_layout
        
        # Title
        title = QLabel("<b>Pit Optimisation</b>")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 10px;")
        self.layout().addWidget(title)
        
        # Form layout for parameters
        form = QFormLayout()
        self.layout().addLayout(form)
        
        # Column mapping button (important for user control)
        mapping_row = QHBoxLayout()
        self.btn_configure_columns = QPushButton("⚙ Configure Column Mapping")
        self.btn_configure_columns.setToolTip(
            "Configure which columns in your block model correspond to:\n"
            "- X, Y, Z coordinates (can be named anything: EASTING, zx, centroid, etc.)\n"
            "- Volume, Density, Tonnes\n"
            "Gives you full control over column naming"
        )
        self.btn_configure_columns.clicked.connect(self._configure_column_mapping)
        self.btn_configure_columns.setEnabled(False)  # Enabled when data loaded
        mapping_row.addWidget(self.btn_configure_columns)
        mapping_row.addStretch()
        form.addRow("", mapping_row)
        
        # Status label for column mapping
        self.mapping_status = QLabel("<i>Using auto-detected columns</i>")
        self.mapping_status.setStyleSheet("color: gray; font-size: 9pt;")
        form.addRow("", self.mapping_status)
        
        # Grade column selector (auto-detects from block model)
        grade_row = QHBoxLayout()
        self.grade_combo = QComboBox()
        self.grade_combo.setEditable(True)
        self.grade_combo.setToolTip("Select or type grade column name (auto-detected from block model)")
        self.grade_combo.setCurrentText("Fe")  # Default
        grade_row.addWidget(self.grade_combo)
        
        # Auto-detect button
        self.btn_auto_detect = QPushButton("Auto-detect")
        self.btn_auto_detect.setToolTip("Re-scan block model for grade columns")
        self.btn_auto_detect.clicked.connect(self._auto_detect_grade)
        grade_row.addWidget(self.btn_auto_detect)
        
        form.addRow("Grade column:", grade_row)
        
        # Price per unit
        self.price = QDoubleSpinBox()
        self.price.setRange(0, 1e9)
        self.price.setDecimals(2)
        self.price.setValue(100.0)
        self.price.setToolTip(
            "Commodity price per unit\n"
            "Units depend on commodity:\n"
            "  • Base metals (Cu, Fe, Zn): $/tonne (e.g., $9,000/t for Cu)\n"
            "  • Precious metals (Au, Ag): $/oz (e.g., $2,000/oz for Au)\n"
            "Make sure your grade units match the price units!"
        )
        form.addRow("Price per unit ($):", self.price)
        
        # Recovery
        self.recovery = QDoubleSpinBox()
        self.recovery.setRange(0.0, 1.0)
        self.recovery.setDecimals(3)
        self.recovery.setValue(0.85)
        self.recovery.setSingleStep(0.01)
        self.recovery.setToolTip(
            "Metallurgical recovery as decimal (0.0 to 1.0)\n"
            "Fraction of metal recovered in processing\n"
            "Typical values:\n"
            "  • 0.85 = 85% recovery (common for base metals)\n"
            "  • 0.90 = 90% recovery (good recovery)\n"
            "  • 0.75 = 75% recovery (lower grade or complex ore)"
        )
        form.addRow("Recovery (0-1):", self.recovery)
        
        # Mining cost
        self.c_mine = QDoubleSpinBox()
        self.c_mine.setRange(0, 1e6)
        self.c_mine.setDecimals(2)
        self.c_mine.setValue(3.0)
        self.c_mine.setToolTip(
            "Cost to mine one tonne of material ($/tonne)\n"
            "Includes drilling, blasting, loading, hauling\n"
            "Typical values:\n"
            "  • Small operations: $5-$10/t\n"
            "  • Medium operations: $2-$5/t\n"
            "  • Large operations: $1-$3/t"
        )
        form.addRow("Mining cost ($/t):", self.c_mine)
        
        # Processing cost
        self.c_proc = QDoubleSpinBox()
        self.c_proc.setRange(0, 1e6)
        self.c_proc.setDecimals(2)
        self.c_proc.setValue(15.0)
        self.c_proc.setToolTip(
            "Cost to process one tonne of ore ($/tonne)\n"
            "Includes crushing, grinding, flotation, etc.\n"
            "Typical values:\n"
            "  • Simple processing: $8-$15/t\n"
            "  • Standard processing: $15-$30/t\n"
            "  • Complex processing: $30-$50/t"
        )
        form.addRow("Processing cost ($/t):", self.c_proc)
        
        # Slope angle
        self.slope = QDoubleSpinBox()
        self.slope.setRange(1.0, 80.0)
        self.slope.setDecimals(2)
        self.slope.setValue(45.0)
        self.slope.setToolTip(
            "Overall pit wall slope angle (degrees from horizontal)\n"
            "Depends on rock strength and geotechnical conditions\n"
            "Typical range: 35-50° for hard rock, 25-40° for soft rock"
        )
        form.addRow("Slope angle (deg):", self.slope)
        
        # Bench height
        self.bench_h = QDoubleSpinBox()
        self.bench_h.setRange(0.0, 1e5)
        self.bench_h.setDecimals(3)
        self.bench_h.setValue(0.0)
        self.bench_h.setSpecialValueText("Use grid spacing")
        self.bench_h.setToolTip(
            "Individual bench height in meters\n"
            "Set to 0 to use block model Z spacing\n"
            "Typical bench heights: 5-15m depending on equipment scale"
        )
        form.addRow("Bench height (m):", self.bench_h)
        
        # Grade cutoff
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.0, 100.0)
        self.cutoff.setDecimals(3)
        self.cutoff.setValue(0.0)
        self.cutoff.setSpecialValueText("No cutoff (economic only)")
        self.cutoff.setToolTip(
            "Minimum grade threshold for ore classification\n"
            "Blocks below cutoff are treated as waste (not processed)\n"
            "Set to 0 to use economic cutoff only\n"
            "Example values:\n"
            "  • Copper: 0.3-0.5% Cu\n"
            "  • Gold: 0.3-0.8 g/t Au\n"
            "  • Iron ore: 55-60% Fe"
        )
        form.addRow("Grade cutoff:", self.cutoff)
        
        # Nested shells controls
        form.addRow(QLabel("<b>Nested Shells (Whittle-style):</b>"), QLabel(""))
        
        self.factor_max = QDoubleSpinBox()
        self.factor_max.setRange(0.1, 5.0)
        self.factor_max.setDecimals(2)
        self.factor_max.setValue(1.30)
        self.factor_max.setToolTip(
            "Maximum revenue factor for nested shells\n"
            "Creates larger pit shells for optimistic scenarios\n"
            "Typical values:\n"
            "  • 1.20 = 20% higher revenue (conservative)\n"
            "  • 1.30 = 30% higher revenue (standard)\n"
            "  • 1.50 = 50% higher revenue (optimistic)"
        )
        form.addRow("Revenue factor max:", self.factor_max)
        
        self.factor_min = QDoubleSpinBox()
        self.factor_min.setRange(0.1, 5.0)
        self.factor_min.setDecimals(2)
        self.factor_min.setValue(0.60)
        self.factor_min.setToolTip(
            "Minimum revenue factor for nested shells\n"
            "Creates smaller pit shells for conservative scenarios\n"
            "Typical values:\n"
            "  • 0.80 = 20% lower revenue (conservative)\n"
            "  • 0.70 = 30% lower revenue (standard)\n"
            "  • 0.60 = 40% lower revenue (pessimistic)"
        )
        form.addRow("Revenue factor min:", self.factor_min)
        
        self.factor_steps = QDoubleSpinBox()
        self.factor_steps.setRange(2, 50)
        self.factor_steps.setDecimals(0)
        self.factor_steps.setValue(7)
        self.factor_steps.setToolTip(
            "Number of nested pit shells to generate\n"
            "More shells = finer resolution for phase planning\n"
            "Typical values:\n"
            "  • 5 shells = Coarse phases\n"
            "  • 7-10 shells = Standard (recommended)\n"
            "  • 15-20 shells = Fine detail for scheduling"
        )
        form.addRow("Number of shells:", self.factor_steps)
        
        # Buttons
        button_row = QHBoxLayout()
        self.layout().addLayout(button_row)
        
        self.btn_run = QPushButton("Run Pit Optimisation")
        self.btn_run.setToolTip(
            "Run Lerchs-Grossmann algorithm for optimal pit design\n"
            "Finds ultimate pit shell that maximizes economic value\n"
            "Considers slope constraints and block economics"
        )
        self.btn_run.clicked.connect(lambda: self._start_pit_analysis('single_pit'))
        button_row.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Stop the current optimization")
        self.btn_cancel.clicked.connect(self._cancel_optimization)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #d32f2f; color: white; }")
        button_row.addWidget(self.btn_cancel)
        
        self.btn_show_shell = QPushButton("Extract Pit Shell")
        self.btn_show_shell.setToolTip(
            "Extract and visualize pit shell surface\n"
            "Creates 3D mesh of final pit boundary\n"
            "Useful for visual inspection and reporting"
        )
        self.btn_show_shell.clicked.connect(self._extract_shell)
        button_row.addWidget(self.btn_show_shell)
        
        self.btn_nested = QPushButton("Run Nested Shells")
        self.btn_nested.setToolTip(
            "Generate series of nested pit shells\n"
            "Creates multiple shells at different economic scenarios\n"
            "Essential for mine scheduling and phase design"
        )
        self.btn_nested.clicked.connect(lambda: self._start_pit_analysis('nested_shells'))
        button_row.addWidget(self.btn_nested)
        
        # Add table viewer buttons
        self.btn_view_pit_table = QPushButton("View Pit Table")
        self.btn_view_pit_table.setToolTip(
            "View ultimate pit block data in table format\n"
            "Shows block coordinates, grades, and selection status"
        )
        self.btn_view_pit_table.clicked.connect(self._view_pit_table)
        self.btn_view_pit_table.setEnabled(False)
        button_row.addWidget(self.btn_view_pit_table)
        
        self.btn_view_shells_table = QPushButton("View Shells Table")
        self.btn_view_shells_table.setToolTip(
            "View nested shells summary in table format\n"
            "Shows shell statistics and economic factors"
        )
        self.btn_view_shells_table.clicked.connect(self._view_shells_table)
        self.btn_view_shells_table.setEnabled(False)
        button_row.addWidget(self.btn_view_shells_table)
        
        # Results statistics label
        self.results_label = QLabel("")
        self.results_label.setStyleSheet(
            f"background-color: {ModernColors.PANEL_BG}; color: #4fc3f7; padding: 8px; "
            "border-radius: 3px; font-weight: bold; border: 1px solid #333;"
        )
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_label.setVisible(False)
        self.layout().addWidget(self.results_label)
        
        # Status bar and progress indicator
        status_group = QVBoxLayout()
        status_group.setSpacing(5)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p% - %v")
        status_group.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            f"background-color: #2b2b2b; color: {ModernColors.TEXT_PRIMARY}; padding: 5px; border-radius: 3px;"
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_group.addWidget(self.status_label)
        
        self.layout().addLayout(status_group)
        
        # Timer for elapsed time updates
        self.timer = QTimer()
        self.start_time = None
        self.timer.timeout.connect(self._update_elapsed_time)
    
    def _auto_detect_column_mapping(self, df: pd.DataFrame) -> Optional[ColumnMapping]:
        """Auto-detect column mapping from dataframe columns."""
        try:
            columns_lower = {col: col.lower().strip() for col in df.columns}
            
            # Try to detect coordinate columns
            x_col = None
            y_col = None
            z_col = None
            volume_col = None
            density_col = None
            
            x_patterns = ['x', 'xc', 'x_centre', 'x_center', 'xcenter', 'easting', 'east']
            y_patterns = ['y', 'yc', 'y_centre', 'y_center', 'ycenter', 'northing', 'north']
            z_patterns = ['z', 'zc', 'z_centre', 'z_center', 'zcenter', 'zmid', 'rl', 'elevation', 'centroid']
            vol_patterns = ['volume', 'vol', 'block_volume', 'vol_m3']
            dens_patterns = ['density', 'dens', 'sg', 'specific_gravity']
            
            for col, col_lower in columns_lower.items():
                if not x_col and any(p in col_lower for p in x_patterns):
                    x_col = col
                if not y_col and any(p in col_lower for p in y_patterns):
                    y_col = col
                if not z_col and any(p in col_lower for p in z_patterns):
                    z_col = col
                if not volume_col and any(p in col_lower for p in vol_patterns):
                    volume_col = col
                if not density_col and any(p in col_lower for p in dens_patterns):
                    density_col = col
            
            # If we found all required columns, create mapping
            if all([x_col, y_col, z_col, volume_col, density_col]):
                return ColumnMapping(
                    x_col=x_col,
                    y_col=y_col,
                    z_col=z_col,
                    volume_col=volume_col,
                    density_col=density_col
                )
            else:
                logger.warning(f"Could not auto-detect all required columns. Found: X={x_col}, Y={y_col}, Z={z_col}, Vol={volume_col}, Dens={density_col}")
                return None
                
        except Exception as e:
            logger.error(f"Error auto-detecting column mapping: {e}")
            return None
    
    def _update_mapping_status(self):
        """Update the column mapping status label."""
        if self.column_mapping:
            self.mapping_status.setText(
                f"<i>Columns: X='{self.column_mapping.x_col}', "
                f"Y='{self.column_mapping.y_col}', "
                f"Z='{self.column_mapping.z_col}'</i>"
            )
            self.mapping_status.setStyleSheet("color: green; font-size: 9pt;")
        else:
            self.mapping_status.setText("<i>Using legacy column detection</i>")
            self.mapping_status.setStyleSheet("color: gray; font-size: 9pt;")
    
    def _configure_column_mapping(self):
        """Show column mapping configuration dialog."""
        if self.block_df is None:
            QMessageBox.warning(self, "No Data", "Please load a block model first.")
            return
        
        from .column_mapping_config_dialog import show_column_mapping_dialog
        
        new_mapping, constant_densities = show_column_mapping_dialog(
            self.block_df,
            current_mapping=self.column_mapping,
            parent=self
        )
        
        if new_mapping:
            self.column_mapping = new_mapping
            self.constant_densities = constant_densities
            
            # Apply constant densities if specified
            if constant_densities:
                self._apply_constant_densities()
            
            # Re-apply mapping
            self.block_df = self.column_mapping.apply_mapping(self.block_df, keep_original=True)
            self._update_mapping_status()
            
            # Recompute grid spec with new columns
            self.grid_spec = self._compute_grid_spec(self.block_df)
            
            density_info = ""
            if constant_densities:
                if '__DEFAULT__' in constant_densities:
                    density_info = f"\nUsing constant density: {constant_densities['__DEFAULT__']:.2f} t/m³"
                else:
                    zone_col = constant_densities.get('__ZONE_COLUMN__', 'zone')
                    num_zones = len([k for k in constant_densities.keys() if not k.startswith('__')])
                    density_info = f"\nUsing {num_zones} density values by {zone_col}"
            
            logger.info(f"Column mapping updated: X='{new_mapping.x_col}', Y='{new_mapping.y_col}', Z='{new_mapping.z_col}'{density_info}")
            QMessageBox.information(
                self,
                "Column Mapping Updated",
                f"Column mapping has been updated:\n\n"
                f"X Coordinate: {new_mapping.x_col}\n"
                f"Y Coordinate: {new_mapping.y_col}\n"
                f"Z Coordinate: {new_mapping.z_col}\n"
                f"Volume: {new_mapping.volume_col}\n"
                f"Density: {new_mapping.density_col if not constant_densities else 'Constant values'}"
                f"{density_info}"
            )
    
    def _apply_constant_densities(self):
        """Apply constant density values to the block model."""
        if not self.constant_densities:
            return
        
        if '__DEFAULT__' in self.constant_densities:
            # Single constant density for all blocks
            default_density = self.constant_densities['__DEFAULT__']
            self.block_df['density'] = default_density
            logger.info(f"Applied constant density: {default_density:.2f} t/m³ to all blocks")
        else:
            # Multiple densities by zone
            zone_col = self.constant_densities.get('__ZONE_COLUMN__')
            if zone_col and zone_col in self.block_df.columns:
                # Create density column based on zone mapping
                def get_density_for_zone(zone):
                    return self.constant_densities.get(str(zone), 2.70)  # Default to 2.70 if not found
                
                self.block_df['density'] = self.block_df[zone_col].apply(get_density_for_zone)
                
                num_zones = len([k for k in self.constant_densities.keys() if not k.startswith('__')])
                logger.info(f"Applied {num_zones} density values based on column '{zone_col}'")
            else:
                logger.warning(f"Zone column '{zone_col}' not found, using default density 2.70")
                self.block_df['density'] = 2.70
    
    def _detect_grade_columns(self, df: pd.DataFrame) -> list:
        """
        Detect possible grade columns from DataFrame.
        
        Returns list of numeric column names, excluding:
        - Coordinate columns (X, Y, Z and variants)
        - Geometric columns (dx, dy, dz, volume, tonnage)
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of candidate grade column names
        """
        exclude = {'x', 'y', 'z', 'dx', 'dy', 'dz', 'volume', 'tonnage', 
                  'xc', 'yc', 'zc', 'easting', 'northing', 'rl', 'elevation',
                  'x_centre', 'y_centre', 'z_centre', 'zmid', 'zmin', 'zmax'}
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        
        # Filter out excluded columns (case-insensitive)
        candidates = [
            c for c in numeric_cols 
            if c.lower().strip() not in exclude
        ]
        
        return sorted(candidates)  # Sort alphabetically for consistency
    
    def _auto_detect_grade(self):
        """Re-scan block model and update grade column dropdown."""
        if self.block_df is None or self.block_df.empty:
            QMessageBox.information(
                self,
                "No Block Model",
                "Please load a block model first."
            )
            return
        
        candidates = self._detect_grade_columns(self.block_df)
        
        # Save current selection
        current_text = self.grade_combo.currentText().strip()
        
        # Update combo box
        self.grade_combo.clear()
        
        if candidates:
            self.grade_combo.addItems(candidates)
            # Try to restore previous selection if still valid
            if current_text in candidates:
                self.grade_combo.setCurrentText(current_text)
            else:
                self.grade_combo.setCurrentText(candidates[0])
            
            logger.info(f"Auto-detected {len(candidates)} grade column candidates: {candidates}")
        else:
            self.grade_combo.setCurrentText("Fe")
            logger.warning("No grade column candidates found, using default 'Fe'")
        
        # Enable column mapping configuration
        self.btn_configure_columns.setEnabled(True)
        
        logger.info(f"Block model loaded: {len(self.block_df)} blocks, grid={self.grid_spec}")
    
    def _update_elapsed_time(self):
        """Update elapsed time display in status bar."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.status_label.setText(
                f"{self.status_label.text().split(' - ')[0]} - Elapsed: {minutes}m {seconds}s"
            )
    
    def _update_status(self, message: str, progress: Optional[int] = None):
        """Update status bar and progress bar."""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setValue(progress)
            self.progress_bar.setVisible(True)
        QApplication.processEvents()  # Update UI
    
    def _reset_status(self):
        """Reset status bar to ready state."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.start_time = None
        self.timer.stop()
    
    def _base_params(self) -> PitParams:
        """Create base PitParams from UI values."""
        return PitParams(
            grade_col=self.grade_combo.currentText().strip(),
            price_per_unit=self.price.value(),
            recovery=self.recovery.value(),
            mining_cost_per_t=self.c_mine.value(),
            processing_cost_per_t=self.c_proc.value(),
            slope_deg=self.slope.value(),
            bench_height=(self.bench_h.value() if self.bench_h.value() > 0 else None),
            cutoff=(self.cutoff.value() if self.cutoff.value() > 0 else None),
        )
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        try:
            # Convert BlockModel to DataFrame if needed
            if hasattr(block_model, 'to_dataframe'):
                block_df = block_model.to_dataframe()
                # Try to extract grid spec from BlockModel metadata
                grid_spec = None
                if hasattr(block_model, 'grid_spec'):
                    grid_spec = block_model.grid_spec
            elif isinstance(block_model, pd.DataFrame):
                block_df = block_model
                grid_spec = None
            else:
                logger.warning(f"Unexpected block model type: {type(block_model)}")
                return
            
            # Set block model using existing method
            self.set_block_model(block_df, grid_spec)
            logger.info(f"Pit Optimisation Panel auto-received block model: {len(block_df)} blocks")
        except Exception as e:
            logger.error(f"Error processing block model in Pit Optimisation Panel: {e}", exc_info=True)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        # Use same handler as generated
        self._on_block_model_generated(block_model)
    
    def set_block_model(self, block_df: pd.DataFrame, grid_spec: Optional[Dict] = None):
        """
        Set the block model data for optimization.
        
        Args:
            block_df: DataFrame with columns X, Y, Z (or any custom names) and grade/property columns
            grid_spec: Optional grid specification dict. If None, will be computed from data.
        
        Note:
            Column mappings can be configured via the "Configure Column Mapping" button.
            Auto-detection attempts to find common coordinate column patterns.
        """
        # Store original dataframe (don't normalize yet - let user configure)
        self.block_df = block_df.copy()
        
        # Try to auto-detect column mapping
        self.column_mapping = self._auto_detect_column_mapping(self.block_df)
        
        # Apply mapping to create standardized columns
        if self.column_mapping:
            self.block_df = self.column_mapping.apply_mapping(self.block_df, keep_original=True)
            self._update_mapping_status()
        else:
            # Fallback to legacy normalization
            self.block_df = normalize_coordinate_columns(self.block_df)
        
        if grid_spec is None:
            self.grid_spec = self._compute_grid_spec(self.block_df)
        else:
            self.grid_spec = grid_spec
        
        # Auto-detect and populate grade columns
        candidates = self._detect_grade_columns(self.block_df)
        self.grade_combo.clear()
        
        if candidates:
            self.grade_combo.addItems(candidates)
            # Auto-select first candidate
            self.grade_combo.setCurrentText(candidates[0])
            if len(candidates) > 1:
                logger.info(f"Multiple grade columns detected: {candidates}. Auto-selected '{candidates[0]}'")
            else:
                logger.info(f"Auto-detected grade column: '{candidates[0]}'")
        else:
            # Fallback to default if no candidates found
            self.grade_combo.setCurrentText("Fe")
            logger.warning("No grade column candidates found, using default 'Fe'")
        
        logger.info(f"Set block model: {len(self.block_df)} blocks, grid: {self.grid_spec}")
    
    def _compute_grid_spec(self, df: pd.DataFrame) -> Dict:
        """
        Compute grid specification from block model data.
        
        Assumes regular grid. Finds unique coordinates and computes spacing.
        Coordinate columns should already be normalized to x, y, z (lowercase).
        """
        # Verify coordinate columns exist (standardized lowercase after mapping)
        if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
            raise ValueError(
                f"Coordinate columns x, y, z not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Get unique sorted coordinates
        x_coords = np.unique(df['x'].values)
        y_coords = np.unique(df['y'].values)
        z_coords = np.unique(df['z'].values)
        
        nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
        
        # Compute increments
        if nx > 1:
            xinc = float(np.mean(np.diff(np.sort(x_coords))))
            xmin = float(np.min(x_coords)) - xinc/2
        else:
            xinc = 1.0
            xmin = float(x_coords[0]) - xinc/2
        
        if ny > 1:
            yinc = float(np.mean(np.diff(np.sort(y_coords))))
            ymin = float(np.min(y_coords)) - yinc/2
        else:
            yinc = 1.0
            ymin = float(y_coords[0]) - yinc/2
        
        if nz > 1:
            zinc = float(np.mean(np.diff(np.sort(z_coords))))
            zmin = float(np.min(z_coords)) - zinc/2
        else:
            zinc = 1.0
            zmin = float(z_coords[0]) - zinc/2
        
        return {
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'xmin': xmin,
            'ymin': ymin,
            'zmin': zmin,
            'xinc': xinc,
            'yinc': yinc,
            'zinc': zinc
        }
    
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------
    
    def _start_pit_analysis(self, operation: str) -> None:
        """Start pit optimization with specified operation."""
        self.current_operation = operation
        self.run_analysis()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI based on current operation."""
        if self.block_df is None or self.block_df.empty:
            raise ValueError("Please load a block model in the main viewer first.")
        
        if self.grid_spec is None:
            raise ValueError("Could not determine grid specification from block model.")
        
        grade_col = self.grade_combo.currentText().strip()
        if grade_col not in self.block_df.columns:
            raise ValueError(f"Grade column '{grade_col}' not found in block model.")
        
        base_params = self._base_params()
        
        if self.current_operation == "single_pit":
            # Convert to format expected by pit_opt task function
            return {
                "block_model": self.block_df.copy(),
                "price": base_params.price_per_unit,
                "mining_cost": base_params.mining_cost_per_t,
                "proc_cost": base_params.processing_cost_per_t,
                "recovery": base_params.recovery,
                "grade_col": base_params.grade_col,
                "use_fast_solver": True
            }
        elif self.current_operation == "nested_shells":
            pmax = float(self.factor_max.value())
            pmin = float(self.factor_min.value())
            steps = int(self.factor_steps.value())
            
            if steps < 2 or pmax <= pmin:
                raise ValueError("Ensure steps >= 2 and factor_max > factor_min.")
            
            factors = np.linspace(pmax, pmin, steps)
            
            return {
                "operation": "nested_shells",
                "block_df": self.block_df.copy(),
                "grid_spec": self.grid_spec,
                "base_params": base_params,
                "factors": factors,
                "column_mapping": self.column_mapping,
            }
        else:
            raise ValueError(f"Unknown pit operation: {self.current_operation}")
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if self.block_df is None or self.block_df.empty:
            self.show_error("No Block Model", "Please load a block model in the main viewer first.")
            return False
        
        if self.grid_spec is None:
            self.show_error("No Grid Spec", "Could not determine grid specification from block model.")
            return False
        
        grade_col = self.grade_combo.currentText().strip()
        if grade_col not in self.block_df.columns:
            self.show_error("Column Missing", f"Grade column '{grade_col}' not found in block model.")
            return False
        
        # Economic parameter validation (warnings only, not blocking)
        metal_price = self.price.value()
        mining_cost = self.c_mine.value()
        processing_cost = self.c_proc.value()
        recovery = self.recovery.value()
        
        warnings = []
        if metal_price < 10 or metal_price > 100000:
            warnings.append(f"Price (${metal_price:.2f}/unit) appears unusual")
        if mining_cost < 0.5 or mining_cost > 50:
            warnings.append(f"Mining cost (${mining_cost:.2f}/t) appears unusual")
        if processing_cost < 2 or processing_cost > 100:
            warnings.append(f"Processing cost (${processing_cost:.2f}/t) appears unusual")
        if recovery < 0.5 or recovery > 1.0:
            warnings.append(f"Recovery ({recovery*100:.1f}%) appears unusual")
        
        if warnings:
            # Show warning but allow continuation
            logger.warning(f"Economic parameter warnings: {warnings}")
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display pit optimization results."""
        if payload is None:
            return
        
        if payload.get("error"):
            self.show_error("Optimization Error", payload["error"])
            return
        
        operation = self.current_operation or payload.get("operation", "single_pit")
        
        # --- NEW: Handle BlockModel API (standard) ---
        from ..models.block_model import BlockModel
        
        if operation == "single_pit":
            # Check if BlockModel was returned (new standard API)
            block_model = payload.get("block_model")
            if isinstance(block_model, BlockModel):
                # ✅ STANDARD API: Results already added to BlockModel
                logger.info("✅ BlockModel API: Results added to BlockModel")
                
                # Extract arrays for visualization
                in_pit = block_model.get_property('IN_PIT')
                lg_value = block_model.get_property('LG_VALUE')
                
                # CRITICAL: Check for NSR data in result (for proper color mapping)
                nsr_data = block_model.get_property('NSR') or block_model.get_property('NSR_TOTAL')
                if nsr_data is None and 'nsr' in payload:
                    # NSR data in payload but not in BlockModel - add it
                    nsr_array = payload.get('nsr')
                    if nsr_array is not None:
                        block_model.add_property('NSR_TOTAL', np.asarray(nsr_array))
                        logger.info("Added NSR_TOTAL property to BlockModel from pit optimization result")
                
                if in_pit is not None and lg_value is not None:
                    # Reshape to 3D for visualization (if needed)
                    # Get grid dimensions from block model
                    positions = block_model.positions
                    if positions is not None:
                        # Estimate grid dimensions (simplified)
                        # In production, store grid_spec in BlockModel metadata
                        selected = in_pit.astype(bool)
                        V = lg_value.astype(float)
                        rgrid = None
                        self._on_worker_finished(selected, V, rgrid)
                    else:
                        logger.warning("BlockModel missing positions for visualization")
                
                # Register updated BlockModel
                try:
                    if hasattr(self, 'registry') and self.registry:
                        self.registry.register_block_model(block_model, source_panel="PitOptimisationPanel")
                        logger.info("✅ Registered updated BlockModel with pit optimization results")
                except Exception as e:
                    logger.warning(f"Failed to register BlockModel: {e}")
                
                return
            
            # --- LEGACY: Handle DataFrame (backward compatibility) ---
            result_df = payload.get("result_df")
            if result_df is not None and isinstance(result_df, pd.DataFrame):
                # Convert to format expected by _on_worker_finished
                selected = result_df['IN_PIT'].values.astype(bool) if 'IN_PIT' in result_df.columns else np.array([False] * len(result_df))
                V = result_df['LG_VALUE'].values if 'LG_VALUE' in result_df.columns else np.array([0.0] * len(result_df))
                rgrid = None
                self._on_worker_finished(selected, V, rgrid)
                
                # Try to add to existing BlockModel if available
                try:
                    if hasattr(self, 'registry') and self.registry:
                        existing_bm = self.registry.get_block_model()
                        if isinstance(existing_bm, BlockModel):
                            # Add results to existing BlockModel
                            existing_bm.add_property('IN_PIT', selected.astype(int))
                            existing_bm.add_property('LG_VALUE', V)
                            
                            # CRITICAL: Add NSR data if available (for proper color mapping)
                            if 'NSR_TOTAL' in result_df.columns:
                                existing_bm.add_property('NSR_TOTAL', result_df['NSR_TOTAL'].values)
                                logger.info("Added NSR_TOTAL property to BlockModel from DataFrame")
                            elif 'nsr' in payload:
                                # NSR in payload metadata
                                nsr_array = payload.get('nsr')
                                if nsr_array is not None:
                                    existing_bm.add_property('NSR_TOTAL', np.asarray(nsr_array))
                                    logger.info("Added NSR_TOTAL property to BlockModel from payload metadata")
                            
                            self.registry.register_block_model(existing_bm, source_panel="PitOptimisationPanel")
                            logger.info("✅ Added pit optimization results to existing BlockModel")
                except Exception as e:
                    logger.warning(f"Failed to add results to BlockModel: {e}")
            else:
                logger.warning("No result_df or block_model in payload")
                self.show_error("Result Error", "Invalid result format from pit optimization")
        
        elif operation == "nested_shells":
            # Nested shells - handle separately if needed
            shells_summary_data = payload.get("shells_summary_data")
            if shells_summary_data is not None:
                self._on_nested_finished(shells_summary_data)
        
        # Publish results to DataRegistry (legacy format for backward compatibility)
        try:
            if hasattr(self, 'registry') and self.registry:
                results = {
                    'operation': operation,
                    'payload': payload
                }
                self.registry.register_pit_optimization_results(results, source_panel="PitOptimisationPanel")
                logger.info(f"Published pit optimization results ({operation}) to DataRegistry")
        except Exception as e:
            logger.warning(f"Failed to publish pit optimization results to DataRegistry: {e}")
    
    def _run(self):
        """Run pit optimisation (legacy method - now routed through BaseAnalysisPanel)."""
        # This method is kept for backward compatibility but is no longer used
        # Button connections now call _start_pit_analysis('single_pit')
        pass
    
    def _run_legacy(self):
        """Legacy run method - removed, now uses BaseAnalysisPanel.run_analysis() via task system."""
        # This method is kept for reference but is no longer used
        # The panel now uses BaseAnalysisPanel.run_analysis() which calls controller.run_task()
        pass
    
    def _on_worker_progress(self, progress: int, message: str):
        """Handle progress updates (now via BaseAnalysisPanel progress handling)."""
        self._update_status(message, progress)
    
    def _on_worker_finished(self, selected, V, rgrid):
        """Handle completion from pit optimization."""
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        try:
            self._update_status("Preparing visualization...", 95)
            
            # Store ultimate pit data for table viewing
            self.ultimate_pit_df = self.block_df.copy()
            # Flatten 3D selected array to 1D to match DataFrame length
            # Use Fortran order to match VTK grid ordering
            selected_flat = selected.ravel(order='F')
            self.ultimate_pit_df['Selected'] = selected_flat
            # Also flatten V array
            V_flat = V.ravel(order='F')
            self.ultimate_pit_df['Net_Value'] = V_flat
            
            # Enable table view button
            self.btn_view_pit_table.setEnabled(True)
            
            # Create proper pit surface from exposed block faces
            self._update_status("Generating pit surface from exposed faces...", 96)
            pit_surface_payload = self._create_pit_surface_payload(selected, self.grid_spec, name='pit_optimisation')
            
            self._update_status("Sending to 3D viewer...", 98)
            
            # Send to renderer via controller
            if pit_surface_payload and self.controller:
                self.controller.apply_render_payload(pit_surface_payload)
            
            # Statistics
            n_selected = np.sum(selected)
            total_value = np.sum(V[selected])
            
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            
            # Update results display
            self.results_label.setText(
                f"✓ Ultimate Pit: {n_selected:,} blocks | "
                f"Value: ${total_value/1e6:.1f}M | Time: {elapsed_str}"
            )
            self.results_label.setVisible(True)
            
            self._update_status(f"Complete! Selected {n_selected:,} blocks (${total_value:,.2f} value)", 100)
            
            self._enable_buttons()
            
            QMessageBox.information(
                self,
                "Pit Optimisation Complete",
                f"Ultimate pit computed and visualized.\n\n"
                f"Selected blocks: {n_selected:,}\n"
                f"Total net value: ${total_value:,.2f}\n"
                f"Processing time: {elapsed_str}\n\n"
                f"The pit has been added to the 3D viewer."
            )
            
            logger.info(f"Pit optimization complete: {n_selected} blocks selected, value: ${total_value:,.2f}, time: {elapsed_str}")
            
            # Reset status after 3 seconds
            QTimer.singleShot(3000, self._reset_status)
            
        except Exception as e:
            self._enable_buttons()
            logger.error(f"Error handling pit optimization results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error visualizing results:\n\n{str(e)}"
            )
            QTimer.singleShot(5000, self._reset_status)
    
    def _on_worker_error(self, error_msg: str):
        """Handle error from pit optimization (now via BaseAnalysisPanel error handling)."""
        if hasattr(self, 'timer'):
            self.timer.stop()
        self._enable_buttons()
        self._update_status(f"Error: {error_msg}", 0)
        logger.error(f"Pit optimization error: {error_msg}")
        QMessageBox.critical(
            self,
            "Optimization Error",
            f"Error during pit optimization:\n\n{error_msg}\n\n"
            f"Please check the log for details."
        )
        # Reset status after error
        QTimer.singleShot(5000, self._reset_status)
    
    def _enable_buttons(self):
        """Re-enable all buttons."""
        self.btn_run.setEnabled(True)
        self.btn_nested.setEnabled(True)
        self.btn_show_shell.setEnabled(True)
        self.btn_cancel.setEnabled(False)
    
    def _cancel_optimization(self):
        """Cancel the currently running optimization."""
        self._cancel_requested = True
        
        # Cancel via controller task system
        if self.controller and hasattr(self.controller, 'cancel_task'):
            self.controller.cancel_task(self.task_name)
            logger.info("Cancellation requested via controller")
        
        # Stop timer
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        # Update UI
        self._update_status("Cancelled by user", 0)
        self._enable_buttons()
        
        QMessageBox.information(
            self,
            "Optimization Cancelled",
            "The optimization process has been cancelled."
        )
        
        # Reset status after a delay
        QTimer.singleShot(3000, self._reset_status)
    
    def _create_pit_surface_payload(self, selected: np.ndarray, grid_spec: Dict, name: str = "pit_surface") -> Optional['MeshPayload']:
        """
        Create pit surface mesh payload from exposed block faces.
        
        Only creates faces for blocks that are:
        1. Selected (in the pit)
        2. Have at least one face exposed (not touching another pit block)
        
        Parameters:
        -----------
        selected : np.ndarray (nx, ny, nz)
            Boolean array of selected blocks
        grid_spec : dict
            Grid specification with dimensions and spacing
        name : str
            Name for the mesh layer
        
        Returns:
        --------
        MeshPayload or None if no faces found
        """
        from ..visualization.render_payloads import MeshPayload
        
        nx, ny, nz = selected.shape
        dx, dy, dz = grid_spec['xinc'], grid_spec['yinc'], grid_spec['zinc']
        xmin, ymin, zmin = grid_spec['xmin'], grid_spec['ymin'], grid_spec['zmin']
        
        vertices = []
        faces = []
        
        logger.info(f"Generating pit surface from {np.sum(selected)} selected blocks...")
        
        # For each selected block, check which faces are exposed
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not selected[i, j, k]:
                        continue
                    
                    # Block center and corners
                    x0 = xmin + i * dx
                    y0 = ymin + j * dy
                    z0 = zmin + k * dz
                    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
                    
                    # Define 8 corners of the block
                    corners = [
                        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # Bottom face (z=z0)
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # Top face (z=z1)
                    ]
                    
                    base_idx = len(vertices)
                    
                    # Top face (k+1) - always exposed if at top or neighbor above not selected
                    if k == nz - 1 or not selected[i, j, k + 1]:
                        for corner in [corners[4], corners[5], corners[6], corners[7]]:
                            vertices.append(corner)
                        faces.extend([
                            [3, base_idx, base_idx + 1, base_idx + 2],
                            [3, base_idx, base_idx + 2, base_idx + 3]
                        ])
                        base_idx += 4
                    
                    # Bottom face (k-1)
                    if k == 0 or not selected[i, j, k - 1]:
                        for corner in [corners[0], corners[1], corners[2], corners[3]]:
                            vertices.append(corner)
                        faces.extend([
                            [3, base_idx, base_idx + 2, base_idx + 1],
                            [3, base_idx, base_idx + 3, base_idx + 2]
                        ])
                        base_idx += 4
                    
                    # Front face (j-1)
                    if j == 0 or not selected[i, j - 1, k]:
                        for corner in [corners[0], corners[1], corners[5], corners[4]]:
                            vertices.append(corner)
                        faces.extend([
                            [3, base_idx, base_idx + 1, base_idx + 2],
                            [3, base_idx, base_idx + 2, base_idx + 3]
                        ])
                        base_idx += 4
                    
                    # Back face (j+1)
                    if j == ny - 1 or not selected[i, j + 1, k]:
                        for corner in [corners[3], corners[2], corners[6], corners[7]]:
                            vertices.append(corner)
                        faces.extend([
                            [3, base_idx, base_idx + 1, base_idx + 2],
                            [3, base_idx, base_idx + 2, base_idx + 3]
                        ])
                        base_idx += 4
                    
                    # Left face (i-1)
                    if i == 0 or not selected[i - 1, j, k]:
                        for corner in [corners[0], corners[4], corners[7], corners[3]]:
                            vertices.append(corner)
                        faces.extend([
                            [3, base_idx, base_idx + 1, base_idx + 2],
                            [3, base_idx, base_idx + 2, base_idx + 3]
                        ])
                        base_idx += 4
                    
                    # Right face (i+1)
                    if i == nx - 1 or not selected[i + 1, j, k]:
                        for corner in [corners[1], corners[2], corners[6], corners[5]]:
                            vertices.append(corner)
                        faces.extend([
                            [3, base_idx, base_idx + 1, base_idx + 2],
                            [3, base_idx, base_idx + 2, base_idx + 3]
                        ])
                        base_idx += 4
        
        # Convert to payload
        if not vertices:
            logger.warning("No exposed faces found - creating empty mesh")
            return None
        
        vertices = np.array(vertices, dtype=np.float64)
        faces_flat = []
        for face in faces:
            faces_flat.extend(face)
        faces_array = np.array(faces_flat, dtype=np.int32)
        
        logger.info(f"Created pit surface payload: {len(vertices)} vertices, {len(faces)} triangles")
        
        return MeshPayload(
            name=name,
            vertices=vertices,
            faces=faces_array,
            scalars=None,
            opacity=0.7,
            visible=True,
            metadata={'type': 'pit_surface'}
        )
    
    def _extract_shell(self):
        """Extract pit shell surface (legacy method - now handled by controller)."""
        # This method is deprecated - shell extraction is now handled by the controller
        # when pit optimization completes. Keeping for backward compatibility.
        QMessageBox.information(
            self,
            "Shell Extraction",
            "Pit shell visualization is now handled automatically when pit optimization completes.\n\n"
            "If you need to extract a shell manually, please re-run the pit optimization."
        )
    
    def _run_nested(self):
        """Run nested shell optimization with varying price factors (using background thread)."""
        # Validation
        if self.block_df is None or self.block_df.empty:
            QMessageBox.warning(
                self,
                "No Block Model",
                "Please load a block model in the main viewer first."
            )
            return
        
        if self.grid_spec is None:
            QMessageBox.warning(
                self,
                "No Grid Spec",
                "Could not determine grid specification from block model."
            )
            return
        
        grade_col = self.grade_combo.currentText().strip()
        if grade_col not in self.block_df.columns:
            QMessageBox.warning(
                self,
                "Column Missing",
                f"Grade column '{grade_col}' not found."
            )
            return
        
        # Check if already running
        if self.nested_worker and self.nested_worker.isRunning():
            QMessageBox.information(
                self,
                "Already Running",
                "Nested shells optimization is already in progress. Please wait..."
            )
            return
        
        pmax = float(self.factor_max.value())
        pmin = float(self.factor_min.value())
        steps = int(self.factor_steps.value())
        
        if steps < 2 or pmax <= pmin:
            QMessageBox.warning(
                self,
                "Invalid Parameters",
                "Ensure steps >= 2 and factor_max > factor_min."
            )
            return
        
        try:
            # Reset cancellation flag
            self._cancel_requested = False
            
            # Disable buttons during processing, enable cancel
            self.btn_run.setEnabled(False)
            self.btn_nested.setEnabled(False)
            self.btn_show_shell.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            
            # Initialize progress
            self.start_time = time.time()
            self.timer.start(100)  # Update every 100ms
            
            # Normalize coordinate columns before processing
            self._update_status("Initializing nested shells optimization...", 0)
            block_df_normalized = normalize_coordinate_columns(self.block_df.copy())
            
            base = self._base_params()
            factors = np.linspace(pmax, pmin, steps)
            
            logger.info(f"Running nested shells: {steps} shells from {pmax:.2f} to {pmin:.2f}")
            
            # Create optimizer
            from ..models.pit_optimizer import ProductionPitOptimizer
            optimizer = ProductionPitOptimizer(self.grid_spec, column_mapping=self.column_mapping)
            
            # Add element
            element = ElementSpec(
                name="Primary",
                grade_col=base.grade_col,
                unit_type=UnitType.PERCENT,
                price_per_unit=base.price_per_unit,
                recovery_primary=base.recovery
            )
            optimizer.add_element(element)
            
            # Set costs
            costs = CostStructure(
                mining_cost_per_t=base.mining_cost_per_t,
                primary_processing_cost=base.processing_cost_per_t
            )
            optimizer.set_cost_structure(costs)
            
            # Add geotech sector
            sector = GeoTechSector(
                name="Uniform",
                azimuth_min=0,
                azimuth_max=360,
                slope_angle=base.slope_deg,
                bench_height=base.bench_height or self.grid_spec['zinc']
            )
            optimizer.add_geotech_sector(sector)
            
            # Store for later use
            self._nested_shells_data = []
            
            # Create and start worker thread
            self.nested_worker = NestedShellsWorker(
                block_df_normalized,
                self.grid_spec,
                base,
                factors,
                self.column_mapping,
                optimizer
            )
            
            # Connect signals
            self.nested_worker.progress.connect(self._on_nested_progress)
            self.nested_worker.shell_ready.connect(self._on_shell_ready)
            self.nested_worker.finished.connect(self._on_nested_finished)
            self.nested_worker.error.connect(self._on_nested_error)
            
            # Start worker
            self.nested_worker.start()
            
        except Exception as e:
            self.timer.stop()
            self._update_status(f"Error: {str(e)}", 0)
            self._enable_buttons()
            logger.error(f"Nested shells error: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Nested Shells Error",
                f"Error starting nested shells optimization:\n\n{str(e)}\n\n"
                f"Please check the log for details."
            )
            QTimer.singleShot(5000, self._reset_status)
    
    def _on_nested_progress(self, progress: int, message: str):
        """Handle progress updates from nested shells worker."""
        self._update_status(message, progress)
    
    def _on_shell_ready(self, revenue_factor: float, selected: np.ndarray, layer_name: str):
        """Handle a shell being ready for visualization."""
        try:
            # Create pit surface in main thread (PyVista operations)
            shell_payload = self._create_pit_surface_payload(selected, self.grid_spec, name=layer_name)
            
            # Send to renderer via controller
            if shell_payload and self.controller:
                self.controller.apply_render_payload(shell_payload)
            
            logger.info(f"Visualized shell {layer_name}")
            
        except Exception as e:
            logger.error(f"Error visualizing shell: {e}", exc_info=True)
    
    def _on_nested_finished(self, shells_summary_data: list):
        """Handle completion of nested shells optimization."""
        self.timer.stop()
        self.nested_worker = None
        
        try:
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            
            # Store shells summary for table viewing
            self.nested_shells_summary = pd.DataFrame(shells_summary_data)
            self.btn_view_shells_table.setEnabled(True)
            
            # Calculate total blocks and value across all shells
            total_blocks = sum(item['Blocks'] for item in shells_summary_data)
            total_value = sum(item['Total_Value'] for item in shells_summary_data)
            
            # Update results display
            self.results_label.setText(
                f"✓ Nested Shells: {len(shells_summary_data)} shells extracted | "
                f"Total blocks: {total_blocks:,} | "
                f"Combined value: ${total_value/1e6:.1f}M | "
                f"Time: {elapsed_str}"
            )
            self.results_label.setVisible(True)
            
            self._update_status(f"Complete! Generated {len(shells_summary_data)} nested shells in {elapsed_str}", 100)
            
            self._enable_buttons()
            
            QMessageBox.information(
                self,
                "Nested Shells Complete",
                f"Successfully generated {len(shells_summary_data)} nested pit shells.\n\n"
                f"Total blocks (all shells): {total_blocks:,}\n"
                f"Combined value: ${total_value/1e6:.1f}M\n"
                f"Time elapsed: {elapsed_str}\n\n"
                f"View the shells in the 3D viewer and check the summary table."
            )
            
        except Exception as e:
            logger.error(f"Error finalizing nested shells: {e}", exc_info=True)
            self._enable_buttons()
        
        # Reset status after delay
        QTimer.singleShot(5000, self._reset_status)
    
    def _on_nested_error(self, error_msg: str):
        """Handle error from nested shells worker."""
        self.timer.stop()
        self.nested_worker = None
        
        self._update_status(f"Error: {error_msg}", 0)
        self._enable_buttons()
        
        logger.error(f"Nested shells worker error: {error_msg}")
        
        QMessageBox.critical(
            self,
            "Nested Shells Error",
            f"Error during nested shell generation:\n\n{error_msg}"
        )
        
        # Reset status after delay
        QTimer.singleShot(5000, self._reset_status)
    
    def _view_pit_table(self):
        """Open ultimate pit results in table viewer."""
        if self.ultimate_pit_df is None or self.ultimate_pit_df.empty:
            QMessageBox.information(
                self,
                "No Results",
                "No ultimate pit results available.\n\nPlease run pit optimization first."
            )
            return
        
        try:
            # Get main window - traverse parent hierarchy
            main_window = self.parent()
            while main_window and not hasattr(main_window, 'open_table_viewer_window_from_df'):
                main_window = main_window.parent()
            
            if main_window and hasattr(main_window, 'open_table_viewer_window_from_df'):
                main_window.open_table_viewer_window_from_df(
                    self.ultimate_pit_df,
                    "Ultimate Pit - Block Data"
                )
                logger.info("Opened ultimate pit table viewer")
            else:
                QMessageBox.warning(
                    self,
                    "Feature Unavailable",
                    "Table viewer feature is not available in the main window."
                )
        except Exception as e:
            logger.error(f"Error opening pit table viewer: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Error opening table viewer:\n\n{str(e)}"
            )
    
    def _view_shells_table(self):
        """Open nested shells summary in table viewer."""
        if self.nested_shells_summary is None or self.nested_shells_summary.empty:
            QMessageBox.information(
                self,
                "No Results",
                "No nested shells results available.\n\nPlease run nested shells optimization first."
            )
            return
        
        try:
            # Get main window - traverse parent hierarchy
            main_window = self.parent()
            while main_window and not hasattr(main_window, 'open_table_viewer_window_from_df'):
                main_window = main_window.parent()
            
            if main_window and hasattr(main_window, 'open_table_viewer_window_from_df'):
                main_window.open_table_viewer_window_from_df(
                    self.nested_shells_summary,
                    "Nested Pit Shells - Summary"
                )
                logger.info("Opened nested shells table viewer")
            else:
                QMessageBox.warning(
                    self,
                    "Feature Unavailable",
                    "Table viewer feature is not available in the main window."
                )
        except Exception as e:
            logger.error(f"Error opening shells table viewer: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Error opening table viewer:\n\n{str(e)}"
            )


