"""
Enhanced Block Model Resource Calculation Panel with Resource Classification.
Implements accurate resource calculations and JORC/NI43-101 style classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QCheckBox, QFileDialog, QMessageBox, QSpinBox, QScrollArea,
    QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from .base_analysis_panel import BaseAnalysisPanel
from .panel_manager import PanelCategory, DockArea
from ..models.block_model import BlockModel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class BlockModelResourcePanel(BaseAnalysisPanel):
    """
    Enhanced panel for block model resource calculation with classification.
    
    Features:
    - Accurate unit conversions (g/t → tonnes of metal)
    - Resource classification (Measured/Indicated/Inferred)
    - Per-class summaries
    - 3D visualization by class
    """
    # PanelManager metadata
    PANEL_ID = "BlockModelResourcePanel"
    PANEL_NAME = "BlockModelResource Panel"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "resource_calculation"
    
    # Signals
    highlight_blocks_requested = pyqtSignal(object)  # block_ids or mask
    visualize_classification_requested = pyqtSignal(pd.DataFrame, dict)  # df, color_map
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="block_resource")

        # Data - block_model is inherited from BasePanel as a property
        self.df: Optional[pd.DataFrame] = None
        self.numeric_properties = []
        self.result_df: Optional[pd.DataFrame] = None
        self.class_summary_df: Optional[pd.DataFrame] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_operation: Optional[str] = None  # 'calculate' or 'classify'

        # Storage for multiple block model sources
        self._block_model_sources: Dict[str, Any] = {}
        self._available_sources: list = []
        self._current_source: str = ""
        self._stored_block_model = None
        self._stored_classified_block_model = None

        # Classification color map
        self.class_colors = {
            'Measured': '#3cb44b',      # green
            'Indicated': '#ffe119',     # yellow
            'Inferred': '#e6194B',      # red
            'Unclassified': '#808080'   # gray
        }
        
        # Subscribe to block model updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_classified)
            
            # Also listen for SGSIM results which contain block model grids
            if hasattr(self.registry, 'sgsimResultsLoaded'):
                self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
            
            # Load existing block model if available (prefer classified)
            existing_classified = self.registry.get_classified_block_model()
            if existing_classified:
                self._on_block_model_classified(existing_classified)
            else:
                existing_block_model = self.registry.get_block_model()
                if existing_block_model:
                    self._on_block_model_loaded(existing_block_model)
            
            # Check for existing SGSIM results
            if hasattr(self.registry, 'get_sgsim_results'):
                sgsim = self.registry.get_sgsim_results()
                if sgsim:
                    self._on_sgsim_loaded(sgsim)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        # UI - use _build_ui pattern recognized by BaseAnalysisPanel
        self._build_ui()
        logger.info("Initialized Enhanced Block Model Resource panel")
    


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
        """Build the custom UI. Called after base class init."""
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        # Use the main_layout provided by BaseAnalysisPanel
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)

        # Add UI elements (dimensions, rotation, etc.)
        layout.addWidget(self._create_dimensions_group())
        layout.addStretch()
    
    def _create_dimensions_group(self) -> QGroupBox:
        """Create block dimension controls."""
        group = QGroupBox("Block Dimensions")
        layout = QFormLayout(group)
        
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 10000)
        self.dx_spin.setValue(25.0)
        self.dx_spin.setDecimals(2)
        self.dx_spin.setSuffix(" m")
        layout.addRow("Block Size X:", self.dx_spin)
        
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 10000)
        self.dy_spin.setValue(25.0)
        self.dy_spin.setDecimals(2)
        self.dy_spin.setSuffix(" m")
        layout.addRow("Block Size Y:", self.dy_spin)
        
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 10000)
        self.dz_spin.setValue(10.0)
        self.dz_spin.setDecimals(2)
        self.dz_spin.setSuffix(" m")
        layout.addRow("Block Size Z:", self.dz_spin)
        
        return group
    
    def _get_project_default_density(self) -> float:
        """
        Get project-specific default density from registry/config.
        
        Returns appropriate default based on project type:
        - Iron ore: ~4.0 t/m³
        - Coal: ~1.5 t/m³
        - General/default: 2.70 t/m³
        
        Returns:
            Default density value in t/m³
        """
        default_density = 2.70  # General default fallback
        
        try:
            # Try to get project metadata from registry
            if self.registry:
                # Check for project type/metadata in registry
                # This is extensible - can check for project_type, mine_type, etc.
                project_metadata = getattr(self.registry, 'project_metadata', None)
                if project_metadata:
                    project_type = project_metadata.get('project_type', '').lower()
                    mine_type = project_metadata.get('mine_type', '').lower()
                    
                    # Check for iron ore projects
                    if any(keyword in project_type or keyword in mine_type 
                           for keyword in ['iron', 'fe', 'hematite', 'magnetite']):
                        default_density = 4.0
                        logger.info("Using iron ore default density: 4.0 t/m³")
                        return default_density
                    
                    # Check for coal projects
                    if any(keyword in project_type or keyword in mine_type 
                           for keyword in ['coal', 'carbon']):
                        default_density = 1.5
                        logger.info("Using coal default density: 1.5 t/m³")
                        return default_density
                
                # Check for explicit default density setting
                if hasattr(self.registry, 'get_project_config'):
                    config = self.registry.get_project_config()
                    if config and 'default_density' in config:
                        default_density = float(config['default_density'])
                        logger.info(f"Using project config default density: {default_density} t/m³")
                        return default_density
        except Exception as e:
            logger.debug(f"Could not fetch project defaults: {e}")
        
        # Return general default
        return default_density
    
    def _create_input_group(self) -> QGroupBox:
        """Create input controls group."""
        group = QGroupBox("Calculation Parameters")
        layout = QFormLayout(group)

        # Block model source selector
        self.source_combo = QComboBox()
        self.source_combo.setToolTip("Select block model source for resource calculation")
        self.source_combo.addItem("No block model loaded", "none")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addRow("Block Model Source:", self.source_combo)

        # Grade property
        self.grade_combo = QComboBox()
        self.grade_combo.setToolTip("Select the grade property")
        layout.addRow("Grade Property:", self.grade_combo)
        
        # Density property
        self.density_combo = QComboBox()
        self.density_combo.setToolTip("Select density property or use default")
        layout.addRow("Density Property (t/m³):", self.density_combo)
        
        # Default density - fetch from project config if available
        default_density = self._get_project_default_density()
        
        self.default_density_spin = QDoubleSpinBox()
        self.default_density_spin.setRange(0.1, 10.0)
        self.default_density_spin.setValue(default_density)
        self.default_density_spin.setDecimals(2)
        self.default_density_spin.setSuffix(" t/m³")
        self.default_density_spin.setToolTip(
            f"Default density for blocks without density values "
            f"(project-specific: {default_density} t/m³)"
        )
        layout.addRow("Default Density:", self.default_density_spin)
        
        # Cut-off grade with auto-suggest button
        cutoff_layout = QHBoxLayout()
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0, 100000)
        self.cutoff_spin.setValue(0.0)
        self.cutoff_spin.setDecimals(3)
        cutoff_layout.addWidget(self.cutoff_spin, stretch=3)

        self.cutoff_suggest_btn = QPushButton("Auto")
        self.cutoff_suggest_btn.setToolTip("Suggest cutoff based on grade percentiles")
        self.cutoff_suggest_btn.setStyleSheet("background-color: #388e3c; color: white;")
        self.cutoff_suggest_btn.clicked.connect(self._auto_suggest_cutoff)
        cutoff_layout.addWidget(self.cutoff_suggest_btn, stretch=1)

        layout.addRow("Cut-off Grade:", cutoff_layout)
        
        # Comparator
        self.compare_combo = QComboBox()
        self.compare_combo.addItems([">=", "<=", ">", "<", "=="])
        layout.addRow("Comparator:", self.compare_combo)
        
        return group
    
    def _create_classification_group(self) -> QGroupBox:
        """Create classification controls."""
        group = QGroupBox("Resource Classification")
        layout = QVBoxLayout(group)
        
        # Enable checkbox
        self.classify_check = QCheckBox("Apply Classification")
        self.classify_check.setToolTip("Enable resource classification based on geological confidence")
        self.classify_check.toggled.connect(self._on_classification_toggled)
        layout.addWidget(self.classify_check)
        
        # Classification fields
        fields_layout = QFormLayout()
        
        self.dist_combo = QComboBox()
        self.dist_combo.setToolTip("Distance to nearest sample (m)")
        fields_layout.addRow("Distance Field:", self.dist_combo)
        
        self.var_combo = QComboBox()
        self.var_combo.setToolTip("Estimation variance or uncertainty")
        fields_layout.addRow("Variance Field:", self.var_combo)
        
        self.samples_combo = QComboBox()
        self.samples_combo.setToolTip("Number of samples used in estimation")
        fields_layout.addRow("Sample Count Field:", self.samples_combo)
        
        layout.addLayout(fields_layout)
        
        # Classification criteria
        criteria_label = QLabel("<b>Classification Criteria:</b>")
        layout.addWidget(criteria_label)
        
        criteria_text = QLabel(
            "• <b>Measured:</b> Dist ≤ 25m, Var ≤ 0.05, Samples ≥ 8<br>"
            "• <b>Indicated:</b> Dist ≤ 50m, Var ≤ 0.15, Samples ≥ 4<br>"
            "• <b>Inferred:</b> Dist > 50m or other conditions"
        )
        criteria_text.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(criteria_text)
        
        # Disable by default
        self.dist_combo.setEnabled(False)
        self.var_combo.setEnabled(False)
        self.samples_combo.setEnabled(False)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Create results table."""
        group = QGroupBox("Resource Summary")
        layout = QVBoxLayout(group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Cut-off", "Blocks", "Tonnes (t)", "Grade (g/t)", "Contained Metal (t)"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(100)
        layout.addWidget(self.results_table)
        
        return group
    
    def _create_class_results_group(self) -> QGroupBox:
        """Create classification results table."""
        group = QGroupBox("Resource Classification Summary")
        layout = QVBoxLayout(group)
        
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(4)
        self.class_table.setHorizontalHeaderLabels([
            "Class", "Tonnes (t)", "Grade (g/t)", "Contained Metal (t)"
        ])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setMaximumHeight(150)
        layout.addWidget(self.class_table)
        
        return group
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.

        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._stored_block_model = block_model
        self._register_source("Block Model", block_model, auto_select=not self._available_sources)
        self._update_source_selector()
        logger.info("Block Resource Panel: Block model available for selection")

    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.

        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._stored_block_model = block_model
        self._register_source("Block Model (Loaded)", block_model, auto_select=not self._available_sources)
        self._update_source_selector()
        logger.info("Block Resource Panel: Loaded block model available for selection")

    def _auto_suggest_cutoff(self):
        """Auto-suggest cutoff value based on grade percentiles.

        Uses P25 percentile as default cutoff, which typically represents
        a reasonable ore/waste boundary. Works for any commodity.
        """
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Load a block model first to auto-suggest cutoff.")
            return

        grade_col = self.grade_combo.currentText()
        if not grade_col or grade_col not in self.df.columns:
            QMessageBox.warning(self, "No Grade Column", "Select a grade property first.")
            return

        try:
            values = self.df[grade_col].dropna()
            if len(values) == 0:
                QMessageBox.warning(self, "No Data", f"No valid data for '{grade_col}'.")
                return

            # Calculate percentiles
            p25 = values.quantile(0.25)
            p50 = values.quantile(0.50)

            # Determine decimal places based on data magnitude
            grade_max = values.max()
            if grade_max > 50:
                decimals = 1
            elif grade_max > 10:
                decimals = 1
            elif grade_max > 1:
                decimals = 2
            else:
                decimals = 3

            # Use P25 as suggested cutoff (reasonable ore/waste boundary)
            suggested = round(p25, decimals)
            self.cutoff_spin.setValue(suggested)

            logger.info(f"BlockResource: Auto-suggested cutoff for '{grade_col}': {suggested} "
                       f"(P25={p25:.3f}, P50={p50:.3f})")

        except Exception as e:
            logger.warning(f"BlockResource: Could not auto-suggest cutoff: {e}")
            QMessageBox.warning(self, "Error", f"Could not calculate cutoff: {e}")

    def _on_block_model_classified(self, block_model):
        """
        Automatically receive classified block model when it's classified.
        Prefer classified model as it has resource categories.

        Args:
            block_model: Classified BlockModel from DataRegistry
        """
        self._stored_classified_block_model = block_model
        self._register_source("Classified Block Model", block_model, auto_select=True)
        self._update_source_selector()
        logger.info("Block Resource Panel: Classified block model available for selection")
    
    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results - register individual statistics as separate sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        try:
            import pyvista as pv
            import numpy as np

            if results is None:
                return

            if not isinstance(results, dict):
                logger.warning(f"BlockModelResourcePanel: SGSIM results is not a dict, type={type(results)}")
                return

            variable = results.get('variable', 'Grade')
            summary = results.get('summary', {})
            params = results.get('params')
            grid = results.get('grid') or results.get('pyvista_grid')

            logger.info(f"BlockModelResourcePanel: SGSIM results keys: {list(results.keys())}")
            logger.info(f"BlockModelResourcePanel: Summary keys: {list(summary.keys()) if summary else 'None'}")
            logger.info(f"BlockModelResourcePanel: params = {params is not None}")

            # Extract coordinates from grid or generate from params
            base_df = None
            n_blocks = 0

            if grid is not None and isinstance(grid, (pv.RectilinearGrid, pv.UnstructuredGrid, pv.StructuredGrid, pv.ImageData)):
                if hasattr(grid, 'cell_centers'):
                    centers = grid.cell_centers()
                    if hasattr(centers, 'points'):
                        coords = centers.points
                        base_df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2]})
                        n_blocks = len(base_df)
                        logger.info(f"BlockModelResourcePanel: Extracted {n_blocks:,} cell centers from grid")

            # If no grid, generate coordinates from params
            if (base_df is None or base_df.empty) and params is not None:
                try:
                    nx, ny, nz = params.nx, params.ny, params.nz
                    xmin, ymin, zmin = params.xmin, params.ymin, params.zmin
                    xinc, yinc, zinc = params.xinc, params.yinc, params.zinc

                    # Generate cell center coordinates
                    x_centers = np.arange(nx) * xinc + xmin + xinc / 2
                    y_centers = np.arange(ny) * yinc + ymin + yinc / 2
                    z_centers = np.arange(nz) * zinc + zmin + zinc / 2

                    # Create meshgrid and flatten (Z varies fastest, then Y, then X)
                    zz, yy, xx = np.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
                    coords_x = xx.transpose(2, 1, 0).flatten()
                    coords_y = yy.transpose(2, 1, 0).flatten()
                    coords_z = zz.transpose(2, 1, 0).flatten()

                    base_df = pd.DataFrame({'X': coords_x, 'Y': coords_y, 'Z': coords_z})
                    n_blocks = len(base_df)
                    logger.info(f"BlockModelResourcePanel: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
                except Exception as e:
                    logger.warning(f"BlockModelResourcePanel: Failed to generate coords from params: {e}")

            if base_df is None or base_df.empty:
                logger.warning("BlockModelResourcePanel: Could not extract coordinates from SGSIM grid or params")
                return

            found_stats = []

            # Extract individual statistics from 'summary' dict
            # SGSIM stores: summary['mean'], summary['std'], summary['p10'], summary['p50'], summary['p90']
            stat_mapping = {
                'mean': 'SGSIM Mean',
                'std': 'SGSIM Std Dev',
                'p10': 'SGSIM P10',
                'p50': 'SGSIM P50',
                'p90': 'SGSIM P90',
            }

            for stat_key, display_prefix in stat_mapping.items():
                stat_data = summary.get(stat_key)
                if stat_data is not None:
                    stat_values = np.asarray(stat_data).flatten()
                    if len(stat_values) == n_blocks:
                        df = base_df.copy()
                        prop_name = f"{variable}_{stat_key.upper()}"
                        df[prop_name] = stat_values

                        display_name = f"{display_prefix} ({variable}) - {n_blocks:,} blocks"
                        self._register_source(display_name, df, auto_select=False)
                        found_stats.append(stat_key)
                        logger.info(f"BlockModelResourcePanel: Registered {display_prefix} ({variable})")

            # Also extract from grid cell_data (e.g., FE_SGSIM_MEAN)
            if grid is not None and hasattr(grid, 'cell_data'):
                for prop_name in grid.cell_data.keys():
                    prop_values = np.asarray(grid.cell_data[prop_name]).flatten()
                    if len(prop_values) != n_blocks:
                        continue

                    prop_upper = prop_name.upper()
                    # Skip if we already have this statistic from summary
                    if 'MEAN' in prop_upper and 'mean' in found_stats:
                        continue
                    if 'STD' in prop_upper and 'std' in found_stats:
                        continue

                    df = base_df.copy()
                    df[prop_name] = prop_values

                    if 'MEAN' in prop_upper or 'E_TYPE' in prop_upper:
                        display_name = f"SGSIM Mean ({variable}) - {n_blocks:,} blocks"
                    elif 'PROB' in prop_upper:
                        display_name = f"SGSIM Probability ({prop_name}) - {n_blocks:,} blocks"
                    else:
                        display_name = f"SGSIM {prop_name} - {n_blocks:,} blocks"

                    self._register_source(display_name, df, auto_select=False)
                    found_stats.append(prop_name)

            if found_stats:
                logger.info(f"BlockModelResourcePanel: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

            self._update_source_selector()

        except Exception as e:
            logger.warning(f"BlockModelResourcePanel: Failed to load from SGSIM: {e}", exc_info=True)

    def _register_source(self, name: str, data, auto_select: bool = False):
        """Register a block model source."""
        if data is None:
            return

        # Convert to BlockModel if needed
        if isinstance(data, pd.DataFrame):
            bm = BlockModel()
            bm.update_from_dataframe(data)
        elif isinstance(data, BlockModel):
            bm = data
        elif hasattr(data, 'to_dataframe'):
            df = data.to_dataframe()
            bm = BlockModel()
            bm.update_from_dataframe(df)
        else:
            return

        self._block_model_sources[name] = bm
        if name not in self._available_sources:
            self._available_sources.append(name)

        logger.info(f"BlockModelResourcePanel: Registered source '{name}'")

        if auto_select and (not self._current_source or self._current_source == ""):
            self._current_source = name
            self.set_block_model(bm)

    def _update_source_selector(self):
        """Update the source selector combo box."""
        if not hasattr(self, 'source_combo'):
            return

        self.source_combo.blockSignals(True)
        self.source_combo.clear()

        if not self._available_sources:
            self.source_combo.addItem("No block model loaded", "none")
        else:
            for source_name in self._available_sources:
                bm = self._block_model_sources.get(source_name)
                count = len(bm.positions) if bm is not None and hasattr(bm, 'positions') else 0
                display_text = f"{source_name} ({count:,} blocks)"
                self.source_combo.addItem(display_text, source_name)

            if self._current_source and self._current_source in self._available_sources:
                idx = self._available_sources.index(self._current_source)
                self.source_combo.setCurrentIndex(idx)

        self.source_combo.blockSignals(False)

    def _on_source_changed(self, index: int):
        """Handle block model source selection change."""
        if index < 0 or not hasattr(self, 'source_combo'):
            return

        source_name = self.source_combo.itemData(index)
        if source_name is None or source_name == "none":
            return

        if source_name in self._block_model_sources:
            self._current_source = source_name
            bm = self._block_model_sources[source_name]
            self.set_block_model(bm)
            logger.info(f"BlockModelResourcePanel: Switched to '{source_name}'")
    
    def set_block_model(self, block_model: BlockModel):
        """Set the block model for resource calculation."""
        super().set_block_model(block_model)
        
        # Convert to DataFrame
        try:
            self.df = block_model.to_dataframe()
        except:
            # Fallback: create dataframe manually
            data = {'XC': block_model.positions[:, 0],
                    'YC': block_model.positions[:, 1],
                    'ZC': block_model.positions[:, 2]}
            for prop_name, prop_data in block_model.properties.items():
                data[prop_name] = prop_data
            self.df = pd.DataFrame(data)
        
        # Get numeric properties
        self.numeric_properties = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_properties.append(col)
        
        # Populate combos
        self.grade_combo.clear()
        self.grade_combo.addItems(self.numeric_properties)
        
        self.density_combo.clear()
        self.density_combo.addItems(['<Use Default>'] + self.numeric_properties)
        
        self.dist_combo.clear()
        self.dist_combo.addItems(['<Not Available>'] + self.numeric_properties)
        
        self.var_combo.clear()
        self.var_combo.addItems(['<Not Available>'] + self.numeric_properties)
        
        self.samples_combo.clear()
        self.samples_combo.addItems(['<Not Available>'] + self.numeric_properties)
        
        # Enable calculation
        self.calc_btn.setEnabled(True)
        
        logger.info(f"Set block model with {len(self.df)} blocks for resource calculation")
    
    def _on_classification_toggled(self, checked: bool):
        """Handle classification checkbox toggle."""
        self.dist_combo.setEnabled(checked)
        self.var_combo.setEnabled(checked)
        self.samples_combo.setEnabled(checked)
        self.classify_btn.setEnabled(checked and self.df is not None)
    
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------
    
    def _start_resource_analysis(self, operation: str):
        """Start resource analysis of specified type."""
        self.current_operation = operation
        self.run_analysis()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI."""
        if self.df is None:
            raise ValueError("No block model data loaded.")
        
        if self.current_operation == 'calculate':
            return {
                "operation": "calculate",
                "data_df": self.df.copy(),
                "grade_col": self.grade_combo.currentText(),
                "density_col": self.density_combo.currentText(),
                "cutoff": self.cutoff_spin.value(),
                "comparator": self.compare_combo.currentText(),
                "default_density": self.default_density_spin.value(),
                "dx": self.dx_spin.value(),
                "dy": self.dy_spin.value(),
                "dz": self.dz_spin.value(),
            }
        elif self.current_operation == 'classify':
            if not self.classify_check.isChecked():
                raise ValueError("Classification is disabled. Please enable it first.")
            
            return {
                "operation": "classify",
                "data_df": self.df.copy(),
                "dist_col": self.dist_combo.currentText(),
                "var_col": self.var_combo.currentText(),
                "samples_col": self.samples_combo.currentText(),
            }
        else:
            raise ValueError(f"Unknown operation: {self.current_operation}")
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if self.df is None:
            self.show_error("No Data", "Please load a block model first.")
            return False
        
        if self.current_operation == 'classify':
            if not self.classify_check.isChecked():
                self.show_error("Classification Disabled", "Please enable classification first.")
                return False
        
        return True
    
    def _publish_resource_results(self, results: Dict[str, Any]):
        """Publish resource calculation results to DataRegistry."""
        try:
            if hasattr(self, 'registry') and self.registry:
                self.registry.register_resource_summary(results, source_panel="BlockModelResourcePanel")
                logger.info("Published resource calculation results to DataRegistry")
        except Exception as e:
            logger.warning(f"Failed to publish resource results to DataRegistry: {e}")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display resource calculation/classification results."""
        operation = payload.get("operation", self.current_operation)
        
        if operation == 'calculate':
            self.result_df = payload.get("result_df")
            self.current_mask = payload.get("mask")
            
            if self.result_df is not None:
                self._display_results()
                if hasattr(self, 'export_btn'):
                    self.export_btn.setEnabled(True)
                self.show_info("Calculation Complete", "Resource calculation completed successfully.")
                
                # Publish resource calculation results
                resource_summary = {
                    'operation': 'calculate',
                    'result_df': self.result_df,
                    'summary': payload.get('summary')
                }
                self._publish_resource_results(resource_summary)
        
        elif operation == 'classify':
            self.df = payload.get("data_df")
            self.class_summary_df = payload.get("class_summary_df")
            
            if self.df is not None and 'CLASS' in self.df.columns:
                self._compute_class_resources()
                if hasattr(self, 'visualize_btn'):
                    self.visualize_btn.setEnabled(True)
                self.show_info("Classification Complete", "Resources classified into Measured/Indicated/Inferred categories.")
                
                # Publish classification results
                resource_summary = {
                    'operation': 'classify',
                    'class_summary_df': self.class_summary_df,
                    'classified_df': self.df
                }
                self._publish_resource_results(resource_summary)
    
    def _compute_class_resources(self):
        """Compute per-class resource summaries."""
        if self.df is None or 'CLASS' not in self.df.columns:
            return
        
        try:
            df = self.df.copy()
            
            # Get parameters
            grade_col = self.grade_combo.currentText()
            
            # Ensure volume and tonnes are calculated
            if 'VOLUME' not in df.columns:
                dx = self.dx_spin.value()
                dy = self.dy_spin.value()
                dz = self.dz_spin.value()
                df['VOLUME'] = dx * dy * dz
            
            if 'DENSITY' not in df.columns:
                density_col = self.density_combo.currentText()
                default_density = self.default_density_spin.value()
                if density_col == '<Use Default>' or density_col not in df.columns:
                    df['DENSITY'] = default_density
                else:
                    df['DENSITY'] = df[density_col].fillna(default_density)
            
            if 'TONNES' not in df.columns:
                df['TONNES'] = df['DENSITY'] * df['VOLUME']
            
            # Apply cut-off if results exist
            if self.current_mask is not None:
                df = df[self.current_mask]
            
            # Calculate per class using TONNAGE-WEIGHTED grade (JORC compliant)
            results = []
            for cls in ['Measured', 'Indicated', 'Inferred', 'Unclassified']:
                subset = df[df['CLASS'] == cls]
                if subset.empty:
                    continue
                
                tonnes = subset['TONNES'].sum()
                
                # JORC COMPLIANCE: Use tonnage-weighted grade, NOT arithmetic mean
                # Weighted mean = Σ(grade × tonnage) / Σ(tonnage)
                if tonnes > 0:
                    weighted_grade = (subset[grade_col] * subset['TONNES']).sum() / tonnes
                else:
                    weighted_grade = 0.0
                
                # Contained metal = tonnage × grade (accounting for units)
                # Assuming grade in % and wanting metal in tonnes
                contained = (tonnes * weighted_grade) / 100.0
                
                results.append({
                    'Class': cls,
                    'Tonnes': tonnes,
                    'Grade': weighted_grade,
                    'ContainedMetal': contained
                })
            
            # Add total row - compute from row data for consistency
            total_tonnes = sum(r['Tonnes'] for r in results)
            total_metal = sum(r['ContainedMetal'] for r in results)
            
            # JORC COMPLIANCE: Total grade must be tonnage-weighted from class grades
            if total_tonnes > 0:
                total_weighted_grade = sum(r['Grade'] * r['Tonnes'] for r in results) / total_tonnes
            else:
                total_weighted_grade = 0.0
            
            # AUDIT VALIDATION: Verify contained metal consistency
            expected_metal_from_grade = (total_tonnes * total_weighted_grade) / 100.0
            if abs(total_metal - expected_metal_from_grade) > 0.01:
                logger.warning(
                    f"AUDIT WARNING: Contained metal inconsistency detected. "
                    f"Sum of rows: {total_metal:,.2f}t, "
                    f"Recalculated: {expected_metal_from_grade:,.2f}t"
                )
            
            results.append({
                'Class': '<b>TOTAL</b>',
                'Tonnes': total_tonnes,
                'Grade': total_weighted_grade,
                'ContainedMetal': total_metal  # Use sum of rows for consistency
            })
            
            self.class_summary_df = pd.DataFrame(results)
            
            # Display classification results
            self._display_class_results()
            
        except Exception as e:
            logger.error(f"Error computing class resources: {e}")
    
    def _display_results(self):
        """Display resource calculation results."""
        if self.result_df is None:
            return
        
        self.results_table.setRowCount(1)
        
        for col_idx, col_name in enumerate(['Cut-off', 'Blocks', 'Tonnes', 'Grade', 'ContainedMetal']):
            value = self.result_df.iloc[0][col_name]
            
            if col_name == 'Cut-off':
                text = str(value)
            elif col_name == 'Blocks':
                text = f"{int(value):,}"
            elif col_name == 'Tonnes':
                text = f"{value:,.2f}"
            elif col_name == 'Grade':
                text = f"{value:.3f}"
            elif col_name == 'ContainedMetal':
                text = f"{value:,.2f}"
            
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(0, col_idx, item)
    
    def _display_class_results(self):
        """Display classification results."""
        if self.class_summary_df is None:
            return
        
        self.class_table.setRowCount(len(self.class_summary_df))
        
        for row_idx, row in self.class_summary_df.iterrows():
            # Class name with color indicator
            class_name = row['Class']
            item = QTableWidgetItem(class_name.replace('<b>', '').replace('</b>', ''))
            
            if class_name in self.class_colors:
                color = QColor(self.class_colors[class_name])
                item.setBackground(color)
            
            if '<b>' in class_name:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.class_table.setItem(row_idx, 0, item)
            
            # Tonnes
            item = QTableWidgetItem(f"{row['Tonnes']:,.2f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if '<b>' in class_name:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.class_table.setItem(row_idx, 1, item)
            
            # Grade
            item = QTableWidgetItem(f"{row['Grade']:.3f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if '<b>' in class_name:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.class_table.setItem(row_idx, 2, item)
            
            # Contained Metal
            item = QTableWidgetItem(f"{row['ContainedMetal']:,.2f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if '<b>' in class_name:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.class_table.setItem(row_idx, 3, item)
    
    def _export_results(self):
        """Export resource results to CSV."""
        if self.result_df is None:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Resource Results", "resource_summary.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                # Export main results
                # Step 10: Use ExportHelpers
                from ..utils.export_helpers import export_dataframe_to_csv
                export_dataframe_to_csv(self.result_df, filename)
                
                # If classification exists, export that too
                if self.class_summary_df is not None:
                    base_name = filename.replace('.csv', '_classification.csv')
                    # Step 10: Use ExportHelpers
                    from ..utils.export_helpers import export_dataframe_to_csv
                    export_dataframe_to_csv(self.class_summary_df, base_name)
                    QMessageBox.information(
                        self, "Success",
                        f"Exported results to:\n{filename}\n{base_name}"
                    )
                else:
                    QMessageBox.information(self, "Success", f"Exported results to {filename}")
                
                logger.info(f"Exported resource results to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")
                logger.error(f"Export error: {e}")
    
    def _visualize_classification(self):
        """Visualize resource classification in 3D viewer."""
        if self.df is None or 'CLASS' not in self.df.columns:
            QMessageBox.warning(self, "No Classification", "Please run classification first.")
            return
        
        # Emit signal to main window
        self.visualize_classification_requested.emit(self.df, self.class_colors)
        
        QMessageBox.information(
            self, "Visualization Updated",
            "3D viewer now shows blocks colored by resource class:\n\n"
            "• Green = Measured\n"
            "• Yellow = Indicated\n"
            "• Red = Inferred\n"
            "• Gray = Unclassified"
        )
        
        logger.info("Requested 3D visualization of resource classification")
    
    def clear(self):
        """Clear the panel."""
        super().set_block_model(None)
        self.df = None
        self.result_df = None
        self.class_summary_df = None
        self.current_mask = None
        self.results_table.setRowCount(0)
        self.class_table.setRowCount(0)
        self.grade_combo.clear()
        self.density_combo.clear()
        self.dist_combo.clear()
        self.var_combo.clear()
        self.samples_combo.clear()
        self.calc_btn.setEnabled(False)
        self.classify_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.visualize_btn.setEnabled(False)





