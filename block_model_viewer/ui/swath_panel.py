"""
Swath Plot Analysis Panel - Modern, professional interface for swath analysis.

Designed with Leapfrog-level professionalism:
- Clean, frameless design with typography hierarchy
- Split layout with live preview
- Context-aware inline plot display
- Integrated 3D interaction
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QLabel, QSpinBox, QCheckBox, QMessageBox,
    QSplitter, QFrame, QFileDialog, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QAction

from .panel_manager import PanelCategory, DockArea
from .base_analysis_panel import BaseAnalysisPanel
from .comparison_utils import ComparisonColors, SourceSelectionWidget, create_comparison_legend
from ..models.block_model import BlockModel

# Matplotlib imports
try:
    # Matplotlib backend is set in main.py
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class SwathPanel(BaseAnalysisPanel):
    """
    Panel for creating interactive swath plots with 3D highlighting.
    
    DATA SOURCES:
    - Supports both block models AND drillhole data
    - Drillhole data: composites, assays, or raw DataFrames
    - Block models: from loader, kriging, SGSIM, etc.
    """
    # PanelManager metadata
    PANEL_ID = "SwathPanel"
    PANEL_NAME = "Swath Plot Analysis"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    # Signals
    swath_highlight_requested = pyqtSignal(str, float, float)  # direction, lower, upper
    swath_highlight_cleared = pyqtSignal()
    
    task_name = "swath"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="swath")
        self.current_model: Optional[BlockModel] = None
        self.current_figure: Optional[Figure] = None
        self.swath_canvas: Optional[FigureCanvasQTAgg] = None
        self.current_canvas: Optional[FigureCanvasQTAgg] = None  # For preview pane
        self.swath_bins: Optional[np.ndarray] = None
        self.swath_direction: Optional[str] = None
        self.swath_axis_col: Optional[str] = None
        self.swath_link_enabled: bool = True
        self.swath_click_cid = None
        
        # References for 3D highlighting
        self.plotter = None
        self.grid = None
        self.df = None
        
        # Drillhole data support
        self._drillhole_df: Optional[pd.DataFrame] = None
        self._data_mode: str = "none"  # "block_model", "drillhole", or "none"
        self._data_source_type: str = "unknown"  # "composites", "assays", "dataframe"
        
        # Stored data sources (for user selection)
        self._stored_drillhole_df: Optional[pd.DataFrame] = None
        self._stored_block_model: Optional[BlockModel] = None
        self._stored_sgsim_df: Optional[pd.DataFrame] = None  # SGSIM results as DataFrame
        self._stored_data_source_type: str = "unknown"
        self._available_sources: list = []

        # Storage for individual SGSIM statistics (Mean, P10, P50, P90, Std Dev)
        self._block_model_sources: Dict[str, Any] = {}

        # Comparison mode support
        self._comparison_mode: bool = False
        self._comparison_results: Dict[str, Dict[str, Any]] = {}  # Store results per source
        
        # Subscribe to data from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_classified)
            
            # Connect to drillhole signals
            if hasattr(self.registry, 'drillholeDataLoaded'):
                self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                logger.info("SwathPanel: Connected to drillholeDataLoaded signal")

            # Connect to SGSIM results signal
            if hasattr(self.registry, 'sgsimResultsLoaded'):
                self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
                logger.info("SwathPanel: Connected to sgsimResultsLoaded signal")

            # Data will be refreshed after _setup_ui() completes
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        # Setup UI and then refresh data
        self._setup_ui()
        
        # Refresh data AFTER UI is set up
        if hasattr(self, 'registry') and self.registry:
            self._refresh_available_data()
        
        logger.info("Initialized Swath Plot Analysis panel")
    
    def _refresh_available_data(self):
        """Refresh the list of available data from all sources (drillholes AND block models)."""
        if not self.registry:
            return
        
        try:
            # Store available data sources (no priority - user selects)
            self._available_sources = []
            
            # Check for drillhole data
            drillhole_data = self.registry.get_drillhole_data(copy_data=False)
            if drillhole_data is not None:
                self._store_drillhole_data(drillhole_data)
                self._available_sources.append("drillhole")
            
            # Check block models
            block_model = self.registry.get_block_model(copy_data=False)
            if block_model is not None:
                self._stored_block_model = block_model
                self._available_sources.append("block_model")

            # Check classified block model (separate from regular block model)
            classified_model = self.registry.get_classified_block_model(copy_data=False)
            if classified_model is not None:
                self._stored_classified_block_model = classified_model
                if "classified_block_model" not in self._available_sources:
                    self._available_sources.append("classified_block_model")

            # Check for SGSIM results
            if hasattr(self.registry, 'get_sgsim_results'):
                sgsim = self.registry.get_sgsim_results()
                if sgsim is not None:
                    self._on_sgsim_loaded(sgsim)

            # Update data source selector
            self._update_data_source_selector()
            
        except Exception as e:
            logger.warning(f"SwathPanel: Failed to refresh available data: {e}")
    
    def _store_drillhole_data(self, drillhole_data):
        """Store drillhole data without switching to it."""
        df = None
        self._stored_data_source_type = "unknown"
        
        if isinstance(drillhole_data, dict):
            comp = drillhole_data.get("composites")
            assays = drillhole_data.get("assays")
            if comp is not None and getattr(comp, "empty", False) is False:
                df = comp
                self._stored_data_source_type = "composites"
            elif assays is not None and getattr(assays, "empty", False) is False:
                df = assays
                self._stored_data_source_type = "assays"
        elif isinstance(drillhole_data, pd.DataFrame):
            df = drillhole_data
            self._stored_data_source_type = "dataframe"
        
        self._stored_drillhole_df = df
    
    def _update_data_source_selector(self):
        """Update the data source selector combo box with all sources including individual SGSIM stats."""
        if not hasattr(self, 'data_source_box'):
            return

        # Store current selection
        current_data = self.data_source_box.currentData() if self.data_source_box.count() > 0 else None

        self.data_source_box.blockSignals(True)
        self.data_source_box.clear()

        if "drillhole" in self._available_sources:
            label = f"Drillhole Data ({self._stored_data_source_type})"
            self.data_source_box.addItem(label, "drillhole")

        if "block_model" in self._available_sources:
            bm = self._stored_block_model
            count = len(bm) if bm is not None and hasattr(bm, '__len__') else 0
            self.data_source_box.addItem(f"Block Model ({count:,} blocks)", "block_model")

        if "classified_block_model" in self._available_sources:
            bm = getattr(self, '_stored_classified_block_model', None)
            count = len(bm) if bm is not None and hasattr(bm, '__len__') else 0
            self.data_source_box.addItem(f"Classified Block Model ({count:,} blocks)", "classified_block_model")

        # Add individual SGSIM sources
        for source_key in self._available_sources:
            if source_key.startswith('sgsim_') and source_key in self._block_model_sources:
                source_info = self._block_model_sources[source_key]
                df = source_info.get('df')
                count = len(df) if df is not None else 0
                display_name = source_info.get('display_name', source_key)
                self.data_source_box.addItem(f"{display_name} ({count:,} blocks)", source_key)

        # Legacy: single SGSIM source (if not using individual stats)
        if "sgsim" in self._available_sources and not any(s.startswith('sgsim_') for s in self._available_sources):
            df = self._stored_sgsim_df
            count = len(df) if df is not None else 0
            self.data_source_box.addItem(f"SGSIM Results ({count:,} blocks)", "sgsim")

        if self.data_source_box.count() == 0:
            self.data_source_box.addItem("No data loaded", "none")
            self.data_source_box.setEnabled(False)
        else:
            self.data_source_box.setEnabled(True)

        self.data_source_box.blockSignals(False)

        # Try to restore previous selection, otherwise select first available
        if self.data_source_box.count() > 0:
            # Try to find and restore previous selection
            restored = False
            if current_data:
                for i in range(self.data_source_box.count()):
                    if self.data_source_box.itemData(i) == current_data:
                        self.data_source_box.setCurrentIndex(i)
                        restored = True
                        break

            # If couldn't restore or no previous selection, select first and activate it
            if not restored:
                self.data_source_box.setCurrentIndex(0)
                # Manually trigger the change to ensure data is set
                self._on_data_source_changed(0)

        # Also update comparison sources if the widget exists
        if hasattr(self, '_source_selection_widget'):
            self._update_comparison_sources()
    
    def _on_data_source_changed(self, index):
        """Handle user changing data source selection."""
        if not hasattr(self, 'data_source_box'):
            return

        if index < 0 or index >= self.data_source_box.count():
            return

        source_type = self.data_source_box.itemData(index)
        logger.info(f"SwathPanel: Data source changed to index {index}, type: {source_type}")

        if source_type == "drillhole":
            if self._stored_drillhole_df is not None:
                self._data_mode = "drillhole"
                self._drillhole_df = self._stored_drillhole_df
                self.df = self._stored_drillhole_df  # For compatibility
                self._data_source_type = self._stored_data_source_type
                self.current_model = None
                self._populate_drillhole_properties()
                logger.info(f"SwathPanel: Switched to drillhole data ({self._data_source_type}, {len(self._drillhole_df)} rows)")
            else:
                logger.warning("SwathPanel: Drillhole data selected but _stored_drillhole_df is None")
                self._data_mode = "none"
        elif source_type == "block_model":
            if self._stored_block_model is not None:
                self._data_mode = "block_model"
                self._drillhole_df = None
                self.set_block_model(self._stored_block_model)
                logger.info("SwathPanel: Switched to block model")
            else:
                logger.warning("SwathPanel: Block model selected but _stored_block_model is None")
                self._data_mode = "none"
        elif source_type == "classified_block_model":
            if hasattr(self, '_stored_classified_block_model') and self._stored_classified_block_model is not None:
                self._data_mode = "block_model"  # Same mode, just different source
                self._drillhole_df = None
                self.set_block_model(self._stored_classified_block_model)
                logger.info("SwathPanel: Switched to classified block model")
            else:
                logger.warning("SwathPanel: Classified block model selected but not available")
                self._data_mode = "none"
        elif source_type == "sgsim":
            if self._stored_sgsim_df is not None:
                self._data_mode = "block_model"  # Treat as block model
                self._drillhole_df = None
                self.set_block_model(self._stored_sgsim_df)
                logger.info("SwathPanel: Switched to SGSIM results")
            else:
                logger.warning("SwathPanel: SGSIM selected but _stored_sgsim_df is None")
                self._data_mode = "none"
        elif source_type and source_type.startswith('sgsim_') and source_type in self._block_model_sources:
            # Handle individual SGSIM statistics
            source_info = self._block_model_sources[source_type]
            df = source_info.get('df')
            if df is not None:
                self._data_mode = "block_model"
                self._drillhole_df = None
                self.set_block_model(df)
                logger.info(f"SwathPanel: Switched to {source_info.get('display_name', source_type)}")
            else:
                logger.warning(f"SwathPanel: SGSIM source {source_type} has no data")
                self._data_mode = "none"
        else:
            logger.warning(f"SwathPanel: Unknown source type or no data: {source_type}")
            self._data_mode = "none"
            if hasattr(self, 'swath_property_box'):
                self.swath_property_box.clear()
            if hasattr(self, 'swath_btn'):
                self.swath_btn.setEnabled(False)
    
    def _populate_drillhole_properties(self):
        """Populate property selector from drillhole DataFrame."""
        if self._drillhole_df is None:
            return
        
        df = self._drillhole_df
        
        # Extract numeric properties
        numeric_props = []
        exclude_cols = {'HOLEID', 'HOLE_ID', 'BHID', 'FROM', 'TO', 'FROM_M', 'TO_M', 
                        'X', 'Y', 'Z', 'MID_X', 'MID_Y', 'MID_Z', 'LENGTH', 'SAMPLE_ID'}
        
        for col in df.columns:
            if col.upper() not in exclude_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_props.append(col)
        
        # Update property selector
        if hasattr(self, 'swath_property_box'):
            self.swath_property_box.clear()
            self.swath_property_box.addItems(numeric_props)
        
        # Enable button if properties available
        if hasattr(self, 'swath_btn') and numeric_props:
            self.swath_btn.setEnabled(True)
        
        logger.info(f"SwathPanel: Loaded {len(numeric_props)} numeric columns from {self._data_source_type}")
    
    def _refresh_available_block_models(self):
        """Refresh the list of available block models from all sources."""
        if not self.registry:
            return
        
        try:
            # Check regular block model
            block_model = self.registry.get_block_model()
            if block_model is not None:
                self._on_block_model_loaded(block_model)
            
            # Check classified block model
            try:
                classified_model = self.registry.get_classified_block_model()
                if classified_model is not None:
                    self._on_block_model_classified(classified_model)
            except Exception as e:
                logger.error(f"Failed to load classified block model: {e}", exc_info=True)
            
            # Also check renderer layers for block models that might be displayed
            self._check_renderer_layers_for_block_models()
            
        except Exception as e:
            logger.warning(f"Failed to refresh available block models: {e}")
    
    def _check_renderer_layers_for_block_models(self):
        """Check renderer layers for block models that might be displayed."""
        try:
            # Try to get renderer from parent/main window
            parent = self.parent()
            while parent:
                if hasattr(parent, 'viewer_widget') and hasattr(parent.viewer_widget, 'renderer'):
                    renderer = parent.viewer_widget.renderer
                    if hasattr(renderer, 'active_layers') and renderer.active_layers:
                        # Look for block model layers (SGSIM, kriging, etc.)
                        for layer_name, layer_info in renderer.active_layers.items():
                            layer_type = layer_info.get('type', '')
                            layer_data = layer_info.get('data', None)
                            
                            # Check if it's a block model layer
                            if layer_type in ('blocks', 'volume') and layer_data is not None:
                                # If we don't have a block model yet, try to use this one
                                if self.current_model is None:
                                    if hasattr(parent, '_extract_block_model_from_grid'):
                                        block_model = parent._extract_block_model_from_grid(layer_data, layer_name)
                                        if block_model is not None:
                                            self._on_block_model_generated(block_model)
                                            logger.info(f"Found block model in renderer layer: {layer_name}")
                    break
                parent = parent.parent() if parent else None
        except Exception as e:
            logger.debug(f"Could not check renderer layers: {e}")
    
    def _on_block_model_classified(self, block_model):
        """Handle block model classification changes."""
        # Store as SEPARATE classified block model (don't overwrite regular block model)
        self._stored_classified_block_model = block_model
        if "classified_block_model" not in self._available_sources:
            self._available_sources.append("classified_block_model")
        self._update_data_source_selector()
        logger.info("SwathPanel: Classified block model available for selection")

    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results - register individual statistics as separate sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        try:
            import numpy as np
            import pyvista as pv

            if results is None:
                return

            if not isinstance(results, dict):
                logger.warning(f"SwathPanel: SGSIM results is not a dict, type={type(results)}")
                return

            variable = results.get('variable', 'Grade')
            summary = results.get('summary', {})
            params = results.get('params')
            grid = results.get('grid') or results.get('pyvista_grid')

            logger.info(f"SwathPanel: SGSIM results keys: {list(results.keys())}")
            logger.info(f"SwathPanel: Summary keys: {list(summary.keys()) if summary else 'None'}")
            logger.info(f"SwathPanel: params = {params is not None}")

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
                        logger.info(f"SwathPanel: Extracted {n_blocks:,} cell centers from grid")

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
                    logger.info(f"SwathPanel: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
                except Exception as e:
                    logger.warning(f"SwathPanel: Failed to generate coords from params: {e}")

            if base_df is None or base_df.empty:
                # Fallback: realizations array with grid coordinates
                reals = results.get('realizations') or results.get('realizations_raw')
                if reals is not None:
                    grid_x, grid_y, grid_z = results.get('grid_x'), results.get('grid_y'), results.get('grid_z')
                    if grid_x is not None and isinstance(reals, np.ndarray):
                        mean_estimate = np.mean(reals, axis=0) if reals.ndim == 2 else reals.ravel()
                        df = pd.DataFrame({
                            'X': np.asarray(grid_x).ravel(), 'Y': np.asarray(grid_y).ravel(),
                            'Z': np.asarray(grid_z).ravel(), variable: mean_estimate
                        })
                        self._stored_sgsim_df = df

                        source_key = f"sgsim_etype_{variable}"
                        self._block_model_sources[source_key] = {
                            'df': df,
                            'display_name': f"SGSIM E-type Mean ({variable})",
                            'property': variable
                        }
                        if source_key not in self._available_sources:
                            self._available_sources.append(source_key)
                        self._update_data_source_selector()
                        logger.info(f"SwathPanel: Registered SGSIM E-type Mean from fallback")
                else:
                    logger.warning("SwathPanel: No grid, params, or realizations found in SGSIM results")
                return

            found_stats = []

            # Extract individual statistics from 'summary' dict
            # SGSIM stores: summary['mean'], summary['std'], summary['p10'], summary['p50'], summary['p90']
            stat_mapping = {
                'mean': ('sgsim_mean', 'SGSIM Mean', 'sgsim_mean'),
                'std': ('sgsim_std', 'SGSIM Std Dev', 'sgsim_std'),
                'p10': ('sgsim_p10', 'SGSIM P10', 'sgsim_p10'),
                'p50': ('sgsim_p50', 'SGSIM P50', 'sgsim_p50'),
                'p90': ('sgsim_p90', 'SGSIM P90', 'sgsim_p90'),
            }

            for stat_key, (key_prefix, display_prefix, source_type) in stat_mapping.items():
                stat_data = summary.get(stat_key)
                if stat_data is not None:
                    stat_values = np.asarray(stat_data).flatten()
                    if len(stat_values) == n_blocks:
                        df = base_df.copy()
                        prop_name = f"{variable}_{stat_key.upper()}"
                        df[prop_name] = stat_values

                        source_key = f"{key_prefix}_{variable}"
                        display_name = f"{display_prefix} ({variable}) - {n_blocks:,} blocks"
                        self._block_model_sources[source_key] = {
                            'df': df,
                            'display_name': display_name,
                            'property': prop_name
                        }

                        if source_key not in self._available_sources:
                            self._available_sources.append(source_key)

                        found_stats.append(stat_key)
                        logger.info(f"SwathPanel: Registered {display_prefix} ({variable})")

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
                        source_key = f"sgsim_mean_{variable}"
                        display_name = f"SGSIM Mean ({variable}) - {n_blocks:,} blocks"
                    elif 'PROB' in prop_upper:
                        source_key = f"sgsim_prob_{prop_name}"
                        display_name = f"SGSIM Probability ({prop_name}) - {n_blocks:,} blocks"
                    else:
                        source_key = f"sgsim_{prop_name}"
                        display_name = f"SGSIM {prop_name} - {n_blocks:,} blocks"

                    if source_key not in self._block_model_sources:
                        self._block_model_sources[source_key] = {
                            'df': df,
                            'display_name': display_name,
                            'property': prop_name
                        }
                        if source_key not in self._available_sources:
                            self._available_sources.append(source_key)
                        found_stats.append(prop_name)

            # Store combined DataFrame for backward compatibility
            if found_stats:
                df_all = base_df.copy()
                for stat_key in ['mean', 'std', 'p10', 'p50', 'p90']:
                    stat_data = summary.get(stat_key)
                    if stat_data is not None:
                        stat_values = np.asarray(stat_data).flatten()
                        if len(stat_values) == n_blocks:
                            df_all[f"{variable}_{stat_key.upper()}"] = stat_values
                self._stored_sgsim_df = df_all
                logger.info(f"SwathPanel: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

            self._update_data_source_selector()

        except Exception as e:
            logger.warning(f"SwathPanel: Failed to load SGSIM results: {e}", exc_info=True)

    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        # Store the block model and update selector
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        logger.info("SwathPanel: Block model available for selection")
    
    def _update_properties_from_dataframe(self, df: pd.DataFrame):
        """Update property selector from DataFrame."""
        if not hasattr(self, 'swath_property_box'):
            return
        
        # Extract numeric properties
        numeric_props = []
        for col in df.columns:
            if col not in ['X', 'Y', 'Z', 'DX', 'DY', 'DZ']:
                if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                    numeric_props.append(col)
        
        # Update property selector
        self.swath_property_box.clear()
        self.swath_property_box.addItems(numeric_props)
        
        # Enable button if properties available
        if hasattr(self, 'swath_btn') and numeric_props:
            self.swath_btn.setEnabled(True)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        # Store the block model and update selector (same as generated)
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        logger.info("SwathPanel: Block model available for selection")
    
    def _on_drillhole_data_loaded(self, drillhole_data):
        """
        Receive drillhole data from DataRegistry.
        
        Stores the data and updates the selector - user must choose to use it.
        """
        logger.info("SwathPanel: received drillhole data from DataRegistry")
        
        # Store the drillhole data
        self._store_drillhole_data(drillhole_data)
        
        # Add to available sources and update selector
        if self._stored_drillhole_df is not None:
            if "drillhole" not in self._available_sources:
                self._available_sources.append("drillhole")
            self._update_data_source_selector()
            logger.info("SwathPanel: Drillhole data available for selection")
    
    def _set_drillhole_data(self, df: pd.DataFrame):
        """Set drillhole DataFrame and update UI."""
        self._drillhole_df = df
        self._data_mode = "drillhole"
        self.current_model = None  # Clear block model when drillhole data is set
        self.df = df  # Also set the df attribute for compatibility
        
        # Update property selector
        self._update_properties_from_dataframe(df)
        
        logger.info(f"SwathPanel: Loaded drillhole data with {len(df)} records from {self._data_source_type}")
    
    def _setup_ui(self):
        """Setup the modern UI components with split layout and live preview."""
        from .modern_styles import ModernColors, get_button_stylesheet, get_combo_box_stylesheet, get_spin_box_stylesheet
        
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        if not MATPLOTLIB_AVAILABLE:
            error_label = QLabel("Matplotlib not available. Install matplotlib to use swath plot features.")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            main_layout.addWidget(error_label)
            return
        
        # Hide Stop and Close buttons if they exist
        if hasattr(self, 'stop_button'):
            self.stop_button.hide()
        if hasattr(self, 'close_button'):
            self.close_button.hide()
        
        # Create split layout: Controls (left) | Preview (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {ModernColors.BORDER}; }}")
        
        # Left side: Controls
        controls_widget = QFrame()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(16)
        
        # Title
        title_label = QLabel("Swath Plot Analysis")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title_label.setFont(title_font)
        controls_layout.addWidget(title_label)
        
        # Separator line
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"color: {ModernColors.BORDER};")
        controls_layout.addWidget(sep1)
        
        controls_layout.addSpacing(4)
        
        # Data section
        data_label = QLabel("Data")
        data_font = QFont()
        data_font.setPointSize(10)
        data_label.setFont(data_font)
        data_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        controls_layout.addWidget(data_label)
        
        self.data_source_box = QComboBox()
        self.data_source_box.addItem("No data available")
        self.data_source_box.setEnabled(False)
        self.data_source_box.currentIndexChanged.connect(self._on_data_source_changed)
        self.data_source_box.setStyleSheet(get_combo_box_stylesheet())
        controls_layout.addWidget(self.data_source_box)

        # Comparison mode widget
        self._source_selection_widget = SourceSelectionWidget()
        self._source_selection_widget.comparison_mode_changed.connect(self._on_comparison_mode_changed)
        self._source_selection_widget.sources_changed.connect(self._on_comparison_sources_changed)
        controls_layout.addWidget(self._source_selection_widget)

        controls_layout.addSpacing(8)
        
        # Configuration section
        config_label = QLabel("Swath Configuration")
        config_label.setFont(data_font)
        config_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        controls_layout.addWidget(config_label)
        
        # Direction
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(8)
        dir_label = QLabel("Direction")
        dir_label.setMinimumWidth(90)
        dir_layout.addWidget(dir_label)
        
        self.swath_direction_box = QComboBox()
        self.swath_direction_box.addItems(["X", "Y", "Z"])
        self.swath_direction_box.setStyleSheet(get_combo_box_stylesheet())
        dir_layout.addWidget(self.swath_direction_box, 1)
        controls_layout.addLayout(dir_layout)
        
        # Property
        prop_layout = QHBoxLayout()
        prop_layout.setSpacing(8)
        prop_label = QLabel("Property")
        prop_label.setMinimumWidth(90)
        prop_layout.addWidget(prop_label)
        
        self.swath_property_box = QComboBox()
        self.swath_property_box.setStyleSheet(get_combo_box_stylesheet())
        prop_layout.addWidget(self.swath_property_box, 1)
        controls_layout.addLayout(prop_layout)
        
        # Bins
        bins_layout = QHBoxLayout()
        bins_layout.setSpacing(8)
        bins_label = QLabel("Bins")
        bins_label.setMinimumWidth(90)
        bins_layout.addWidget(bins_label)
        
        self.swath_bins_spin = QSpinBox()
        self.swath_bins_spin.setRange(5, 100)
        self.swath_bins_spin.setValue(20)
        self.swath_bins_spin.setToolTip("Number of bins to divide the swath into")
        self.swath_bins_spin.setMaximumWidth(100)
        self.swath_bins_spin.setStyleSheet(get_spin_box_stylesheet())
        bins_layout.addWidget(self.swath_bins_spin)
        bins_layout.addStretch()
        controls_layout.addLayout(bins_layout)
        
        controls_layout.addSpacing(4)
        
        # 3D Linking checkbox
        self.swath_link_check = QCheckBox("Link 3D Highlight")
        self.swath_link_check.setChecked(True)
        self.swath_link_check.setToolTip("Enable 3D highlighting when clicking swath bins")
        self.swath_link_check.toggled.connect(self._on_swath_link_toggled)
        self.swath_link_check.setStyleSheet(f"""
            QCheckBox {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {ModernColors.BORDER};
                border-radius: 3px;
                background-color: {ModernColors.CARD_BG};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ModernColors.ACCENT_PRIMARY};
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
        """)
        controls_layout.addWidget(self.swath_link_check)
        
        controls_layout.addSpacing(12)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {ModernColors.BORDER};")
        controls_layout.addWidget(sep2)
        
        controls_layout.addSpacing(8)
        
        # Bottom buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        
        self.swath_btn = QPushButton("Generate Swath Plot")
        self.swath_btn.setToolTip("Create interactive swath plot with 3D linking")
        self.swath_btn.clicked.connect(self.run_analysis)
        self.swath_btn.setEnabled(False)
        self.swath_btn.setMinimumHeight(38)
        self.swath_btn.setStyleSheet(get_button_stylesheet("primary"))
        buttons_layout.addWidget(self.swath_btn, 1)
        
        self.clear_swath_btn = QPushButton("Clear")
        self.clear_swath_btn.setToolTip("Remove swath interval highlight from 3D viewer")
        self.clear_swath_btn.clicked.connect(self._on_clear_swath_highlight)
        self.clear_swath_btn.setEnabled(False)
        self.clear_swath_btn.setMinimumHeight(38)
        self.clear_swath_btn.setMaximumWidth(100)
        self.clear_swath_btn.setStyleSheet(get_button_stylesheet("secondary"))
        buttons_layout.addWidget(self.clear_swath_btn)
        
        self.export_btn = QPushButton("Export ▼")
        self.export_btn.setToolTip("Export swath plot or data")
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumHeight(38)
        self.export_btn.setMaximumWidth(120)
        self.export_btn.setStyleSheet(get_button_stylesheet("secondary"))
        buttons_layout.addWidget(self.export_btn)
        
        controls_layout.addLayout(buttons_layout)
        
        controls_layout.addSpacing(12)
        
        # Info/Instructions section
        instructions = QLabel(
            "<b>Instructions:</b><br>"
            "1. Select swath direction (X, Y, or Z)<br>"
            "2. Choose property to analyze<br>"
            "3. Set number of bins<br>"
            "4. Click 'Generate Swath Plot'<br>"
            "5. Click any bin in the plot to highlight the corresponding 3D interval"
        )
        instructions.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 10px;
                padding: 12px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        instructions.setWordWrap(True)
        controls_layout.addWidget(instructions)
        
        controls_layout.addStretch()
        
        # Right side: Preview
        preview_widget = QFrame()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(12)
        
        # Preview title
        preview_title = QLabel("Plot Preview")
        preview_title.setFont(data_font)
        preview_title.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        preview_layout.addWidget(preview_title)
        
        # Plot preview area
        self.preview_frame = QFrame()
        self.preview_frame.setFrameShape(QFrame.Shape.Box)
        self.preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        self.preview_frame_layout = QVBoxLayout(self.preview_frame)
        self.preview_frame_layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder label
        self.placeholder_label = QLabel("Generate a swath plot to see preview")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 12px;")
        self.preview_frame_layout.addWidget(self.placeholder_label)
        
        preview_layout.addWidget(self.preview_frame, 1)
        
        # Status label
        self.swath_info_label = QLabel("Configure swath parameters and click 'Generate Swath Plot'")
        self.swath_info_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 10px;
                font-style: italic;
                padding: 8px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        self.swath_info_label.setWordWrap(True)
        self.swath_info_label.setMinimumHeight(50)
        preview_layout.addWidget(self.swath_info_label)
        
        # Add widgets to splitter
        splitter.addWidget(controls_widget)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 3)  # Controls: 30%
        splitter.setStretchFactor(1, 7)  # Preview: 70%
        
        main_layout.addWidget(splitter)
        
        # Apply panel-level styles
        self.setObjectName("SwathPanel")
        self.setStyleSheet(f"""
            QWidget#SwathPanel {{
                background-color: {ModernColors.PANEL_BG};
            }}
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
            }}
        """)

    def set_block_model(self, block_model):
        """Set the block model for swath analysis."""
        # Store the block model (DataFrame or BlockModel instance)
        self.current_model = block_model
        self._data_mode = "block_model"
        self._drillhole_df = None
        
        # Handle DataFrame block models
        if isinstance(block_model, pd.DataFrame):
            # Extract numeric properties from DataFrame columns
            numeric_props = []
            exclude_cols = {'X', 'Y', 'Z', 'XC', 'YC', 'ZC', 'BLOCK_ID', 'DOMAIN', 'ZONE'}
            for col in block_model.columns:
                if col.upper() not in exclude_cols:
                    if pd.api.types.is_numeric_dtype(block_model[col]):
                        numeric_props.append(col)
            
            logger.info(f"Set DataFrame block model with {len(numeric_props)} numeric properties in Swath panel")
        else:
            # BlockModel instance
            numeric_props = []
            if hasattr(block_model, 'properties') and block_model.properties:
                for prop_name, prop_data in block_model.properties.items():
                    if hasattr(prop_data, 'dtype') and np.issubdtype(prop_data.dtype, np.number):
                        numeric_props.append(prop_name)
                    elif isinstance(prop_data, np.ndarray) and np.issubdtype(prop_data.dtype, np.number):
                        numeric_props.append(prop_name)
            
            logger.info(f"Set BlockModel instance with {len(numeric_props)} numeric properties in Swath panel")
        
        # Update property selector
        if hasattr(self, 'swath_property_box'):
            self.swath_property_box.clear()
            self.swath_property_box.addItems(numeric_props)
        
        # Enable button
        if MATPLOTLIB_AVAILABLE and numeric_props:
            if hasattr(self, 'swath_btn'):
                self.swath_btn.setEnabled(True)
    
    def set_grid_data(self, grid, layer_name: str = "Grid"):
        """
        Set PyVista grid data for swath analysis (for SGSIM results, kriging, etc.).
        
        Args:
            grid: PyVista RectilinearGrid or StructuredGrid with cell_data properties
            layer_name: Name of the layer for logging
        """
        try:
            # Step 12: Legacy PyVista import - should use RenderPayload instead
            import pyvista as pv
            
            # Convert PyVista grid to BlockModel-like structure
            # Create a mock BlockModel that stores grid data
            class GridBlockModel:
                def __init__(self, grid, layer_name):
                    self.grid = grid
                    self.layer_name = layer_name
                    self.properties = {}
                    
                    # Extract cell centers as positions
                    try:
                        if hasattr(grid, 'cell_centers'):
                            centers = grid.cell_centers()
                            self.positions = np.column_stack([
                                centers.points[:, 0],  # X
                                centers.points[:, 1],  # Y
                                centers.points[:, 2]   # Z
                            ])
                        else:
                            # Fallback: use points directly
                            self.positions = grid.points
                    except Exception as e:
                        logger.warning(f"Error extracting cell centers, using grid points: {e}")
                        self.positions = grid.points
                    
                    # Extract all numeric cell_data properties
                    if hasattr(grid, 'cell_data'):
                        for prop_name in grid.cell_data.keys():
                            try:
                                prop_data = grid.cell_data[prop_name]
                                if len(prop_data) > 0 and np.issubdtype(prop_data.dtype, np.number):
                                    self.properties[prop_name] = prop_data
                            except Exception as e:
                                logger.debug(f"Skipping property '{prop_name}': {e}")
                    
                    # If no cell_data, try point_data
                    if not self.properties and hasattr(grid, 'point_data'):
                        for prop_name in grid.point_data.keys():
                            try:
                                prop_data = grid.point_data[prop_name]
                                if len(prop_data) > 0 and np.issubdtype(prop_data.dtype, np.number):
                                    self.properties[prop_name] = prop_data
                            except Exception as e:
                                logger.debug(f"Skipping property '{prop_name}': {e}")
                
                def get_property(self, prop_name):
                    """Get property data by name."""
                    return self.properties.get(prop_name, None)
            
            # Create mock block model
            self.current_model = GridBlockModel(grid, layer_name)
            
            # Get numeric properties
            numeric_props = list(self.current_model.properties.keys())
            
            if not numeric_props:
                logger.warning(f"No numeric properties found in grid '{layer_name}'")
                return
            
            # Update property selector
            self.swath_property_box.clear()
            self.swath_property_box.addItems(numeric_props)
            
            # Enable button
            if MATPLOTLIB_AVAILABLE and numeric_props:
                self.swath_btn.setEnabled(True)
            
            logger.info(f"Set grid data '{layer_name}' with {len(numeric_props)} properties in Swath panel")
            
        except Exception as e:
            logger.error(f"Error setting grid data in Swath panel: {e}", exc_info=True)
            raise
    
    def set_plotter_reference(self, plotter, grid, df):
        """Set references to PyVista plotter and grid for 3D highlighting."""
        self.plotter = plotter
        self.grid = grid
        self.df = df
    
    def _on_swath_link_toggled(self, checked: bool):
        """Handle swath 3D link toggle."""
        self.swath_link_enabled = checked
        logger.info(f"Swath 3D linking {'enabled' if checked else 'disabled'}")
    
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------

    def run_analysis(self):
        """Run swath analysis - handles both single source and comparison mode."""
        if self._comparison_mode:
            # Use comparison flow
            self._run_comparison_analysis()
        else:
            # Use standard single-source flow from base class
            super().run_analysis()

    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI."""
        if not self._has_data():
            raise ValueError("No data loaded (block model or drillhole data).")
        
        direction = self.swath_direction_box.currentText()
        property_name = self.swath_property_box.currentText()
        
        if not property_name:
            raise ValueError("Please select a property for swath analysis.")
        
        # Validate data mode and corresponding data
        if self._data_mode == "drillhole":
            if self._drillhole_df is None:
                raise ValueError("Drillhole mode selected but no drillhole data available.")
            logger.info(f"SwathPanel.gather_parameters: Using drillhole data ({len(self._drillhole_df)} rows)")
        elif self._data_mode == "block_model":
            if self.current_model is None:
                raise ValueError("Block model mode selected but no block model available.")
            logger.info(f"SwathPanel.gather_parameters: Using block model")
        else:
            raise ValueError(f"Invalid data mode: {self._data_mode}. Please select a data source from the dropdown.")
        
        return {
            "block_model": self.current_model,
            "drillhole_df": self._drillhole_df,
            "data_mode": self._data_mode,
            "direction": direction,
            "property_name": property_name,
            "n_bins": self.swath_bins_spin.value(),
        }
    
    def _has_data(self) -> bool:
        """Check if any data (block model or drillhole) is available."""
        if self._data_mode == "drillhole" and self._drillhole_df is not None:
            return True
        elif self._data_mode == "block_model" and self.current_model is not None:
            return True
        # Also check for raw df
        if self.df is not None:
            return True
        return False
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if not self._has_data():
            self.show_error("No Data", "Please load a block model or drillhole data first.")
            return False
        
        property_name = self.swath_property_box.currentText()
        if not property_name:
            self.show_error("No Property", "Please select a property for swath analysis.")
            return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display swath plot results."""
        grouped_data = payload.get("grouped_data")
        bins = payload.get("bins")
        direction = payload.get("direction")
        property_name = payload.get("property_name")
        n_bins = payload.get("n_bins")
        
        if grouped_data is not None and bins is not None:
            # Store swath state for click handling
            self.swath_bins = bins
            self.swath_direction = direction
            axis_map = {"X": 0, "Y": 1, "Z": 2}
            self.swath_axis_col = f"{direction}C"
            
            # Create figure with dark theme
            from .modern_styles import ModernColors
            self.current_figure = Figure(figsize=(10, 6), facecolor=ModernColors.CARD_BG)
            ax = self.current_figure.add_subplot(111)
            ax.set_facecolor(ModernColors.CARD_BG)
            
            # Plot swath with error bands  
            x = grouped_data['centre'].values
            y = grouped_data['mean'].values
            err = grouped_data['std'].values
            
            ax.plot(x, y, color=ModernColors.ACCENT_PRIMARY, linewidth=2.5, marker='o', markersize=6, label='Mean')
            ax.fill_between(x, y - err, y + err, color=ModernColors.ACCENT_PRIMARY, alpha=0.2, label='±1 Std Dev')
            
            ax.set_xlabel(f"{direction} (m)", fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_ylabel(property_name, fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_title(f"Swath Plot: {property_name} along {direction}-axis", fontsize=12, 
                        fontweight='bold', color=ModernColors.TEXT_PRIMARY, pad=15)
            ax.grid(True, linestyle='--', alpha=0.2, color=ModernColors.TEXT_SECONDARY)
            ax.tick_params(colors=ModernColors.TEXT_PRIMARY, labelsize=9)
            ax.legend(loc='best', fontsize=10, facecolor=ModernColors.CARD_BG, 
                     edgecolor=ModernColors.BORDER, labelcolor=ModernColors.TEXT_PRIMARY)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_edgecolor(ModernColors.BORDER)
            
            self.current_figure.tight_layout()
            
            # Show chart in preview pane
            self._show_plot_in_preview()
            
            # Update info label
            from .modern_styles import ModernColors
            self.swath_info_label.setText(
                f"✓ Swath plot generated ({n_bins} bins). "
                f"Click bins to highlight 3D intervals." if self.swath_link_enabled 
                else f"✓ Swath plot generated ({n_bins} bins). 3D linking disabled."
            )
            self.swath_info_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernColors.SUCCESS};
                    font-size: 10px;
                    font-style: italic;
                    padding: 8px;
                    background-color: {ModernColors.CARD_BG};
                    border: 1px solid {ModernColors.BORDER};
                    border-radius: 4px;
                }}
            """)
            
            # Enable clear button
            if hasattr(self, 'clear_swath_btn'):
                self.clear_swath_btn.setEnabled(True)
            if hasattr(self, 'export_btn'):
                self.export_btn.setEnabled(True)
            
            # Publish swath results to DataRegistry (as analysis metadata)
            if hasattr(self, 'registry') and self.registry:
                try:
                    swath_results = {
                        'direction': direction,
                        'property_name': property_name,
                        'bins': bins.tolist() if bins is not None else None,
                        'grouped_data': grouped_data.to_dict() if grouped_data is not None else None,
                        'n_bins': n_bins,
                        'source': 'swath_analysis'
                    }
                    # Store as metadata in block model if available
                    # For now, just log it - could extend DataRegistry with analysis results storage
                    logger.info(f"Swath analysis completed: {direction} direction, {n_bins} bins, property: {property_name}")
                except Exception as e:
                    logger.warning(f"Failed to process swath results for publication: {e}")
            
            logger.info(f"Generated swath plot: {property_name} along {direction} with {n_bins} bins")
    
    def _show_plot_in_preview(self):
        """Show the current figure in the preview pane with interactive clicking."""
        if not self.current_figure:
            return
        
        # Remove old canvas if exists
        if self.current_canvas:
            if self.swath_click_cid:
                self.current_canvas.mpl_disconnect(self.swath_click_cid)
            self.preview_frame_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None
        
        # Remove placeholder
        if self.placeholder_label:
            self.placeholder_label.setVisible(False)
        
        # Create new canvas
        self.current_canvas = FigureCanvasQTAgg(self.current_figure)
        self.preview_frame_layout.addWidget(self.current_canvas)
        
        # Connect click event for 3D highlighting
        self.swath_click_cid = self.current_canvas.mpl_connect('button_press_event', self._on_swath_click)
    
    def _show_swath_window(self):
        """Show swath plot in a popup window with interactive click handling."""
        if not self.current_figure:
            return
        
        from PyQt6.QtWidgets import QDialog, QVBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Swath Plot - {self.swath_property_box.currentText()}")
        dialog.resize(1200, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Disconnect previous canvas if exists
        if self.swath_canvas and self.swath_click_cid:
            self.swath_canvas.mpl_disconnect(self.swath_click_cid)
        
        # Create new canvas
        self.swath_canvas = FigureCanvasQTAgg(self.current_figure)
        layout.addWidget(self.swath_canvas)
        
        # Connect click event for 3D highlighting
        self.swath_click_cid = self.swath_canvas.mpl_connect('button_press_event', self._on_swath_click)
        
        dialog.show()
    
    def _on_swath_click(self, event):
        """Handle click on swath plot to highlight corresponding 3D slice."""
        if event.xdata is None or not hasattr(self, 'swath_bins'):
            return
        
        if not self.swath_link_enabled:
            return
        
        xval = event.xdata
        bins = self.swath_bins
        direction = self.swath_direction
        
        # Find which bin was clicked
        idx = np.searchsorted(bins, xval) - 1
        if idx < 0 or idx >= len(bins) - 1:
            return
        
        lower, upper = bins[idx], bins[idx + 1]
        
        logger.info(f"Swath bin {idx} clicked: {lower:.2f}–{upper:.2f} ({direction})")
        
        # Emit signal for 3D highlighting
        self.swath_highlight_requested.emit(direction, float(lower), float(upper))
        
        # Update info label
        from .modern_styles import ModernColors
        self.swath_info_label.setText(
            f"✓ Highlighted {direction} interval: [{lower:.2f}, {upper:.2f}] in 3D viewer"
        )
        self.swath_info_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.HIGHLIGHT};
                font-size: 10px;
                font-style: italic;
                font-weight: bold;
                padding: 8px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        
        # Enable clear button
        self.clear_swath_btn.setEnabled(True)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(True)
    
    def _on_clear_swath_highlight(self):
        """Clear swath highlight from 3D viewer."""
        self.swath_highlight_cleared.emit()
        
        # Update info label
        from .modern_styles import ModernColors
        self.swath_info_label.setText("Cleared 3D highlight")
        self.swath_info_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 10px;
                font-style: italic;
                padding: 8px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)
        
        # Disable clear button until next highlight
        self.clear_swath_btn.setEnabled(False)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(False)
        
        logger.info("Cleared swath highlight")

    # ------------------------------------------------------------------
    # Comparison mode methods
    # ------------------------------------------------------------------

    def _on_comparison_mode_changed(self, enabled: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = enabled
        # Hide single source selector when in comparison mode
        self.data_source_box.setVisible(not enabled)
        logger.info(f"SwathPanel: Comparison mode {'enabled' if enabled else 'disabled'}")

        if enabled:
            # Update comparison widget with available sources
            self._update_comparison_sources()

    def _on_comparison_sources_changed(self, selected_keys: List[str]):
        """Handle comparison source selection changes."""
        logger.info(f"SwathPanel: Comparison sources changed: {selected_keys}")

        # Populate properties from selected sources
        if selected_keys:
            self._populate_comparison_properties(selected_keys)

        # Enable/disable generate button based on selection
        if hasattr(self, 'swath_btn'):
            # Need at least 2 sources for comparison
            self.swath_btn.setEnabled(len(selected_keys) >= 2)

    def _populate_comparison_properties(self, selected_keys: List[str]):
        """Populate property dropdown with properties from selected sources."""
        import pandas as pd

        all_properties = set()

        for source_key in selected_keys:
            logger.debug(f"SwathPanel: Getting properties for source: {source_key}")

            if source_key == "drillhole":
                df = self._stored_drillhole_df
                if df is not None:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH'):
                            all_properties.add(col)
                    logger.debug(f"SwathPanel: Drillhole has {len(df.columns)} columns")

            elif source_key == "block_model":
                bm = self._stored_block_model
                if bm is not None:
                    logger.debug(f"SwathPanel: Block model type: {type(bm).__name__}")

                    # Check if it's a DataFrame first (common case)
                    if isinstance(bm, pd.DataFrame):
                        for col in bm.columns:
                            if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG'):
                                try:
                                    if pd.api.types.is_numeric_dtype(bm[col]):
                                        all_properties.add(col)
                                except:
                                    all_properties.add(col)
                        logger.debug(f"SwathPanel: Block model DataFrame has columns: {list(bm.columns)[:10]}")
                    else:
                        # BlockModel class
                        # Try properties attribute first
                        if hasattr(bm, 'properties') and bm.properties:
                            for prop in bm.properties.keys():
                                all_properties.add(prop)
                            logger.debug(f"SwathPanel: Block model has properties: {list(bm.properties.keys())[:5]}")
                        # Try to_dataframe method
                        if hasattr(bm, 'to_dataframe'):
                            try:
                                df = bm.to_dataframe()
                                for col in df.columns:
                                    if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            all_properties.add(col)
                                logger.debug(f"SwathPanel: Block model DataFrame has {len(df.columns)} columns, properties: {list(df.columns)[:10]}")
                            except Exception as e:
                                logger.debug(f"SwathPanel: Could not convert block model to dataframe: {e}")

            elif source_key == "classified_block_model":
                bm = getattr(self, '_stored_classified_block_model', None)
                if bm is not None:
                    if hasattr(bm, 'properties') and bm.properties:
                        all_properties.update(bm.properties.keys())
                    if hasattr(bm, 'columns'):
                        for col in bm.columns:
                            if col.upper() not in ('X', 'Y', 'Z'):
                                all_properties.add(col)

            elif source_key in self._block_model_sources:
                source_info = self._block_model_sources[source_key]
                prop = source_info.get('property')
                if prop:
                    all_properties.add(prop)
                df = source_info.get('df')
                if df is not None:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z'):
                            all_properties.add(col)
                    logger.debug(f"SwathPanel: SGSIM source {source_key} has columns: {list(df.columns)[:5]}")

            else:
                # Try to get from comparison widget sources
                source_info = self._source_selection_widget._sources.get(source_key, {})
                df = source_info.get('df')
                if df is not None and hasattr(df, 'columns'):
                    for col in df.columns:
                        if col.upper() not in ('X', 'Y', 'Z'):
                            all_properties.add(col)
                    logger.debug(f"SwathPanel: Widget source {source_key} has columns: {list(df.columns)[:5]}")

        logger.info(f"SwathPanel: Collected {len(all_properties)} properties: {sorted(all_properties)[:10]}")

        # Update property dropdown
        if hasattr(self, 'swath_property_box'):
            current_prop = self.swath_property_box.currentText()
            self.swath_property_box.blockSignals(True)
            self.swath_property_box.clear()

            sorted_props = sorted(all_properties)
            self.swath_property_box.addItems(sorted_props)

            # Try to restore previous selection or select first grade-like property
            if current_prop and current_prop in sorted_props:
                self.swath_property_box.setCurrentText(current_prop)
            else:
                for prop in sorted_props:
                    if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')):
                        self.swath_property_box.setCurrentText(prop)
                        break

            self.swath_property_box.blockSignals(False)
            logger.info(f"SwathPanel: Populated {len(sorted_props)} properties for comparison: {sorted_props[:5]}")

    def _update_comparison_sources(self):
        """Update the comparison widget with available sources."""
        sources = {}
        logger.info(f"SwathPanel: Updating comparison sources. Available: {self._available_sources}")
        logger.debug(f"SwathPanel: _stored_block_model is {'set' if self._stored_block_model is not None else 'None'}")
        logger.debug(f"SwathPanel: _block_model_sources has {len(self._block_model_sources)} entries")

        # Add drillhole data
        if "drillhole" in self._available_sources and self._stored_drillhole_df is not None:
            sources["drillhole"] = {
                'display_name': f"Drillhole Data ({self._stored_data_source_type})",
                'block_count': len(self._stored_drillhole_df),
                'df': self._stored_drillhole_df
            }
            logger.debug(f"SwathPanel: Added drillhole source with {len(self._stored_drillhole_df)} rows")

        # Add block model
        if "block_model" in self._available_sources and self._stored_block_model is not None:
            bm = self._stored_block_model
            count = len(bm) if hasattr(bm, '__len__') else 0
            sources["block_model"] = {
                'display_name': "Block Model",
                'block_count': count,
                'data': bm
            }
            logger.info(f"SwathPanel: Added block_model source with {count} blocks, type: {type(bm).__name__}")

        # Add classified block model
        if "classified_block_model" in self._available_sources:
            bm = getattr(self, '_stored_classified_block_model', None)
            if bm is not None:
                count = len(bm) if hasattr(bm, '__len__') else 0
                sources["classified_block_model"] = {
                    'display_name': "Classified Block Model",
                    'block_count': count,
                    'data': bm
                }
                logger.debug(f"SwathPanel: Added classified_block_model source with {count} blocks")

        # Add individual SGSIM sources
        for source_key, source_info in self._block_model_sources.items():
            df = source_info.get('df')
            if df is not None:
                sources[source_key] = {
                    'display_name': source_info.get('display_name', source_key),
                    'block_count': len(df),
                    'df': df,
                    'property': source_info.get('property')
                }
                logger.debug(f"SwathPanel: Added SGSIM source {source_key} with {len(df)} rows")

        logger.info(f"SwathPanel: Final comparison sources: {list(sources.keys())}")
        self._source_selection_widget.update_sources(sources)

    def _get_data_for_source(self, source_key: str):
        """Get the data (DataFrame or BlockModel) for a given source key.

        Returns tuple of (data, data_mode, property_hint)
        """
        if source_key == "drillhole":
            return self._stored_drillhole_df, "drillhole", None

        if source_key == "block_model":
            return self._stored_block_model, "block_model", None

        if source_key == "classified_block_model":
            bm = getattr(self, '_stored_classified_block_model', None)
            return bm, "block_model", None

        if source_key in self._block_model_sources:
            source_info = self._block_model_sources[source_key]
            return source_info.get('df'), "block_model", source_info.get('property')

        return None, "none", None

    def _run_comparison_analysis(self):
        """Run swath analysis for multiple sources and plot comparison."""
        selected_keys = self._source_selection_widget.get_selected_sources()

        if len(selected_keys) < 2:
            self.show_error("Selection Error", "Please select at least 2 sources for comparison.")
            return

        direction = self.swath_direction_box.currentText()
        property_name = self.swath_property_box.currentText()
        n_bins = self.swath_bins_spin.value()

        if not property_name:
            self.show_error("No Property", "Please select a property for swath analysis.")
            return

        # Collect results for each source
        self._comparison_results = {}
        source_display_names = []

        for source_key in selected_keys:
            data, data_mode, property_hint = self._get_data_for_source(source_key)

            if data is None:
                logger.warning(f"SwathPanel: No data for source {source_key}")
                continue

            # Determine which property to use
            use_property = property_hint if property_hint else property_name

            # Get DataFrame for analysis
            if data_mode == "drillhole":
                df = data
            elif isinstance(data, pd.DataFrame):
                df = data
            elif hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            elif hasattr(data, 'positions') and hasattr(data, 'properties'):
                # BlockModel-like object
                df = pd.DataFrame({
                    'X': data.positions[:, 0],
                    'Y': data.positions[:, 1],
                    'Z': data.positions[:, 2],
                })
                for pname, pdata in data.properties.items():
                    df[pname] = pdata
            else:
                logger.warning(f"SwathPanel: Cannot convert source {source_key} to DataFrame")
                continue

            # Validate property exists in data - try multiple matching strategies
            if use_property not in df.columns:
                found_match = False

                # Strategy 1: Property name contains column name (e.g., "FE_SGSIM_MEAN" -> "FE")
                for col in df.columns:
                    if col.upper() in property_name.upper() and col.upper() not in ('X', 'Y', 'Z'):
                        use_property = col
                        found_match = True
                        logger.info(f"SwathPanel: Matched {property_name} -> {col} (property contains column)")
                        break

                # Strategy 2: Column name contains property name
                if not found_match:
                    for col in df.columns:
                        if property_name.upper() in col.upper():
                            use_property = col
                            found_match = True
                            logger.info(f"SwathPanel: Matched {property_name} -> {col} (column contains property)")
                            break

                # Strategy 3: Extract base name from SGSIM property (e.g., "Grade_MEAN" -> look for "Grade")
                if not found_match:
                    # Try to extract base variable name
                    base_name = property_name.split('_')[0] if '_' in property_name else property_name
                    for col in df.columns:
                        col_base = col.split('_')[0] if '_' in col else col
                        if base_name.upper() == col_base.upper() and col.upper() not in ('X', 'Y', 'Z'):
                            use_property = col
                            found_match = True
                            logger.info(f"SwathPanel: Matched {property_name} -> {col} (base name match)")
                            break

                # Strategy 4: Look for any grade-like property
                if not found_match:
                    for col in df.columns:
                        if any(k in col.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')) and col.upper() not in ('X', 'Y', 'Z'):
                            use_property = col
                            found_match = True
                            logger.info(f"SwathPanel: Using grade-like property {col} for source {source_key}")
                            break

                if not found_match:
                    logger.warning(f"SwathPanel: No matching property for '{property_name}' in source {source_key}. Available: {list(df.columns)[:10]}")
                    continue

            # Calculate swath for this source
            try:
                grouped_data, bins = self._calculate_swath(df, direction, use_property, n_bins)

                if grouped_data is not None:
                    # Get display name
                    source_info = self._source_selection_widget._sources.get(source_key, {})
                    display_name = source_info.get('display_name', source_key)

                    self._comparison_results[source_key] = {
                        'grouped_data': grouped_data,
                        'bins': bins,
                        'display_name': display_name,
                        'property': use_property,
                    }
                    source_display_names.append(display_name)
            except Exception as e:
                logger.warning(f"SwathPanel: Failed to calculate swath for {source_key}: {e}")
                continue

        if len(self._comparison_results) < 2:
            self.show_error("Analysis Error", "Could not compute swath for at least 2 sources.")
            return

        # Plot comparison
        self._plot_comparison_swath(direction, property_name, n_bins)

    def _calculate_swath(self, df: pd.DataFrame, direction: str, property_name: str, n_bins: int):
        """Calculate swath statistics for a single DataFrame.

        Returns (grouped_data_df, bins_array)
        """
        # Determine axis column
        axis_col = direction.upper()
        if axis_col not in df.columns:
            # Try common variations
            for col in df.columns:
                if col.upper() == axis_col or col.upper() == f"{axis_col}C":
                    axis_col = col
                    break
            else:
                logger.warning(f"SwathPanel: Axis column {direction} not found in DataFrame")
                return None, None

        # Filter valid data
        mask = df[property_name].notna() & df[axis_col].notna()
        valid_df = df[mask]

        if len(valid_df) == 0:
            return None, None

        # Create bins
        axis_values = valid_df[axis_col].values
        bins = np.linspace(axis_values.min(), axis_values.max(), n_bins + 1)

        # Compute bin indices
        bin_indices = np.digitize(axis_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate statistics per bin
        bin_centers = []
        means = []
        stds = []

        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.sum() > 0:
                bin_values = valid_df.loc[bin_mask, property_name].values
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                means.append(np.nanmean(bin_values))
                stds.append(np.nanstd(bin_values))

        if len(bin_centers) == 0:
            return None, None

        grouped_data = pd.DataFrame({
            'centre': bin_centers,
            'mean': means,
            'std': stds,
        })

        return grouped_data, bins

    def _plot_comparison_swath(self, direction: str, property_name: str, n_bins: int):
        """Plot overlaid swath curves for multiple sources."""
        from .modern_styles import ModernColors

        # Create figure
        self.current_figure = Figure(figsize=(10, 6), facecolor=ModernColors.CARD_BG)
        ax = self.current_figure.add_subplot(111)
        ax.set_facecolor(ModernColors.CARD_BG)

        # Plot each source with different colors
        source_names = []
        for i, (source_key, result) in enumerate(self._comparison_results.items()):
            grouped_data = result['grouped_data']
            display_name = result['display_name']

            x = grouped_data['centre'].values
            y = grouped_data['mean'].values
            err = grouped_data['std'].values

            # Get color and style
            style = ComparisonColors.get_style(i)
            color = style['color']
            linestyle = style['linestyle']
            marker = style['marker']

            # Plot mean line
            ax.plot(x, y, color=color, linewidth=2.5, linestyle=linestyle,
                   marker=marker, markersize=5, label=display_name)

            # Plot uncertainty band with same color
            fill_color = ComparisonColors.get_fill_color(i)
            ax.fill_between(x, y - err, y + err, color=fill_color)

            source_names.append(display_name)

        # Styling
        ax.set_xlabel(f"{direction} (m)", fontsize=11, color=ModernColors.TEXT_PRIMARY)
        ax.set_ylabel(property_name, fontsize=11, color=ModernColors.TEXT_PRIMARY)
        ax.set_title(f"Swath Plot Comparison: {property_name} along {direction}-axis",
                    fontsize=12, fontweight='bold', color=ModernColors.TEXT_PRIMARY, pad=15)
        ax.grid(True, linestyle='--', alpha=0.2, color=ModernColors.TEXT_SECONDARY)
        ax.tick_params(colors=ModernColors.TEXT_PRIMARY, labelsize=9)

        # Create legend
        legend = ax.legend(loc='upper right', fontsize=9, facecolor=ModernColors.CARD_BG,
                          edgecolor=ModernColors.BORDER, labelcolor=ModernColors.TEXT_PRIMARY)
        legend.get_frame().set_alpha(0.9)

        # Style spines
        for spine in ax.spines.values():
            spine.set_edgecolor(ModernColors.BORDER)

        self.current_figure.tight_layout()

        # Show plot
        self._show_plot_in_preview()

        # Update info label
        self.swath_info_label.setText(
            f"Comparison plot generated ({len(self._comparison_results)} sources, {n_bins} bins). "
            f"Click bins to highlight 3D intervals." if self.swath_link_enabled
            else f"Comparison plot generated ({len(self._comparison_results)} sources, {n_bins} bins)."
        )
        self.swath_info_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.SUCCESS};
                font-size: 10px;
                font-style: italic;
                padding: 8px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)

        # Enable buttons
        if hasattr(self, 'clear_swath_btn'):
            self.clear_swath_btn.setEnabled(True)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(True)

        # Store bins for click handling (use first source's bins)
        first_result = next(iter(self._comparison_results.values()))
        self.swath_bins = first_result['bins']
        self.swath_direction = direction

        logger.info(f"Generated comparison swath plot: {len(self._comparison_results)} sources, {direction} direction, {n_bins} bins")

    def clear(self):
        """Clear the panel."""
        self.current_model = None
        self.current_figure = None
        self.swath_canvas = None
        
        if self.current_canvas:
            if self.swath_click_cid:
                self.current_canvas.mpl_disconnect(self.swath_click_cid)
            self.preview_frame_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None
        
        if self.placeholder_label:
            self.placeholder_label.setVisible(True)
        
        self.swath_bins = None
        self.swath_direction = None
        self.swath_axis_col = None
        self.swath_property_box.clear()
        self.swath_btn.setEnabled(False)
        self.clear_swath_btn.setEnabled(False)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(False)
        
        # Reset info label
        from .modern_styles import ModernColors
        self.swath_info_label.setText("Configure swath parameters and click 'Generate Swath Plot'")
        self.swath_info_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: 10px;
                font-style: italic;
                padding: 8px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
            }}
        """)

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        # Clear comparison results
        self._comparison_results = {}
        # Use existing clear() method
        self.clear()
        super().clear_panel()
        logger.info("SwathPanel: Panel fully cleared")

    def _on_export_clicked(self):
        """Handle Export button click - show dropdown menu."""
        if not self.current_figure:
            return
        
        from .modern_styles import ModernColors
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {ModernColors.CARD_BG};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
            }}
            QMenu::item:selected {{
                background-color: {ModernColors.CARD_HOVER};
            }}
        """)
        
        export_png = QAction("PNG", self)
        export_png.triggered.connect(lambda: self._export_plot("png"))
        menu.addAction(export_png)
        
        export_pdf = QAction("PDF", self)
        export_pdf.triggered.connect(lambda: self._export_plot("pdf"))
        menu.addAction(export_pdf)
        
        export_svg = QAction("SVG", self)
        export_svg.triggered.connect(lambda: self._export_plot("svg"))
        menu.addAction(export_svg)
        
        menu.addSeparator()
        
        export_csv = QAction("CSV Data", self)
        export_csv.triggered.connect(self._export_data_csv)
        menu.addAction(export_csv)
        
        # Show menu at button position
        menu.exec(self.export_btn.mapToGlobal(self.export_btn.rect().bottomLeft()))
    
    def _export_plot(self, format_type: str):
        """Export current swath plot to file."""
        if not self.current_figure:
            return
        
        format_map = {
            "png": ("PNG Files (*.png)", "png"),
            "pdf": ("PDF Files (*.pdf)", "pdf"),
            "svg": ("SVG Files (*.svg)", "svg"),
        }
        
        filter_str, ext = format_map.get(format_type, ("PNG Files (*.png)", "png"))
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export Swath Plot as {ext.upper()}", f"swath_plot.{ext}", filter_str
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight', 
                                           facecolor=self.current_figure.get_facecolor())
                QMessageBox.information(self, "Success", f"Swath plot exported to {filename}")
                logger.info(f"Exported swath plot to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot:\n{e}")
                logger.error(f"Swath export error: {e}")
    
    def _export_data_csv(self):
        """Export current swath data to CSV."""
        if self.swath_bins is None or not hasattr(self, 'swath_axis_col'):
            QMessageBox.warning(self, "No Data", "No swath data available to export.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Swath Data as CSV", "swath_data.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                # Export the swath bins data
                df = pd.DataFrame(self.swath_bins)
                df.to_csv(filename, index=False)
                
                QMessageBox.information(self, "Success", f"Swath data exported to {filename}")
                logger.info(f"Exported swath data to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")
                logger.error(f"Swath data export error: {e}")

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet
        self.setStyleSheet(get_analysis_panel_stylesheet())










