"""
Charts & Visualization Panel - Modern, professional analytical interface.

Designed with Leapfrog-level professionalism:
- Clean, frameless design with typography hierarchy
- Split layout with live preview
- Horizontal segmented chart type controls
- Context-aware settings (dynamic UI)
- Integrated statistics for investor-grade analysis
- Export dropdown with multiple formats
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Dict, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QLabel, QSpinBox, QFileDialog, QMessageBox,
    QButtonGroup, QRadioButton, QSplitter, QFrame, QMenu
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction

from ..models.block_model import BlockModel

try:
    from .base_analysis_panel import BaseAnalysisPanel
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

# Matplotlib imports
try:
    # Matplotlib backend is set in main.py
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .modern_styles import (
    get_theme_colors, ModernColors,
    get_button_stylesheet,
    get_combo_box_stylesheet,
    get_group_box_stylesheet,
    get_spin_box_stylesheet,
)
from .comparison_utils import ComparisonColors, SourceSelectionWidget

logger = logging.getLogger(__name__)


class ChartsPanel(BaseAnalysisPanel if BASE_AVAILABLE else QWidget):
    """
    Modern, professional panel for creating charts and visualizations.
    
    DESIGN PHILOSOPHY:
    - Professional, Leapfrog-level interface
    - Clean spacing and typography instead of heavy frames
    - Split layout: controls (30%) | preview (70%)
    - Horizontal segmented chart type selectors
    - Context-aware settings (only show relevant options)
    - Live preview with integrated statistics
    - Investor-grade analytics presentation
    
    DATA SOURCES:
    - Supports both block models AND drillhole data
    - Drillhole data: composites, assays, or raw DataFrames
    - Block models: from loader, kriging, SGSIM, etc.
    """

    # PanelManager metadata (only if BaseAnalysisPanel available)
    if BASE_AVAILABLE:
        PANEL_ID = "ChartsPanel"
        PANEL_NAME = "Charts & Visualization"
        PANEL_CATEGORY = "Visualization"
        PANEL_DEFAULT_VISIBLE = False
        PANEL_DEFAULT_DOCK_AREA = "Right"

    def __init__(self, parent=None):
        # Initialize data attributes before calling super().__init__
        self.current_model: Optional[BlockModel] = None
        self.active_property: Optional[str] = None
        self.current_figure: Optional[Figure] = None
        self.current_canvas: Optional[FigureCanvasQTAgg] = None
        
        # Drillhole data support
        self._drillhole_df: Optional[Any] = None  # pd.DataFrame
        self._data_mode: str = "none"  # "block_model", "drillhole", or "none"
        self._data_source_type: str = "unknown"  # "composites", "assays", "dataframe"
        
        # Stored data sources (for user selection)
        self._stored_drillhole_df: Optional[Any] = None
        self._stored_block_model: Optional[BlockModel] = None
        self._stored_sgsim_df: Optional[Any] = None  # SGSIM results as DataFrame
        self._stored_data_source_type: str = "unknown"
        self._available_sources: list = []

        # Storage for individual SGSIM statistics (Mean, P10, P50, P90, Std Dev)
        self._block_model_sources: dict = {}

        # Comparison mode support
        self._comparison_mode: bool = False
        self._comparison_data: Dict[str, Any] = {}  # Store data per source

        # Current chart type
        self._current_chart_type: str = "histogram"

        super().__init__(parent=parent, panel_id="charts" if BASE_AVAILABLE else None)

        # Connect to DataRegistry if available
        if BASE_AVAILABLE:
            self._connect_registry()

        self._setup_ui()
        
        # Refresh data AFTER UI is set up
        if self.registry:
            self._refresh_available_data()
        
        logger.info("Initialized Charts & Visualization panel")

    def _connect_registry(self):
        """Connect to DataRegistry for automatic block model and drillhole updates."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                # Connect to block model signals
                self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
                self.registry.blockModelGenerated.connect(self._on_block_model_generated)
                self.registry.blockModelClassified.connect(self._on_block_model_classified)
                
                # Connect to drillhole signals - NEW
                if hasattr(self.registry, 'drillholeDataLoaded'):
                    self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                    logger.info("ChartsPanel: Connected to drillholeDataLoaded signal")

                # Connect to SGSIM results signal
                if hasattr(self.registry, 'sgsimResultsLoaded'):
                    self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
                    logger.info("ChartsPanel: Connected to sgsimResultsLoaded signal")

                # Data will be refreshed after _setup_ui() completes
                logger.info("Charts panel connected to DataRegistry")
            else:
                logger.info("DataRegistry not available, charts panel running standalone")
                self.registry = None
        except Exception as e:
            logger.warning(f"Failed to connect charts panel to DataRegistry: {e}")
            self.registry = None
    
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

            # Check for SGSIM results
            if hasattr(self.registry, 'get_sgsim_results'):
                sgsim = self.registry.get_sgsim_results()
                if sgsim is not None:
                    self._on_sgsim_loaded(sgsim)

            # Update data source selector
            self._update_data_source_selector()
            
        except Exception as e:
            logger.warning(f"ChartsPanel: Failed to refresh available data: {e}")
    
    def _store_drillhole_data(self, drillhole_data):
        """Store drillhole data without switching to it."""
        import pandas as pd
        
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

        self.data_source_box.blockSignals(False)

        # If we have real data sources and nothing is selected yet, select the first one
        if len(self._available_sources) > 0 and self._data_mode == "none":
            self.data_source_box.setCurrentIndex(0)
            self._on_data_source_changed(0)

        # Also update comparison sources widget
        if hasattr(self, '_source_selection_widget'):
            self._update_comparison_sources()
            logger.info(f"ChartsPanel: Available sources = {self._available_sources}")

    def _on_data_source_changed(self, index):
        """Handle user changing data source selection."""
        if not hasattr(self, 'data_source_box'):
            return

        source_type = self.data_source_box.itemData(index)
        if source_type == "drillhole" and self._stored_drillhole_df is not None:
            self._data_mode = "drillhole"
            self._drillhole_df = self._stored_drillhole_df
            self._data_source_type = self._stored_data_source_type
            self.current_model = None
            self._populate_drillhole_properties()
            logger.info(f"ChartsPanel: Switched to drillhole data ({self._data_source_type})")
        elif source_type == "block_model" and self._stored_block_model is not None:
            self._data_mode = "block_model"
            self._drillhole_df = None
            self._update_block_model(self._stored_block_model)
            logger.info("ChartsPanel: Switched to block model")
        elif source_type == "classified_block_model" and hasattr(self, '_stored_classified_block_model') and self._stored_classified_block_model is not None:
            self._data_mode = "block_model"  # Same mode, just different source
            self._drillhole_df = None
            self._update_block_model(self._stored_classified_block_model)
            logger.info("ChartsPanel: Switched to classified block model")
        elif source_type == "sgsim" and self._stored_sgsim_df is not None:
            self._data_mode = "block_model"  # Treat as block model
            self._drillhole_df = None
            self._update_block_model(self._stored_sgsim_df)
            logger.info("ChartsPanel: Switched to SGSIM results")
        elif source_type and source_type.startswith('sgsim_') and source_type in self._block_model_sources:
            # Handle individual SGSIM statistics
            source_info = self._block_model_sources[source_type]
            df = source_info.get('df')
            if df is not None:
                self._data_mode = "block_model"
                self._drillhole_df = None
                self._update_block_model(df)
                logger.info(f"ChartsPanel: Switched to {source_info.get('display_name', source_type)}")
        else:
            self._data_mode = "none"
            self.property_box.clear()
            self.property2_box.clear()
            self._set_generate_enabled(False)
    
    def _populate_drillhole_properties(self):
        """Populate property selector from drillhole DataFrame."""
        import pandas as pd
        
        if self._drillhole_df is None:
            return
        
        df = self._drillhole_df
        
        # Extract numeric columns for property selection
        numeric_cols = []
        exclude_cols = {'HOLEID', 'HOLE_ID', 'BHID', 'FROM', 'TO', 'FROM_M', 'TO_M', 
                        'X', 'Y', 'Z', 'MID_X', 'MID_Y', 'MID_Z', 'LENGTH', 'SAMPLE_ID'}
        
        for col in df.columns:
            if col.upper() not in exclude_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
        
        # Update property combo boxes
        self.property_box.clear()
        self.property_box.addItems(numeric_cols)
        
        self.property2_box.clear()
        self.property2_box.addItem("Optional")
        self.property2_box.addItems(numeric_cols)
        
        if numeric_cols:
            self.active_property = numeric_cols[0]
        
        # Enable generate button
        if MATPLOTLIB_AVAILABLE and numeric_cols:
            self._set_generate_enabled(True)
        
        logger.info(f"ChartsPanel: Loaded {len(numeric_cols)} numeric columns from {self._data_source_type}")
    
    def _refresh_available_block_models(self, update_ui: bool = True):
        """Refresh the list of available block models from all sources."""
        if not self.registry:
            return
        
        try:
            # Check regular block model
            block_model = self.registry.get_block_model()
            if block_model is not None:
                if update_ui:
                    self._update_block_model(block_model)
                else:
                    # Just store it without triggering UI update
                    self.current_model = block_model
            
            # Check classified block model
            try:
                classified_model = self.registry.get_classified_block_model()
                if classified_model is not None:
                    # Use classified model if we don't have a current one
                    if block_model is None:
                        if update_ui:
                            self._update_block_model(classified_model)
                        else:
                            self.current_model = classified_model
            except Exception:
                pass
            
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

    def _on_block_model_loaded(self, block_model):
        """Handle new block model loaded."""
        # Store the block model and update selector
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        logger.info("ChartsPanel: Block model available for selection")

    def _on_block_model_generated(self, block_model):
        """Handle new block model generated."""
        # Store the block model and update selector
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        logger.info("ChartsPanel: Block model available for selection")

    def _on_block_model_classified(self, block_model):
        """Handle block model classification changes."""
        # Store as SEPARATE classified block model (don't overwrite regular block model)
        self._stored_classified_block_model = block_model
        if "classified_block_model" not in self._available_sources:
            self._available_sources.append("classified_block_model")
        self._update_data_source_selector()
        logger.info("ChartsPanel: Classified block model available for selection")

    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results - register individual statistics as separate sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        try:
            import pandas as pd
            import numpy as np
            import pyvista as pv

            if results is None:
                return

            if not isinstance(results, dict):
                logger.warning(f"ChartsPanel: SGSIM results is not a dict, type={type(results)}")
                return

            variable = results.get('variable', 'Grade')
            summary = results.get('summary', {})
            params = results.get('params')
            grid = results.get('grid') or results.get('pyvista_grid')

            logger.info(f"ChartsPanel: SGSIM results keys: {list(results.keys())}")
            logger.info(f"ChartsPanel: Summary keys: {list(summary.keys()) if summary else 'None'}")
            logger.info(f"ChartsPanel: params = {params is not None}")

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
                        logger.info(f"ChartsPanel: Extracted {n_blocks:,} cell centers from grid")

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
                    logger.info(f"ChartsPanel: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
                except Exception as e:
                    logger.warning(f"ChartsPanel: Failed to generate coords from params: {e}")

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
                        logger.info(f"ChartsPanel: Registered SGSIM E-type Mean from fallback")
                else:
                    logger.warning("ChartsPanel: No grid, params, or realizations found in SGSIM results")
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
                        logger.info(f"ChartsPanel: Registered {display_prefix} ({variable})")

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
                logger.info(f"ChartsPanel: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

            self._update_data_source_selector()

        except Exception as e:
            logger.warning(f"ChartsPanel: Failed to load SGSIM results: {e}", exc_info=True)

    def _on_drillhole_data_loaded(self, drillhole_data):
        """
        Receive drillhole data from DataRegistry.
        
        Stores the data and updates the selector - user must choose to use it.
        """
        logger.info("ChartsPanel: received drillhole data from DataRegistry")
        
        # Store the drillhole data
        self._store_drillhole_data(drillhole_data)
        
        # Add to available sources and update selector
        if self._stored_drillhole_df is not None:
            if "drillhole" not in self._available_sources:
                self._available_sources.append("drillhole")
            self._update_data_source_selector()
            logger.info("ChartsPanel: Drillhole data available for selection")

    def _update_block_model(self, block_model):
        """Update internal block model reference and refresh UI."""
        self.current_model = block_model
        
        # Update UI if block model is available
        if block_model is not None:
            # Check if it's a BlockModel object or DataFrame
            if hasattr(block_model, 'properties'):
                # BlockModel API
                numeric_props = []
                if block_model.properties:
                    for prop_name, prop_data in block_model.properties.items():
                        if hasattr(prop_data, 'dtype') and np.issubdtype(prop_data.dtype, np.number):
                            numeric_props.append(prop_name)
                        elif isinstance(prop_data, np.ndarray) and np.issubdtype(prop_data.dtype, np.number):
                            numeric_props.append(prop_name)
                
                # Update property combo boxes
                if hasattr(self, 'property_box'):
                    self.property_box.clear()
                    self.property_box.addItems(numeric_props)
                    
                    self.property2_box.clear()
                    self.property2_box.addItems(numeric_props)
                    
                    if numeric_props:
                        self.active_property = numeric_props[0]
                    
                    # Enable generate button
                    if MATPLOTLIB_AVAILABLE and numeric_props:
                        self._set_generate_enabled(True)
            elif hasattr(block_model, 'columns'):
                # DataFrame - extract numeric columns
                numeric_props = []
                for col in block_model.columns:
                    if col not in ['X', 'Y', 'Z', 'DX', 'DY', 'DZ']:
                        if block_model[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                            numeric_props.append(col)
                
                # Update property combo boxes
                if hasattr(self, 'property_box'):
                    self.property_box.clear()
                    self.property_box.addItems(numeric_props)
                    
                    self.property2_box.clear()
                    self.property2_box.addItems(numeric_props)
                    
                    if numeric_props:
                        self.active_property = numeric_props[0]
                    
                    # Enable generate button
                    if MATPLOTLIB_AVAILABLE and numeric_props:
                        self._set_generate_enabled(True)
        else:
            # Clear UI if no block model
            if hasattr(self, 'property_box'):
                self.property_box.clear()
                self.property2_box.clear()
                self._set_generate_enabled(False)
        
        logger.info(f"Charts panel: Block model updated ({type(block_model)})")
    
    def _setup_ui(self):
        """Setup the modern UI components with split layout and live preview."""
        # Use the main_layout from BaseAnalysisPanel
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        if not MATPLOTLIB_AVAILABLE:
            error_label = QLabel("Matplotlib not available. Install matplotlib to use charting features.")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            main_layout.addWidget(error_label)
            return
        
        # Create split layout: Controls (left) | Preview (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {ModernColors.BORDER}; }}")
        
        # Left side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(16)
        
        # Title
        title_label = QLabel("Charts & Visualization")
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
        data_font.setBold(False)
        data_label.setFont(data_font)
        data_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        controls_layout.addWidget(data_label)
        
        self.data_source_box = QComboBox()
        self.data_source_box.addItem("No data loaded", "none")
        self.data_source_box.currentIndexChanged.connect(self._on_data_source_changed)
        controls_layout.addWidget(self.data_source_box)

        # Comparison mode widget
        self._source_selection_widget = SourceSelectionWidget()
        self._source_selection_widget.comparison_mode_changed.connect(self._on_comparison_mode_changed)
        self._source_selection_widget.sources_changed.connect(self._on_comparison_sources_changed)
        controls_layout.addWidget(self._source_selection_widget)

        controls_layout.addSpacing(8)
        
        # Variables section
        vars_label = QLabel("Variables")
        vars_label.setFont(data_font)
        vars_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        controls_layout.addWidget(vars_label)
        
        # Primary property
        prim_layout = QHBoxLayout()
        prim_layout.setSpacing(8)
        prim_label = QLabel("Primary")
        prim_label.setMinimumWidth(80)
        prim_layout.addWidget(prim_label)
        
        self.property_box = QComboBox()
        self.property_box.currentTextChanged.connect(self._on_property_changed)
        prim_layout.addWidget(self.property_box, 1)
        controls_layout.addLayout(prim_layout)
        
        # Secondary property
        sec_layout = QHBoxLayout()
        sec_layout.setSpacing(8)
        sec_label = QLabel("Secondary")
        sec_label.setMinimumWidth(80)
        sec_layout.addWidget(sec_label)
        
        self.property2_box = QComboBox()
        self.property2_box.addItem("Optional")
        sec_layout.addWidget(self.property2_box, 1)
        controls_layout.addLayout(sec_layout)
        
        controls_layout.addSpacing(8)
        
        # Chart Type section with horizontal segmented buttons
        chart_label = QLabel("Chart Type")
        chart_label.setFont(data_font)
        chart_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        controls_layout.addWidget(chart_label)
        
        # Horizontal segmented radio buttons for chart types
        chart_buttons_widget = QWidget()
        chart_buttons_layout = QHBoxLayout(chart_buttons_widget)
        chart_buttons_layout.setSpacing(0)
        chart_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chart_button_group = QButtonGroup()
        
        self.hist_radio = QRadioButton("Histogram")
        self.hist_radio.setChecked(True)
        self.hist_radio.toggled.connect(lambda checked: self._on_chart_type_changed("histogram") if checked else None)
        self.chart_button_group.addButton(self.hist_radio)
        chart_buttons_layout.addWidget(self.hist_radio)
        
        self.scatter_radio = QRadioButton("Scatter")
        self.scatter_radio.toggled.connect(lambda checked: self._on_chart_type_changed("scatter") if checked else None)
        self.chart_button_group.addButton(self.scatter_radio)
        chart_buttons_layout.addWidget(self.scatter_radio)
        
        self.gt_radio = QRadioButton("GT Curve")
        self.gt_radio.toggled.connect(lambda checked: self._on_chart_type_changed("gt_curve") if checked else None)
        self.chart_button_group.addButton(self.gt_radio)
        chart_buttons_layout.addWidget(self.gt_radio)
        
        self.box_radio = QRadioButton("Box")
        self.box_radio.toggled.connect(lambda checked: self._on_chart_type_changed("box") if checked else None)
        self.chart_button_group.addButton(self.box_radio)
        chart_buttons_layout.addWidget(self.box_radio)
        
        controls_layout.addWidget(chart_buttons_widget)
        
        controls_layout.addSpacing(8)
        
        # Chart Settings section (dynamic - only shown for relevant charts)
        self.settings_label = QLabel("Chart Settings")
        self.settings_label.setFont(data_font)
        self.settings_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        controls_layout.addWidget(self.settings_label)
        
        # Container for dynamic settings
        self.settings_container = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_layout.setSpacing(8)
        
        # Bins setting (for histogram)
        bins_layout = QHBoxLayout()
        bins_layout.setSpacing(8)
        bins_label = QLabel("Bins")
        bins_label.setMinimumWidth(80)
        bins_layout.addWidget(bins_label)
        
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 200)
        self.bins_spin.setValue(30)
        self.bins_spin.setMaximumWidth(100)
        bins_layout.addWidget(self.bins_spin)
        bins_layout.addStretch()
        
        self.bins_widget = QWidget()
        self.bins_widget.setLayout(bins_layout)
        self.settings_layout.addWidget(self.bins_widget)
        
        controls_layout.addWidget(self.settings_container)
        
        controls_layout.addSpacing(12)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {ModernColors.BORDER};")
        controls_layout.addWidget(sep2)
        
        controls_layout.addSpacing(8)
        
        # Bottom buttons (Generate + Export)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        
        self.generate_btn = QPushButton("Generate Plot")
        self.generate_btn.setMinimumHeight(38)
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        buttons_layout.addWidget(self.generate_btn, 1)
        
        self.export_btn = QPushButton("Export ▼")
        self.export_btn.setMinimumHeight(38)
        self.export_btn.setMaximumWidth(120)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.export_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_btn)
        
        controls_layout.addLayout(buttons_layout)
        
        controls_layout.addStretch()
        
        # Right side: Preview + Statistics
        preview_widget = QWidget()
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
        self.placeholder_label = QLabel("Generate a plot to see preview")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 12px;")
        self.preview_frame_layout.addWidget(self.placeholder_label)
        
        preview_layout.addWidget(self.preview_frame, 1)
        
        # Statistics panel
        stats_title = QLabel("Statistics")
        stats_title.setFont(data_font)
        stats_title.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY};")
        preview_layout.addWidget(stats_title)
        
        self.stats_label = QLabel("―")
        self.stats_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
                font-family: 'Consolas', 'Courier New', monospace;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        self.stats_label.setMinimumHeight(60)
        self.stats_label.setWordWrap(True)
        preview_layout.addWidget(self.stats_label)
        
        # Add widgets to splitter
        splitter.addWidget(controls_widget)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 3)  # Controls: 30%
        splitter.setStretchFactor(1, 7)  # Preview: 70%
        
        main_layout.addWidget(splitter)
        
        # Apply modern styles
        self._apply_modern_ui_styles()
        
        # Initialize dynamic settings
        self._update_dynamic_settings()
        
        # Disable generate initially
        self._set_generate_enabled(False)
        
        # Hide stop and close buttons (inherited from BaseAnalysisPanel)
        if hasattr(self, "stop_button"):
            self.stop_button.hide()
        if hasattr(self, "close_button"):
            self.close_button.hide()

    def _apply_modern_ui_styles(self) -> None:
        """Apply modern, professional UI styles."""
        self.setObjectName("ChartsPanel")
        self.setStyleSheet(f"""
            QWidget#ChartsPanel {{
                background-color: {ModernColors.PANEL_BG};
            }}
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
            }}
            QRadioButton {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 11px;
                padding: 8px 12px;
                background-color: {ModernColors.CARD_BG};
                border: 1px solid {ModernColors.BORDER};
                spacing: 4px;
            }}
            QRadioButton:hover {{
                background-color: {ModernColors.CARD_HOVER};
                border-color: {ModernColors.ACCENT_PRIMARY};
            }}
            QRadioButton:checked {{
                background-color: {ModernColors.ACCENT_PRIMARY};
                color: white;
                border-color: {ModernColors.ACCENT_PRIMARY};
                font-weight: bold;
            }}
            QRadioButton::indicator {{
                width: 0px;
                height: 0px;
            }}
        """)

        combo_style = get_combo_box_stylesheet()
        self.data_source_box.setStyleSheet(combo_style)
        self.property_box.setStyleSheet(combo_style)
        self.property2_box.setStyleSheet(combo_style)
        self.bins_spin.setStyleSheet(get_spin_box_stylesheet())

        self.generate_btn.setStyleSheet(get_button_stylesheet("primary"))
        self.export_btn.setStyleSheet(get_button_stylesheet("secondary"))
    
    def _on_chart_type_changed(self, chart_type: str):
        """Handle chart type selection change."""
        self._current_chart_type = chart_type
        self._update_dynamic_settings()
    
    def _update_dynamic_settings(self):
        """Update settings visibility based on selected chart type."""
        # Show/hide bins setting based on chart type
        if self._current_chart_type == "histogram":
            self.bins_widget.setVisible(True)
            self.settings_label.setVisible(True)
            self.settings_container.setVisible(True)
        else:
            self.bins_widget.setVisible(False)
            self.settings_label.setVisible(False)
            self.settings_container.setVisible(False)
    
    def _set_generate_enabled(self, enabled: bool):
        """Enable/disable generate button."""
        if hasattr(self, 'generate_btn'):
            self.generate_btn.setEnabled(enabled)
    
    def _on_generate_clicked(self):
        """Handle Generate Plot button click."""
        # Check for comparison mode
        if self._comparison_mode:
            self._run_comparison_chart()
            return

        if self._current_chart_type == "histogram":
            self._plot_histogram()
        elif self._current_chart_type == "scatter":
            self._plot_scatter()
        elif self._current_chart_type == "gt_curve":
            self._plot_grade_tonnage()
        elif self._current_chart_type == "box":
            self._plot_boxplot()
    
    def _on_export_clicked(self):
        """Handle Export button click - show dropdown menu."""
        if not self.current_figure:
            return
        
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
    
    def set_block_model(self, block_model):
        """Set the block model for charting."""
        # Use _update_block_model which handles both BlockModel and DataFrame
        self._update_block_model(block_model)
    
    def set_grid_data(self, grid, layer_name: str = "Grid"):
        """
        Set PyVista grid data for charting (for SGSIM results, kriging, etc.).
        
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
            
            # Update property selectors
            self.property_box.clear()
            self.property_box.addItems(numeric_props)
            
            self.property2_box.clear()
            self.property2_box.addItem("Optional")
            self.property2_box.addItems(numeric_props)
            
            if numeric_props:
                self.active_property = numeric_props[0]
            
            # Enable generate button
            if MATPLOTLIB_AVAILABLE and numeric_props:
                self._set_generate_enabled(True)
            
            logger.info(f"Set grid data '{layer_name}' with {len(numeric_props)} properties in Charts panel")
            
        except Exception as e:
            logger.error(f"Error setting grid data in Charts panel: {e}", exc_info=True)
            raise
    
    def _on_property_changed(self, property_name: str):
        """Handle property selection change."""
        if property_name:
            self.active_property = property_name
            # Update statistics if we already have a plot  
            if self.current_canvas:
                self._update_statistics()
    
    def _get_property_data(self, property_name: str):
        """
        Get property data from either block model or drillhole data.
        
        Returns:
            numpy array of property values, or None if not found
        """
        if self._data_mode == "drillhole" and self._drillhole_df is not None:
            if property_name in self._drillhole_df.columns:
                return self._drillhole_df[property_name].values
            return None
        elif self._data_mode == "block_model" and self.current_model is not None:
            # Handle both BlockModel objects and DataFrames
            if hasattr(self.current_model, 'get_property'):
                # BlockModel API
                return self.current_model.get_property(property_name)
            elif hasattr(self.current_model, 'columns'):
                # DataFrame
                if property_name in self.current_model.columns:
                    return self.current_model[property_name].values
            return None
        return None
    
    def _has_data(self) -> bool:
        """Check if any data (block model or drillhole) is available."""
        if self._data_mode == "drillhole" and self._drillhole_df is not None:
            return True
        elif self._data_mode == "block_model" and self.current_model is not None:
            return True
        return False
    
    def _plot_histogram(self):
        """Plot histogram of active property in the preview pane."""
        if not self.active_property or not self._has_data():
            QMessageBox.warning(self, "No Data", 
                                "Please load drillhole data or a block model first.")
            return
        
        try:
            prop_data = self._get_property_data(self.active_property)
            if prop_data is None:
                return
            
            valid_data = prop_data[np.isfinite(prop_data)]
            
            # Create figure with dark theme
            self.current_figure = Figure(figsize=(8, 6), facecolor=ModernColors.CARD_BG)
            ax = self.current_figure.add_subplot(111)
            ax.set_facecolor(ModernColors.CARD_BG)
            
            # Plot histogram
            n, bins, patches = ax.hist(valid_data, bins=self.bins_spin.value(), 
                                       edgecolor='black', alpha=0.8, color=ModernColors.ACCENT_PRIMARY)
            ax.set_xlabel(self.active_property, fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_ylabel('Frequency', fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_title(f'Histogram: {self.active_property}', fontsize=12, 
                        fontweight='bold', color=ModernColors.TEXT_PRIMARY, pad=15)
            ax.grid(True, alpha=0.2, color=ModernColors.TEXT_SECONDARY)
            ax.tick_params(colors=ModernColors.TEXT_PRIMARY, labelsize=9)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_edgecolor(ModernColors.BORDER)
            
            self.current_figure.tight_layout()
            
            self._show_plot_in_preview()
            self._update_statistics()
            
            logger.info(f"Plotted histogram for {self.active_property}")
            
        except Exception as e:
            logger.error(f"Error plotting histogram: {e}")
            QMessageBox.warning(self, "Plot Error", f"Failed to create histogram:\n{e}")
    
    def _plot_scatter(self):
        """Plot scatter plot between two properties in the preview pane."""
        if not self._has_data():
            QMessageBox.warning(self, "No Data", 
                                "Please load drillhole data or a block model first.")
            return
        
        try:
            prop1 = self.property_box.currentText()
            prop2 = self.property2_box.currentText()
            
            if not prop1 or not prop2 or prop2 == "Optional":
                QMessageBox.warning(self, "Select Properties", 
                                   "Please select both primary and secondary properties.")
                return
            
            data1 = self._get_property_data(prop1)
            data2 = self._get_property_data(prop2)
            
            if data1 is None or data2 is None:
                return
            
            # Create figure with dark theme
            self.current_figure = Figure(figsize=(8, 6), facecolor=ModernColors.CARD_BG)
            ax = self.current_figure.add_subplot(111)
            ax.set_facecolor(ModernColors.CARD_BG)
            
            # Scatter plot
            ax.scatter(data1, data2, alpha=0.5, s=20, c=ModernColors.ACCENT_PRIMARY, 
                      edgecolors='black', linewidth=0.3)
            ax.set_xlabel(prop1, fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_ylabel(prop2, fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_title(f'Scatter: {prop1} vs {prop2}', fontsize=12, 
                        fontweight='bold', color=ModernColors.TEXT_PRIMARY, pad=15)
            ax.grid(True, alpha=0.2, color=ModernColors.TEXT_SECONDARY)
            ax.tick_params(colors=ModernColors.TEXT_PRIMARY, labelsize=9)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_edgecolor(ModernColors.BORDER)
            
            self.current_figure.tight_layout()
            
            self._show_plot_in_preview()
            self._update_statistics()
            
            logger.info(f"Plotted scatter: {prop1} vs {prop2}")
            
        except Exception as e:
            logger.error(f"Error plotting scatter: {e}")
            QMessageBox.warning(self, "Plot Error", f"Failed to create scatter plot:\n{e}")
    
    def _plot_grade_tonnage(self):
        """Plot grade-tonnage curve in the preview pane."""
        if not self.active_property or not self._has_data():
            QMessageBox.warning(self, "No Data", 
                                "Please load drillhole data or a block model first.")
            return
        
        try:
            prop_data = self._get_property_data(self.active_property)
            if prop_data is None:
                return
            
            # Calculate tonnage (assume uniform block size)
            sorted_data = np.sort(prop_data)[::-1]  # Descending order
            cumulative_tonnage = np.arange(1, len(sorted_data) + 1)
            cumulative_grade = np.cumsum(sorted_data) / cumulative_tonnage
            
            # Create figure with two y-axes and dark theme
            self.current_figure = Figure(figsize=(8, 6), facecolor=ModernColors.CARD_BG)
            ax1 = self.current_figure.add_subplot(111)
            ax1.set_facecolor(ModernColors.CARD_BG)
            ax2 = ax1.twinx()
            
            # Plot tonnage
            ax1.plot(sorted_data, cumulative_tonnage, color=ModernColors.ACCENT_PRIMARY, 
                    linewidth=2, label='Tonnage')
            ax1.set_xlabel(f'Cut-off Grade ({self.active_property})', 
                          fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax1.set_ylabel('Cumulative Tonnage (blocks)', 
                          fontsize=11, color=ModernColors.ACCENT_PRIMARY)
            ax1.tick_params(axis='y', labelcolor=ModernColors.ACCENT_PRIMARY, labelsize=9)
            ax1.tick_params(axis='x', colors=ModernColors.TEXT_PRIMARY, labelsize=9)
            ax1.grid(True, alpha=0.2, color=ModernColors.TEXT_SECONDARY)
            
            # Plot grade
            ax2.plot(sorted_data, cumulative_grade, color='#E74C3C', linewidth=2, label='Grade')
            ax2.set_ylabel(f'Average Grade ({self.active_property})', 
                          fontsize=11, color='#E74C3C')
            ax2.tick_params(axis='y', labelcolor='#E74C3C', labelsize=9)
            
            ax1.set_title(f'Grade-Tonnage: {self.active_property}', fontsize=12, 
                         fontweight='bold', color=ModernColors.TEXT_PRIMARY, pad=15)
            
            # Style spines
            for spine in ax1.spines.values():
                spine.set_edgecolor(ModernColors.BORDER)
            for spine in ax2.spines.values():
                spine.set_edgecolor(ModernColors.BORDER)
            
            self.current_figure.tight_layout()
            
            self._show_plot_in_preview()
            self._update_statistics()
            
            logger.info(f"Plotted grade-tonnage for {self.active_property}")
            
        except Exception as e:
            logger.error(f"Error plotting grade-tonnage: {e}")
            QMessageBox.warning(self, "Plot Error", f"Failed to create grade-tonnage curve:\n{e}")
    
    def _plot_boxplot(self):
        """Plot box plot of active property in the preview pane."""
        if not self.active_property or not self._has_data():
            QMessageBox.warning(self, "No Data", 
                                "Please load drillhole data or a block model first.")
            return
        
        try:
            prop_data = self._get_property_data(self.active_property)
            if prop_data is None:
                return
            
            valid_data = prop_data[np.isfinite(prop_data)]
            
            # Create figure with dark theme
            self.current_figure = Figure(figsize=(8, 6), facecolor=ModernColors.CARD_BG)
            ax = self.current_figure.add_subplot(111)
            ax.set_facecolor(ModernColors.CARD_BG)
            
            # Box plot
            bp = ax.boxplot([valid_data], vert=True, patch_artist=True, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor(ModernColors.ACCENT_PRIMARY)
                patch.set_edgecolor(ModernColors.TEXT_PRIMARY)
            for whisker in bp['whiskers']:
                whisker.set_color(ModernColors.TEXT_PRIMARY)
            for cap in bp['caps']:
                cap.set_color(ModernColors.TEXT_PRIMARY)
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(2)
            
            ax.set_ylabel(self.active_property, fontsize=11, color=ModernColors.TEXT_PRIMARY)
            ax.set_title(f'Box Plot: {self.active_property}', fontsize=12, 
                        fontweight='bold', color=ModernColors.TEXT_PRIMARY, pad=15)
            ax.set_xticklabels([self.active_property])
            ax.tick_params(colors=ModernColors.TEXT_PRIMARY, labelsize=9)
            ax.grid(True, alpha=0.2, axis='y', color=ModernColors.TEXT_SECONDARY)
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_edgecolor(ModernColors.BORDER)
            
            self.current_figure.tight_layout()
            
            self._show_plot_in_preview()
            self._update_statistics()
            
            logger.info(f"Plotted box plot for {self.active_property}")
            
        except Exception as e:
            logger.error(f"Error plotting box plot: {e}")
            QMessageBox.warning(self, "Plot Error", f"Failed to create box plot:\n{e}")
    
    def _show_plot_in_preview(self):
        """Show the current figure in the preview pane."""
        if not self.current_figure:
            return
        
        # Remove old canvas if exists
        if self.current_canvas:
            self.preview_frame_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None
        
        # Remove placeholder
        if self.placeholder_label:
            self.placeholder_label.setVisible(False)
        
        # Create new canvas
        self.current_canvas = FigureCanvasQTAgg(self.current_figure)
        self.preview_frame_layout.addWidget(self.current_canvas)
        
        # Enable export button
        self.export_btn.setEnabled(True)
    
    def _update_statistics(self):
        """Update the statistics panel with current data."""
        if not self.active_property or not self._has_data():
            self.stats_label.setText("―")
            return
        
        try:
            prop_data = self._get_property_data(self.active_property)
            if prop_data is None:
                self.stats_label.setText("―")
                return
            
            valid_data = prop_data[np.isfinite(prop_data)]
            
            n = len(valid_data)
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            p10 = np.percentile(valid_data, 10)
            p50 = np.percentile(valid_data, 50)
            p90 = np.percentile(valid_data, 90)
            cv = (std / mean) if mean != 0 else 0
            
            stats_text = (
                f"N: {n:,}   "
                f"Mean: {mean:.2f}   "
                f"SD: {std:.2f}   "
                f"CV: {cv:.2f}\\n"
                f"Min: {min_val:.2f}   "
                f"Max: {max_val:.2f}   "
                f"P10: {p10:.2f}   "
                f"P50: {p50:.2f}   "
                f"P90: {p90:.2f}"
            )
            
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
            self.stats_label.setText("Error calculating statistics")
    
    def _export_plot(self, format_type: str):
        """Export current plot to file."""
        if not self.current_figure:
            return
        
        format_map = {
            "png": ("PNG Files (*.png)", "png"),
            "pdf": ("PDF Files (*.pdf)", "pdf"),
            "svg": ("SVG Files (*.svg)", "svg"),
        }
        
        filter_str, ext = format_map.get(format_type, ("PNG Files (*.png)", "png"))
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export Plot as {ext.upper()}", f"plot.{ext}", filter_str
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight', 
                                           facecolor=self.current_figure.get_facecolor())
                QMessageBox.information(self, "Success", f"Plot exported to {filename}")
                logger.info(f"Exported plot to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot:\\n{e}")
                logger.error(f"Export error: {e}")
    
    def _export_data_csv(self):
        """Export current plot data to CSV."""
        if not self.active_property or not self._has_data():
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data as CSV", "data.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                prop_data = self._get_property_data(self.active_property)
                if prop_data is None:
                    return
                
                import pandas as pd
                df = pd.DataFrame({self.active_property: prop_data})
                df.to_csv(filename, index=False)
                
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
                logger.info(f"Exported data to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data:\\n{e}")
                logger.error(f"Export error: {e}")

    # ------------------------------------------------------------------
    # Comparison mode methods
    # ------------------------------------------------------------------

    def _on_comparison_mode_changed(self, enabled: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = enabled
        # Hide single source selector when in comparison mode (but NOT the parent widget!)
        self.data_source_box.setVisible(not enabled)
        logger.info(f"ChartsPanel: Comparison mode {'enabled' if enabled else 'disabled'}")

        if enabled:
            self._update_comparison_sources()

    def _on_comparison_sources_changed(self, selected_keys: List[str]):
        """Handle comparison source selection changes."""
        logger.info(f"ChartsPanel: Comparison sources changed: {selected_keys}")

        # Populate properties from selected sources
        if selected_keys:
            self._populate_comparison_properties(selected_keys)

        if len(selected_keys) >= 2:
            self._generate_btn_enabled = True
            if hasattr(self, 'generate_btn'):
                self.generate_btn.setEnabled(True)

    def _populate_comparison_properties(self, selected_keys: List[str]):
        """Populate property dropdown with properties from selected sources."""
        import pandas as pd

        all_properties = set()
        logger.info(f"ChartsPanel: _populate_comparison_properties called with: {selected_keys}")
        logger.info(f"ChartsPanel: Available _block_model_sources: {list(self._block_model_sources.keys())}")

        for source_key in selected_keys:
            if source_key == "drillhole":
                df = self._stored_drillhole_df
                if df is not None:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH'):
                            all_properties.add(col)

            elif source_key == "block_model":
                bm = self._stored_block_model
                if bm is not None:
                    logger.debug(f"ChartsPanel: Block model type: {type(bm).__name__}")
                    # Check if it's a DataFrame first
                    if isinstance(bm, pd.DataFrame):
                        for col in bm.columns:
                            if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG'):
                                try:
                                    if pd.api.types.is_numeric_dtype(bm[col]):
                                        all_properties.add(col)
                                except:
                                    all_properties.add(col)
                        logger.debug(f"ChartsPanel: Block model DataFrame columns: {list(bm.columns)[:10]}")
                    else:
                        # BlockModel class
                        if hasattr(bm, 'properties') and bm.properties:
                            all_properties.update(bm.properties.keys())
                            logger.debug(f"ChartsPanel: Block model properties: {list(bm.properties.keys())[:5]}")
                        if hasattr(bm, 'to_dataframe'):
                            try:
                                df = bm.to_dataframe()
                                for col in df.columns:
                                    if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            all_properties.add(col)
                                logger.debug(f"ChartsPanel: Block model DataFrame columns: {list(df.columns)[:10]}")
                            except Exception as e:
                                logger.debug(f"ChartsPanel: to_dataframe failed: {e}")

            elif source_key == "classified_block_model":
                bm = getattr(self, '_stored_classified_block_model', None)
                if bm is not None:
                    if isinstance(bm, pd.DataFrame):
                        for col in bm.columns:
                            if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG'):
                                try:
                                    if pd.api.types.is_numeric_dtype(bm[col]):
                                        all_properties.add(col)
                                except:
                                    all_properties.add(col)
                    else:
                        if hasattr(bm, 'properties') and bm.properties:
                            all_properties.update(bm.properties.keys())
                        if hasattr(bm, 'to_dataframe'):
                            try:
                                df = bm.to_dataframe()
                                for col in df.columns:
                                    if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            all_properties.add(col)
                            except:
                                pass

            elif source_key in self._block_model_sources:
                source_info = self._block_model_sources[source_key]
                prop = source_info.get('property')
                logger.debug(f"ChartsPanel: Found source '{source_key}' in _block_model_sources, prop={prop}")
                if prop:
                    all_properties.add(prop)
                df = source_info.get('df')
                if df is not None:
                    logger.debug(f"ChartsPanel: Source '{source_key}' has df with columns: {list(df.columns)[:8]}")
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z'):
                            all_properties.add(col)
                else:
                    logger.warning(f"ChartsPanel: Source '{source_key}' found but df is None")

            else:
                logger.warning(f"ChartsPanel: Source '{source_key}' not found in any storage location")

        logger.info(f"ChartsPanel: Collected {len(all_properties)} properties: {sorted(all_properties)[:10]}")

        # Update primary property dropdown
        current_prop = self.property_box.currentText()
        self.property_box.blockSignals(True)
        self.property_box.clear()

        sorted_props = sorted(all_properties)
        self.property_box.addItems(sorted_props)

        # Try to restore previous selection or select first grade-like property
        if current_prop and current_prop in sorted_props:
            self.property_box.setCurrentText(current_prop)
        else:
            for prop in sorted_props:
                if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')):
                    self.property_box.setCurrentText(prop)
                    break

        self.property_box.blockSignals(False)

        # Also update secondary property dropdown
        if hasattr(self, 'property2_box'):
            current_prop2 = self.property2_box.currentText()
            self.property2_box.blockSignals(True)
            self.property2_box.clear()
            self.property2_box.addItem("Optional")  # Keep the optional item
            self.property2_box.addItems(sorted_props)

            # Try to restore previous selection
            if current_prop2 and current_prop2 in sorted_props:
                self.property2_box.setCurrentText(current_prop2)

            self.property2_box.blockSignals(False)

        logger.info(f"ChartsPanel: Populated {len(sorted_props)} properties for comparison: {sorted_props[:5]}")

    def _update_comparison_sources(self):
        """Update the comparison widget with available sources."""
        sources = {}

        # Add drillhole data
        if "drillhole" in self._available_sources and self._stored_drillhole_df is not None:
            sources["drillhole"] = {
                'display_name': f"Drillhole ({self._stored_data_source_type})",
                'block_count': len(self._stored_drillhole_df),
                'df': self._stored_drillhole_df
            }

        # Add block model
        if "block_model" in self._available_sources and self._stored_block_model is not None:
            bm = self._stored_block_model
            count = len(bm) if hasattr(bm, '__len__') else 0
            sources["block_model"] = {
                'display_name': "Block Model",
                'block_count': count,
            }

        # Add classified block model
        if "classified_block_model" in self._available_sources:
            bm = getattr(self, '_stored_classified_block_model', None)
            if bm is not None:
                count = len(bm) if hasattr(bm, '__len__') else 0
                sources["classified_block_model"] = {
                    'display_name': "Classified Block Model",
                    'block_count': count,
                }

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

        logger.info(f"ChartsPanel._update_comparison_sources: Found {len(sources)} sources: {list(sources.keys())}")
        self._source_selection_widget.update_sources(sources)

    def _get_comparison_data(self, source_key: str, property_name: str = None):
        """Get data array for a given source.

        Returns tuple of (data_array, actual_property_name)
        """
        import pandas as pd

        if source_key == "drillhole":
            df = self._stored_drillhole_df
            if df is None:
                logger.debug(f"ChartsPanel: drillhole source has no data")
                return None, None
            return self._find_property_in_df(df, property_name)

        if source_key == "block_model":
            bm = self._stored_block_model
            if bm is None:
                logger.debug(f"ChartsPanel: block_model source has no data")
                return None, None
            return self._find_property_in_model(bm, property_name)

        if source_key == "classified_block_model":
            bm = getattr(self, '_stored_classified_block_model', None)
            if bm is None:
                logger.debug(f"ChartsPanel: classified_block_model source has no data")
                return None, None
            return self._find_property_in_model(bm, property_name)

        if source_key in self._block_model_sources:
            source_info = self._block_model_sources[source_key]
            df = source_info.get('df')
            prop = source_info.get('property')  # SGSIM sources have specific property
            logger.debug(f"ChartsPanel: Found '{source_key}' in _block_model_sources, df={df is not None}, prop={prop}")
            if df is not None:
                logger.debug(f"ChartsPanel: DataFrame columns for '{source_key}': {list(df.columns)[:10]}")
                # Use the built-in property if available
                if prop and prop in df.columns:
                    logger.debug(f"ChartsPanel: Using built-in property {prop} for {source_key}")
                    return df[prop].values, prop
                # Otherwise try to find matching property
                logger.debug(f"ChartsPanel: Built-in property '{prop}' not found, trying flexible match with '{property_name}'")
                return self._find_property_in_df(df, property_name)
            logger.warning(f"ChartsPanel: Source '{source_key}' found in _block_model_sources but df is None")
            return None, None

        # Check widget sources as fallback
        logger.debug(f"ChartsPanel: Source '{source_key}' not in _block_model_sources, checking widget sources")
        source_info = self._source_selection_widget._sources.get(source_key, {})
        df = source_info.get('df')
        if df is not None:
            prop = source_info.get('property')
            logger.debug(f"ChartsPanel: Found '{source_key}' in widget sources, prop={prop}")
            if prop and prop in df.columns:
                return df[prop].values, prop
            return self._find_property_in_df(df, property_name)

        logger.warning(f"ChartsPanel: Unknown source '{source_key}' - not found anywhere")
        return None, None

    def _find_property_in_df(self, df, property_name: str):
        """Find property in DataFrame with flexible matching."""
        import pandas as pd

        if df is None:
            return None, None

        # If property_name is provided, try exact and flexible matching first
        if property_name:
            # Strategy 1: Exact match
            if property_name in df.columns:
                return df[property_name].values, property_name

            # Strategy 2: Property name contains column name (e.g., "Grade_MEAN" -> "Grade")
            for col in df.columns:
                if col.upper() in property_name.upper() and col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                    logger.debug(f"ChartsPanel: Matched {property_name} -> {col} (property contains column)")
                    return df[col].values, col

            # Strategy 3: Column name contains property name
            for col in df.columns:
                if property_name.upper() in col.upper():
                    logger.debug(f"ChartsPanel: Matched {property_name} -> {col} (column contains property)")
                    return df[col].values, col

            # Strategy 4: Base name match (e.g., "Grade_MEAN" -> "Grade")
            base_name = property_name.split('_')[0] if '_' in property_name else property_name
            for col in df.columns:
                col_base = col.split('_')[0] if '_' in col else col
                if base_name.upper() == col_base.upper() and col.upper() not in ('X', 'Y', 'Z'):
                    logger.debug(f"ChartsPanel: Matched {property_name} -> {col} (base name match)")
                    return df[col].values, col

        # Strategy 5: Find any grade-like property (works even when property_name is None)
        for col in df.columns:
            if any(k in col.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')) and col.upper() not in ('X', 'Y', 'Z'):
                logger.debug(f"ChartsPanel: Using grade-like property {col}")
                return df[col].values, col

        logger.warning(f"ChartsPanel: No matching property for '{property_name}' in DataFrame. Available: {list(df.columns)[:10]}")
        return None, None

    def _find_property_in_model(self, bm, property_name: str):
        """Find property in block model with flexible matching."""
        import pandas as pd

        if bm is None:
            return None, None

        # Check if it's a DataFrame first
        if isinstance(bm, pd.DataFrame):
            return self._find_property_in_df(bm, property_name)

        # Get available properties from BlockModel
        available_props = []
        if hasattr(bm, 'properties') and bm.properties:
            available_props = list(bm.properties.keys())
        elif hasattr(bm, 'to_dataframe'):
            try:
                df = bm.to_dataframe()
                return self._find_property_in_df(df, property_name)
            except:
                pass

        if not available_props:
            logger.debug(f"ChartsPanel: Block model has no properties")
            return None, None

        # Strategy 1: Exact match
        if property_name and property_name in available_props:
            return bm.properties[property_name], property_name

        # Strategy 2: Property name contains column name
        if property_name:
            for prop in available_props:
                if prop.upper() in property_name.upper() and prop.upper() not in ('X', 'Y', 'Z'):
                    logger.debug(f"ChartsPanel: Matched {property_name} -> {prop} (property contains)")
                    return bm.properties[prop], prop

        # Strategy 3: Column contains property name
        if property_name:
            for prop in available_props:
                if property_name.upper() in prop.upper():
                    logger.debug(f"ChartsPanel: Matched {property_name} -> {prop} (contains)")
                    return bm.properties[prop], prop

        # Strategy 4: Base name match
        if property_name:
            base_name = property_name.split('_')[0] if '_' in property_name else property_name
            for prop in available_props:
                prop_base = prop.split('_')[0] if '_' in prop else prop
                if base_name.upper() == prop_base.upper() and prop.upper() not in ('X', 'Y', 'Z'):
                    logger.debug(f"ChartsPanel: Matched {property_name} -> {prop} (base name)")
                    return bm.properties[prop], prop

        # Strategy 5: Find any grade-like property
        for prop in available_props:
            if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')) and prop.upper() not in ('X', 'Y', 'Z'):
                logger.debug(f"ChartsPanel: Using grade-like property {prop}")
                return bm.properties[prop], prop

        logger.warning(f"ChartsPanel: No matching property for '{property_name}' in block model. Available: {available_props[:10]}")
        return None, None

    def _run_comparison_chart(self):
        """Generate comparison chart for selected sources."""
        if not self._comparison_mode:
            return

        selected_keys = self._source_selection_widget.get_selected_sources()
        if len(selected_keys) < 2:
            return

        property_name = self.property_box.currentText() if self.property_box.currentText() else None
        logger.info(f"ChartsPanel: Running comparison for {selected_keys} with property '{property_name}'")
        logger.info(f"ChartsPanel: Available block_model_sources: {list(self._block_model_sources.keys())}")

        # Collect data for each source
        self._comparison_data = {}
        failed_sources = []
        for source_key in selected_keys:
            logger.debug(f"ChartsPanel: Getting data for source '{source_key}'")
            data, actual_prop = self._get_comparison_data(source_key, property_name)
            if data is not None:
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    source_info = self._source_selection_widget._sources.get(source_key, {})
                    self._comparison_data[source_key] = {
                        'data': valid_data,
                        'display_name': source_info.get('display_name', source_key),
                        'property': actual_prop
                    }
                    logger.info(f"ChartsPanel: Got {len(valid_data)} values for '{source_key}' (property: {actual_prop})")
                else:
                    failed_sources.append(f"{source_key} (no valid data)")
                    logger.warning(f"ChartsPanel: Source '{source_key}' has no valid (non-NaN) data")
            else:
                failed_sources.append(f"{source_key} (data not found)")
                logger.warning(f"ChartsPanel: Could not get data for source '{source_key}'")

        if len(self._comparison_data) < 2:
            msg = f"Could not get data for at least 2 sources.\n\nSuccessful: {list(self._comparison_data.keys())}\nFailed: {failed_sources}"
            QMessageBox.warning(self, "Error", msg)
            return

        # Generate appropriate comparison chart
        if self._current_chart_type == "histogram":
            self._plot_comparison_histogram()
        elif self._current_chart_type == "box":
            self._plot_comparison_boxplot()
        else:
            self._plot_comparison_histogram()  # Default to histogram

    def _plot_comparison_histogram(self):
        """Plot overlaid histograms for comparison."""
        if not MATPLOTLIB_AVAILABLE or not self._comparison_data:
            return

        # Create figure
        self.current_figure = Figure(figsize=(10, 6), facecolor=ModernColors.CARD_BG)
        ax = self.current_figure.add_subplot(111)
        ax.set_facecolor(ModernColors.CARD_BG)

        # Determine common bin edges
        all_data = np.concatenate([d['data'] for d in self._comparison_data.values()])
        bins = np.linspace(all_data.min(), all_data.max(), self.bins_spin.value())

        # Plot each source
        for i, (source_key, result) in enumerate(self._comparison_data.items()):
            data = result['data']
            display_name = result['display_name']
            color = ComparisonColors.get_color(i)

            ax.hist(data, bins=bins, alpha=0.5, color=color, label=display_name,
                   edgecolor=color, linewidth=1.2)

        ax.set_title("Histogram Comparison", color=ModernColors.TEXT_PRIMARY, fontweight='bold')
        ax.set_xlabel("Value", color=ModernColors.TEXT_PRIMARY)
        ax.set_ylabel("Frequency", color=ModernColors.TEXT_PRIMARY)
        ax.tick_params(colors=ModernColors.TEXT_PRIMARY)
        ax.grid(True, linestyle='--', alpha=0.2, color=ModernColors.TEXT_SECONDARY)

        for spine in ax.spines.values():
            spine.set_edgecolor(ModernColors.BORDER)

        legend = ax.legend(loc='upper right', frameon=True,
                          facecolor=ModernColors.CARD_BG, edgecolor=ModernColors.BORDER,
                          fontsize=9, labelcolor=ModernColors.TEXT_PRIMARY)
        legend.get_frame().set_alpha(0.9)

        self.current_figure.tight_layout()
        self._show_plot_in_preview()

        # Update statistics
        stats_parts = []
        for source_key, result in self._comparison_data.items():
            data = result['data']
            name = result['display_name'][:15]
            stats_parts.append(f"{name}: N={len(data):,} Mean={np.mean(data):.3f}")
        self.stats_label.setText(" | ".join(stats_parts))

        self.export_btn.setEnabled(True)

    def _plot_comparison_boxplot(self):
        """Plot side-by-side boxplots for comparison."""
        if not MATPLOTLIB_AVAILABLE or not self._comparison_data:
            return

        # Create figure
        self.current_figure = Figure(figsize=(10, 6), facecolor=ModernColors.CARD_BG)
        ax = self.current_figure.add_subplot(111)
        ax.set_facecolor(ModernColors.CARD_BG)

        # Prepare data
        data_list = []
        labels = []
        colors = []

        for i, (source_key, result) in enumerate(self._comparison_data.items()):
            data_list.append(result['data'])
            labels.append(result['display_name'][:20])
            colors.append(ComparisonColors.get_color(i))

        bp = ax.boxplot(data_list, labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color(ModernColors.TEXT_PRIMARY)

        ax.set_title("Box Plot Comparison", color=ModernColors.TEXT_PRIMARY, fontweight='bold')
        ax.set_ylabel("Value", color=ModernColors.TEXT_PRIMARY)
        ax.tick_params(colors=ModernColors.TEXT_PRIMARY)
        ax.grid(True, linestyle='--', alpha=0.2, axis='y', color=ModernColors.TEXT_SECONDARY)

        for spine in ax.spines.values():
            spine.set_edgecolor(ModernColors.BORDER)

        self.current_figure.tight_layout()
        self._show_plot_in_preview()

        self.export_btn.setEnabled(True)

    def clear(self):
        """Clear the panel."""
        self.current_model = None
        self.active_property = None
        self.current_figure = None

        if self.current_canvas:
            self.preview_frame_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None

        if self.placeholder_label:
            self.placeholder_label.setVisible(True)

        self.property_box.clear()
        self.property2_box.clear()
        self.export_btn.setEnabled(False)
        self._set_generate_enabled(False)
        self.stats_label.setText("―")

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        self.clear()
        super().clear_panel()
        logger.info("ChartsPanel: Panel fully cleared")

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors
        self.setStyleSheet(get_analysis_panel_stylesheet())










