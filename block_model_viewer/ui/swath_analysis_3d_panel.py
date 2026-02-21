"""
3D Swath Analysis Panel for Geostatistical Estimation Reliability Assessment

Performs swath analysis along X, Y, Z axes or full 3D volumes to quantify
smoothing, bias, and continuity distortion in block model estimates by comparing
them with input composites across moving volumetric windows.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QRadioButton,
    QCheckBox, QSpinBox, QSplitter, QTabWidget, QHeaderView,
    QProgressDialog, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
from scipy.spatial import cKDTree

# Matplotlib backend is set in main.py
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..models.block_model import BlockModel
from .comparison_utils import ComparisonColors, SourceSelectionWidget, create_comparison_legend

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class SwathMode(Enum):
    """Swath analysis modes."""
    X_AXIS = "X-Axis Swath"
    Y_AXIS = "Y-Axis Swath"
    Z_AXIS = "Z-Axis Swath"
    FULL_3D = "Full 3D Cube"


class SwathAnalysis3DPanel(QWidget):
    """
    3D Swath Analysis Panel for assessing estimation reliability.
    
    Compares block model estimates against composite samples to quantify:
    - Estimation bias
    - Smoothing effects
    - Continuity distortion
    """
    
    # Signals
    analysis_completed = pyqtSignal(dict)
    swath_selected = pyqtSignal(int)  # Swath ID for 3D visualization
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.block_model: Optional[BlockModel] = None
        self.block_model_df: Optional[pd.DataFrame] = None  # For DataFrame storage
        self.composite_data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.swath_mode: SwathMode = SwathMode.X_AXIS
        self.current_grade_field: Optional[str] = None
        self.controller = None  # Optional controller reference

        # Spatial trees for efficient queries
        self._block_tree: Optional[cKDTree] = None
        self._composite_tree: Optional[cKDTree] = None

        # Separate storage for different block model sources
        self._stored_block_model = None
        self._stored_classified_block_model = None
        self._stored_sgsim_df = None  # SGSIM results as DataFrame
        self._available_sources: List[str] = []
        self._current_source: str = "block_model"

        # Storage for individual SGSIM statistics (Mean, P10, P50, P90, Std Dev)
        self._block_model_sources: Dict[str, Any] = {}

        # Comparison mode state
        self._comparison_mode: bool = False
        self._comparison_results: Dict[str, Any] = {}

        self._setup_ui()
        self._connect_to_registry()
        logger.info("Initialized 3D Swath Analysis Panel")
    


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
    def _connect_to_registry(self):
        """Connect to DataRegistry signals for automatic data updates."""
        try:
            from ..core.data_registry import DataRegistry
            registry = DataRegistry.instance()
            
            # Connect to block model signals
            registry.blockModelLoaded.connect(self._on_block_model_loaded)
            registry.blockModelGenerated.connect(self._on_block_model_generated)
            registry.blockModelClassified.connect(self._on_block_model_classified)
            
            # Connect to drillhole data signal for composites
            registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)

            # Connect to SGSIM results signal
            if hasattr(registry, 'sgsimResultsLoaded'):
                registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
                logger.info("SwathAnalysis3DPanel: Connected to sgsimResultsLoaded signal")

            # Load any existing data from registry
            self._refresh_available_data()
            
            logger.info("SwathAnalysis3DPanel: Connected to DataRegistry")
        except Exception as e:
            logger.warning(f"SwathAnalysis3DPanel: Could not connect to DataRegistry: {e}")
    
    def _refresh_available_data(self):
        """Fetch existing data from registry."""
        try:
            from ..core.data_registry import DataRegistry
            registry = DataRegistry.instance()

            # Check for regular block model
            try:
                block_model = registry.get_data("block_model", copy_data=False)
                if block_model is not None:
                    self._stored_block_model = block_model
                    if "block_model" not in self._available_sources:
                        self._available_sources.append("block_model")
                    logger.info("SwathAnalysis3DPanel: Loaded block_model from registry")
            except KeyError:
                pass

            # Check for classified block model (SEPARATE from regular block model)
            try:
                classified_model = registry.get_data("classified_block_model", copy_data=False)
                if classified_model is not None:
                    self._stored_classified_block_model = classified_model
                    if "classified_block_model" not in self._available_sources:
                        self._available_sources.append("classified_block_model")
                    logger.info("SwathAnalysis3DPanel: Loaded classified_block_model from registry")
            except KeyError:
                pass

            # Check for SGSIM results
            try:
                if hasattr(registry, 'get_sgsim_results'):
                    sgsim = registry.get_sgsim_results()
                    if sgsim is not None:
                        self._on_sgsim_loaded(sgsim)
                        logger.info("SwathAnalysis3DPanel: Loaded SGSIM results from registry")
            except Exception:
                pass

            # Update data source selector
            self._update_data_source_selector()

            # Set the first available block model as active
            if self._stored_classified_block_model is not None:
                self.set_block_model(self._stored_classified_block_model)
            elif self._stored_block_model is not None:
                self.set_block_model(self._stored_block_model)

            # Check for drillhole composites
            drillhole_data = registry.get_data("drillhole_data", copy_data=False)
            if drillhole_data is not None:
                self._on_drillhole_data_loaded(drillhole_data)

        except Exception as e:
            logger.debug(f"SwathAnalysis3DPanel: Could not refresh from registry: {e}")
    
    def _on_block_model_loaded(self, block_model):
        """Handle block model loaded from DataRegistry."""
        logger.info("SwathAnalysis3DPanel: Block model received from registry")
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        # Only set if no classified model is currently active
        if self._current_source == "block_model":
            self.set_block_model(block_model)

    def _on_block_model_generated(self, block_model):
        """Handle block model generated (simulation/estimation)."""
        logger.info("SwathAnalysis3DPanel: Generated block model received from registry")
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        if self._current_source == "block_model":
            self.set_block_model(block_model)

    def _on_block_model_classified(self, block_model):
        """Handle classified block model from DataRegistry."""
        logger.info("SwathAnalysis3DPanel: Classified block model received from registry")
        # Store as SEPARATE classified block model (don't overwrite regular block model)
        self._stored_classified_block_model = block_model
        if "classified_block_model" not in self._available_sources:
            self._available_sources.append("classified_block_model")
        self._update_data_source_selector()
        # Auto-switch to classified model when it becomes available
        self._current_source = "classified_block_model"
        if hasattr(self, 'data_source_box'):
            idx = self.data_source_box.findData("classified_block_model")
            if idx >= 0:
                self.data_source_box.setCurrentIndex(idx)
        self.set_block_model(block_model)

    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results - register individual statistics as separate sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        try:
            import pyvista as pv

            if results is None:
                return

            if not isinstance(results, dict):
                logger.warning(f"SwathAnalysis3DPanel: SGSIM results is not a dict, type={type(results)}")
                return

            variable = results.get('variable', 'Grade')
            summary = results.get('summary', {})
            params = results.get('params')
            grid = results.get('grid') or results.get('pyvista_grid')

            logger.info(f"SwathAnalysis3DPanel: SGSIM results keys: {list(results.keys())}")
            logger.info(f"SwathAnalysis3DPanel: Summary keys: {list(summary.keys()) if summary else 'None'}")
            logger.info(f"SwathAnalysis3DPanel: params = {params is not None}")

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
                        logger.info(f"SwathAnalysis3DPanel: Extracted {n_blocks:,} cell centers from grid")

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
                    logger.info(f"SwathAnalysis3DPanel: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
                except Exception as e:
                    logger.warning(f"SwathAnalysis3DPanel: Failed to generate coords from params: {e}")

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
                        logger.info(f"SwathAnalysis3DPanel: Registered SGSIM E-type Mean from fallback")
                else:
                    logger.warning("SwathAnalysis3DPanel: No grid, params, or realizations found in SGSIM results")
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
                        logger.info(f"SwathAnalysis3DPanel: Registered {display_prefix} ({variable})")

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
                logger.info(f"SwathAnalysis3DPanel: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

            self._update_data_source_selector()

        except Exception as e:
            logger.warning(f"SwathAnalysis3DPanel: Failed to load SGSIM results: {e}", exc_info=True)

    def _on_drillhole_data_loaded(self, drillhole_data):
        """Handle drillhole data from DataRegistry - extract composites."""
        logger.info("SwathAnalysis3DPanel: Drillhole data received from registry")

        df = None
        source = "DataRegistry"

        if isinstance(drillhole_data, dict):
            # Prefer composites for swath analysis
            comp = drillhole_data.get("composites")
            if comp is not None and not getattr(comp, "empty", True):
                df = comp
                source = "composites"
                logger.info("SwathAnalysis3DPanel: Using composites from registry")
            else:
                # Fall back to assays if no composites
                assays = drillhole_data.get("assays")
                if assays is not None and not getattr(assays, "empty", True):
                    df = assays
                    source = "assays"
                    logger.info("SwathAnalysis3DPanel: Using assays from registry (no composites)")
        elif isinstance(drillhole_data, pd.DataFrame):
            df = drillhole_data
            source = "DataFrame"

        if df is not None:
            self.set_composite_data(df, source=f"From {source}")

    def _update_data_source_selector(self):
        """Update the data source dropdown with available sources including individual SGSIM stats."""
        if not hasattr(self, 'data_source_box'):
            return

        # Remember current selection
        current_data = self.data_source_box.currentData()

        # Block signals while updating
        self.data_source_box.blockSignals(True)
        self.data_source_box.clear()

        # Add available sources
        if "block_model" in self._available_sources:
            bm = self._stored_block_model
            count = len(bm) if bm is not None and hasattr(bm, '__len__') else 0
            self.data_source_box.addItem(f"Block Model ({count:,} blocks)", "block_model")

        if "classified_block_model" in self._available_sources:
            bm = self._stored_classified_block_model
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

        # Restore selection if possible
        if current_data:
            idx = self.data_source_box.findData(current_data)
            if idx >= 0:
                self.data_source_box.setCurrentIndex(idx)

        self.data_source_box.blockSignals(False)
        logger.debug(f"SwathAnalysis3DPanel: Updated data source selector with {len(self._available_sources)} sources")

        # Also update comparison sources widget
        if hasattr(self, '_source_selection_widget'):
            self._update_comparison_sources()

    def _on_data_source_changed(self, index: int):
        """Handle data source selection change."""
        if index < 0:
            return

        source_type = self.data_source_box.currentData()
        if source_type is None:
            return

        logger.info(f"SwathAnalysis3DPanel: Data source changed to {source_type}")
        self._current_source = source_type

        if source_type == "block_model":
            if self._stored_block_model is not None:
                self.set_block_model(self._stored_block_model)
                logger.info("SwathAnalysis3DPanel: Switched to regular block model")
        elif source_type == "classified_block_model":
            if self._stored_classified_block_model is not None:
                self.set_block_model(self._stored_classified_block_model)
                logger.info("SwathAnalysis3DPanel: Switched to classified block model")
        elif source_type == "sgsim":
            if self._stored_sgsim_df is not None:
                self.set_block_model(self._stored_sgsim_df)
                logger.info("SwathAnalysis3DPanel: Switched to SGSIM results")
        elif source_type and source_type.startswith('sgsim_') and source_type in self._block_model_sources:
            # Handle individual SGSIM statistics
            source_info = self._block_model_sources[source_type]
            df = source_info.get('df')
            if df is not None:
                self.set_block_model(df)
                logger.info(f"SwathAnalysis3DPanel: Switched to {source_info.get('display_name', source_type)}")

    def _on_comparison_mode_changed(self, enabled: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = enabled
        if enabled:
            self._update_comparison_sources()
            # Disable the single data source selector in comparison mode
            self.data_source_box.setEnabled(False)
        else:
            self.data_source_box.setEnabled(True)
            self._comparison_results.clear()
        logger.info(f"SwathAnalysis3DPanel: Comparison mode {'enabled' if enabled else 'disabled'}")

    def _on_comparison_sources_changed(self, selected_sources: list):
        """Handle comparison source selection change."""
        logger.debug(f"SwathAnalysis3DPanel: Comparison sources changed: {selected_sources}")

        # Populate properties from selected sources
        if selected_sources:
            self._populate_comparison_properties(selected_sources)

        # Enable run button if at least 2 sources selected
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(len(selected_sources) >= 2)

    def _populate_comparison_properties(self, selected_keys: list):
        """Populate property dropdown with properties from selected sources."""

        all_properties = set()

        for source_key in selected_keys:
            data = None
            if source_key == 'block_model':
                data = self._stored_block_model
            elif source_key == 'classified_block_model':
                data = self._stored_classified_block_model
            elif source_key in self._block_model_sources:
                data = self._block_model_sources[source_key].get('df')

            if data is not None:
                logger.debug(f"SwathAnalysis3DPanel: Source {source_key} type: {type(data).__name__}")
                if isinstance(data, pd.DataFrame):
                    for col in data.columns:
                        if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG'):
                            try:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    all_properties.add(col)
                            except:
                                all_properties.add(col)
                    logger.debug(f"SwathAnalysis3DPanel: DataFrame columns: {list(data.columns)[:10]}")
                else:
                    # BlockModel class
                    if hasattr(data, 'properties') and data.properties:
                        for prop in data.properties.keys():
                            all_properties.add(prop)
                        logger.debug(f"SwathAnalysis3DPanel: BlockModel properties: {list(data.properties.keys())[:5]}")
                    if hasattr(data, 'to_dataframe'):
                        try:
                            df = data.to_dataframe()
                            for col in df.columns:
                                if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        all_properties.add(col)
                            logger.debug(f"SwathAnalysis3DPanel: to_dataframe columns: {list(df.columns)[:10]}")
                        except Exception as e:
                            logger.debug(f"SwathAnalysis3DPanel: to_dataframe failed: {e}")

        # Update property dropdown
        if hasattr(self, 'property_combo'):
            current = self.property_combo.currentText()
            self.property_combo.blockSignals(True)
            self.property_combo.clear()

            sorted_props = sorted(all_properties)
            self.property_combo.addItems(sorted_props)

            if current and current in sorted_props:
                self.property_combo.setCurrentText(current)
            else:
                for prop in sorted_props:
                    if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')):
                        self.property_combo.setCurrentText(prop)
                        break

            self.property_combo.blockSignals(False)
            logger.info(f"SwathAnalysis3DPanel: Populated {len(sorted_props)} properties for comparison")

    def _update_comparison_sources(self):
        """Update the comparison source list with available block model sources."""
        sources = {}

        # Add regular block model
        if self._stored_block_model is not None:
            df = self._stored_block_model
            if isinstance(df, pd.DataFrame):
                count = len(df)
            elif hasattr(df, 'to_dataframe'):
                count = len(df.to_dataframe())
            else:
                count = 0
            sources['block_model'] = {
                'display_name': 'Block Model',
                'block_count': count,
                'df': df
            }

        # Add classified block model
        if self._stored_classified_block_model is not None:
            df = self._stored_classified_block_model
            if isinstance(df, pd.DataFrame):
                count = len(df)
            elif hasattr(df, 'to_dataframe'):
                count = len(df.to_dataframe())
            else:
                count = 0
            sources['classified_block_model'] = {
                'display_name': 'Classified Block Model',
                'block_count': count,
                'df': df
            }

        # Add individual SGSIM sources
        for source_key, source_info in self._block_model_sources.items():
            if source_key.startswith('sgsim_'):
                df = source_info.get('df')
                if df is not None:
                    sources[source_key] = {
                        'display_name': source_info.get('display_name', source_key),
                        'block_count': len(df) if isinstance(df, pd.DataFrame) else 0,
                        'df': df
                    }

        self._source_selection_widget.update_sources(sources)
        logger.debug(f"SwathAnalysis3DPanel: Updated comparison sources with {len(sources)} sources")

    def _get_data_for_source(self, source_key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame for a given source key."""
        data = None
        if source_key == 'block_model':
            data = self._stored_block_model
        elif source_key == 'classified_block_model':
            data = self._stored_classified_block_model
        elif source_key == 'sgsim':
            data = self._stored_sgsim_df
        elif source_key.startswith('sgsim_') and source_key in self._block_model_sources:
            data = self._block_model_sources[source_key].get('df')

        # Convert BlockModel to DataFrame if needed
        if data is not None:
            if isinstance(data, pd.DataFrame):
                return data
            elif hasattr(data, 'to_dataframe'):
                return data.to_dataframe()
        return None

    def bind_controller(self, controller):
        """
        Bind the application controller.
        
        Args:
            controller: AppController instance (optional, for future use)
        """
        self.controller = controller
        logger.debug("SwathAnalysis3DPanel: Controller bound")
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Input configuration group
        input_group = self._create_input_group()
        layout.addWidget(input_group)
        
        # Swath parameters group
        swath_group = self._create_swath_parameters_group()
        layout.addWidget(swath_group)
        
        # Create splitter for results and charts
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Results table
        self.results_table = self._create_results_table()
        splitter.addWidget(self.results_table)
        
        # Charts in tabs
        self.charts_tabs = QTabWidget()
        self._create_chart_tabs()
        splitter.addWidget(self.charts_tabs)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Action buttons
        action_layout = self._create_action_buttons()
        layout.addWidget(action_layout)
    
    def _create_input_group(self) -> QGroupBox:
        """Create input data configuration group."""
        group = QGroupBox("Data Sources")
        layout = QVBoxLayout()

        # Data source selector
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Data Source:"))
        self.data_source_box = QComboBox()
        self.data_source_box.setMinimumWidth(200)
        self.data_source_box.currentIndexChanged.connect(self._on_data_source_changed)
        source_layout.addWidget(self.data_source_box)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # Multi-source comparison widget
        self._source_selection_widget = SourceSelectionWidget()
        self._source_selection_widget.comparison_mode_changed.connect(self._on_comparison_mode_changed)
        self._source_selection_widget.sources_changed.connect(self._on_comparison_sources_changed)
        layout.addWidget(self._source_selection_widget)

        # Block model info
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel("Block Model:"))
        self.block_model_label = QLabel("No block model loaded")
        self.block_model_label.setStyleSheet("color: #888; font-style: italic;")
        block_layout.addWidget(self.block_model_label)
        block_layout.addStretch()
        layout.addLayout(block_layout)
        
        # Composite data loading
        composite_layout = QHBoxLayout()
        composite_layout.addWidget(QLabel("Composite Data:"))
        self.composite_label = QLabel("No composite data loaded")
        self.composite_label.setStyleSheet("color: #888; font-style: italic;")
        composite_layout.addWidget(self.composite_label)
        
        self.load_composite_btn = QPushButton("Load Composites")
        self.load_composite_btn.clicked.connect(self._load_composite_data)
        composite_layout.addWidget(self.load_composite_btn)
        composite_layout.addStretch()
        layout.addLayout(composite_layout)
        
        # Grade field selection
        grade_layout = QHBoxLayout()
        grade_layout.addWidget(QLabel("Grade Field:"))
        self.grade_field_combo = QComboBox()
        self.grade_field_combo.setMinimumWidth(150)
        self.grade_field_combo.currentIndexChanged.connect(self._check_ready_state)
        grade_layout.addWidget(self.grade_field_combo)
        
        grade_layout.addWidget(QLabel("Composite Grade Field:"))
        self.composite_grade_combo = QComboBox()
        self.composite_grade_combo.setMinimumWidth(150)
        self.composite_grade_combo.currentIndexChanged.connect(self._check_ready_state)
        grade_layout.addWidget(self.composite_grade_combo)
        
        grade_layout.addStretch()
        layout.addLayout(grade_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_swath_parameters_group(self) -> QGroupBox:
        """Create swath parameters configuration group."""
        group = QGroupBox("Swath Analysis Parameters")
        layout = QVBoxLayout()
        
        # Swath mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Swath Mode:"))
        
        self.mode_group = QButtonGroup()
        self.x_axis_radio = QRadioButton("X-Axis")
        self.y_axis_radio = QRadioButton("Y-Axis")
        self.z_axis_radio = QRadioButton("Z-Axis")
        self.full_3d_radio = QRadioButton("Full 3D Cube")
        
        self.x_axis_radio.setChecked(True)
        self.mode_group.addButton(self.x_axis_radio, 0)
        self.mode_group.addButton(self.y_axis_radio, 1)
        self.mode_group.addButton(self.z_axis_radio, 2)
        self.mode_group.addButton(self.full_3d_radio, 3)
        
        mode_layout.addWidget(self.x_axis_radio)
        mode_layout.addWidget(self.y_axis_radio)
        mode_layout.addWidget(self.z_axis_radio)
        mode_layout.addWidget(self.full_3d_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # Window size and step
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Window Size (m):"))
        self.window_size_spin = QDoubleSpinBox()
        self.window_size_spin.setRange(1.0, 10000.0)
        self.window_size_spin.setValue(50.0)
        self.window_size_spin.setDecimals(1)
        self.window_size_spin.setSingleStep(10.0)
        params_layout.addWidget(self.window_size_spin)
        
        params_layout.addWidget(QLabel("Step Size (m):"))
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(1.0, 1000.0)
        self.step_size_spin.setValue(25.0)
        self.step_size_spin.setDecimals(1)
        self.step_size_spin.setSingleStep(5.0)
        params_layout.addWidget(self.step_size_spin)
        
        params_layout.addStretch()
        layout.addLayout(params_layout)
        
        # Filtering options
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filters:"))
        
        self.filter_classification = QCheckBox("Classification")
        filter_layout.addWidget(self.filter_classification)
        
        self.classification_combo = QComboBox()
        self.classification_combo.setEnabled(False)
        self.filter_classification.toggled.connect(self.classification_combo.setEnabled)
        filter_layout.addWidget(self.classification_combo)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Run button
        run_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Swath Analysis")
        self.run_button.setMinimumHeight(40)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_analysis)
        run_layout.addStretch()
        run_layout.addWidget(self.run_button)
        run_layout.addStretch()
        layout.addLayout(run_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_results_table(self) -> QTableWidget:
        """Create results table widget."""
        table = QTableWidget()
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels([
            "Swath ID",
            "X Center",
            "Y Center",
            "Z Center",
            "Mean Est",
            "Mean Comp",
            "ΔGrade",
            "Ratio",
            "#Blocks",
            "#Samples"
        ])
        
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.itemSelectionChanged.connect(self._on_swath_selected)
        
        return table
    
    def _create_chart_tabs(self):
        """Create chart tab widgets."""
        # 2D Swath Plot
        self.swath_2d_figure = Figure(figsize=(10, 6))
        self.swath_2d_canvas = FigureCanvas(self.swath_2d_figure)
        self.charts_tabs.addTab(self.swath_2d_canvas, "2D Swath Plot")
        
        # Bias Distribution
        self.bias_figure = Figure(figsize=(10, 6))
        self.bias_canvas = FigureCanvas(self.bias_figure)
        self.charts_tabs.addTab(self.bias_canvas, "Bias Distribution")
        
        # Ratio Analysis
        self.ratio_figure = Figure(figsize=(10, 6))
        self.ratio_canvas = FigureCanvas(self.ratio_figure)
        self.charts_tabs.addTab(self.ratio_canvas, "Ratio Analysis")
        
        # 3D Bias Map
        self.bias_3d_figure = Figure(figsize=(10, 8))
        self.bias_3d_canvas = FigureCanvas(self.bias_3d_figure)
        self.charts_tabs.addTab(self.bias_3d_canvas, "3D Bias Map")
    
    def _create_action_buttons(self) -> QWidget:
        """Create action buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self._export_csv)
        layout.addWidget(export_csv_btn)
        
        export_excel_btn = QPushButton("Export to Excel")
        export_excel_btn.clicked.connect(self._export_excel)
        layout.addWidget(export_excel_btn)
        
        export_charts_btn = QPushButton("Export Charts")
        export_charts_btn.clicked.connect(self._export_charts)
        layout.addWidget(export_charts_btn)
        
        self.visualize_3d_btn = QPushButton("Visualize in 3D View")
        self.visualize_3d_btn.setEnabled(False)
        self.visualize_3d_btn.clicked.connect(self._visualize_in_3d)
        layout.addWidget(self.visualize_3d_btn)
        
        layout.addStretch()
        
        return widget
    
    def set_block_model(self, block_model):
        """Set the block model for analysis."""
        # Handle both BlockModel and DataFrame
        if isinstance(block_model, pd.DataFrame):
            # Store DataFrame directly - will need to convert to BlockModel if needed
            self.block_model_df = block_model
            self.block_model = None  # No BlockModel object available
            self._populate_block_model_info_from_dataframe(block_model)
        elif hasattr(block_model, 'block_count') or hasattr(block_model, 'properties'):
            # BlockModel instance
            self.block_model = block_model
            self.block_model_df = None
            self._populate_block_model_info()
        else:
            logger.warning(f"Unknown block model type: {type(block_model)}")
            return
        
        self._check_ready_state()
        block_count = len(block_model) if isinstance(block_model, pd.DataFrame) else (block_model.block_count if hasattr(block_model, 'block_count') else 0)
        logger.info(f"Set block model with {block_count} blocks")
    
    def _populate_block_model_info_from_dataframe(self, df: pd.DataFrame):
        """Populate block model information from DataFrame."""
        if df is None or df.empty:
            return

        # Update label
        self.block_model_label.setText(f"{len(df)} blocks loaded")
        self.block_model_label.setStyleSheet("color: #2ecc71; font-weight: bold;")

        # Block signals while populating to avoid premature checks
        self.grade_field_combo.blockSignals(True)

        # Populate grade field combo
        self.grade_field_combo.clear()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude coordinate and dimension columns, system IDs, and compositing metadata
        exclude_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'X', 'Y', 'Z',
                       'XINC', 'YINC', 'ZINC', 'DISTANCE', 'VARIANCE',
                       'GLOBAL_INTERVAL_ID', 'global_interval_id',
                       # Compositing metadata columns
                       'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                       'sample_count', 'total_mass', 'total_length', 'support', 'is_partial']

        grade_fields = [col for col in numeric_cols if col.upper() not in [c.upper() for c in exclude_cols]]

        logger.info(f"SwathAnalysis3DPanel: Found grade fields: {grade_fields}")
        self.grade_field_combo.addItems(grade_fields)

        if grade_fields:
            self.current_grade_field = grade_fields[0]
            self.grade_field_combo.setCurrentText(grade_fields[0])
            logger.info(f"SwathAnalysis3DPanel: Selected grade field: {grade_fields[0]}")

        self.grade_field_combo.blockSignals(False)
    
    def _populate_block_model_info(self):
        """Populate block model information and grade fields."""
        if self.block_model is None:
            return

        # Get DataFrame - handle both BlockModel and DataFrame
        if isinstance(self.block_model, pd.DataFrame):
            df = self.block_model
        elif hasattr(self.block_model, 'to_dataframe'):
            df = self.block_model.to_dataframe()
        else:
            logger.warning(f"Cannot convert block model to DataFrame: {type(self.block_model)}")
            return

        # Update label
        block_count = self.block_model.block_count if hasattr(self.block_model, 'block_count') else len(df)
        self.block_model_label.setText(f"{block_count} blocks loaded")
        self.block_model_label.setStyleSheet("color: #2ecc71; font-weight: bold;")

        # Block signals while populating to avoid premature checks
        self.grade_field_combo.blockSignals(True)

        # Populate grade field combo
        self.grade_field_combo.clear()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude coordinate and dimension columns, system IDs, and compositing metadata
        exclude_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'X', 'Y', 'Z',
                       'XINC', 'YINC', 'ZINC', 'DISTANCE', 'VARIANCE',
                       'GLOBAL_INTERVAL_ID', 'global_interval_id',
                       # Compositing metadata columns
                       'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                       'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO',
                       'sample_count', 'total_mass', 'total_length', 'support', 'is_partial',
                       'method', 'weighting', 'element_weights', 'merged_partial', 'merged_partial_auto']
        grade_cols = [col for col in numeric_cols if col not in exclude_cols]

        if grade_cols:
            self.grade_field_combo.addItems(grade_cols)
            self.current_grade_field = grade_cols[0]
            self.grade_field_combo.setCurrentText(grade_cols[0])
            logger.info(f"SwathAnalysis3DPanel: Found and selected grade field: {grade_cols[0]}")

        self.grade_field_combo.blockSignals(False)
        
        # Populate classification filter if available
        if 'Classification' in df.columns:
            classifications = df['Classification'].unique().tolist()
            self.classification_combo.addItems(classifications)
            self.filter_classification.setEnabled(True)
        
        # Build spatial tree for blocks
        if hasattr(self.block_model, 'positions') and self.block_model.positions is not None:
            positions = self.block_model.positions
            self._block_tree = cKDTree(positions)
            logger.debug(f"Built spatial tree for {len(positions)} blocks")
        else:
            # Try to get coordinates from DataFrame - handle multiple naming conventions
            coord_cols = None
            columns_upper = {col.upper(): col for col in df.columns}
            
            # Check for standard coordinate columns first
            if all(col in df.columns for col in ['X', 'Y', 'Z']):
                coord_cols = ['X', 'Y', 'Z']
            elif all(col in df.columns for col in ['x', 'y', 'z']):
                coord_cols = ['x', 'y', 'z']
            # Check for centroid coordinates (common in block models)
            elif 'XC' in columns_upper and 'YC' in columns_upper and 'ZC' in columns_upper:
                coord_cols = [columns_upper['XC'], columns_upper['YC'], columns_upper['ZC']]
            
            if coord_cols:
                positions = df[coord_cols].values
                self._block_tree = cKDTree(positions)
                logger.debug(f"Built spatial tree for {len(positions)} blocks from DataFrame using {coord_cols}")
    
    def set_composite_data(self, df: pd.DataFrame, source: str = "From drillholes"):
        """
        Set composite data programmatically (auto-detected from drillholes).
        
        Args:
            df: DataFrame with composite data (must have x, y, z columns)
            source: Description of data source for UI label
        """
        try:
            # Validate DataFrame
            if df is None or df.empty:
                logger.warning("Empty composite DataFrame provided")
                return
            
            # Validate required columns
            required_coords = ['x', 'y', 'z']
            missing_coords = [col for col in required_coords 
                            if col.lower() not in df.columns.str.lower()]
            
            if missing_coords:
                # Try uppercase
                required_coords_upper = ['X', 'Y', 'Z']
                missing_coords_upper = [col for col in required_coords_upper 
                                       if col not in df.columns]
                
                if missing_coords_upper:
                    logger.error(f"Composite data missing coordinate columns: {missing_coords}")
                    return
                else:
                    # Standardize to lowercase
                    df = df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'})
            else:
                # Ensure lowercase
                df = df.rename(columns={col: col.lower() for col in ['X', 'Y', 'Z'] 
                                       if col in df.columns})
            
            self.composite_data = df
            
            # Update label with source info
            self.composite_label.setText(f"{len(df)} composites ({source})")
            self.composite_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            
            # Populate composite grade field combo
            self.composite_grade_combo.clear()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude coordinate columns, system IDs, and compositing metadata
            exclude_cols = ['x', 'y', 'z', 'from', 'to', 'length', 'holeid', 'hole_id', 'global_interval_id',
                           # Compositing metadata columns
                           'sample_count', 'total_mass', 'total_length', 'support', 'is_partial',
                           'method', 'weighting', 'element_weights', 'merged_partial', 'merged_partial_auto']
            grade_cols = [col for col in numeric_cols 
                         if col.lower() not in [e.lower() for e in exclude_cols]]
            
            if grade_cols:
                self.composite_grade_combo.addItems(grade_cols)
            
            # Build spatial tree for composites
            coords = df[['x', 'y', 'z']].values
            self._composite_tree = cKDTree(coords)
            logger.info(f"Auto-loaded {len(df)} composites from {source}")
            logger.debug(f"Built spatial tree for {len(coords)} composites")
            
            self._check_ready_state()
            
        except Exception as e:
            logger.error(f"Error setting composite data: {e}", exc_info=True)
    
    def _load_composite_data(self):
        """Load composite data from CSV file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Composite Data",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return
            
            # Load CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded composite data: {len(df)} records from {file_path}")
            
            # Validate required columns
            required_coords = ['x', 'y', 'z']
            missing_coords = [col for col in required_coords 
                            if col not in df.columns.str.lower()]
            
            if missing_coords:
                # Try uppercase
                required_coords_upper = ['X', 'Y', 'Z']
                missing_coords_upper = [col for col in required_coords_upper 
                                       if col not in df.columns]
                
                if missing_coords_upper:
                    QMessageBox.warning(
                        self,
                        "Invalid Data",
                        f"Composite data must contain coordinate columns (x, y, z or X, Y, Z).\n"
                        f"Missing: {', '.join(missing_coords)}"
                    )
                    return
                else:
                    # Standardize to lowercase
                    df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
            else:
                # Ensure lowercase
                df.rename(columns={col: col.lower() for col in ['X', 'Y', 'Z'] 
                                 if col in df.columns}, inplace=True)
            
            self.composite_data = df
            
            # Update label
            self.composite_label.setText(f"{len(df)} composites loaded")
            self.composite_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            
            # Populate composite grade field combo
            self.composite_grade_combo.clear()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude coordinate columns, system IDs, and compositing metadata
            exclude_cols = ['x', 'y', 'z', 'from', 'to', 'length', 'holeid', 'hole_id', 'global_interval_id',
                           # Compositing metadata columns
                           'sample_count', 'total_mass', 'total_length', 'support', 'is_partial',
                           'method', 'weighting', 'element_weights', 'merged_partial', 'merged_partial_auto']
            grade_cols = [col for col in numeric_cols 
                         if col.lower() not in [e.lower() for e in exclude_cols]]
            
            if grade_cols:
                self.composite_grade_combo.addItems(grade_cols)
            
            # Build spatial tree for composites
            coords = df[['x', 'y', 'z']].values
            self._composite_tree = cKDTree(coords)
            logger.debug(f"Built spatial tree for {len(coords)} composites")
            
            self._check_ready_state()
            
            QMessageBox.information(
                self,
                "Success",
                f"Loaded {len(df)} composites with {len(grade_cols)} grade fields"
            )
            
        except Exception as e:
            logger.error(f"Error loading composite data: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load composite data:\n{str(e)}"
            )
    
    def _check_ready_state(self):
        """Check if all required data is loaded and enable run button."""
        has_block_model = (self.block_model is not None or 
                          (hasattr(self, 'block_model_df') and self.block_model_df is not None))
        ready = (has_block_model and 
                self.composite_data is not None and
                self.grade_field_combo.currentText() != "" and
                self.composite_grade_combo.currentText() != "")
        
        self.run_button.setEnabled(ready)
        
        if ready:
            logger.debug("Swath analysis ready to run")
        else:
            logger.debug(f"Swath analysis not ready: block_model={has_block_model}, "
                        f"composite_data={self.composite_data is not None}, "
                        f"grade_field={self.grade_field_combo.currentText()}, "
                        f"composite_field={self.composite_grade_combo.currentText()}")
    
    def _normalize_coordinate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize coordinate columns to lowercase x, y, z.
        
        Handles multiple common coordinate column naming conventions:
        - x, y, z (lowercase)
        - X, Y, Z (uppercase)
        - xc, yc, zc (lowercase centroid)
        - XC, YC, ZC (uppercase centroid - common in block models)
        
        Args:
            df: DataFrame with coordinate columns
            
        Returns:
            DataFrame with standardized x, y, z columns
        """
        df = df.copy()
        columns_upper = {col.upper(): col for col in df.columns}
        
        # Priority order: exact x/y/z first, then XC/YC/ZC variants
        coord_mappings = [
            # Check for exact X, Y, Z first
            ('X', 'x'),
            ('Y', 'y'),
            ('Z', 'z'),
            # Then check for XC, YC, ZC (centroid coordinates)
            ('XC', 'x'),
            ('YC', 'y'),
            ('ZC', 'z'),
        ]
        
        rename_map = {}
        found_coords = set()
        
        for source_upper, target in coord_mappings:
            # Skip if we already have this target coordinate
            if target in found_coords:
                continue
            
            # Check if this column exists (case-insensitive)
            if source_upper in columns_upper:
                original_col = columns_upper[source_upper]
                # Only rename if not already the target name
                if original_col != target:
                    rename_map[original_col] = target
                found_coords.add(target)
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Normalized coordinate columns: {rename_map}")
        
        return df
    
    def _get_swath_mode(self) -> SwathMode:
        """Get selected swath mode."""
        if self.x_axis_radio.isChecked():
            return SwathMode.X_AXIS
        elif self.y_axis_radio.isChecked():
            return SwathMode.Y_AXIS
        elif self.z_axis_radio.isChecked():
            return SwathMode.Z_AXIS
        else:
            return SwathMode.FULL_3D
    
    def _run_analysis(self):
        """Run the swath analysis."""
        # Check for comparison mode
        if self._comparison_mode:
            self._run_comparison_analysis()
            return

        try:
            # Validate inputs
            has_block_model = (self.block_model is not None or
                              (hasattr(self, 'block_model_df') and self.block_model_df is not None))
            if not has_block_model or self.composite_data is None:
                QMessageBox.warning(self, "No Data", "Please load both block model and composite data.")
                return
            
            grade_field = self.grade_field_combo.currentText()
            composite_grade_field = self.composite_grade_combo.currentText()
            
            if not grade_field or not composite_grade_field:
                QMessageBox.warning(self, "Invalid Selection", 
                                  "Please select grade fields for both datasets.")
                return
            
            # Get parameters
            window_size = self.window_size_spin.value()
            step_size = self.step_size_spin.value()
            swath_mode = self._get_swath_mode()
            
            # Get data - handle both BlockModel and DataFrame
            if isinstance(self.block_model, pd.DataFrame):
                block_df = self.block_model.copy()
            elif hasattr(self, 'block_model_df') and self.block_model_df is not None:
                block_df = self.block_model_df.copy()
            elif hasattr(self.block_model, 'to_dataframe'):
                block_df = self.block_model.to_dataframe()
            else:
                QMessageBox.warning(self, "No Block Model", "Block model data is not available.")
                return
            
            # Apply classification filter if enabled
            if self.filter_classification.isChecked():
                classification = self.classification_combo.currentText()
                if 'Classification' in block_df.columns:
                    block_df = block_df[block_df['Classification'] == classification]
                    logger.info(f"Filtered to {len(block_df)} blocks with classification: {classification}")
            
            # Show progress dialog
            progress = QProgressDialog("Running swath analysis...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            # Run analysis
            self.results = self._perform_swath_analysis(
                block_df,
                self.composite_data,
                grade_field,
                composite_grade_field,
                window_size,
                step_size,
                swath_mode,
                progress
            )
            
            progress.setValue(100)
            progress.close()
            
            if self.results is None or len(self.results) == 0:
                QMessageBox.warning(self, "No Results", 
                                  "No swaths generated. Try adjusting window or step size.")
                return
            
            # Update displays
            self.current_grade_field = grade_field
            self._update_results_table()
            self._update_charts()
            
            # Enable visualization button
            self.visualize_3d_btn.setEnabled(True)
            
            # Emit signal
            self.analysis_completed.emit({
                'results': self.results,
                'mode': swath_mode.value,
                'grade_field': grade_field
            })
            
            QMessageBox.information(
                self,
                "Success",
                f"Swath analysis completed:\n"
                f"- {len(self.results)} swaths analyzed\n"
                f"- Mode: {swath_mode.value}\n"
                f"- Window size: {window_size}m\n"
                f"- Step size: {step_size}m"
            )
            
            logger.info(f"Swath analysis completed: {len(self.results)} swaths")
            
        except Exception as e:
            logger.error(f"Error running swath analysis: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to run analysis:\n{str(e)}")
    
    def _perform_swath_analysis(
        self,
        block_df: pd.DataFrame,
        composite_df: pd.DataFrame,
        block_grade_field: str,
        composite_grade_field: str,
        window_size: float,
        step_size: float,
        mode: SwathMode,
        progress: QProgressDialog
    ) -> pd.DataFrame:
        """
        Perform swath analysis across the specified mode.
        
        Returns DataFrame with swath statistics.
        """
        results = []
        
        # Normalize coordinate columns for block DataFrame
        # Supports: x/y/z, X/Y/Z, XC/YC/ZC, xc/yc/zc (centroid coordinates)
        block_df = self._normalize_coordinate_columns(block_df)
        
        # Normalize coordinate columns for composite DataFrame
        composite_df = self._normalize_coordinate_columns(composite_df)
        
        # Validate coordinates exist
        if not all(col in block_df.columns for col in ['x', 'y', 'z']):
            logger.error(f"Block DataFrame missing coordinate columns. Available: {list(block_df.columns)}")
            return pd.DataFrame()
        
        if not all(col in composite_df.columns for col in ['x', 'y', 'z']):
            logger.error(f"Composite DataFrame missing coordinate columns. Available: {list(composite_df.columns)}")
            return pd.DataFrame()
        
        # Get block positions and grades
        block_positions = block_df[['x', 'y', 'z']].values
        block_grades = block_df[block_grade_field].values
        
        # Get composite positions and grades
        comp_positions = composite_df[['x', 'y', 'z']].values
        comp_grades = composite_df[composite_grade_field].values
        
        # Debug: Log spatial extents
        logger.info(f"Swath analysis - Block extent: X=[{block_positions[:, 0].min():.1f}, {block_positions[:, 0].max():.1f}], "
                   f"Y=[{block_positions[:, 1].min():.1f}, {block_positions[:, 1].max():.1f}], "
                   f"Z=[{block_positions[:, 2].min():.1f}, {block_positions[:, 2].max():.1f}]")
        logger.info(f"Swath analysis - Composite extent: X=[{comp_positions[:, 0].min():.1f}, {comp_positions[:, 0].max():.1f}], "
                   f"Y=[{comp_positions[:, 1].min():.1f}, {comp_positions[:, 1].max():.1f}], "
                   f"Z=[{comp_positions[:, 2].min():.1f}, {comp_positions[:, 2].max():.1f}]")
        
        # Determine swath range based on mode
        if mode == SwathMode.X_AXIS:
            axis_idx = 0
            axis_name = 'X'
        elif mode == SwathMode.Y_AXIS:
            axis_idx = 1
            axis_name = 'Y'
        elif mode == SwathMode.Z_AXIS:
            axis_idx = 2
            axis_name = 'Z'
        else:  # Full 3D
            axis_idx = None
            axis_name = '3D'
        
        if mode != SwathMode.FULL_3D:
            # 1D swath (along single axis)
            axis_values = block_positions[:, axis_idx]
            min_val = axis_values.min()
            max_val = axis_values.max()
            
            swath_centers = np.arange(
                min_val + window_size / 2,
                max_val - window_size / 2 + step_size,
                step_size
            )
            
            total_swaths = len(swath_centers)
            logger.info(f"Swath analysis - Axis {axis_name}: range=[{min_val:.1f}, {max_val:.1f}], "
                       f"window={window_size}, step={step_size}, potential swaths={total_swaths}")
            
            # Track why swaths are being skipped
            skipped_no_blocks = 0
            skipped_no_samples = 0
            
            for i, center in enumerate(swath_centers):
                if progress.wasCanceled():
                    return None
                
                progress.setValue(int((i / total_swaths) * 100))
                
                # Define swath window
                lower = center - window_size / 2
                upper = center + window_size / 2
                
                # Find blocks in swath
                block_mask = (block_positions[:, axis_idx] >= lower) & \
                            (block_positions[:, axis_idx] <= upper)
                
                # Find composites in swath
                comp_mask = (comp_positions[:, axis_idx] >= lower) & \
                           (comp_positions[:, axis_idx] <= upper)
                
                n_blocks = block_mask.sum()
                n_samples = comp_mask.sum()
                
                if n_blocks > 0 and n_samples > 0:
                    # Calculate statistics
                    block_grades_in_swath = block_grades[block_mask]
                    comp_grades_in_swath = comp_grades[comp_mask]
                    
                    mean_est = np.mean(block_grades_in_swath)
                    mean_comp = np.mean(comp_grades_in_swath)
                    delta_grade = mean_est - mean_comp
                    ratio = mean_est / mean_comp if mean_comp > 0 else np.nan
                    
                    # Calculate centroid
                    block_centroid = block_positions[block_mask].mean(axis=0)
                    
                    results.append({
                        'Swath_ID': i,
                        'X_Center': block_centroid[0],
                        'Y_Center': block_centroid[1],
                        'Z_Center': block_centroid[2],
                        'Mean_Est': mean_est,
                        'Mean_Comp': mean_comp,
                        'Delta_Grade': delta_grade,
                        'Ratio': ratio,
                        'N_Blocks': n_blocks,
                        'N_Samples': n_samples,
                        'Axis_Position': center
                    })
                else:
                    if n_blocks == 0:
                        skipped_no_blocks += 1
                    if n_samples == 0:
                        skipped_no_samples += 1
            
            # Log skip summary
            if skipped_no_blocks > 0 or skipped_no_samples > 0:
                logger.info(f"Swath analysis - Skipped swaths: {skipped_no_blocks} with no blocks, "
                           f"{skipped_no_samples} with no composites")
        
        else:
            # Full 3D cube analysis
            # Create 3D grid of swath positions
            x_min, y_min, z_min = block_positions.min(axis=0)
            x_max, y_max, z_max = block_positions.max(axis=0)
            
            x_centers = np.arange(x_min + window_size / 2, x_max - window_size / 2 + step_size, step_size)
            y_centers = np.arange(y_min + window_size / 2, y_max - window_size / 2 + step_size, step_size)
            z_centers = np.arange(z_min + window_size / 2, z_max - window_size / 2 + step_size, step_size)
            
            total_swaths = len(x_centers) * len(y_centers) * len(z_centers)
            current_swath = 0
            swath_id = 0
            
            for x_center in x_centers:
                for y_center in y_centers:
                    for z_center in z_centers:
                        if progress.wasCanceled():
                            return None
                        
                        current_swath += 1
                        progress.setValue(int((current_swath / total_swaths) * 100))
                        
                        # Define 3D cube window
                        half_window = window_size / 2
                        
                        # Find blocks in cube
                        block_mask = (
                            (block_positions[:, 0] >= x_center - half_window) &
                            (block_positions[:, 0] <= x_center + half_window) &
                            (block_positions[:, 1] >= y_center - half_window) &
                            (block_positions[:, 1] <= y_center + half_window) &
                            (block_positions[:, 2] >= z_center - half_window) &
                            (block_positions[:, 2] <= z_center + half_window)
                        )
                        
                        # Find composites in cube
                        comp_mask = (
                            (comp_positions[:, 0] >= x_center - half_window) &
                            (comp_positions[:, 0] <= x_center + half_window) &
                            (comp_positions[:, 1] >= y_center - half_window) &
                            (comp_positions[:, 1] <= y_center + half_window) &
                            (comp_positions[:, 2] >= z_center - half_window) &
                            (comp_positions[:, 2] <= z_center + half_window)
                        )
                        
                        n_blocks = block_mask.sum()
                        n_samples = comp_mask.sum()
                        
                        if n_blocks > 0 and n_samples > 0:
                            # Calculate statistics
                            block_grades_in_swath = block_grades[block_mask]
                            comp_grades_in_swath = comp_grades[comp_mask]
                            
                            mean_est = np.mean(block_grades_in_swath)
                            mean_comp = np.mean(comp_grades_in_swath)
                            delta_grade = mean_est - mean_comp
                            ratio = mean_est / mean_comp if mean_comp > 0 else np.nan
                            
                            results.append({
                                'Swath_ID': swath_id,
                                'X_Center': x_center,
                                'Y_Center': y_center,
                                'Z_Center': z_center,
                                'Mean_Est': mean_est,
                                'Mean_Comp': mean_comp,
                                'Delta_Grade': delta_grade,
                                'Ratio': ratio,
                                'N_Blocks': n_blocks,
                                'N_Samples': n_samples
                            })
                            swath_id += 1
        
        return pd.DataFrame(results)
    
    def _update_results_table(self):
        """Update the results table with swath data."""
        if self.results is None:
            return
        
        self.results_table.setRowCount(len(self.results))
        
        for row, data in self.results.iterrows():
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{int(data['Swath_ID'])}"))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{data['X_Center']:.1f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{data['Y_Center']:.1f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{data['Z_Center']:.1f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{data['Mean_Est']:.3f}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{data['Mean_Comp']:.3f}"))
            
            # Color-code delta grade
            delta = data['Delta_Grade']
            delta_item = QTableWidgetItem(f"{delta:.3f}")
            if delta > 0:
                # Convert matplotlib color (RGBA tuple) to QColor
                rgba = plt.cm.Reds(min(abs(delta) / 5, 1.0))
                color = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))
                delta_item.setBackground(color)
            elif delta < 0:
                # Convert matplotlib color (RGBA tuple) to QColor
                rgba = plt.cm.Blues(min(abs(delta) / 5, 1.0))
                color = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))
                delta_item.setBackground(color)
            self.results_table.setItem(row, 6, delta_item)
            
            ratio_text = f"{data['Ratio']:.2f}" if np.isfinite(data['Ratio']) else "N/A"
            self.results_table.setItem(row, 7, QTableWidgetItem(ratio_text))
            
            self.results_table.setItem(row, 8, QTableWidgetItem(f"{int(data['N_Blocks'])}"))
            self.results_table.setItem(row, 9, QTableWidgetItem(f"{int(data['N_Samples'])}"))
    
    def _update_charts(self):
        """Update all charts with results."""
        if self.results is None:
            return
        
        self._plot_2d_swath()
        self._plot_bias_distribution()
        self._plot_ratio_analysis()
        self._plot_3d_bias_map()
    
    def _plot_2d_swath(self):
        """Plot 2D swath profile."""
        self.swath_2d_figure.clear()
        ax = self.swath_2d_figure.add_subplot(111)
        
        # Determine axis for plotting
        mode = self._get_swath_mode()
        
        if mode != SwathMode.FULL_3D and 'Axis_Position' in self.results.columns:
            x_vals = self.results['Axis_Position']
            x_label = mode.value.split('-')[0]
        else:
            x_vals = self.results['Swath_ID']
            x_label = 'Swath ID'
        
        # Plot estimated and composite grades
        ax.plot(x_vals, self.results['Mean_Est'], 'o-', color='#3498db', 
               linewidth=2, markersize=5, label='Estimated Grade', alpha=0.8)
        ax.plot(x_vals, self.results['Mean_Comp'], 's-', color='#e74c3c', 
               linewidth=2, markersize=5, label='Composite Grade', alpha=0.8)
        
        ax.set_xlabel(f'{x_label} Position (m)' if mode != SwathMode.FULL_3D else 'Swath ID', 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Grade', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        self.swath_2d_figure.suptitle('Swath Analysis: Grade Comparison', 
                                     fontsize=13, fontweight='bold')
        self.swath_2d_figure.tight_layout()
        self.swath_2d_canvas.draw()
    
    def _plot_bias_distribution(self):
        """Plot bias distribution histogram and statistics."""
        self.bias_figure.clear()
        
        # Create 2x2 subplot grid
        ax1 = self.bias_figure.add_subplot(221)
        ax2 = self.bias_figure.add_subplot(222)
        ax3 = self.bias_figure.add_subplot(223)
        ax4 = self.bias_figure.add_subplot(224)
        
        delta_grades = self.results['Delta_Grade'].values
        
        # Histogram of delta grade
        ax1.hist(delta_grades, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Bias')
        ax1.axvline(x=np.mean(delta_grades), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(delta_grades):.3f}')
        ax1.set_xlabel('ΔGrade (Est - Comp)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(delta_grades, vert=True)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_ylabel('ΔGrade', fontweight='bold')
        ax2.set_title('Bias Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(delta_grades, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)')
        ax3.grid(True, alpha=0.3)
        
        # Statistics summary
        ax4.axis('off')
        stats_text = (
            f"Bias Statistics:\n\n"
            f"Mean ΔGrade: {np.mean(delta_grades):.4f}\n"
            f"Median ΔGrade: {np.median(delta_grades):.4f}\n"
            f"Std Dev: {np.std(delta_grades):.4f}\n"
            f"Min: {np.min(delta_grades):.4f}\n"
            f"Max: {np.max(delta_grades):.4f}\n\n"
            f"Interpretation:\n"
        )
        
        mean_bias = np.mean(delta_grades)
        if abs(mean_bias) < 0.01:
            stats_text += "✓ Well-calibrated estimation"
        elif mean_bias > 0:
            stats_text += "⚠ Overestimation bias detected"
        else:
            stats_text += "⚠ Underestimation bias detected"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        self.bias_figure.suptitle('Estimation Bias Analysis', fontsize=13, fontweight='bold')
        self.bias_figure.tight_layout()
        self.bias_canvas.draw()
    
    def _plot_ratio_analysis(self):
        """Plot ratio analysis."""
        self.ratio_figure.clear()
        
        ax1 = self.ratio_figure.add_subplot(121)
        ax2 = self.ratio_figure.add_subplot(122)
        
        ratios = self.results['Ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Ratio histogram
        ax1.hist(ratios, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Match')
        ax1.axvline(x=np.mean(ratios), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(ratios):.3f}')
        ax1.set_xlabel('Ratio (Est / Comp)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: Est vs Comp
        ax2.scatter(self.results['Mean_Comp'], self.results['Mean_Est'], 
                   alpha=0.6, s=50, c='#3498db')
        
        # Add 1:1 line
        min_val = min(self.results['Mean_Comp'].min(), self.results['Mean_Est'].min())
        max_val = max(self.results['Mean_Comp'].max(), self.results['Mean_Est'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        ax2.set_xlabel('Composite Grade', fontweight='bold')
        ax2.set_ylabel('Estimated Grade', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.ratio_figure.suptitle('Ratio Analysis: Estimation Smoothing', 
                                   fontsize=13, fontweight='bold')
        self.ratio_figure.tight_layout()
        self.ratio_canvas.draw()
    
    def _plot_3d_bias_map(self):
        """Plot 3D bias map."""
        self.bias_3d_figure.clear()
        ax = self.bias_3d_figure.add_subplot(111, projection='3d')
        
        # Plot swath centers colored by delta grade
        x = self.results['X_Center']
        y = self.results['Y_Center']
        z = self.results['Z_Center']
        delta = self.results['Delta_Grade']
        
        # Normalize colors
        vmin, vmax = delta.min(), delta.max()
        
        scatter = ax.scatter(x, y, z, c=delta, cmap='RdBu_r', 
                           s=100, alpha=0.6, vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('X (m)', fontweight='bold')
        ax.set_ylabel('Y (m)', fontweight='bold')
        ax.set_zlabel('Z (m)', fontweight='bold')
        
        # Add colorbar
        cbar = self.bias_3d_figure.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('ΔGrade (Est - Comp)', fontweight='bold')
        
        self.bias_3d_figure.suptitle('3D Bias Distribution Map',
                                    fontsize=13, fontweight='bold')
        self.bias_3d_figure.tight_layout()
        self.bias_3d_canvas.draw()

    def _run_comparison_analysis(self):
        """Run swath analysis for multiple sources and create comparison plots."""
        selected_sources = self._source_selection_widget.get_selected_sources()

        if len(selected_sources) < 2:
            QMessageBox.warning(
                self, "Comparison Mode",
                "Please select at least 2 sources to compare."
            )
            return

        if self.composite_data is None:
            QMessageBox.warning(self, "No Data", "Please load composite data first.")
            return

        grade_field = self.grade_field_combo.currentText()
        composite_grade_field = self.composite_grade_combo.currentText()

        if not grade_field or not composite_grade_field:
            QMessageBox.warning(self, "Invalid Selection",
                              "Please select grade fields for both datasets.")
            return

        try:
            # Get parameters
            window_size = self.window_size_spin.value()
            step_size = self.step_size_spin.value()
            swath_mode = self._get_swath_mode()

            # Show progress dialog
            progress = QProgressDialog("Running comparison analysis...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # Clear previous comparison results
            self._comparison_results.clear()

            # Process each source
            for i, source_key in enumerate(selected_sources):
                block_df = self._get_data_for_source(source_key)
                if block_df is None:
                    logger.warning(f"SwathAnalysis3DPanel: No data for source {source_key}")
                    continue

                # Get display name
                display_name = source_key
                if source_key in self._block_model_sources:
                    display_name = self._block_model_sources[source_key].get('display_name', source_key)
                elif source_key == 'block_model':
                    display_name = 'Block Model'
                elif source_key == 'classified_block_model':
                    display_name = 'Classified Block Model'

                # Apply classification filter if enabled
                if self.filter_classification.isChecked():
                    classification = self.classification_combo.currentText()
                    if 'Classification' in block_df.columns:
                        block_df = block_df[block_df['Classification'] == classification].copy()

                # Run swath analysis for this source
                try:
                    result = self._perform_swath_analysis(
                        block_df,
                        self.composite_data,
                        grade_field,
                        composite_grade_field,
                        window_size,
                        step_size,
                        swath_mode,
                        progress=None  # Don't show sub-progress
                    )

                    if result is not None and len(result) > 0:
                        self._comparison_results[source_key] = {
                            'display_name': display_name,
                            'results': result
                        }
                except Exception as e:
                    logger.error(f"SwathAnalysis3DPanel: Error running swath analysis for {source_key}: {e}")

                # Update progress
                pct = int((i + 1) / len(selected_sources) * 100)
                progress.setValue(pct)

                if progress.wasCanceled():
                    break

            progress.close()

            # Plot comparison results
            if len(self._comparison_results) >= 2:
                self._plot_comparison_swaths()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Comparison analysis completed:\n"
                    f"- {len(self._comparison_results)} sources analyzed\n"
                    f"- Mode: {swath_mode.value}\n"
                    f"- Window size: {window_size}m"
                )
            else:
                QMessageBox.warning(self, "No Results",
                                  "Not enough valid sources for comparison.")

        except Exception as e:
            logger.error(f"SwathAnalysis3DPanel: Comparison analysis error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Comparison analysis failed:\n{str(e)}")

    def _plot_comparison_swaths(self):
        """Plot overlaid swath curves for multiple sources."""
        if not self._comparison_results:
            return

        try:
            # Clear existing plot
            self.swath_2d_figure.clear()
            ax = self.swath_2d_figure.add_subplot(111)

            # Dark theme styling
            ax.set_facecolor(f'{ModernColors.PANEL_BG}')
            self.swath_2d_figure.set_facecolor(f'{ModernColors.CARD_BG}')

            swath_mode = self._get_swath_mode()
            source_names = []

            for i, (source_key, data) in enumerate(self._comparison_results.items()):
                style = ComparisonColors.get_style(i)
                display_name = data['display_name']
                source_names.append(display_name)
                results = data['results']

                # Determine x-axis
                if swath_mode != SwathMode.FULL_3D and 'Axis_Position' in results.columns:
                    x_vals = results['Axis_Position']
                else:
                    x_vals = results['Swath_ID']

                # Plot estimated grade for each source
                ax.plot(
                    x_vals, results['Mean_Est'],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=2,
                    marker=style['marker'],
                    markersize=4,
                    label=f'{display_name}',
                    alpha=style['alpha']
                )

            # Plot composite grade (single line - same for all sources)
            first_result = list(self._comparison_results.values())[0]['results']
            if swath_mode != SwathMode.FULL_3D and 'Axis_Position' in first_result.columns:
                x_vals = first_result['Axis_Position']
                x_label = swath_mode.value.split('-')[0]
            else:
                x_vals = first_result['Swath_ID']
                x_label = 'Swath ID'

            ax.plot(
                x_vals, first_result['Mean_Comp'],
                color='#e74c3c',
                linestyle='-',
                linewidth=2.5,
                marker='s',
                markersize=5,
                label='Composite Grade',
                alpha=0.9
            )

            # Styling
            ax.set_xlabel(f'{x_label} Position (m)' if swath_mode != SwathMode.FULL_3D else 'Swath ID',
                         fontsize=11, color=f'{ModernColors.TEXT_PRIMARY}')
            ax.set_ylabel('Grade', fontsize=11, color=f'{ModernColors.TEXT_PRIMARY}')

            ax.tick_params(colors=f'{ModernColors.TEXT_PRIMARY}')
            ax.spines['bottom'].set_color('#555')
            ax.spines['left'].set_color('#555')
            ax.spines['top'].set_color('#555')
            ax.spines['right'].set_color('#555')

            ax.legend(
                loc='best',
                frameon=True,
                facecolor=f'{ModernColors.CARD_BG}',
                edgecolor='#444',
                fontsize=9,
                labelcolor=f'{ModernColors.TEXT_PRIMARY}'
            )
            ax.grid(True, alpha=0.3, color='#555')

            self.swath_2d_figure.suptitle(
                'Swath Comparison: Multiple Estimation Methods',
                fontsize=13,
                fontweight='bold',
                color=f'{ModernColors.TEXT_PRIMARY}'
            )
            self.swath_2d_figure.tight_layout()
            self.swath_2d_canvas.draw()

            # Also update the bias comparison plot
            self._plot_comparison_bias()

            logger.info(f"SwathAnalysis3DPanel: Plotted comparison for {len(self._comparison_results)} sources")

        except Exception as e:
            logger.error(f"Error plotting comparison swaths: {e}", exc_info=True)

    def _plot_comparison_bias(self):
        """Plot bias comparison for multiple sources."""
        if not self._comparison_results:
            return

        try:
            self.bias_figure.clear()
            ax = self.bias_figure.add_subplot(111)

            # Dark theme styling
            ax.set_facecolor(f'{ModernColors.PANEL_BG}')
            self.bias_figure.set_facecolor(f'{ModernColors.CARD_BG}')

            source_names = []
            bias_data = []

            for i, (source_key, data) in enumerate(self._comparison_results.items()):
                style = ComparisonColors.get_style(i)
                display_name = data['display_name']
                source_names.append(display_name)

                results = data['results']
                delta_grades = results['Delta_Grade'].values
                bias_data.append(delta_grades)

                # Calculate statistics
                mean_bias = np.mean(delta_grades)
                std_bias = np.std(delta_grades)

                # Plot histogram with alpha
                ax.hist(
                    delta_grades,
                    bins=25,
                    alpha=0.4,
                    color=style['color'],
                    label=f'{display_name} (μ={mean_bias:.3f}, σ={std_bias:.3f})',
                    edgecolor=style['color'],
                    linewidth=1.5
                )

            # Add zero bias line
            ax.axvline(x=0, color='#ff5555', linestyle='--', linewidth=2, label='Zero Bias')

            # Styling
            ax.set_xlabel('ΔGrade (Estimated - Composite)', fontsize=11, color=f'{ModernColors.TEXT_PRIMARY}')
            ax.set_ylabel('Frequency', fontsize=11, color=f'{ModernColors.TEXT_PRIMARY}')

            ax.tick_params(colors=f'{ModernColors.TEXT_PRIMARY}')
            ax.spines['bottom'].set_color('#555')
            ax.spines['left'].set_color('#555')
            ax.spines['top'].set_color('#555')
            ax.spines['right'].set_color('#555')

            ax.legend(
                loc='upper right',
                frameon=True,
                facecolor=f'{ModernColors.CARD_BG}',
                edgecolor='#444',
                fontsize=9,
                labelcolor=f'{ModernColors.TEXT_PRIMARY}'
            )
            ax.grid(True, alpha=0.3, color='#555')

            self.bias_figure.suptitle(
                'Bias Comparison: Multiple Estimation Methods',
                fontsize=13,
                fontweight='bold',
                color=f'{ModernColors.TEXT_PRIMARY}'
            )
            self.bias_figure.tight_layout()
            self.bias_canvas.draw()

        except Exception as e:
            logger.error(f"Error plotting comparison bias: {e}", exc_info=True)

    def _on_swath_selected(self):
        """Handle swath selection in table."""
        selected_rows = self.results_table.selectedIndexes()
        if selected_rows:
            row = selected_rows[0].row()
            swath_id = int(self.results_table.item(row, 0).text())
            self.swath_selected.emit(swath_id)
            logger.debug(f"Selected swath: {swath_id}")
    
    def _export_csv(self):
        """Export results to CSV."""
        if self.results is None:
            QMessageBox.warning(self, "No Data", "No results available to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Swath Analysis Results",
            f"swath_analysis_{self.current_grade_field}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Step 10: Use ExportHelpers
                from ..utils.export_helpers import export_dataframe_to_csv
                export_dataframe_to_csv(self.results, file_path)
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
                logger.info(f"Exported swath results to {file_path}")
            except Exception as e:
                logger.error(f"Error exporting CSV: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def _export_excel(self):
        """Export results to Excel with statistics."""
        if self.results is None:
            QMessageBox.warning(self, "No Data", "No results available to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Swath Analysis Results",
            f"swath_analysis_{self.current_grade_field}.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                # Step 10: Use ExportHelpers
                from ..utils.export_helpers import export_multiple_sheets_to_excel
                
                # Build frames dictionary
                frames = {'Swath Results': self.results}
                
                # Summary statistics
                delta_grades = self.results['Delta_Grade']
                ratios = self.results['Ratio'].replace([np.inf, -np.inf], np.nan)
                
                summary = pd.DataFrame({
                    'Metric': [
                        'Number of Swaths',
                        'Mean ΔGrade',
                        'Median ΔGrade',
                        'Std Dev ΔGrade',
                        'Mean Ratio',
                        'Median Ratio',
                        'Std Dev Ratio'
                    ],
                    'Value': [
                        len(self.results),
                        delta_grades.mean(),
                        delta_grades.median(),
                        delta_grades.std(),
                        ratios.mean(),
                        ratios.median(),
                        ratios.std()
                    ]
                })
                frames['Summary'] = summary
                
                # Parameters
                params = pd.DataFrame({
                    'Parameter': [
                        'Grade Field',
                        'Swath Mode',
                        'Window Size (m)',
                        'Step Size (m)'
                    ],
                    'Value': [
                        self.current_grade_field,
                        self._get_swath_mode().value,
                        self.window_size_spin.value(),
                        self.step_size_spin.value()
                    ]
                })
                frames['Parameters'] = params
                
                # Export all sheets at once
                export_multiple_sheets_to_excel(frames, file_path)
                
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
                logger.info(f"Exported swath results to Excel: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting Excel: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def _export_charts(self):
        """Export all charts as images."""
        if self.results is None:
            QMessageBox.warning(self, "No Data", "No results available to export.")
            return
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for Chart Export"
        )
        
        if directory:
            try:
                base_name = f"swath_analysis_{self.current_grade_field}"
                
                self.swath_2d_figure.savefig(
                    Path(directory) / f"{base_name}_2d_swath.png", 
                    dpi=300, bbox_inches='tight'
                )
                self.bias_figure.savefig(
                    Path(directory) / f"{base_name}_bias_dist.png", 
                    dpi=300, bbox_inches='tight'
                )
                self.ratio_figure.savefig(
                    Path(directory) / f"{base_name}_ratio.png", 
                    dpi=300, bbox_inches='tight'
                )
                self.bias_3d_figure.savefig(
                    Path(directory) / f"{base_name}_3d_map.png", 
                    dpi=300, bbox_inches='tight'
                )
                
                QMessageBox.information(self, "Success", 
                                      f"Charts exported to:\n{directory}")
                logger.info(f"Exported charts to {directory}")
            except Exception as e:
                logger.error(f"Error exporting charts: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")
    
    def _visualize_in_3d(self):
        """Trigger 3D visualization in main viewer."""
        QMessageBox.information(
            self,
            "3D Visualization",
            "3D visualization integration with main viewer is available.\n\n"
            "Selected swaths will be highlighted in the 3D view.\n"
            "This feature requires additional integration with the renderer."
        )
        # This would connect to the main viewer's renderer
        # to visualize selected swaths in 3D space
