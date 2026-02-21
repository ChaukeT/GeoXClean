from __future__ import annotations

import logging
from typing import Optional, Any, Dict, List
from datetime import datetime

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QSplitter, QTabWidget,
    QHeaderView, QFrame, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

# Optional Matplotlib for Histogram
try:
    # Matplotlib backend is set in main.py
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# If BlockModel isn't available in this context, use a dummy type or Any
try:
    from ..models.block_model import BlockModel
except ImportError:
    BlockModel = Any

from .comparison_utils import ComparisonColors, SourceSelectionWidget
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class StatisticsPanel(QWidget):
    """
    Panel for property statistics, histograms, filtering, and spatial slicing.
    
    PROVENANCE TRACKING (STAT-004):
    - Tracks source block model identity
    - Records timestamp of statistics computation
    - Exports include full provenance metadata for JORC/SAMREC audit trail
    
    STATISTICAL CORRECTNESS:
    - CV calculation guards against division by zero
    - NaN values are explicitly handled and excluded
    - All statistics computed on the same filtered dataset
    
    DATA SOURCES:
    - Supports both block models AND drillhole data
    - Drillhole data: composites, assays, or raw DataFrames
    - Block models: from loader, kriging, SGSIM, etc.
    """
    
    # Signals
    property_changed = pyqtSignal(str)
    filter_applied = pyqtSignal(str, float, float)
    filter_cleared = pyqtSignal()
    slice_applied = pyqtSignal(str, float, str)
    slice_cleared = pyqtSignal()
    
    def __init__(self, registry=None):
        super().__init__()
        self.registry = registry
        self.current_model: Optional[BlockModel] = None
        self.active_property: Optional[str] = None
        
        # Drillhole data support
        self._drillhole_df: Optional[Any] = None  # pd.DataFrame
        self._data_mode: str = "none"  # "block_model", "drillhole", or "none"
        self._data_source_type: str = "unknown"  # "composites", "assays", "dataframe"
        
        # Stored data sources (for user selection)
        self._stored_drillhole_df: Optional[Any] = None
        self._stored_block_model: Optional[BlockModel] = None
        self._stored_classified_block_model: Optional[BlockModel] = None  # Separate from regular
        self._stored_sgsim_df: Optional[Any] = None  # SGSIM results as DataFrame
        self._stored_data_source_type: str = "unknown"
        self._available_sources: list = []

        # Storage for individual SGSIM statistics (Mean, P10, P50, P90, Std Dev)
        self._block_model_sources: Dict[str, Any] = {}

        # Comparison mode support
        self._comparison_mode: bool = False
        self._comparison_stats: Dict[str, Dict[str, Any]] = {}  # Store stats per source

        # STAT-004: Provenance tracking
        self._provenance: Dict[str, Any] = {}
        self._model_source: str = "unknown"
        self._stats_timestamp: Optional[str] = None
        
        # Registry Connection
        self._connect_registry()

        self.setWindowTitle("Block Model Statistics")
        self.setStyleSheet(get_analysis_panel_stylesheet())
        self._setup_ui()
        
        # Refresh data AFTER UI is set up
        if self.registry:
            self._refresh_available_data()
        
        logger.info("Initialized Statistics panel")

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self.setStyleSheet(get_analysis_panel_stylesheet())
        # Redraw the chart with new theme colors
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'figure') and hasattr(self, 'canvas'):
            try:
                colors = get_theme_colors()
                self.figure.patch.set_facecolor(colors.PANEL_BG)
                for ax in self.figure.axes:
                    ax.set_facecolor(colors.CARD_BG)
                    ax.tick_params(colors=colors.TEXT_SECONDARY)
                    ax.title.set_color(colors.TEXT_PRIMARY)
                    ax.xaxis.label.set_color(colors.TEXT_PRIMARY)
                    ax.yaxis.label.set_color(colors.TEXT_PRIMARY)
                    for spine in ax.spines.values():
                        spine.set_color(colors.BORDER)
                self.canvas.draw_idle()
            except Exception:
                pass  # Ignore errors during refresh

    def _connect_registry(self):
        try:
            if self.registry is None:
                self.registry = self.get_registry()
            if self.registry is None:
                logger.info("DataRegistry not found, running standalone.")
                return

            # Connect only available Qt signals; do not fail the panel if a signal is missing.
            signal_map = [
                ("blockModelGenerated", self._on_block_model_generated),
                ("blockModelLoaded", self._on_block_model_loaded),
                ("blockModelClassified", self._on_block_model_classified),
                ("drillholeDataLoaded", self._on_drillhole_data_loaded),
                ("sgsimResultsLoaded", self._on_sgsim_loaded),
            ]
            connected = 0
            for signal_name, handler in signal_map:
                try:
                    signal = getattr(self.registry, signal_name, None)
                    if signal is not None and hasattr(signal, "connect"):
                        signal.connect(handler)
                        connected += 1
                except Exception:
                    logger.debug("StatisticsPanel: Failed to connect signal %s", signal_name, exc_info=True)

            logger.info("StatisticsPanel: Connected %d registry signals", connected)
        except ImportError:
            logger.info("DataRegistry not found, running standalone.")
            self.registry = None
        except Exception as e:
            logger.warning(f"Registry connection failed: {e}")
            # Keep the registry reference even if signal hookups fail.
    
    def _refresh_available_data(self):
        """Refresh the list of available data from all sources (block models AND drillholes)."""
        if not self.registry:
            logger.debug("StatisticsPanel._refresh_available_data: No registry")
            return
        
        try:
            # Store available data sources (no priority - user selects)
            self._available_sources = []
            
            # Check for drillhole data
            drillhole_data = self.registry.get_drillhole_data(copy_data=False)
            logger.debug(
                "StatisticsPanel._refresh_available_data: drillhole_data = %s",
                type(drillhole_data) if drillhole_data is not None else None,
            )
            if drillhole_data is not None:
                self._store_drillhole_data(drillhole_data)
                self._available_sources.append("drillhole")
                logger.info("StatisticsPanel: Found drillhole data in registry")
            
            # Check regular block model
            block_model = self.registry.get_block_model(copy_data=False)
            logger.debug(
                "StatisticsPanel._refresh_available_data: block_model = %s",
                type(block_model) if block_model is not None else None,
            )
            if block_model is not None:
                self._stored_block_model = block_model
                if "block_model" not in self._available_sources:
                    self._available_sources.append("block_model")
                logger.info("StatisticsPanel: Found regular block model in registry")

            # Check classified block model (SEPARATE from regular block model)
            if hasattr(self.registry, "get_classified_block_model"):
                try:
                    classified_model = self.registry.get_classified_block_model(copy_data=False)
                    logger.debug(
                        "StatisticsPanel._refresh_available_data: classified_block_model = %s",
                        type(classified_model) if classified_model is not None else None
                    )
                    if classified_model is not None:
                        self._stored_classified_block_model = classified_model
                        if "classified_block_model" not in self._available_sources:
                            self._available_sources.append("classified_block_model")
                        logger.info("StatisticsPanel: Found classified block model in registry")
                except Exception:
                    pass

            # Check for SGSIM results
            try:
                if hasattr(self.registry, 'get_sgsim_results'):
                    sgsim = self.registry.get_sgsim_results()
                    if sgsim is not None:
                        self._on_sgsim_loaded(sgsim)
                        logger.info("StatisticsPanel: Found SGSIM results in registry")
            except Exception:
                pass

            logger.info(f"StatisticsPanel: Available sources = {self._available_sources}")
            
            # Update data source selector
            self._update_data_source_selector()
            
        except Exception as e:
            logger.warning(f"Failed to refresh available data: {e}", exc_info=True)
    
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
            logger.warning("StatisticsPanel._update_data_source_selector: data_source_box not found")
            return

        logger.debug(f"StatisticsPanel._update_data_source_selector: Available sources = {self._available_sources}, mode = {self._data_mode}")

        self.data_source_box.blockSignals(True)
        self.data_source_box.clear()

        if "drillhole" in self._available_sources:
            label = f"Drillhole Data ({self._stored_data_source_type})"
            self.data_source_box.addItem(label, "drillhole")
            logger.debug(f"StatisticsPanel: Added drillhole item: {label}")

        if "block_model" in self._available_sources:
            bm = self._stored_block_model
            count = len(bm) if bm is not None and hasattr(bm, '__len__') else 0
            self.data_source_box.addItem(f"Block Model ({count:,} blocks)", "block_model")
            logger.debug("StatisticsPanel: Added block model item")

        if "classified_block_model" in self._available_sources:
            bm = self._stored_classified_block_model
            count = len(bm) if bm is not None and hasattr(bm, '__len__') else 0
            self.data_source_box.addItem(f"Classified Block Model ({count:,} blocks)", "classified_block_model")
            logger.debug("StatisticsPanel: Added classified block model item")

        # Add individual SGSIM sources
        for source_key in self._available_sources:
            if source_key.startswith('sgsim_') and source_key in self._block_model_sources:
                source_info = self._block_model_sources[source_key]
                df = source_info.get('df')
                count = len(df) if df is not None else 0
                display_name = source_info.get('display_name', source_key)
                self.data_source_box.addItem(f"{display_name} ({count:,} blocks)", source_key)
                logger.debug(f"StatisticsPanel: Added SGSIM source: {display_name}")

        # Legacy: single SGSIM source (if not using individual stats)
        if "sgsim" in self._available_sources and not any(s.startswith('sgsim_') for s in self._available_sources):
            df = self._stored_sgsim_df
            count = len(df) if df is not None else 0
            self.data_source_box.addItem(f"SGSIM Results ({count:,} blocks)", "sgsim")
            logger.debug("StatisticsPanel: Added legacy SGSIM results item")

        if self.data_source_box.count() == 0:
            self.data_source_box.addItem("No data loaded", "none")
            logger.debug("StatisticsPanel: No data sources, added 'No data loaded'")

        self.data_source_box.blockSignals(False)

        # If we have real data sources and nothing is selected yet, select the first one
        if len(self._available_sources) > 0 and self._data_mode == "none":
            logger.info(f"StatisticsPanel: Selecting first data source (sources={self._available_sources})")
            self.data_source_box.setCurrentIndex(0)
            self._on_data_source_changed(0)
        else:
            logger.debug(f"StatisticsPanel: Not auto-selecting (sources={len(self._available_sources)}, mode={self._data_mode})")

        # Also update comparison sources widget
        if hasattr(self, '_source_selection_widget'):
            self._update_comparison_sources()
            logger.info(f"StatisticsPanel: Available sources = {self._available_sources}")

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
    
    def _on_block_model_classified(self, block_model):
        """Handle block model classification changes."""
        # Store as SEPARATE classified block model (don't overwrite regular block model)
        self._stored_classified_block_model = block_model
        if "classified_block_model" not in self._available_sources:
            self._available_sources.append("classified_block_model")
        self._update_data_source_selector()
        # Auto-switch to classified model when it becomes available
        if hasattr(self, 'data_source_box'):
            idx = self.data_source_box.findData("classified_block_model")
            if idx >= 0:
                self.data_source_box.setCurrentIndex(idx)
        self.set_block_model(block_model)
        logger.info("StatisticsPanel: Classified block model available for selection")
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Top Bar: Data Source & Property Selector ---
        top_bar = QFrame()
        top_bar.setStyleSheet(f"background-color: #333; border-bottom: 1px solid {ModernColors.PANEL_BG};")
        top_bar.setFixedHeight(50)
        tb_layout = QHBoxLayout(top_bar)
        
        # Data source selector
        tb_layout.addWidget(QLabel("<b>Data Source:</b>"))
        self.data_source_box = QComboBox()
        self.data_source_box.setMinimumWidth(180)
        self.data_source_box.addItem("No data loaded", "none")
        self.data_source_box.currentIndexChanged.connect(self._on_data_source_changed)
        tb_layout.addWidget(self.data_source_box)

        tb_layout.addWidget(QLabel("  "))  # Spacer
        
        tb_layout.addWidget(QLabel("<b>Property:</b>"))
        
        self.property_box = QComboBox()
        self.property_box.setMinimumWidth(150)
        self.property_box.currentTextChanged.connect(self._on_property_changed)
        tb_layout.addWidget(self.property_box)
        
        tb_layout.addStretch()
        
        self.lbl_count = QLabel("N: 0")
        self.lbl_count.setStyleSheet("color: #aaa; margin-right: 10px;")
        tb_layout.addWidget(self.lbl_count)

        layout.addWidget(top_bar)

        # --- Comparison Mode Widget (expandable) ---
        self._source_selection_widget = SourceSelectionWidget()
        self._source_selection_widget.comparison_mode_changed.connect(self._on_comparison_mode_changed)
        self._source_selection_widget.sources_changed.connect(self._on_comparison_sources_changed)
        self._source_selection_widget.setStyleSheet("background-color: #2b2b2b; padding: 5px;")
        layout.addWidget(self._source_selection_widget)

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # === Left Panel: Visualization (Stats & Charts) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.tabs = QTabWidget()
        
        # Tab 1: Table
        self.tab_table = QWidget()
        tbl_layout = QVBoxLayout(self.tab_table)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setAlternatingRowColors(True)
        tbl_layout.addWidget(self.stats_table)
        
        btn_export = QPushButton("Export CSV")
        btn_export.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        btn_export.clicked.connect(self._export_statistics)
        tbl_layout.addWidget(btn_export)
        
        self.tabs.addTab(self.tab_table, "Summary Table")
        
        # Tab 2: Chart
        self.tab_chart = QWidget()
        chart_layout = QVBoxLayout(self.tab_chart)
        if MATPLOTLIB_AVAILABLE:
            colors = get_theme_colors()
            self.figure = Figure(facecolor=colors.PANEL_BG)
            self.canvas = FigureCanvas(self.figure)
            chart_layout.addWidget(self.canvas)
        else:
            lbl = QLabel("Matplotlib not installed.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chart_layout.addWidget(lbl)
            
        self.tabs.addTab(self.tab_chart, "Histogram")

        left_layout.addWidget(self.tabs)
        splitter.addWidget(left_widget)

        # === Right Panel: Controls (Filter & Slice) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Group 1: Filter
        gb_filter = QGroupBox("Value Filter")
        form_filter = QFormLayout()
        
        self.filter_min = QDoubleSpinBox()
        self.filter_min.setRange(-1e9, 1e9)
        self.filter_min.setDecimals(3)
        self.filter_min.setPrefix("Min: ")
        form_filter.addRow(self.filter_min)
        
        self.filter_max = QDoubleSpinBox()
        self.filter_max.setRange(-1e9, 1e9)
        self.filter_max.setDecimals(3)
        self.filter_max.setPrefix("Max: ")
        form_filter.addRow(self.filter_max)
        
        h_filter_btns = QHBoxLayout()
        self.apply_filter_btn = QPushButton("Apply")
        self.apply_filter_btn.setObjectName("PrimaryButton")
        self.apply_filter_btn.clicked.connect(self._on_apply_filter)
        
        self.clear_filter_btn = QPushButton("Reset")
        self.clear_filter_btn.clicked.connect(self._on_clear_filter)
        
        h_filter_btns.addWidget(self.apply_filter_btn)
        h_filter_btns.addWidget(self.clear_filter_btn)
        form_filter.addRow(h_filter_btns)
        
        gb_filter.setLayout(form_filter)
        right_layout.addWidget(gb_filter)
        
        # Group 2: Slicing
        gb_slice = QGroupBox("Spatial Slicing")
        form_slice = QFormLayout()
        
        self.slice_axis = QComboBox()
        self.slice_axis.addItems(["X (Easting)", "Y (Northing)", "Z (Elevation)"])
        form_slice.addRow("Axis:", self.slice_axis)
        
        self.slice_position = QDoubleSpinBox()
        self.slice_position.setRange(-1e9, 1e9)
        form_slice.addRow("Position:", self.slice_position)
        
        self.slice_side = QComboBox()
        self.slice_side.addItems(["Keep Above / Right", "Keep Below / Left"])
        form_slice.addRow("Direction:", self.slice_side)
        
        h_slice_btns = QHBoxLayout()
        self.apply_slice_btn = QPushButton("Cut")
        self.apply_slice_btn.setObjectName("PrimaryButton")
        self.apply_slice_btn.clicked.connect(self._on_apply_slice)
        
        self.clear_slice_btn = QPushButton("Clear")
        self.clear_slice_btn.clicked.connect(self._on_clear_slice)
        
        h_slice_btns.addWidget(self.apply_slice_btn)
        h_slice_btns.addWidget(self.clear_slice_btn)
        form_slice.addRow(h_slice_btns)
        
        gb_slice.setLayout(form_slice)
        right_layout.addWidget(gb_slice)
        
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        
        # Set Splitter Defaults (60% Visuals, 40% Controls)
        splitter.setSizes([600, 400])

    # =========================================================
    # LOGIC
    # =========================================================

    def set_block_model(self, block_model: BlockModel, source: str = "unknown"):
        """
        Load model and populate properties.
        
        STAT-004: Tracks model source for provenance.
        
        Args:
            block_model: The block model to load
            source: Source identifier (e.g., 'kriging', 'sgsim', 'loaded')
        """
        self.current_model = block_model
        self._drillhole_df = None  # Clear drillhole data when block model is set
        self._data_mode = "block_model"
        self._model_source = source
        
        # STAT-004: Update provenance
        self._provenance = {
            'model_source': source,
            'loaded_timestamp': datetime.now().isoformat(),
            'model_id': getattr(block_model, 'model_id', None) or id(block_model),
        }
        
        # Try to get additional provenance from model attrs
        if hasattr(block_model, 'metadata'):
            self._provenance['model_metadata'] = block_model.metadata
        
        # Extract numeric properties only
        numeric_props = []
        if hasattr(block_model, 'properties') and block_model.properties:
            for prop_name, prop_data in block_model.properties.items():
                # Check if numpy array and numeric
                if hasattr(prop_data, 'dtype') and np.issubdtype(prop_data.dtype, np.number):
                    numeric_props.append(prop_name)
        elif hasattr(block_model, 'columns'):
            # Support DataFrame-like block models loaded from registry
            try:
                import pandas as pd
                for col in block_model.columns:
                    if pd.api.types.is_numeric_dtype(block_model[col]):
                        numeric_props.append(str(col))
            except Exception:
                logger.debug("StatisticsPanel: Failed to extract DataFrame numeric columns", exc_info=True)
        
        self.property_box.blockSignals(True)
        self.property_box.clear()
        self.property_box.addItems(numeric_props)
        self.property_box.blockSignals(False)
        
        # Trigger update for the first property
        if numeric_props:
            self.active_property = numeric_props[0]
            self._update_statistics()
        
        logger.info(f"Loaded {len(numeric_props)} properties from source: {source}")

    def _on_property_changed(self, property_name: str):
        if not property_name: return
        self.active_property = property_name

        # In comparison mode, re-run comparison analysis
        if self._comparison_mode:
            selected_keys = self._source_selection_widget.get_selected_sources()
            if len(selected_keys) >= 2:
                self._run_comparison_analysis(selected_keys)
        else:
            self._update_statistics()

        self.property_changed.emit(property_name)

    def _update_statistics(self):
        """
        Calculates stats, updates table and histogram.
        
        STAT-004: Updates provenance with computation timestamp.
        STAT-006: Guards against division by zero in CV calculation.
        
        Supports both block models and drillhole data.
        """
        if not self.active_property:
            return
        
        # Get data based on current mode
        data = None
        
        if self._data_mode == "drillhole" and self._drillhole_df is not None:
            # Drillhole data mode
            if self.active_property in self._drillhole_df.columns:
                data = self._drillhole_df[self.active_property].values
            else:
                logger.warning(f"Property '{self.active_property}' not found in drillhole data")
                return
        elif self._data_mode == "block_model" and self.current_model is not None:
            # Block model mode - handle both BlockModel and DataFrame
            if hasattr(self.current_model, 'properties'):
                # BlockModel API
                data = self.current_model.properties.get(self.active_property)
            elif hasattr(self.current_model, 'columns'):
                # DataFrame
                if self.active_property in self.current_model.columns:
                    data = self.current_model[self.active_property].values
                else:
                    data = None
            else:
                data = None
            
            if data is None:
                logger.warning(f"Property '{self.active_property}' not found in block model")
                return
        else:
            logger.warning("No data loaded (neither block model nor drillhole data)")
            return
        
        # Remove NaNs for calculation
        valid_data = data[~np.isnan(data)]
        count = len(valid_data)
        nan_count = len(data) - count
        self.lbl_count.setText(f"N: {count:,} ({nan_count} NaN)")
        
        if count == 0:
            logger.warning(f"No valid data for property '{self.active_property}'")
            return

        # STAT-004: Record computation timestamp
        self._stats_timestamp = datetime.now().isoformat()
        
        # 1. Compute Statistics
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        var_val = np.var(valid_data)
        
        # STAT-006: Guard against division by zero in CV
        cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0.0
        
        stats_map = {
            "Count": count,
            "NaN Count": nan_count,
            "Mean": mean_val,
            "Median": np.median(valid_data),
            "Std Dev": std_val,
            "CV (%)": cv_val,
            "Variance": var_val,
            "Min": np.min(valid_data),
            "Max": np.max(valid_data),
            "Q1 (25%)": np.percentile(valid_data, 25),
            "Q3 (75%)": np.percentile(valid_data, 75),
            "P10": np.percentile(valid_data, 10),
            "P90": np.percentile(valid_data, 90),
        }
        
        # Store for export
        self._current_stats = stats_map.copy()
        
        self.stats_table.setRowCount(len(stats_map))
        for i, (k, v) in enumerate(stats_map.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(k))
            if isinstance(v, (int, np.integer)):
                val_str = f"{v:,}"
            elif isinstance(v, (float, np.floating)):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)
            self.stats_table.setItem(i, 1, QTableWidgetItem(val_str))
            
        # 2. Update Filter Defaults (Smart UX)
        self.filter_min.setValue(stats_map["Min"])
        self.filter_max.setValue(stats_map["Max"])
        
        # 3. Update Histogram
        self._plot_histogram(valid_data, self.active_property)

    def _plot_histogram(self, data, title):
        if not MATPLOTLIB_AVAILABLE: return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Apply current theme colors
        colors = get_theme_colors()
        ax.set_facecolor(colors.CARD_BG)
        self.figure.patch.set_facecolor(colors.PANEL_BG)

        ax.hist(data, bins=50, color='#3498db', alpha=0.8, edgecolor=colors.CARD_BG)

        ax.set_title(f"Distribution: {title}", color=colors.TEXT_PRIMARY)
        ax.tick_params(colors=colors.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(colors.BORDER)
        ax.grid(True, linestyle='--', alpha=0.3, color=colors.BORDER)
        
        self.figure.tight_layout()
        self.canvas.draw()

    # =========================================================
    # ACTIONS
    # =========================================================

    def _on_apply_filter(self):
        if not self.active_property: return
        mn = self.filter_min.value()
        mx = self.filter_max.value()
        self.filter_applied.emit(self.active_property, mn, mx)
        logger.info(f"Filter applied: {mn} to {mx}")

    def _on_clear_filter(self):
        # Reset spinboxes to data limits
        if self.current_model is not None and self.active_property:
            # Handle both BlockModel and DataFrame
            data = None
            if hasattr(self.current_model, 'properties'):
                # BlockModel API
                data = self.current_model.properties.get(self.active_property)
            elif hasattr(self.current_model, 'columns'):
                # DataFrame
                if self.active_property in self.current_model.columns:
                    data = self.current_model[self.active_property].values
            
            if data is not None:
                self.filter_min.setValue(np.nanmin(data))
                self.filter_max.setValue(np.nanmax(data))
        self.filter_cleared.emit()

    def _on_apply_slice(self):
        axis = self.slice_axis.currentText().split(" ")[0] # "X"
        pos = self.slice_position.value()
        side = "above" if self.slice_side.currentIndex() == 0 else "below"
        self.slice_applied.emit(axis, pos, side)

    def _on_clear_slice(self):
        self.slice_cleared.emit()

    def _export_statistics(self):
        """
        Export statistics with full provenance metadata.
        
        STAT-004: Includes provenance for JORC/SAMREC audit trail.
        """
        if self.stats_table.rowCount() == 0:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Statistics", 
            f"{self.active_property}_stats.csv", 
            "CSV (*.csv)"
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                # STAT-004: Write provenance header
                f.write("# GeoX Block Model Statistics Report\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Property: {self.active_property}\n")
                f.write(f"# Model Source: {self._model_source}\n")
                f.write(f"# Computation Time: {self._stats_timestamp}\n")
                if self._provenance:
                    f.write(f"# Model ID: {self._provenance.get('model_id', 'unknown')}\n")
                    f.write(f"# Loaded: {self._provenance.get('loaded_timestamp', 'unknown')}\n")
                f.write("#\n")
                f.write("Metric,Value\n")
                
                for r in range(self.stats_table.rowCount()):
                    m = self.stats_table.item(r, 0).text()
                    v = self.stats_table.item(r, 1).text()
                    f.write(f"{m},{v}\n")
            
            QMessageBox.information(self, "Exported", f"Saved to {path}")

    # =========================================================
    # REGISTRY CALLBACKS
    # =========================================================
    def _on_block_model_generated(self, block_model):
        # Store the block model and update selector
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        logger.info("StatisticsPanel: Block model available for selection")

    def _on_block_model_loaded(self, block_model):
        # Store the block model and update selector
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        logger.info("StatisticsPanel: Block model available for selection")

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
                logger.warning(f"StatisticsPanel: SGSIM results is not a dict, type={type(results)}")
                return

            variable = results.get('variable', 'Grade')
            summary = results.get('summary', {})
            params = results.get('params')
            grid = results.get('grid') or results.get('pyvista_grid')

            logger.info(f"StatisticsPanel: SGSIM results keys: {list(results.keys())}")
            logger.info(f"StatisticsPanel: Summary keys: {list(summary.keys()) if summary else 'None'}")
            logger.info(f"StatisticsPanel: params = {params is not None}")

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
                        logger.info(f"StatisticsPanel: Extracted {n_blocks:,} cell centers from grid")

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
                    # Transpose to match SGSIM output order: (nz, ny, nx) -> flatten in C order
                    coords_x = xx.transpose(2, 1, 0).flatten()
                    coords_y = yy.transpose(2, 1, 0).flatten()
                    coords_z = zz.transpose(2, 1, 0).flatten()

                    base_df = pd.DataFrame({'X': coords_x, 'Y': coords_y, 'Z': coords_z})
                    n_blocks = len(base_df)
                    logger.info(f"StatisticsPanel: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
                except Exception as e:
                    logger.warning(f"StatisticsPanel: Failed to generate coords from params: {e}")

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
                        logger.info(f"StatisticsPanel: Registered SGSIM E-type Mean from fallback")
                else:
                    logger.warning("StatisticsPanel: No grid, params, or realizations found in SGSIM results")
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
                        logger.info(f"StatisticsPanel: Registered {display_prefix} ({variable})")

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
                logger.info(f"StatisticsPanel: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

            self._update_data_source_selector()

        except Exception as e:
            logger.warning(f"StatisticsPanel: Failed to load SGSIM results: {e}", exc_info=True)

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
            logger.info(f"StatisticsPanel: Switched to drillhole data ({self._data_source_type})")
        elif source_type == "block_model" and self._stored_block_model is not None:
            self._data_mode = "block_model"
            self._drillhole_df = None
            self.set_block_model(self._stored_block_model)
            logger.info("StatisticsPanel: Switched to block model")
        elif source_type == "classified_block_model" and self._stored_classified_block_model is not None:
            self._data_mode = "block_model"  # Same mode, different source
            self._drillhole_df = None
            self.set_block_model(self._stored_classified_block_model)
            logger.info("StatisticsPanel: Switched to classified block model")
        elif source_type == "sgsim" and self._stored_sgsim_df is not None:
            self._data_mode = "block_model"  # Treat as block model
            self._drillhole_df = None
            self.set_block_model(self._stored_sgsim_df)
            logger.info("StatisticsPanel: Switched to SGSIM results")
        elif source_type and source_type.startswith('sgsim_') and source_type in self._block_model_sources:
            # Handle individual SGSIM statistics
            source_info = self._block_model_sources[source_type]
            df = source_info.get('df')
            if df is not None:
                self._data_mode = "block_model"
                self._drillhole_df = None
                self.set_block_model(df)
                logger.info(f"StatisticsPanel: Switched to {source_info.get('display_name', source_type)}")
        else:
            self._data_mode = "none"
            self.property_box.clear()
            self.lbl_count.setText("N: 0")
    
    def _populate_drillhole_properties(self):
        """Populate property selector from drillhole DataFrame."""
        import pandas as pd
        
        if self._drillhole_df is None:
            return
        
        df = self._drillhole_df
        
        # STAT-004: Update provenance
        self._provenance = {
            'data_source': 'drillhole',
            'source_type': self._data_source_type,
            'loaded_timestamp': datetime.now().isoformat(),
            'row_count': len(df),
        }
        self._model_source = f"drillhole ({self._data_source_type})"
        
        # Extract numeric columns for property selection
        numeric_cols = []
        exclude_cols = {'HOLEID', 'HOLE_ID', 'BHID', 'FROM', 'TO', 'FROM_M', 'TO_M', 
                        'X', 'Y', 'Z', 'MID_X', 'MID_Y', 'MID_Z', 'LENGTH', 'SAMPLE_ID'}
        
        for col in df.columns:
            if col.upper() not in exclude_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
        
        self.property_box.blockSignals(True)
        self.property_box.clear()
        self.property_box.addItems(numeric_cols)
        self.property_box.blockSignals(False)
        
        # Trigger update for the first property
        if numeric_cols:
            self.active_property = numeric_cols[0]
            self._update_statistics()
        
        logger.info(f"StatisticsPanel: Loaded {len(numeric_cols)} numeric columns from {self._data_source_type}")
    
    def _on_drillhole_data_loaded(self, drillhole_data):
        """
        Receive drillhole data from DataRegistry.
        
        drillhole_data: dict with keys like 'composites', 'assays', etc.,
        or a plain DataFrame.
        
        Stores the data and updates the selector - user must choose to use it.
        """
        import pandas as pd
        
        logger.info("StatisticsPanel: received drillhole data from DataRegistry")
        
        # Store the drillhole data
        self._store_drillhole_data(drillhole_data)
        
        # Add to available sources and update selector
        if self._stored_drillhole_df is not None:
            if "drillhole" not in self._available_sources:
                self._available_sources.append("drillhole")
            self._update_data_source_selector()
            logger.info("StatisticsPanel: Drillhole data available for selection")

    # ------------------------------------------------------------------
    # Comparison mode methods
    # ------------------------------------------------------------------

    def _on_comparison_mode_changed(self, enabled: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = enabled
        # Hide single source selector when in comparison mode (property_box stays visible for selection)
        self.data_source_box.setVisible(not enabled)
        # Note: property_box should remain visible so user can select which property to compare
        logger.info(f"StatisticsPanel: Comparison mode {'enabled' if enabled else 'disabled'}")

        if enabled:
            self._update_comparison_sources()

    def _on_comparison_sources_changed(self, selected_keys: List[str]):
        """Handle comparison source selection changes."""
        logger.info(f"StatisticsPanel: Comparison sources changed: {selected_keys}")

        # Populate properties from selected sources
        if selected_keys:
            self._populate_comparison_properties(selected_keys)

        if len(selected_keys) >= 2:
            self._run_comparison_analysis(selected_keys)

    def _populate_comparison_properties(self, selected_keys: List[str]):
        """Populate property dropdown with properties from selected sources."""
        import pandas as pd

        all_properties = set()

        for source_key in selected_keys:
            logger.debug(f"StatisticsPanel: Getting properties from source: {source_key}")

            if source_key == "drillhole":
                df = self._stored_drillhole_df
                if df is not None:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH'):
                            all_properties.add(col)
                    logger.debug(f"StatisticsPanel: Drillhole properties: {len(all_properties)}")

            elif source_key == "block_model":
                bm = self._stored_block_model
                if bm is not None:
                    logger.debug(f"StatisticsPanel: Block model type: {type(bm).__name__}")

                    # Check if it's a DataFrame first (common case)
                    if isinstance(bm, pd.DataFrame):
                        for col in bm.columns:
                            if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG'):
                                try:
                                    if pd.api.types.is_numeric_dtype(bm[col]):
                                        all_properties.add(col)
                                except:
                                    all_properties.add(col)
                        logger.debug(f"StatisticsPanel: Block model DataFrame has columns: {list(bm.columns)[:10]}")
                    else:
                        # BlockModel class
                        if hasattr(bm, 'properties') and bm.properties:
                            props = list(bm.properties.keys())
                            all_properties.update(p for p in props if p.upper() not in ('X', 'Y', 'Z'))
                            logger.debug(f"StatisticsPanel: Block model properties: {props[:5]}")
                        # Try to_dataframe
                        if hasattr(bm, 'to_dataframe'):
                            try:
                                df = bm.to_dataframe()
                                for col in df.columns:
                                    if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            all_properties.add(col)
                                logger.debug(f"StatisticsPanel: Block model DataFrame has columns: {list(df.columns)[:10]}")
                            except Exception as e:
                                logger.debug(f"StatisticsPanel: Could not convert block model to dataframe: {e}")

            elif source_key == "classified_block_model":
                bm = self._stored_classified_block_model
                if bm is not None:
                    if hasattr(bm, 'properties') and bm.properties:
                        all_properties.update(p for p in bm.properties.keys() if p.upper() not in ('X', 'Y', 'Z'))
                    if hasattr(bm, 'columns'):
                        for col in bm.columns:
                            if col.upper() not in ('X', 'Y', 'Z'):
                                all_properties.add(col)

            elif source_key in self._block_model_sources:
                # SGSIM sources have their own property
                source_info = self._block_model_sources[source_key]
                prop = source_info.get('property')
                if prop:
                    all_properties.add(prop)
                df = source_info.get('df')
                if df is not None:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z'):
                            all_properties.add(col)
                    logger.debug(f"StatisticsPanel: SGSIM source {source_key} properties: {list(df.columns)[:5]}")

            else:
                # Check comparison widget sources
                source_info = self._source_selection_widget._sources.get(source_key, {})
                df = source_info.get('df')
                if df is not None and hasattr(df, 'columns'):
                    for col in df.columns:
                        if col.upper() not in ('X', 'Y', 'Z') and pd.api.types.is_numeric_dtype(df[col]):
                            all_properties.add(col)
                    logger.debug(f"StatisticsPanel: Widget source {source_key} properties: {list(df.columns)[:5]}")

        logger.info(f"StatisticsPanel: Total properties collected: {len(all_properties)}")

        # Update property dropdown
        current_prop = self.property_box.currentText()
        self.property_box.blockSignals(True)
        self.property_box.clear()

        sorted_props = sorted(all_properties)
        self.property_box.addItems(sorted_props)

        # Try to restore previous selection or select first grade-like property
        if current_prop and current_prop in sorted_props:
            self.property_box.setCurrentText(current_prop)
        else:
            # Try to find a grade-related property
            for prop in sorted_props:
                if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')):
                    self.property_box.setCurrentText(prop)
                    break

        self.property_box.blockSignals(False)
        logger.info(f"StatisticsPanel: Populated {len(sorted_props)} properties for comparison: {sorted_props[:5]}...")

    def _update_comparison_sources(self):
        """Update the comparison widget with available sources."""
        sources = {}
        logger.info(f"StatisticsPanel: Updating comparison sources. Available: {self._available_sources}")
        logger.debug(f"StatisticsPanel: _stored_block_model is {'set' if self._stored_block_model is not None else 'None'}")
        logger.debug(f"StatisticsPanel: _block_model_sources has {len(self._block_model_sources)} entries")

        # Add drillhole data
        if "drillhole" in self._available_sources and self._stored_drillhole_df is not None:
            sources["drillhole"] = {
                'display_name': f"Drillhole ({self._stored_data_source_type})",
                'block_count': len(self._stored_drillhole_df),
                'df': self._stored_drillhole_df
            }
            logger.debug(f"StatisticsPanel: Added drillhole source with {len(self._stored_drillhole_df)} rows")

        # Add block model
        if "block_model" in self._available_sources and self._stored_block_model is not None:
            bm = self._stored_block_model
            count = len(bm) if hasattr(bm, '__len__') else 0
            sources["block_model"] = {
                'display_name': "Block Model",
                'block_count': count,
                'data': bm  # Store reference for property extraction
            }
            logger.info(f"StatisticsPanel: Added block_model source with {count} blocks, type: {type(bm).__name__}")

        # Add classified block model
        if "classified_block_model" in self._available_sources and self._stored_classified_block_model is not None:
            bm = self._stored_classified_block_model
            count = len(bm) if hasattr(bm, '__len__') else 0
            sources["classified_block_model"] = {
                'display_name': "Classified Block Model",
                'block_count': count,
                'data': bm  # Store reference for property extraction
            }
            logger.debug(f"StatisticsPanel: Added classified_block_model source with {count} blocks")

        # Add individual SGSIM sources from _block_model_sources
        for source_key, source_info in self._block_model_sources.items():
            df = source_info.get('df')
            if df is not None:
                sources[source_key] = {
                    'display_name': source_info.get('display_name', source_key),
                    'block_count': len(df),
                    'df': df,
                    'property': source_info.get('property')
                }
                logger.debug(f"StatisticsPanel: Added SGSIM source {source_key} with {len(df)} rows")

        # Fallback: Try to extract SGSIM sources directly if _block_model_sources is empty
        if not self._block_model_sources and self.registry:
            try:
                sgsim = self.registry.get_sgsim_results() if hasattr(self.registry, 'get_sgsim_results') else None
                if sgsim and isinstance(sgsim, dict):
                    summary = sgsim.get('summary', {})
                    variable = sgsim.get('variable', 'Grade')

                    # Check if we have the stored SGSIM DataFrame with stats
                    if self._stored_sgsim_df is not None and len(self._stored_sgsim_df) > 0:
                        df = self._stored_sgsim_df
                        n_blocks = len(df)

                        # Look for summary statistic columns
                        stat_mapping = {
                            'mean': 'SGSIM Mean',
                            'std': 'SGSIM Std Dev',
                            'p10': 'SGSIM P10',
                            'p50': 'SGSIM P50',
                            'p90': 'SGSIM P90',
                        }

                        for stat_key, display_prefix in stat_mapping.items():
                            prop_name = f"{variable}_{stat_key.upper()}"
                            if prop_name in df.columns:
                                source_key = f"sgsim_{stat_key}_{variable}"
                                sources[source_key] = {
                                    'display_name': f"{display_prefix} ({variable})",
                                    'block_count': n_blocks,
                                    'df': df,
                                    'property': prop_name
                                }
                                logger.info(f"StatisticsPanel: Added SGSIM source from stored df: {source_key}")

                    # Alternative: Extract directly from summary arrays
                    elif summary:
                        import pandas as pd
                        # We need coordinates - try to get from block model
                        base_coords = None
                        if self._stored_block_model is not None:
                            bm = self._stored_block_model
                            if hasattr(bm, 'columns') and 'X' in bm.columns:
                                base_coords = bm[['X', 'Y', 'Z']].copy()
                            elif hasattr(bm, 'positions'):
                                base_coords = pd.DataFrame(bm.positions, columns=['X', 'Y', 'Z'])

                        if base_coords is not None:
                            n_blocks = len(base_coords)
                            for stat_key in ['mean', 'std', 'p10', 'p50', 'p90']:
                                stat_data = summary.get(stat_key)
                                if stat_data is not None:
                                    stat_values = np.asarray(stat_data).flatten()
                                    if len(stat_values) == n_blocks:
                                        df = base_coords.copy()
                                        prop_name = f"{variable}_{stat_key.upper()}"
                                        df[prop_name] = stat_values
                                        source_key = f"sgsim_{stat_key}_{variable}"
                                        sources[source_key] = {
                                            'display_name': f"SGSIM {stat_key.title()} ({variable})",
                                            'block_count': n_blocks,
                                            'df': df,
                                            'property': prop_name
                                        }
                                        logger.info(f"StatisticsPanel: Added SGSIM source from summary: {source_key}")

            except Exception as e:
                logger.warning(f"StatisticsPanel: Failed to extract SGSIM sources for comparison: {e}")

        logger.info(f"StatisticsPanel._update_comparison_sources: Found {len(sources)} sources: {list(sources.keys())}")
        self._source_selection_widget.update_sources(sources)

    def _get_data_array_for_source(self, source_key: str, property_name: str = None):
        """Get data array for a given source and property.

        Returns tuple of (data_array, actual_property_name)
        """
        import pandas as pd

        if source_key == "drillhole":
            df = self._stored_drillhole_df
            if df is None:
                return None, None
            # Find first numeric column if property not specified
            if property_name and property_name in df.columns:
                return df[property_name].values, property_name
            # Find property by pattern
            if property_name:
                for col in df.columns:
                    if property_name.upper() in col.upper():
                        return df[col].values, col
            # Fall back to first numeric column
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z'):
                    return df[col].values, col
            return None, None

        if source_key == "block_model":
            bm = self._stored_block_model
            if bm is None:
                return None, None
            return self._find_property_in_model(bm, property_name)

        if source_key == "classified_block_model":
            bm = self._stored_classified_block_model
            if bm is None:
                return None, None
            return self._find_property_in_model(bm, property_name)

        # Handle SGSIM and other sources from _block_model_sources
        if source_key in self._block_model_sources:
            source_info = self._block_model_sources[source_key]
            df = source_info.get('df')
            prop = source_info.get('property')  # SGSIM sources have built-in property
            if df is not None and prop and prop in df.columns:
                return df[prop].values, prop
            return None, None

        # Also check in comparison widget sources
        source_info = self._source_selection_widget._sources.get(source_key, {})
        df = source_info.get('df')
        if df is not None:
            prop = source_info.get('property')
            if prop and prop in df.columns:
                return df[prop].values, prop
            # Try property matching
            if property_name and property_name in df.columns:
                return df[property_name].values, property_name
            # Try to find matching property
            for col in df.columns:
                if property_name and property_name.upper() in col.upper():
                    return df[col].values, col

        return None, None

    def _find_property_in_model(self, bm, property_name: str):
        """Find property in block model with flexible matching."""
        import pandas as pd

        # Get available properties
        available_props = []
        if hasattr(bm, 'properties') and bm.properties:
            available_props = list(bm.properties.keys())
        elif hasattr(bm, 'columns'):
            available_props = list(bm.columns)

        if not available_props:
            return None, None

        # Strategy 1: Exact match
        if property_name in available_props:
            if hasattr(bm, 'properties'):
                return bm.properties[property_name], property_name
            return bm[property_name].values, property_name

        # Strategy 2: Property name contains column name (e.g., "FE_SGSIM_MEAN" -> "FE")
        for col in available_props:
            if col.upper() in property_name.upper() and col.upper() not in ('X', 'Y', 'Z'):
                if hasattr(bm, 'properties'):
                    return bm.properties[col], col
                return bm[col].values, col

        # Strategy 3: Column name contains property name
        for col in available_props:
            if property_name.upper() in col.upper():
                if hasattr(bm, 'properties'):
                    return bm.properties[col], col
                return bm[col].values, col

        # Strategy 4: Base name match (e.g., "Grade_MEAN" -> "Grade")
        base_name = property_name.split('_')[0] if '_' in property_name else property_name
        for col in available_props:
            col_base = col.split('_')[0] if '_' in col else col
            if base_name.upper() == col_base.upper() and col.upper() not in ('X', 'Y', 'Z'):
                if hasattr(bm, 'properties'):
                    return bm.properties[col], col
                return bm[col].values, col

        # Strategy 5: Find any grade-like property
        for col in available_props:
            if any(k in col.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')) and col.upper() not in ('X', 'Y', 'Z'):
                if hasattr(bm, 'properties'):
                    return bm.properties[col], col
                try:
                    return bm[col].values, col
                except:
                    pass

        logger.warning(f"StatisticsPanel: No matching property for '{property_name}' in block model. Available: {available_props[:10]}")
        return None, None

    def _run_comparison_analysis(self, selected_keys: List[str]):
        """Run comparison analysis for selected sources."""
        self._comparison_stats = {}

        property_name = self.property_box.currentText() if self.property_box.currentText() else None

        for source_key in selected_keys:
            data, actual_prop = self._get_data_array_for_source(source_key, property_name)

            if data is None or len(data) == 0:
                continue

            # Remove NaNs (handle both numeric and non-numeric data)
            import pandas as pd
            series = pd.Series(data)
            # Convert to numeric, coercing errors to NaN
            numeric_series = pd.to_numeric(series, errors='coerce')
            # Remove NaN values
            valid_data = numeric_series.dropna().values
            if len(valid_data) == 0:
                continue

            # Compute statistics
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)

            stats = {
                'Count': len(valid_data),
                'Mean': mean_val,
                'Median': np.median(valid_data),
                'Std Dev': std_val,
                'CV (%)': (std_val / mean_val * 100) if mean_val != 0 else 0.0,
                'Min': np.min(valid_data),
                'Max': np.max(valid_data),
                'P10': np.percentile(valid_data, 10),
                'P50': np.percentile(valid_data, 50),
                'P90': np.percentile(valid_data, 90),
            }

            # Get display name
            source_info = self._source_selection_widget._sources.get(source_key, {})
            display_name = source_info.get('display_name', source_key)

            self._comparison_stats[source_key] = {
                'stats': stats,
                'data': valid_data,
                'display_name': display_name,
                'property': actual_prop,
            }

        if len(self._comparison_stats) >= 2:
            self._update_comparison_table()
            self._plot_comparison_histograms()

    def _update_comparison_table(self):
        """Update stats table with comparison data."""
        if not self._comparison_stats:
            return

        # Build comparison table
        sources = list(self._comparison_stats.keys())
        stat_names = ['Count', 'Mean', 'Median', 'Std Dev', 'CV (%)', 'Min', 'Max', 'P10', 'P50', 'P90']

        # Update table - columns: Metric, Source1, Source2, ...
        n_cols = 1 + len(sources)
        self.stats_table.setColumnCount(n_cols)

        headers = ["Metric"]
        for source_key in sources:
            display_name = self._comparison_stats[source_key]['display_name']
            # Truncate long names
            headers.append(display_name[:20] if len(display_name) > 20 else display_name)
        self.stats_table.setHorizontalHeaderLabels(headers)

        self.stats_table.setRowCount(len(stat_names))

        for row, stat_name in enumerate(stat_names):
            self.stats_table.setItem(row, 0, QTableWidgetItem(stat_name))

            for col, source_key in enumerate(sources, start=1):
                stats = self._comparison_stats[source_key]['stats']
                value = stats.get(stat_name, 0)

                if isinstance(value, (int, np.integer)):
                    val_str = f"{value:,}"
                elif isinstance(value, (float, np.floating)):
                    val_str = f"{value:.4f}"
                else:
                    val_str = str(value)

                item = QTableWidgetItem(val_str)
                # Color code based on comparison
                self.stats_table.setItem(row, col, item)

        # Resize columns to fit
        self.stats_table.resizeColumnsToContents()

        # Update count label
        total_count = sum(s['stats']['Count'] for s in self._comparison_stats.values())
        self.lbl_count.setText(f"Comparing {len(self._comparison_stats)} sources (N: {total_count:,})")

    def _plot_comparison_histograms(self):
        """Plot overlaid histograms for comparison."""
        if not MATPLOTLIB_AVAILABLE or not self._comparison_stats:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Apply current theme colors
        colors = get_theme_colors()
        ax.set_facecolor(colors.CARD_BG)
        self.figure.patch.set_facecolor(colors.PANEL_BG)

        # Determine common bin edges
        all_data = np.concatenate([s['data'] for s in self._comparison_stats.values()])
        bins = np.linspace(all_data.min(), all_data.max(), 50)

        # Plot each source with different colors
        for i, (source_key, result) in enumerate(self._comparison_stats.items()):
            data = result['data']
            display_name = result['display_name']
            color = ComparisonColors.get_color(i)

            ax.hist(data, bins=bins, alpha=0.5, color=color, label=display_name,
                   edgecolor=color, linewidth=1.2)

        ax.set_title("Distribution Comparison", color=colors.TEXT_PRIMARY, fontweight='bold')
        ax.set_xlabel("Value", color=colors.TEXT_PRIMARY)
        ax.set_ylabel("Frequency", color=colors.TEXT_PRIMARY)
        ax.tick_params(colors=colors.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(colors.BORDER)
        ax.grid(True, linestyle='--', alpha=0.3, color=colors.BORDER)

        # Legend
        legend = ax.legend(loc='upper right', frameon=True,
                          facecolor=colors.ELEVATED_BG, edgecolor=colors.BORDER,
                          fontsize=9, labelcolor=colors.TEXT_PRIMARY)
        legend.get_frame().set_alpha(0.9)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear all state and reset provenance."""
        self.current_model = None
        self._drillhole_df = None
        self._data_mode = "none"
        self._data_source_type = "unknown"
        self.active_property = None
        self._provenance = {}
        self._model_source = "unknown"
        self._stats_timestamp = None
        self._current_stats = {}

        self.property_box.clear()
        self.stats_table.setRowCount(0)
        self.lbl_count.setText("N: 0")
        if MATPLOTLIB_AVAILABLE:
            self.figure.clear()
            self.canvas.draw()

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        # Use existing clear() method for state clearing
        self.clear()
        # Call base class to clear common widgets
        super().clear_panel()
        logger.info("StatisticsPanel: Panel fully cleared")

    def get_registry(self):
        """Get the DataRegistry singleton."""
        try:
            from ..core.data_registry import DataRegistry
            # GeoX registry API uses instance(), not get_instance()
            return DataRegistry.instance()
        except ImportError:
            return None
        except Exception:
            logger.debug("StatisticsPanel: Failed to get DataRegistry instance", exc_info=True)
            return None
