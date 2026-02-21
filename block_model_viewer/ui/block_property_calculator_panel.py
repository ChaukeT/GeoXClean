"""
Block Property Calculator Panel.

Calculates and permanently adds fundamental properties to block models:
- Tonnage (from volume × density)
- Volume (from block dimensions)
- Composite grades (weighted averages, etc.)

Properties are saved back to the block model in the registry for permanent storage.

Author: GeoX Mining Software
"""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QMessageBox,
    QFormLayout, QFrame, QScrollArea, QCheckBox, QLineEdit,
    QProgressBar, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont

from .base_analysis_panel import BaseAnalysisPanel
from .panel_manager import PanelCategory, DockArea
from .modern_styles import get_theme_colors, get_analysis_panel_stylesheet

logger = logging.getLogger(__name__)


class PropertyCalculationWorker(QThread):
    """Background worker for calculating block properties."""

    progressUpdate = pyqtSignal(str, int)  # message, percentage
    calculationComplete = pyqtSignal(pd.DataFrame)  # updated dataframe
    calculationError = pyqtSignal(str)  # error message

    def __init__(self, block_model_df: pd.DataFrame, calculation_config: Dict[str, Any]):
        super().__init__()
        self.block_model_df = block_model_df.copy()
        self.config = calculation_config

    def run(self):
        """Execute property calculations in background."""
        try:
            self.progressUpdate.emit("Starting property calculations...", 10)

            df = self.block_model_df
            config = self.config

            # Calculate volume if requested
            if config.get('calculate_volume', False):
                self.progressUpdate.emit("Calculating block volumes...", 30)

                dimension_source = config.get('dimension_source', 'From Constant Values')

                if dimension_source == 'From Constant Values':
                    # Use constant dimensions
                    block_x = config['block_x']
                    block_y = config['block_y']
                    block_z = config['block_z']
                    volume = block_x * block_y * block_z
                    df['VOLUME'] = volume
                    logger.info(f"Added VOLUME column (constant): {volume:.2f} m³ per block")
                else:
                    # Use columns for variable dimensions
                    dim_x_col = config.get('dim_x_column')
                    dim_y_col = config.get('dim_y_column')
                    dim_z_col = config.get('dim_z_column')

                    # Validate columns exist
                    missing_cols = []
                    if dim_x_col not in df.columns:
                        missing_cols.append(dim_x_col)
                    if dim_y_col not in df.columns:
                        missing_cols.append(dim_y_col)
                    if dim_z_col not in df.columns:
                        missing_cols.append(dim_z_col)

                    if missing_cols:
                        raise ValueError(f"Dimension columns not found in block model: {', '.join(missing_cols)}")

                    # Calculate volume from columns (variable per block)
                    df['VOLUME'] = df[dim_x_col] * df[dim_y_col] * df[dim_z_col]

                    valid_volume = df['VOLUME'].dropna()
                    logger.info(
                        f"Added VOLUME column (variable): {valid_volume.min():.2f} - {valid_volume.max():.2f} m³ "
                        f"(mean: {valid_volume.mean():.2f})"
                    )

            # Calculate tonnage if requested
            if config.get('calculate_tonnage', False):
                self.progressUpdate.emit("Calculating block tonnages...", 60)

                density_source = config.get('density_source', 'From Column')

                # Check for volume column
                if 'VOLUME' not in df.columns:
                    # Calculate volume first if not exists
                    dimension_source = config.get('dimension_source', 'From Constant Values')
                    if dimension_source == 'From Constant Values':
                        block_x = config['block_x']
                        block_y = config['block_y']
                        block_z = config['block_z']
                        df['VOLUME'] = block_x * block_y * block_z
                    else:
                        dim_x_col = config.get('dim_x_column')
                        dim_y_col = config.get('dim_y_column')
                        dim_z_col = config.get('dim_z_column')
                        df['VOLUME'] = df[dim_x_col] * df[dim_y_col] * df[dim_z_col]

                # Get density values
                if density_source == 'From Column':
                    density_col = config['density_column']

                    if density_col not in df.columns:
                        raise ValueError(f"Density column '{density_col}' not found in block model")

                    # Validate density values
                    nan_count = df[density_col].isna().sum()
                    if nan_count > 0:
                        logger.warning(f"{nan_count} blocks have NaN density - tonnage will be NaN for these blocks")

                    invalid_density = df[~df[density_col].isna() & (df[density_col] <= 0)]
                    if len(invalid_density) > 0:
                        raise ValueError(
                            f"Density column '{density_col}' contains {len(invalid_density)} blocks with zero or negative values.\n"
                            f"Density must be positive (typical range: 1.5 - 5.0 tonnes/m³).\n"
                            f"Invalid density range: {df[density_col].min():.3f} - {df[density_col].max():.3f}"
                        )

                    # Calculate tonnage = volume × density (from column)
                    df['TONNAGE'] = df['VOLUME'] * df[density_col]
                else:
                    # Use constant density value
                    density_value = config['density_value']
                    density_unit = config['density_unit']

                    # Convert density to t/m³
                    if density_unit == 'kg/m³':
                        density_t_m3 = density_value / 1000.0
                    elif density_unit == 'g/cm³':
                        density_t_m3 = density_value  # g/cm³ ≈ t/m³
                    else:  # 't/m³'
                        density_t_m3 = density_value

                    # Validate density
                    if density_t_m3 <= 0:
                        raise ValueError(f"Density must be positive (got {density_value} {density_unit})")

                    # Calculate tonnage = volume × density (constant)
                    df['TONNAGE'] = df['VOLUME'] * density_t_m3
                    logger.info(f"Using constant density: {density_t_m3:.3f} t/m³")

                valid_tonnage = df['TONNAGE'].dropna()
                logger.info(
                    f"Added TONNAGE column: {valid_tonnage.min():.2f} - {valid_tonnage.max():.2f} tonnes "
                    f"(mean: {valid_tonnage.mean():.2f}, {len(valid_tonnage):,} valid blocks)"
                )

            self.progressUpdate.emit("Calculations complete!", 100)
            self.calculationComplete.emit(df)

        except Exception as e:
            logger.exception("Property calculation failed")
            self.calculationError.emit(str(e))


class BlockPropertyCalculatorPanel(BaseAnalysisPanel):
    """
    Panel for calculating and permanently adding properties to block models.

    Calculates:
    - Volume: From block dimensions (X × Y × Z)
    - Tonnage: From volume × density

    Properties are saved back to the block model in the registry.
    """

    # PanelManager metadata
    PANEL_ID = "BlockPropertyCalculator"
    PANEL_NAME = "Block Property Calculator"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    # Signals
    propertiesCalculated = pyqtSignal(object)  # Updated block model

    def __init__(self, parent=None, panel_id=None):
        # State attributes
        self._block_model = None
        self._block_model_df = None
        self._calculation_worker = None

        # Source registry (like JORC classification panel)
        self._available_sources = []  # List of source keys
        self._block_model_sources = {}  # Dict of {key: {df, display_name, property, source_type}}
        self._current_source = ""  # Currently selected source key

        super().__init__(parent=parent, panel_id=panel_id)

        # Connect to registry
        self._init_registry()

        logger.info("Block Property Calculator Panel initialized")

    def _init_registry(self):
        """Initialize connection to data registry."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                connected_signals = []

                def safe_connect(signal_name, handler):
                    signal = getattr(self.registry, signal_name, None)
                    if signal is not None:
                        try:
                            signal.connect(handler)
                            connected_signals.append(signal_name)
                        except (TypeError, AttributeError) as e:
                            logger.debug(f"Could not connect {signal_name}: {e}")

                # Block model signals
                safe_connect('blockModelLoaded', self._on_model_changed)
                safe_connect('blockModelGenerated', self._on_model_changed)
                safe_connect('blockModelClassified', self._on_model_changed)
                safe_connect('currentBlockModelChanged', self._on_model_changed)

                # Estimation result signals
                for sig in [
                    'krigingResultsLoaded',
                    'simpleKrigingResultsLoaded',
                    'universalKrigingResultsLoaded',
                    'cokrigingResultsLoaded',
                    'indicatorKrigingResultsLoaded',
                    'softKrigingResultsLoaded',
                    'sgsimResultsLoaded',
                ]:
                    safe_connect(sig, self._on_model_changed)

                logger.info(f"Block Property Calculator connected to registry ({len(connected_signals)} signals)")

                # Populate block model selector dropdown
                # Use QTimer to ensure UI is fully set up before populating
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, self._refresh_from_registry)

        except Exception as e:
            logger.error(f"Failed to connect to registry: {e}")
            self.registry = None

    def setup_ui(self):
        """Set up the user interface using modern styles."""
        layout = self.main_layout if hasattr(self, 'main_layout') else QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Configuration panel
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)

        # Right: Preview/Results panel
        right_panel = self._create_preview_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        layout.addWidget(splitter)

        # Apply modern stylesheet
        self.setStyleSheet(get_analysis_panel_stylesheet())

    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Scroll area for configuration options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(15)

        # Block Model Selection
        config_layout.addWidget(self._create_block_model_selection_group())

        # Volume Calculation
        config_layout.addWidget(self._create_volume_calculation_group())

        # Tonnage Calculation
        config_layout.addWidget(self._create_tonnage_calculation_group())

        # Action Buttons
        config_layout.addWidget(self._create_action_buttons_group())

        config_layout.addStretch()
        scroll.setWidget(config_widget)
        layout.addWidget(scroll)

        return panel

    def _create_block_model_selection_group(self) -> QGroupBox:
        """Create block model selection group."""
        group = QGroupBox("1. Block Model Selection")
        colors = get_theme_colors()
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                color: {colors.ACCENT_PRIMARY};
                border: 2px solid {colors.ACCENT_PRIMARY};
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)

        form = QFormLayout(group)
        form.setSpacing(8)

        # Block model selector dropdown
        self.block_model_selector = QComboBox()
        self.block_model_selector.setToolTip("Select which block model to calculate properties for")
        self.block_model_selector.currentIndexChanged.connect(self._on_block_model_selected)
        form.addRow("Block Model:", self.block_model_selector)

        # Block model info
        self.block_model_info = QLabel("No block model loaded")
        self.block_model_info.setWordWrap(True)
        self.block_model_info.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; padding: 5px;")
        form.addRow("Info:", self.block_model_info)

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh Available Models")
        self.refresh_btn.clicked.connect(lambda: self._refresh_from_registry(show_message=True))
        form.addRow("", self.refresh_btn)

        return group

    def _create_volume_calculation_group(self) -> QGroupBox:
        """Create volume calculation configuration group."""
        group = QGroupBox("2. Volume Calculation")
        colors = get_theme_colors()
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                color: {colors.SUCCESS};
                border: 2px solid {colors.SUCCESS};
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)

        form = QFormLayout(group)
        form.setSpacing(8)

        # Enable volume calculation
        self.calculate_volume_check = QCheckBox("Calculate Volume Column")
        self.calculate_volume_check.setChecked(True)
        self.calculate_volume_check.toggled.connect(self._on_volume_toggle)
        form.addRow("", self.calculate_volume_check)

        # Dimension source selector
        self.dimension_source = QComboBox()
        self.dimension_source.addItems([
            "From Constant Values",
            "From Columns (Variable Dimensions)"
        ])
        self.dimension_source.currentIndexChanged.connect(self._on_dimension_source_changed)
        form.addRow("Dimension Source:", self.dimension_source)

        # Constant dimensions (shown by default)
        dim_layout = QHBoxLayout()

        self.block_x = QDoubleSpinBox()
        self.block_x.setRange(0.01, 10000)
        self.block_x.setValue(10.0)
        self.block_x.setDecimals(2)
        self.block_x.setSuffix(" m")
        dim_layout.addWidget(QLabel("X:"))
        dim_layout.addWidget(self.block_x)

        self.block_y = QDoubleSpinBox()
        self.block_y.setRange(0.01, 10000)
        self.block_y.setValue(10.0)
        self.block_y.setDecimals(2)
        self.block_y.setSuffix(" m")
        dim_layout.addWidget(QLabel("Y:"))
        dim_layout.addWidget(self.block_y)

        self.block_z = QDoubleSpinBox()
        self.block_z.setRange(0.01, 10000)
        self.block_z.setValue(10.0)
        self.block_z.setDecimals(2)
        self.block_z.setSuffix(" m")
        dim_layout.addWidget(QLabel("Z:"))
        dim_layout.addWidget(self.block_z)

        self.constant_dims_widget = QWidget()
        self.constant_dims_widget.setLayout(dim_layout)
        form.addRow("Dimensions:", self.constant_dims_widget)

        # Column-based dimensions (hidden by default)
        col_dim_layout = QVBoxLayout()
        col_dim_layout.setSpacing(4)

        x_col_layout = QHBoxLayout()
        x_col_layout.addWidget(QLabel("X Column:"))
        self.dim_x_column = QComboBox()
        self.dim_x_column.setToolTip("Select column containing X dimension (DX) in meters")
        x_col_layout.addWidget(self.dim_x_column)
        col_dim_layout.addLayout(x_col_layout)

        y_col_layout = QHBoxLayout()
        y_col_layout.addWidget(QLabel("Y Column:"))
        self.dim_y_column = QComboBox()
        self.dim_y_column.setToolTip("Select column containing Y dimension (DY) in meters")
        y_col_layout.addWidget(self.dim_y_column)
        col_dim_layout.addLayout(y_col_layout)

        z_col_layout = QHBoxLayout()
        z_col_layout.addWidget(QLabel("Z Column:"))
        self.dim_z_column = QComboBox()
        self.dim_z_column.setToolTip("Select column containing Z dimension (DZ) in meters")
        z_col_layout.addWidget(self.dim_z_column)
        col_dim_layout.addLayout(z_col_layout)

        self.column_dims_widget = QWidget()
        self.column_dims_widget.setLayout(col_dim_layout)
        self.column_dims_widget.setVisible(False)
        form.addRow("", self.column_dims_widget)

        # Auto-detect button
        self.auto_detect_dims_btn = QPushButton("Auto-Detect from Model")
        self.auto_detect_dims_btn.clicked.connect(self._auto_detect_dimensions)
        form.addRow("", self.auto_detect_dims_btn)

        # Formula display
        formula_label = QLabel("Formula: Volume = X × Y × Z")
        formula_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-style: italic; font-size: 9pt;")
        form.addRow("", formula_label)

        return group

    def _create_tonnage_calculation_group(self) -> QGroupBox:
        """Create tonnage calculation configuration group."""
        group = QGroupBox("3. Tonnage Calculation")
        colors = get_theme_colors()
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                color: {colors.WARNING};
                border: 2px solid {colors.WARNING};
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)

        form = QFormLayout(group)
        form.setSpacing(8)

        # Enable tonnage calculation
        self.calculate_tonnage_check = QCheckBox("Calculate Tonnage Column")
        self.calculate_tonnage_check.setChecked(True)
        self.calculate_tonnage_check.toggled.connect(self._on_tonnage_toggle)
        form.addRow("", self.calculate_tonnage_check)

        # Density source selector
        self.density_source = QComboBox()
        self.density_source.addItems([
            "From Column",
            "Constant Value"
        ])
        self.density_source.currentIndexChanged.connect(self._on_density_source_changed)
        form.addRow("Density Source:", self.density_source)

        # Density column selector (shown by default)
        self.density_column = QComboBox()
        self.density_column.setToolTip("Select the density/SG column (tonnes/m³)")
        form.addRow("Density Column:", self.density_column)

        # Constant density value (hidden by default)
        density_value_layout = QHBoxLayout()

        self.density_value = QDoubleSpinBox()
        self.density_value.setRange(0.1, 20.0)
        self.density_value.setValue(2.7)
        self.density_value.setDecimals(3)
        self.density_value.setSingleStep(0.1)
        self.density_value.setToolTip("Constant density value for tonnage calculation")
        density_value_layout.addWidget(self.density_value)

        self.density_unit = QComboBox()
        self.density_unit.addItems(['t/m³', 'g/cm³', 'kg/m³'])
        self.density_unit.setCurrentText('t/m³')
        self.density_unit.currentIndexChanged.connect(self._on_density_unit_changed)
        density_value_layout.addWidget(self.density_unit)

        self.constant_density_widget = QWidget()
        self.constant_density_widget.setLayout(density_value_layout)
        self.constant_density_widget.setVisible(False)
        form.addRow("Density Value:", self.constant_density_widget)

        # Formula display
        formula_label = QLabel("Formula: Tonnage = Volume × Density")
        formula_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-style: italic; font-size: 9pt;")
        form.addRow("", formula_label)

        # Note about dependencies
        note_label = QLabel("⚠️ Requires Volume column (calculated above or existing)")
        note_label.setStyleSheet(f"color: {colors.WARNING}; font-size: 9pt;")
        note_label.setWordWrap(True)
        form.addRow("", note_label)

        return group

    def _create_action_buttons_group(self) -> QGroupBox:
        """Create action buttons group."""
        group = QGroupBox("4. Actions")
        colors = get_theme_colors()
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                color: {colors.ACCENT_SECONDARY};
                border: 2px solid {colors.ACCENT_SECONDARY};
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)

        form = QFormLayout(group)
        form.setSpacing(10)

        # Calculate button
        self.calculate_btn = QPushButton("📊 Calculate Properties")
        self.calculate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 11pt;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        self.calculate_btn.clicked.connect(self._run_calculation)
        self.calculate_btn.setEnabled(False)
        form.addRow("", self.calculate_btn)

        # Save button
        self.save_btn = QPushButton("💾 Save to Block Model")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 11pt;
            }
            QPushButton:hover { background-color: #0b7dda; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        self.save_btn.clicked.connect(self._save_to_block_model)
        self.save_btn.setEnabled(False)
        form.addRow("", self.save_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        form.addRow("Progress:", self.progress_bar)

        return group

    def _create_preview_panel(self) -> QWidget:
        """Create the preview/results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QLabel("Preview & Results")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(header)

        # Preview text
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlainText(
            "Load a block model and configure calculations to see preview.\n\n"
            "This panel will show:\n"
            "- Current block model properties\n"
            "- Properties that will be added\n"
            "- Sample calculations\n"
            "- Validation results"
        )
        layout.addWidget(self.preview_text)

        return panel

    def _on_model_changed(self, block_model=None):
        """Handle when models change in registry - refresh dropdown."""
        try:
            # Refresh the dropdown to show newly available models
            self._refresh_from_registry()
        except Exception as e:
            logger.error(f"Error refreshing model list: {e}", exc_info=True)

    def _on_block_model_loaded(self, block_model):
        """Handle block model loaded from registry."""
        try:
            if block_model is None:
                return

            self._block_model = block_model

            # Convert to dataframe
            if hasattr(block_model, 'to_dataframe'):
                self._block_model_df = block_model.to_dataframe()
            elif isinstance(block_model, pd.DataFrame):
                self._block_model_df = block_model
            elif hasattr(block_model, 'data') and isinstance(block_model.data, pd.DataFrame):
                self._block_model_df = block_model.data
            else:
                logger.warning(f"Unsupported block model type: {type(block_model)}")
                return

            # Update UI
            self._update_block_model_info()
            self._update_column_selectors()
            self._auto_detect_dimensions()

            self.calculate_btn.setEnabled(True)

            logger.info(f"Block model loaded: {len(self._block_model_df):,} blocks")

        except Exception as e:
            logger.exception("Failed to load block model")
            QMessageBox.critical(self, "Error", f"Failed to load block model:\n{str(e)}")

    def _register_source(self, key: str, df: pd.DataFrame, display_name: str,
                         property_name: str = "", source_type: str = "block_model",
                         auto_select: bool = False):
        """Register a block model source (following JORC classification pattern).

        Args:
            key: Unique identifier for this source
            df: DataFrame with block model data
            display_name: Human-readable name for the selector
            property_name: Name of the primary property/column
            source_type: Type of source (block_model, sgsim_mean, kriging, etc.)
            auto_select: If True, automatically select this source
        """
        self._block_model_sources[key] = {
            'df': df,
            'display_name': display_name,
            'property': property_name,
            'source_type': source_type
        }

        if key not in self._available_sources:
            self._available_sources.append(key)

        logger.debug(f"Block Property Calculator: Registered source '{key}' ({source_type}) with {len(df):,} blocks")

        if auto_select:
            self._select_source(key)

    def _select_source(self, key: str):
        """Programmatically select a source by key."""
        if key not in self._block_model_sources:
            return

        self._current_source = key
        src = self._block_model_sources[key]
        df = src.get('df')

        if df is not None and not df.empty:
            # Store the original model object if available
            self._block_model_df = df
            logger.info(f"Block Property Calculator: Selected source '{key}' with {len(df):,} blocks")

            # Update UI
            self._update_block_model_info()
            self._update_column_selectors()
            self._auto_detect_dimensions()
            self.calculate_btn.setEnabled(True)
        else:
            logger.warning(f"Block Property Calculator: Source '{key}' has no data")

    def _update_source_selector(self):
        """Update the source combo box with available sources."""
        if not hasattr(self, 'block_model_selector'):
            logger.warning("Block Property Calculator: block_model_selector not initialized yet")
            return

        logger.info("Updating dropdown with registered sources...")

        self.block_model_selector.blockSignals(True)
        current_key = self._current_source

        self.block_model_selector.clear()

        if not self._available_sources:
            self.block_model_selector.addItem("No models available", "none")
            self.block_model_selector.setEnabled(False)
            logger.info("  Dropdown: No sources - showing 'No models available'")
        else:
            # Add sources in priority order: Block Model, Kriging, SGSIM stats
            priority_order = ['block_model', 'kriging', 'sgsim_mean', 'sgsim_p10', 'sgsim_p50', 'sgsim_p90']

            sorted_sources = []
            for ptype in priority_order:
                for key in self._available_sources:
                    src = self._block_model_sources.get(key, {})
                    if src.get('source_type', '').startswith(ptype) and key not in sorted_sources:
                        sorted_sources.append(key)

            # Add any remaining sources
            for key in self._available_sources:
                if key not in sorted_sources:
                    sorted_sources.append(key)

            logger.info(f"  Dropdown: Adding {len(sorted_sources)} sources in priority order...")

            # Add to combo box
            for idx, key in enumerate(sorted_sources):
                src = self._block_model_sources.get(key, {})
                display_name = src.get('display_name', key)
                self.block_model_selector.addItem(display_name, key)
                logger.info(f"    [{idx+1}] Added: '{display_name}' (key={key})")

            self.block_model_selector.setEnabled(True)

        logger.info(f"  Dropdown now has {self.block_model_selector.count()} items")

        # Restore selection if possible
        if current_key:
            idx = self.block_model_selector.findData(current_key)
            if idx >= 0:
                self.block_model_selector.setCurrentIndex(idx)
                logger.info(f"  Restored previous selection: {current_key} at index {idx}")
            elif self.block_model_selector.count() > 0:
                self.block_model_selector.setCurrentIndex(0)
                self._current_source = self.block_model_selector.currentData()
                logger.info(f"  Selected first item: {self._current_source}")

        self.block_model_selector.blockSignals(False)

    def _refresh_from_registry(self, show_message=False):
        """Refresh available block models from registry.

        Args:
            show_message: If True, show message box when no models found (for manual refresh).
                         If False, silently update dropdown (for auto-refresh).
        """
        if not self.registry:
            if show_message:
                QMessageBox.warning(self, "No Registry", "Registry connection not available.")
            return

        logger.info("Block Property Calculator: Refreshing data from registry...")

        # Clear existing sources
        self._available_sources.clear()
        self._block_model_sources.clear()
        self._current_source = ""

        # Load all available sources
        self._load_existing()

        # Update UI
        self._update_source_selector()

        # Show message if manually triggered and no models found
        if show_message and not self._available_sources:
            QMessageBox.information(
                self,
                "No Block Models",
                "No block models found in registry.\n\nPlease:\n"
                "1. Load a block model from file (Data → Block Model)\n"
                "2. Build a block model (Estimations → Resource Modelling → Build Block Model)\n"
                "3. Run SGSIM or Kriging estimation"
            )

        # Log final status
        n_sources = len(self._available_sources)
        logger.info(f"Block Property Calculator: Found {n_sources} block model sources")

    def _load_existing(self):
        """Load existing data from registry on panel open - registers ALL available sources."""
        if not hasattr(self, 'registry') or self.registry is None:
            logger.warning("Block Property Calculator: Registry not available")
            return

        logger.info("=" * 70)
        logger.info("BLOCK PROPERTY CALCULATOR: Scanning Registry for Block Models")
        logger.info("=" * 70)

        found_any = False

        def _to_df(bm):
            if isinstance(bm, pd.DataFrame):
                return bm
            if hasattr(bm, 'to_dataframe'):
                return bm.to_dataframe()
            return None

        # 1. All registered block models via get_block_model_list() (multi-model support)
        logger.info("[1/4] Checking for registered block models (get_block_model_list)...")
        try:
            models = self.registry.get_block_model_list()
            if models:
                logger.info(f"  Found {len(models)} registered block model(s)")
                for model_info in models:
                    model_id = model_info['model_id']
                    is_current = model_info.get('is_current', False)
                    source_panel = model_info.get('source_panel', '')
                    n_blocks = model_info.get('row_count', 0)
                    bm = self.registry.get_block_model(model_id=model_id)
                    if bm is not None:
                        df = _to_df(bm)
                        if df is not None and not df.empty:
                            n_blocks = len(df)
                            # Build a human-friendly label from source_panel / model_id
                            sp_lower = source_panel.lower()
                            if 'builder' in sp_lower or 'blockmodelbuilder' in model_id.lower():
                                label = "Built Block Model"
                            elif 'import' in sp_lower:
                                raw = model_id.replace('imported_', '').replace('_', ' ').strip()
                                label = f"{raw} (Imported)" if raw else "Imported Block Model"
                            else:
                                label = model_id.replace('_', ' ').title()
                            if is_current:
                                label += " (current)"
                            display = f"{label} - {n_blocks:,} blocks"
                            key = f"block_model_{model_id}"
                            self._register_source(key, df, display, "", "block_model",
                                                  auto_select=is_current or not found_any)
                            found_any = True
                            logger.info(f"  ✓ REGISTERED: {display}")
            else:
                logger.info("  ✗ No models from get_block_model_list, falling back to get_block_model()")
        except Exception as e:
            logger.debug(f"  get_block_model_list() failed: {e}")

        # Fallback: single main block model
        if not found_any:
            logger.info("[1b] Fallback: Checking for single main block model...")
            try:
                bm = self.registry.get_block_model()
                if bm is not None:
                    df = _to_df(bm)
                    if df is not None and not df.empty:
                        n_blocks = len(df)
                        self._register_source("block_model", df,
                                              f"Block Model (Main) - {n_blocks:,} blocks",
                                              "", "block_model", auto_select=True)
                        found_any = True
                        logger.info(f"  ✓ REGISTERED: Block Model (Main) - {n_blocks:,} blocks")
                    else:
                        logger.info("  ✗ Block model found but DataFrame is empty")
                else:
                    logger.info("  ✗ No main block model in registry")
            except Exception as e:
                logger.info(f"  ✗ Failed to load main block model: {e}")

        # 2. Classified block model
        logger.info("[2/4] Checking for classified block model...")
        try:
            classified = self.registry.get_classified_block_model()
            if classified is not None:
                df = _to_df(classified)
                if df is not None and not df.empty:
                    already = any('classified' in k.lower() for k in self._block_model_sources)
                    if not already:
                        n_blocks = len(df)
                        self._register_source("classified_block_model", df,
                                              f"Block Model (Classified) - {n_blocks:,} blocks",
                                              "", "classified", auto_select=not found_any)
                        found_any = True
                        logger.info(f"  ✓ REGISTERED: Block Model (Classified) - {n_blocks:,} blocks")
        except Exception as e:
            logger.debug(f"  No classified block model: {e}")

        # 3. Try all estimation results
        logger.info("[3/4] Checking for estimation results (Kriging, SGSIM)...")
        self._try_load_all_estimation_results(auto_select_first=not found_any)

        # 4. Summary
        logger.info("[4/4] Scan Summary:")
        logger.info(f"  Total sources registered: {len(self._available_sources)}")
        logger.info(f"  Source keys: {self._available_sources}")
        for key in self._available_sources:
            src = self._block_model_sources.get(key, {})
            logger.info(f"    - {key}: {src.get('display_name', 'Unknown')}")
        logger.info("=" * 70)

    def _try_load_all_estimation_results(self, auto_select_first: bool = False):
        """Try to load ALL estimation/simulation results as separate sources.

        Args:
            auto_select_first: If True, auto-select the first successful source
        """
        if not hasattr(self, 'registry') or self.registry is None:
            return

        first_found = True

        # List of estimation result getters to try
        estimation_sources = [
            ('kriging_results', 'get_kriging_results', 'Ordinary Kriging', 'kriging'),
            ('simple_kriging_results', 'get_simple_kriging_results', 'Simple Kriging', 'kriging'),
            ('universal_kriging_results', 'get_universal_kriging_results', 'Universal Kriging', 'kriging'),
            ('cokriging_results', 'get_cokriging_results', 'Co-Kriging', 'kriging'),
            ('indicator_kriging_results', 'get_indicator_kriging_results', 'Indicator Kriging', 'kriging'),
        ]

        for key, getter_name, source_name, source_type in estimation_sources:
            if hasattr(self.registry, getter_name):
                try:
                    results = getattr(self.registry, getter_name)()
                    if results is not None and 'block_model' in results:
                        bm = results['block_model']
                        df = bm if isinstance(bm, pd.DataFrame) else (bm.to_dataframe() if hasattr(bm, 'to_dataframe') else None)
                        if df is not None and not df.empty:
                            n_blocks = len(df)
                            should_select = auto_select_first and first_found
                            self._register_source(key, df,
                                                  f"{source_name} - {n_blocks:,} blocks",
                                                  "", source_type, auto_select=should_select)
                            logger.info(f"  ✓ REGISTERED: {source_name} - {n_blocks:,} blocks")
                            first_found = False
                    else:
                        logger.debug(f"  ✗ {source_name}: No results or no block_model key")
                except Exception as e:
                    logger.debug(f"  ✗ {source_name}: {e}")

        # Handle SGSIM separately - extract individual statistics
        logger.info("  Checking for SGSIM results...")
        if hasattr(self.registry, 'get_sgsim_results'):
            try:
                results = self.registry.get_sgsim_results()
                if results is not None:
                    before = len(self._available_sources)
                    self._register_sgsim_sources(results)
                    added = len(self._available_sources) - before
                    if added:
                        logger.info(f"  ✓ SGSIM: registered {added} source(s)")
                        first_found = False
                    else:
                        logger.info("  ✗ SGSIM results found but no sources extracted")
                else:
                    logger.info("  ✗ No SGSIM results in registry")
            except Exception as e:
                logger.info(f"  ✗ SGSIM check failed: {e}")

    def _register_sgsim_sources(self, sgsim_results: Dict[str, Any]):
        """Register SGSIM results as individual selectable block model sources.

        Handles three SGSIM storage formats:
        1. Unified PyVista grid (results['grid'] or results['pyvista_grid']) - the primary
           format produced by sgsim_panel.py. Each cell data array (e.g.
           'FE_PCT_SGSIM_MEAN', 'FE_PCT_SGSIM_P10', …) becomes its own source.
        2. Legacy 'grids' dict (results['grids']['mean'], ['p10'], etc.) - older format.
        3. Realizations array (results['realizations'] + grid_x/y/z) - computes E-type.
        """
        import pyvista as pv

        if sgsim_results is None:
            return

        variable = sgsim_results.get('variable', 'Grade')

        # ── Format 1: unified PyVista grid ───────────────────────────────────
        grid = sgsim_results.get('grid') or sgsim_results.get('pyvista_grid')
        if grid is not None and hasattr(grid, 'cell_data'):
            try:
                centers = grid.cell_centers()
                coords = centers.points
                base_df = pd.DataFrame({
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                })

                stat_label = {
                    'MEAN': 'SGSIM Mean',
                    'P10':  'SGSIM P10',
                    'P50':  'SGSIM P50 (Median)',
                    'P90':  'SGSIM P90',
                    'STD':  'SGSIM Std Dev',
                    'PROB': 'SGSIM Probability',
                    'VAR':  'SGSIM Variance',
                }

                for prop_name in grid.cell_data.keys():
                    df = base_df.copy()
                    df[prop_name] = grid.cell_data[prop_name]
                    n = len(df)

                    # Pick a human-readable label
                    upper = prop_name.upper()
                    label = next((v for k, v in stat_label.items() if k in upper), f"SGSIM {prop_name}")
                    display = f"{label} ({variable}) - {n:,} blocks"

                    key = f"sgsim_{prop_name}"
                    self._register_source(key, df, display, prop_name, 'sgsim_mean', auto_select=False)
                    logger.info(f"    ✓ {display}")
            except Exception as e:
                logger.debug(f"  SGSIM unified grid extraction failed: {e}")

        # ── Format 2: legacy 'grids' dict ────────────────────────────────────
        elif 'grids' in sgsim_results:
            grids = sgsim_results['grids']
            logger.info(f"  SGSIM legacy grids dict: {list(grids.keys())}")
            legacy_stats = [
                ('mean', f'SGSIM Mean ({variable})',       'sgsim_mean'),
                ('p10',  f'SGSIM P10 ({variable})',        'sgsim_p10'),
                ('p50',  f'SGSIM P50 Median ({variable})', 'sgsim_p50'),
                ('p90',  f'SGSIM P90 ({variable})',        'sgsim_p90'),
                ('std',  f'SGSIM Std Dev ({variable})',    'sgsim_std'),
            ]
            for stat_key, stat_name, source_type in legacy_stats:
                if stat_key not in grids:
                    continue
                g = grids[stat_key]
                try:
                    if hasattr(g, 'cell_data'):
                        centers = g.cell_centers()
                        df = pd.DataFrame({
                            'X': centers.points[:, 0],
                            'Y': centers.points[:, 1],
                            'Z': centers.points[:, 2],
                        })
                        for arr in g.cell_data:
                            df[arr] = g.cell_data[arr]
                    elif isinstance(g, pd.DataFrame):
                        df = g
                    else:
                        continue

                    if not df.empty:
                        n = len(df)
                        key = f"sgsim_{stat_key}"
                        self._register_source(key, df, f"{stat_name} - {n:,} blocks",
                                              "", source_type, auto_select=False)
                        logger.info(f"    ✓ {stat_name} - {n:,} blocks")
                except Exception as e:
                    logger.debug(f"  SGSIM legacy '{stat_key}' failed: {e}")

        # ── Format 4: summary dict from run_full_sgsim_workflow (actual storage) ─
        # Results dict has: 'summary', 'params' (SGSIMParameters), no 'grid_coords'
        # Summary arrays shape: (nz, ny, nx), must ravel with order='C'
        # Grid coords: reconstruct from params using meshgrid(zs,ys,xs,indexing='ij')
        elif 'summary' in sgsim_results:
            summary = sgsim_results['summary']
            params_obj = sgsim_results.get('params')  # SGSIMParameters dataclass
            metadata = sgsim_results.get('metadata', {})
            variable = metadata.get('variable') or sgsim_results.get('variable', 'SGSIM')
            grid_coords = sgsim_results.get('grid_coords')

            if grid_coords is None and params_obj is not None:
                # Reconstruct coords from SGSIMParameters.
                # Summary stats are (nz,ny,nx) C-raveled so use meshgrid(zs,ys,xs,'ij')
                try:
                    nx = int(getattr(params_obj, 'nx'))
                    ny = int(getattr(params_obj, 'ny'))
                    nz = int(getattr(params_obj, 'nz'))
                    xmin = float(getattr(params_obj, 'xmin', 0))
                    ymin = float(getattr(params_obj, 'ymin', 0))
                    zmin = float(getattr(params_obj, 'zmin', 0))
                    xinc = float(getattr(params_obj, 'xinc', 1))
                    yinc = float(getattr(params_obj, 'yinc', 1))
                    zinc = float(getattr(params_obj, 'zinc', 1))
                    xs = np.arange(nx) * xinc + xmin + xinc / 2
                    ys = np.arange(ny) * yinc + ymin + yinc / 2
                    zs = np.arange(nz) * zinc + zmin + zinc / 2
                    # meshgrid(zs,ys,xs,'ij') → shape (nz,ny,nx) matches summary ravel order
                    GZ, GY, GX = np.meshgrid(zs, ys, xs, indexing='ij')
                    grid_coords = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
                    logger.info(f"  SGSIM: Reconstructed {len(grid_coords):,} grid coords "
                                f"from params ({nx}×{ny}×{nz})")
                except Exception as e:
                    logger.info(f"  SGSIM coord reconstruction failed: {e}")

            if grid_coords is not None and summary:
                coords = np.asarray(grid_coords)
                if coords.ndim == 2 and coords.shape[1] >= 3:
                    stat_display = {
                        'mean': 'SGSIM Mean',
                        'p10':  'SGSIM P10',
                        'p25':  'SGSIM P25',
                        'p50':  'SGSIM P50 Median',
                        'p75':  'SGSIM P75',
                        'p90':  'SGSIM P90',
                        'std':  'SGSIM Std Dev',
                        'var':  'SGSIM Variance',
                        'cv':   'SGSIM CV',
                        'iqr':  'SGSIM IQR',
                    }
                    base_df = pd.DataFrame({
                        'X': coords[:, 0],
                        'Y': coords[:, 1],
                        'Z': coords[:, 2],
                    })
                    n_coords = len(base_df)
                    for stat_key, stat_label in stat_display.items():
                        if stat_key not in summary:
                            continue
                        try:
                            arr_flat = np.asarray(summary[stat_key]).ravel(order='C')
                            if len(arr_flat) != n_coords:
                                logger.info(f"  SGSIM '{stat_key}' size mismatch: "
                                            f"{len(arr_flat)} values vs {n_coords} coords")
                                continue
                            col_name = f"{variable}_{stat_key.upper()}"
                            df = base_df.copy()
                            df[col_name] = arr_flat
                            display_name = f"{stat_label} ({variable}) - {n_coords:,} blocks"
                            key = f"sgsim_{stat_key}"
                            auto_sel = (stat_key == 'mean')
                            self._register_source(key, df, display_name, col_name,
                                                  f"sgsim_{stat_key}", auto_select=auto_sel)
                            logger.info(f"    ✓ {display_name}")
                        except Exception as e:
                            logger.info(f"  SGSIM summary '{stat_key}' failed: {e}")
                else:
                    logger.info(f"  SGSIM grid_coords unexpected shape: {coords.shape}")
            else:
                logger.info(f"  SGSIM Format4 failed: grid_coords={grid_coords is not None}, "
                            f"has_params={params_obj is not None}, "
                            f"summary_keys={list(summary.keys()) if summary else 'empty'}")

        # ── Format 3: realizations array → E-type mean ───────────────────────
        if 'realizations' in sgsim_results and 'summary' not in sgsim_results:
            reals = sgsim_results['realizations']
            gx = sgsim_results.get('grid_x')
            gy = sgsim_results.get('grid_y')
            gz = sgsim_results.get('grid_z')
            if gx is not None and isinstance(reals, np.ndarray):
                try:
                    mean_est = np.mean(reals, axis=0) if reals.ndim == 2 else reals.ravel()
                    df = pd.DataFrame({
                        'X': np.asarray(gx).ravel(),
                        'Y': np.asarray(gy).ravel(),
                        'Z': np.asarray(gz).ravel(),
                        variable: mean_est,
                    })
                    if not df.empty:
                        n = len(df)
                        key = 'sgsim_etype_mean'
                        display = f"SGSIM E-type Mean ({variable}) - {n:,} blocks"
                        self._register_source(key, df, display, variable, 'sgsim_mean', auto_select=False)
                        logger.info(f"    ✓ {display}")
                except Exception as e:
                    logger.debug(f"  SGSIM E-type extraction failed: {e}")

    def _update_block_model_info(self):
        """Update block model information display."""
        if self._block_model_df is None:
            self.block_model_info.setText("No block model loaded")
            return

        df = self._block_model_df
        info = f"✅ Block model loaded: {len(df):,} blocks\n"
        info += f"Columns: {len(df.columns)} properties\n"

        # Check existing properties
        has_volume = 'VOLUME' in df.columns
        has_tonnage = 'TONNAGE' in df.columns

        if has_volume:
            info += "⚠️ VOLUME column already exists (will be overwritten)\n"
        if has_tonnage:
            info += "⚠️ TONNAGE column already exists (will be overwritten)\n"

        self.block_model_info.setText(info)

    def _update_column_selectors(self):
        """Update column selectors with available columns."""
        if self._block_model_df is None:
            return

        numeric_cols = self._block_model_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # Update dimension column selectors
        self.dim_x_column.clear()
        self.dim_y_column.clear()
        self.dim_z_column.clear()
        self.dim_x_column.addItems(numeric_cols)
        self.dim_y_column.addItems(numeric_cols)
        self.dim_z_column.addItems(numeric_cols)

        # Auto-select dimension columns
        dim_x_patterns = ['DX', 'XINC', 'X_SIZE', 'XSIZE', 'BLOCK_X']
        dim_y_patterns = ['DY', 'YINC', 'Y_SIZE', 'YSIZE', 'BLOCK_Y']
        dim_z_patterns = ['DZ', 'ZINC', 'Z_SIZE', 'ZSIZE', 'BLOCK_Z']

        for col in numeric_cols:
            col_upper = col.upper()
            if any(p in col_upper for p in dim_x_patterns):
                self.dim_x_column.setCurrentText(col)
            if any(p in col_upper for p in dim_y_patterns):
                self.dim_y_column.setCurrentText(col)
            if any(p in col_upper for p in dim_z_patterns):
                self.dim_z_column.setCurrentText(col)

        # Update density column selector
        self.density_column.clear()
        self.density_column.addItems(numeric_cols)

        # Auto-select density column
        density_patterns = ['DENSITY', 'SG', 'SPECIFIC_GRAVITY', 'DENS', 'RHO']
        for col in numeric_cols:
            if any(p in col.upper() for p in density_patterns):
                self.density_column.setCurrentText(col)
                break

    def _auto_detect_dimensions(self):
        """Auto-detect block dimensions from model metadata, columns, or spacing."""
        if self._block_model_df is None:
            return

        try:
            df = self._block_model_df

            # PRIORITY 1: Check for dimension columns (DX, DY, DZ, etc.)
            dim_x_patterns = ['DX', 'XINC', 'X_SIZE', 'XSIZE', 'BLOCK_X']
            dim_y_patterns = ['DY', 'YINC', 'Y_SIZE', 'YSIZE', 'BLOCK_Y']
            dim_z_patterns = ['DZ', 'ZINC', 'Z_SIZE', 'ZSIZE', 'BLOCK_Z']

            found_dim_cols = {'X': None, 'Y': None, 'Z': None}

            for col in df.columns:
                col_upper = col.upper()
                if any(p in col_upper for p in dim_x_patterns) and found_dim_cols['X'] is None:
                    found_dim_cols['X'] = col
                if any(p in col_upper for p in dim_y_patterns) and found_dim_cols['Y'] is None:
                    found_dim_cols['Y'] = col
                if any(p in col_upper for p in dim_z_patterns) and found_dim_cols['Z'] is None:
                    found_dim_cols['Z'] = col

            # If all dimension columns found, switch to column mode
            if all(found_dim_cols.values()):
                self.dimension_source.setCurrentText("From Columns (Variable Dimensions)")
                self.dim_x_column.setCurrentText(found_dim_cols['X'])
                self.dim_y_column.setCurrentText(found_dim_cols['Y'])
                self.dim_z_column.setCurrentText(found_dim_cols['Z'])
                logger.info(f"Auto-detected dimension columns: {found_dim_cols['X']}, {found_dim_cols['Y']}, {found_dim_cols['Z']}")
                self._update_preview()
                return

            # PRIORITY 2: Try metadata for constant dimensions
            if hasattr(self._block_model, 'metadata'):
                metadata = self._block_model.metadata
                if hasattr(metadata, 'xinc'):
                    self.dimension_source.setCurrentText("From Constant Values")
                    self.block_x.setValue(metadata.xinc)
                    self.block_y.setValue(metadata.yinc)
                    self.block_z.setValue(metadata.zinc)
                    logger.info(f"Auto-detected dimensions from metadata: {metadata.xinc} × {metadata.yinc} × {metadata.zinc}")
                    self._update_preview()
                    return

            # PRIORITY 3: Try to infer from coordinate spacing
            coord_cols = {'X': None, 'Y': None, 'Z': None}

            # Find coordinate columns
            for col in df.columns:
                col_upper = col.upper()
                if 'X' in col_upper and coord_cols['X'] is None and col not in ['DX', 'XINC']:
                    coord_cols['X'] = col
                elif 'Y' in col_upper and coord_cols['Y'] is None and col not in ['DY', 'YINC']:
                    coord_cols['Y'] = col
                elif 'Z' in col_upper and coord_cols['Z'] is None and col not in ['DZ', 'ZINC']:
                    coord_cols['Z'] = col

            if all(coord_cols.values()):
                self.dimension_source.setCurrentText("From Constant Values")
                for axis, col in coord_cols.items():
                    unique = np.sort(df[col].unique())
                    if len(unique) > 1:
                        spacing = np.diff(unique)
                        median_spacing = np.median(spacing[spacing > 0])
                        if median_spacing > 0:
                            if axis == 'X':
                                self.block_x.setValue(median_spacing)
                            elif axis == 'Y':
                                self.block_y.setValue(median_spacing)
                            elif axis == 'Z':
                                self.block_z.setValue(median_spacing)

                logger.info("Auto-detected dimensions from coordinate spacing")
                self._update_preview()

        except Exception as e:
            logger.debug(f"Could not auto-detect dimensions: {e}")

    def _on_volume_toggle(self, checked: bool):
        """Handle volume calculation toggle."""
        self.dimension_source.setEnabled(checked)
        self.block_x.setEnabled(checked)
        self.block_y.setEnabled(checked)
        self.block_z.setEnabled(checked)
        self.dim_x_column.setEnabled(checked)
        self.dim_y_column.setEnabled(checked)
        self.dim_z_column.setEnabled(checked)
        self.auto_detect_dims_btn.setEnabled(checked)
        self._update_preview()

    def _on_tonnage_toggle(self, checked: bool):
        """Handle tonnage calculation toggle."""
        self.density_source.setEnabled(checked)
        self.density_column.setEnabled(checked)
        self.density_value.setEnabled(checked)
        self.density_unit.setEnabled(checked)
        self._update_preview()

    def _on_block_model_selected(self, index: int):
        """Handle block model selection change."""
        if index < 0:
            return

        # Get the source key from the combo box data
        source_key = self.block_model_selector.currentData()
        if not source_key or source_key == "none":
            return

        try:
            # Select the source using the registry pattern
            self._select_source(source_key)
        except Exception as e:
            logger.error(f"Failed to select block model source: {e}", exc_info=True)
            QMessageBox.critical(self, "Selection Error", f"Failed to select block model:\n\n{str(e)}")

    def _on_dimension_source_changed(self, index: int):
        """Handle dimension source change (Constant / From Columns)."""
        source = self.dimension_source.currentText()

        if source == "From Constant Values":
            self.constant_dims_widget.setVisible(True)
            self.column_dims_widget.setVisible(False)
        else:  # "From Columns (Variable Dimensions)"
            self.constant_dims_widget.setVisible(False)
            self.column_dims_widget.setVisible(True)

        self._update_preview()

    def _on_density_source_changed(self, index: int):
        """Handle density source change (From Column / Constant Value)."""
        source = self.density_source.currentText()

        if source == "From Column":
            self.density_column.setVisible(True)
            self.constant_density_widget.setVisible(False)
        else:  # "Constant Value"
            self.density_column.setVisible(False)
            self.constant_density_widget.setVisible(True)

        self._update_preview()

    def _on_density_unit_changed(self, index: int):
        """Handle density unit change - update suffix."""
        unit = self.density_unit.currentText()
        self.density_value.setSuffix(f" {unit}")
        self._update_preview()

    def _update_preview(self):
        """Update the preview display."""
        if self._block_model_df is None:
            return

        preview = "=== CALCULATION PREVIEW ===\n\n"

        # Block model info
        model_name = self.block_model_selector.currentText() if hasattr(self, 'block_model_selector') else "Unknown"
        preview += f"Block Model: {model_name}\n"
        preview += f"Total Blocks: {len(self._block_model_df):,}\n\n"

        # Volume calculation
        if self.calculate_volume_check.isChecked():
            dimension_source = self.dimension_source.currentText()
            preview += "✅ VOLUME Calculation:\n"
            preview += f"  Source: {dimension_source}\n"

            if dimension_source == "From Constant Values":
                block_x = self.block_x.value()
                block_y = self.block_y.value()
                block_z = self.block_z.value()
                volume = block_x * block_y * block_z

                preview += f"  Dimensions: {block_x:.2f}m × {block_y:.2f}m × {block_z:.2f}m\n"
                preview += f"  Volume per block: {volume:.2f} m³ (constant)\n"
            else:  # From Columns
                dim_x_col = self.dim_x_column.currentText()
                dim_y_col = self.dim_y_column.currentText()
                dim_z_col = self.dim_z_column.currentText()

                preview += f"  X Column: '{dim_x_col}'\n"
                preview += f"  Y Column: '{dim_y_col}'\n"
                preview += f"  Z Column: '{dim_z_col}'\n"

                if all(col in self._block_model_df.columns for col in [dim_x_col, dim_y_col, dim_z_col]):
                    vol_calc = self._block_model_df[dim_x_col] * self._block_model_df[dim_y_col] * self._block_model_df[dim_z_col]
                    preview += f"  Volume range: {vol_calc.min():.2f} - {vol_calc.max():.2f} m³\n"
                    preview += f"  Volume mean: {vol_calc.mean():.2f} m³ (variable per block)\n"

            preview += "  Column: 'VOLUME' will be added/updated\n\n"
        else:
            preview += "❌ VOLUME: Not enabled\n\n"

        # Tonnage calculation
        if self.calculate_tonnage_check.isChecked():
            density_source = self.density_source.currentText()
            preview += "✅ TONNAGE Calculation:\n"
            preview += f"  Source: {density_source}\n"

            if density_source == "From Column":
                density_col = self.density_column.currentText()
                if density_col and density_col in self._block_model_df.columns:
                    density_stats = self._block_model_df[density_col].describe()

                    preview += f"  Density column: '{density_col}'\n"
                    preview += f"  Density range: {density_stats['min']:.3f} - {density_stats['max']:.3f} t/m³\n"
                    preview += f"  Density mean: {density_stats['mean']:.3f} t/m³\n"

                    # Example calculation
                    if self.calculate_volume_check.isChecked():
                        if dimension_source == "From Constant Values":
                            example_tonnage = volume * density_stats['mean']
                            preview += f"  Example: {volume:.2f} m³ × {density_stats['mean']:.3f} t/m³ = {example_tonnage:.2f} tonnes\n"
                else:
                    preview += f"  ⚠️ Density column '{density_col}' not found\n"
            else:  # Constant Value
                density_value = self.density_value.value()
                density_unit = self.density_unit.currentText()

                # Convert to t/m³ for display
                if density_unit == 'kg/m³':
                    density_t_m3 = density_value / 1000.0
                elif density_unit == 'g/cm³':
                    density_t_m3 = density_value
                else:
                    density_t_m3 = density_value

                preview += f"  Density value: {density_value:.3f} {density_unit}\n"
                preview += f"  Density (t/m³): {density_t_m3:.3f} t/m³\n"

                # Example calculation
                if self.calculate_volume_check.isChecked():
                    if dimension_source == "From Constant Values":
                        example_tonnage = volume * density_t_m3
                        preview += f"  Example: {volume:.2f} m³ × {density_t_m3:.3f} t/m³ = {example_tonnage:.2f} tonnes\n"

            preview += "  Formula: TONNAGE = VOLUME × DENSITY\n"
            preview += "  Column: 'TONNAGE' will be added/updated\n\n"
        else:
            preview += "❌ TONNAGE: Not enabled\n\n"

        # Existing columns check
        preview += "=== EXISTING COLUMNS CHECK ===\n"
        if 'VOLUME' in self._block_model_df.columns:
            preview += "⚠️ VOLUME column already exists (will be OVERWRITTEN)\n"
        if 'TONNAGE' in self._block_model_df.columns:
            preview += "⚠️ TONNAGE column already exists (will be OVERWRITTEN)\n"

        if 'VOLUME' not in self._block_model_df.columns and 'TONNAGE' not in self._block_model_df.columns:
            preview += "✅ No conflicts - new columns will be added\n"

        self.preview_text.setPlainText(preview)

    def _run_calculation(self):
        """Run property calculations in background."""
        if self._block_model_df is None:
            QMessageBox.warning(self, "No Data", "Load a block model first.")
            return

        # Validate inputs
        if not self.calculate_volume_check.isChecked() and not self.calculate_tonnage_check.isChecked():
            QMessageBox.warning(self, "No Calculations", "Enable at least one calculation (Volume or Tonnage).")
            return

        dimension_source = self.dimension_source.currentText()
        density_source = self.density_source.currentText()

        # Validate volume calculation inputs
        if self.calculate_volume_check.isChecked():
            if dimension_source == "From Columns (Variable Dimensions)":
                # Check dimension columns are selected
                dim_x_col = self.dim_x_column.currentText()
                dim_y_col = self.dim_y_column.currentText()
                dim_z_col = self.dim_z_column.currentText()

                if not dim_x_col or dim_x_col not in self._block_model_df.columns:
                    QMessageBox.warning(self, "Invalid Input", f"X dimension column '{dim_x_col}' not found.")
                    return
                if not dim_y_col or dim_y_col not in self._block_model_df.columns:
                    QMessageBox.warning(self, "Invalid Input", f"Y dimension column '{dim_y_col}' not found.")
                    return
                if not dim_z_col or dim_z_col not in self._block_model_df.columns:
                    QMessageBox.warning(self, "Invalid Input", f"Z dimension column '{dim_z_col}' not found.")
                    return

        # Validate tonnage calculation inputs
        if self.calculate_tonnage_check.isChecked():
            if density_source == "From Column":
                density_col = self.density_column.currentText()
                if not density_col or density_col not in self._block_model_df.columns:
                    QMessageBox.warning(self, "Invalid Input", f"Density column '{density_col}' not found in block model.")
                    return
            else:  # Constant Value
                density_value = self.density_value.value()
                if density_value <= 0:
                    QMessageBox.warning(self, "Invalid Input", "Density value must be positive.")
                    return

        # Build configuration
        config = {
            'calculate_volume': self.calculate_volume_check.isChecked(),
            'calculate_tonnage': self.calculate_tonnage_check.isChecked(),
            'dimension_source': dimension_source,
            'block_x': self.block_x.value(),
            'block_y': self.block_y.value(),
            'block_z': self.block_z.value(),
            'dim_x_column': self.dim_x_column.currentText() if dimension_source == "From Columns (Variable Dimensions)" else None,
            'dim_y_column': self.dim_y_column.currentText() if dimension_source == "From Columns (Variable Dimensions)" else None,
            'dim_z_column': self.dim_z_column.currentText() if dimension_source == "From Columns (Variable Dimensions)" else None,
            'density_source': density_source,
            'density_column': self.density_column.currentText() if density_source == "From Column" else None,
            'density_value': self.density_value.value() if density_source == "Constant Value" else None,
            'density_unit': self.density_unit.currentText() if density_source == "Constant Value" else None
        }

        # Start worker
        self._calculation_worker = PropertyCalculationWorker(self._block_model_df, config)
        self._calculation_worker.progressUpdate.connect(self._on_progress_update)
        self._calculation_worker.calculationComplete.connect(self._on_calculation_complete)
        self._calculation_worker.calculationError.connect(self._on_calculation_error)

        # Update UI
        self.calculate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start calculation
        self._calculation_worker.start()
        logger.info("Started property calculation worker")

    @pyqtSlot(str, int)
    def _on_progress_update(self, message: str, percentage: int):
        """Handle progress updates from worker."""
        self.progress_bar.setValue(percentage)
        logger.debug(f"Progress: {message} ({percentage}%)")

    @pyqtSlot(pd.DataFrame)
    def _on_calculation_complete(self, updated_df: pd.DataFrame):
        """Handle successful calculation completion."""
        self._block_model_df = updated_df

        self.calculate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update preview with results
        preview = "=== CALCULATION RESULTS ===\n\n"
        preview += f"✅ Calculations completed successfully!\n\n"

        if 'VOLUME' in updated_df.columns:
            preview += f"✅ VOLUME column added\n"
            preview += f"   Value: {updated_df['VOLUME'].iloc[0]:.2f} m³ per block\n\n"

        if 'TONNAGE' in updated_df.columns:
            tonnage_stats = updated_df['TONNAGE'].describe()
            preview += f"✅ TONNAGE column added\n"
            preview += f"   Range: {tonnage_stats['min']:.2f} - {tonnage_stats['max']:.2f} tonnes\n"
            preview += f"   Mean: {tonnage_stats['mean']:.2f} tonnes\n"
            preview += f"   Total: {updated_df['TONNAGE'].sum():,.2f} tonnes\n\n"

        preview += "📌 Click 'Save to Block Model' to permanently add these properties.\n"

        self.preview_text.setPlainText(preview)

        QMessageBox.information(
            self,
            "Success",
            "Property calculations completed!\n\nClick 'Save to Block Model' to permanently save these properties."
        )

    @pyqtSlot(str)
    def _on_calculation_error(self, error_message: str):
        """Handle calculation error."""
        self.calculate_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(
            self,
            "Calculation Error",
            f"Property calculation failed:\n\n{error_message}\n\nPlease check your inputs and try again."
        )

    def _save_to_block_model(self):
        """Save calculated properties back to block model in registry."""
        if self._block_model_df is None:
            QMessageBox.warning(self, "No Data", "No calculated properties to save.")
            return

        # Confirm save
        reply = QMessageBox.question(
            self,
            "Confirm Save",
            "This will permanently add/update the calculated properties in the block model.\n\n"
            "Properties to save:\n"
            + ("- VOLUME\n" if 'VOLUME' in self._block_model_df.columns else "")
            + ("- TONNAGE\n" if 'TONNAGE' in self._block_model_df.columns else "")
            + "\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Update block model with new properties
            if hasattr(self._block_model, 'data'):
                # BlockModel object with .data attribute
                self._block_model.data = self._block_model_df
            elif isinstance(self._block_model, pd.DataFrame):
                # Direct dataframe
                self._block_model = self._block_model_df

            # Save back to registry
            if self.registry:
                self.registry.set_block_model(self._block_model)
                logger.info("Saved updated block model to registry")

                # Emit signal
                self.propertiesCalculated.emit(self._block_model)

                QMessageBox.information(
                    self,
                    "Saved",
                    "Properties saved successfully!\n\nThe block model has been updated in the registry."
                )

                # Disable save button (already saved)
                self.save_btn.setEnabled(False)
            else:
                QMessageBox.warning(self, "No Registry", "Registry connection not available.\nProperties calculated but not saved.")

        except Exception as e:
            logger.exception("Failed to save block model")
            QMessageBox.critical(self, "Save Error", f"Failed to save properties:\n{str(e)}")

    def refresh_theme(self):
        """Refresh theme styling."""
        self.setStyleSheet(get_analysis_panel_stylesheet())
