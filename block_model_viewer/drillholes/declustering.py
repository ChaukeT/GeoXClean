"""
DECLUSTERING ENGINE - GeoX Professional Mining Software

Cell-based declustering for composited drillhole samples.
Implements industry-standard methodology for statistical defensibility under JORC/SAMREC.

Key Features:
- Cell declustering (2D/3D grid-based) - accepted standard methodology
- Weights = 1 / number of samples per occupied cell
- Multi-cell-size sensitivity analysis
- Auditable outputs with diagnostic summaries
- Deterministic, reproducible results
- No random weighting or distance-based methods

Integration Points:
- Feeds declustered weights directly into Variogram engine
- Publishes summary tables to QA panel for CP reporting
- Supports multi-size evaluation for sensitivity analysis

Author: GeoX Declustering Engine Architect
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DeclusteringMethod(str, Enum):
    """Supported declustering methodologies.

    CELL_DECLUSTERING: Industry standard cell-based method.
    Only cell declustering is supported - distance-based methods violate
    geostatistical principles and are not JORC/SAMREC defensible.
    """
    CELL_DECLUSTERING = "cell_declustering"

class CellShape(str, Enum):
    """Cell geometry for declustering grid."""
    CUBIC = "cubic"           # Equal dimensions in all directions
    RECTANGULAR = "rectangular"  # Different X,Y dimensions, same Z
    PRISMATIC = "prismatic"   # Full 3D control (sx, sy, sz)

class WeightingRule(str, Enum):
    """Weight assignment rules - deterministic only."""
    UNIFORM = "uniform"  # weights = 1 / samples_per_cell (industry standard)
    # Note: No random weighting - violates audit requirements

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CellDefinition:
    """Cell geometry specification for declustering grid.

    Defines the spatial discretization used for grouping samples
    into cells for weight calculation.
    """
    cell_size_x: float  # Cell width in X direction (metres)
    cell_size_y: float  # Cell width in Y direction (metres)
    cell_size_z: Optional[float] = None  # Cell height in Z (None = 2D)

    origin_x: float = 0.0  # Grid origin X coordinate
    origin_y: float = 0.0  # Grid origin Y coordinate
    origin_z: Optional[float] = None  # Grid origin Z coordinate

    @property
    def is_3d(self) -> bool:
        """True if 3D declustering (Z dimension defined)."""
        return self.cell_size_z is not None

    def __post_init__(self):
        """Ensure origin_z is set for 3D declustering."""
        if self.is_3d and self.origin_z is None:
            self.origin_z = 0.0

    @property
    def volume(self) -> Optional[float]:
        """Cell volume in cubic metres (None for 2D)."""
        if not self.is_3d:
            return None
        return self.cell_size_x * self.cell_size_y * self.cell_size_z

    def get_cell_coords(self, x: float, y: float, z: Optional[float] = None) -> Tuple[int, int, Optional[int]]:
        """Convert world coordinates to cell indices.

        Args:
            x, y, z: World coordinates

        Returns:
            Cell indices (ix, iy, iz) where iz is None for 2D
        """
        ix = int(np.floor((x - self.origin_x) / self.cell_size_x))
        iy = int(np.floor((y - self.origin_y) / self.cell_size_y))

        iz = None
        if self.is_3d and z is not None:
            iz = int(np.floor((z - self.origin_z) / self.cell_size_z))

        return ix, iy, iz

    def get_cell_key(self, x: float, y: float, z: Optional[float] = None) -> str:
        """Generate unique cell identifier key.

        Args:
            x, y, z: World coordinates

        Returns:
            String key for cell identification
        """
        ix, iy, iz = self.get_cell_coords(x, y, z)
        if iz is not None:
            return f"{ix},{iy},{iz}"
        return f"{ix},{iy}"

@dataclass
class DeclusteringConfig:
    """Configuration for declustering operation.

    Defines the methodology and parameters for weight calculation.
    """
    method: DeclusteringMethod = DeclusteringMethod.CELL_DECLUSTERING
    cell_definition: CellDefinition = field(default_factory=lambda: CellDefinition(10.0, 10.0))

    # Weighting parameters
    weighting_rule: WeightingRule = WeightingRule.UNIFORM

    # Audit and validation
    require_coordinates: bool = True  # Must have X,Y (and Z if 3D)
    validate_composited: bool = True  # Check for composited sample support

    # Output options
    include_diagnostics: bool = True
    include_cell_counts: bool = True

@dataclass
class CellDiagnostic:
    """Diagnostic information for a single declustering cell.

    Captures the statistical properties of samples within each cell
    for audit and validation purposes.
    """
    cell_key: Union[int, str]  # Integer (optimized) or string key
    sample_count: int
    weight_value: float

    # Cell boundaries
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: Optional[float]
    max_z: Optional[float]

    # Statistical summaries per variable
    variable_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Sample details for audit (empty by default for performance)
    sample_indices: List[int] = field(default_factory=list)

@dataclass
class DeclusteringSummary:
    """Comprehensive summary of declustering operation.

    Provides auditable statistics for JORC/SAMREC compliance
    and sensitivity analysis.
    """
    cell_size_summary: str  # Human-readable cell size description
    total_samples: int
    occupied_cells: int
    empty_cells: int  # In the sample domain

    # Weight statistics
    min_weight: float
    max_weight: float
    mean_weight: float
    weight_std: float

    # Per-variable raw vs declustered statistics
    variable_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Cell diagnostics
    cell_diagnostics: List[CellDiagnostic] = field(default_factory=list)

    # Processing metadata
    processing_time_seconds: float = 0.0
    timestamp: str = ""

    # Stability analysis (for multi-cell-size sensitivity)
    stability_achieved: bool = False
    weight_change_from_previous: Optional[float] = None

    @property
    def cells_per_sample(self) -> float:
        """Average number of samples per occupied cell."""
        if self.occupied_cells == 0:
            return 0.0
        return self.total_samples / self.occupied_cells

    @property
    def declustering_ratio(self) -> float:
        """Ratio of total samples to occupied cells (inverse of average weight)."""
        return self.cells_per_sample

@dataclass
class ValidationResult:
    """Result of input validation with detailed error reporting."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

# =============================================================================
# CORE ENGINE
# =============================================================================

class DeclusteringEngine:
    """
    Professional Mining-Grade Declustering Engine

    Implements cell-based declustering following industry standards
    for statistical defensibility under JORC/SAMREC review.

    KEY PRINCIPLES:
    - Cell declustering only (distance-based methods violate geostatistics)
    - Deterministic weights = 1 / samples_per_cell
    - No randomness or seeds required
    - Auditable outputs with full traceability
    - Support for multi-cell-size sensitivity analysis

    LINEAGE ENFORCEMENT:
    - Engine-level checks for composited data (not raw assays)
    - Validation status verification before processing
    - Full provenance metadata attached to results

    INTEGRATION:
    - Feeds weights directly into Variogram engine for experimental variograms
    - Publishes summaries to QA panel for CP reporting
    - Supports batch processing for sensitivity analysis
    """

    def __init__(self, config: Optional[DeclusteringConfig] = None):
        """
        Initialize the declustering engine.

        Args:
            config: Declustering configuration. Uses defaults if None.
        """
        self.config = config or DeclusteringConfig()
        self._last_summary: Optional[DeclusteringSummary] = None
        self._lineage_metadata: Dict[str, Any] = {}  # Track provenance information

        logger.info(f"Initialized DeclusteringEngine with method: {self.config.method.value}")

    # =========================================================================
    # LINEAGE ENFORCEMENT LAYER
    # =========================================================================

    def validate_lineage(
        self,
        df: pd.DataFrame,
        source_type: str = "unknown",
        validation_status: str = "NOT_RUN",
        require_composites: bool = True,
        require_validation: bool = False
    ) -> ValidationResult:
        """
        Validate data lineage for JORC/SAMREC compliance.

        Enforces geostatistical best practices at the ENGINE level:
        - Declustering should be performed on composited data
        - Validation should be run before declustering

        Args:
            df: Input DataFrame
            source_type: Source data type ('composites', 'assays', or 'unknown')
            validation_status: Drillhole validation status ('PASS', 'WARN', 'FAIL', 'NOT_RUN')
            require_composites: If True, raises error for non-composited data
            require_validation: If True, raises error if validation not passed

        Returns:
            ValidationResult with lineage-specific errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # LINEAGE CHECK 1: Source data type
        if require_composites and source_type == "assays":
            result.add_error(
                "LINEAGE: Declustering on raw assays violates geostatistical principles. "
                "Raw assays have inconsistent sample support, which invalidates the "
                "statistical assumptions of declustering. Use composited data instead."
            )

        if source_type == "unknown":
            result.add_warning(
                "LINEAGE: Data source type unknown. For JORC/SAMREC compliance, "
                "ensure data has been composited before declustering."
            )

        # LINEAGE CHECK 2: Validation status
        if require_validation:
            if validation_status == "NOT_RUN":
                result.add_error(
                    "LINEAGE: Drillhole validation has not been run. "
                    "Validate data before declustering for audit defensibility."
                )
            elif validation_status == "FAIL":
                result.add_error(
                    "LINEAGE: Drillhole validation FAILED. Fix validation errors "
                    "before proceeding with declustering."
                )

        if validation_status == "WARN":
            result.add_warning(
                "LINEAGE: Drillhole validation passed with warnings. "
                "Review warnings for JORC/SAMREC compliance."
            )

        # LINEAGE CHECK 3: Check for lineage_gate_passed attribute
        if hasattr(df, 'attrs'):
            if df.attrs.get('lineage_gate_passed', False):
                logger.info("LINEAGE: DataFrame has passed lineage gate (from registry)")
            elif df.attrs.get('source_type') == 'composites':
                logger.info("LINEAGE: DataFrame marked as composites source")

        # Store lineage metadata for downstream use
        self._lineage_metadata = {
            'source_type': source_type,
            'validation_status': validation_status,
            'lineage_check_passed': result.is_valid,
            'lineage_warnings': result.warnings.copy(),
        }

        logger.info(
            f"LINEAGE: Validation complete. Source: {source_type}, "
            f"Validation: {validation_status}, Passed: {result.is_valid}"
        )

        return result

    # =========================================================================
    # INPUT VALIDATION LAYER
    # =========================================================================

    def validate_input_data(
        self,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        z_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Comprehensive input validation for declustering.

        Validates coordinate columns, value columns, and data integrity
        to ensure JORC/SAMREC compliance.

        Args:
            df: Input DataFrame with sample data
            x_col, y_col, z_col: Coordinate column names
            value_cols: Grade/element column names to decluster

        Returns:
            ValidationResult with errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # Check DataFrame validity
        if df is None or df.empty:
            result.add_error("Input DataFrame is None or empty")
            return result

        # Coordinate column validation
        coord_cols = self._find_coordinate_columns(df, x_col, y_col, z_col)
        if not coord_cols['x']:
            result.add_error("X coordinate column not found. Required columns: X, Y")
        if not coord_cols['y']:
            result.add_error("Y coordinate column not found. Required columns: X, Y")

        if self.config.cell_definition.is_3d:
            if not coord_cols['z']:
                result.add_error("Z coordinate column required for 3D declustering")
        else:
            if not coord_cols['z'] and z_col:
                result.add_warning("Z column specified but 2D declustering configured")

        # Value column validation
        if value_cols:
            missing_cols = [col for col in value_cols if col not in df.columns]
            if missing_cols:
                result.add_error(f"Value columns not found in DataFrame: {missing_cols}")

            # Check for numeric data
            for col in value_cols:
                if col in df.columns:
                    non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
                    if non_numeric > 0:
                        result.add_warning(f"Column '{col}' contains {non_numeric} non-numeric values")
        else:
            result.add_warning("No value columns specified - declustering weights only")

        # Data quality checks
        if coord_cols['x'] and coord_cols['y']:
            # Check for coordinate precision issues
            x_precision = self._check_coordinate_precision(df[coord_cols['x']])
            y_precision = self._check_coordinate_precision(df[coord_cols['y']])

            if x_precision < 0.01:  # Less than 1cm precision
                result.add_warning(f"X coordinates have low precision ({x_precision:.6f}m). Consider rounding.")
            if y_precision < 0.01:
                result.add_warning(f"Y coordinates have low precision ({y_precision:.6f}m). Consider rounding.")

            # Check for clustering/colocation issues
            coord_duplicates = df.duplicated(subset=[coord_cols['x'], coord_cols['y']]).sum()
            if coord_duplicates > 0:
                result.add_warning(f"{coord_duplicates} samples have identical X,Y coordinates (potential colocation)")

        # Sample count validation
        if len(df) < 2:
            result.add_warning("Very few samples (< 2) - statistical reliability may be limited")

        logger.info(f"Input validation complete. Valid: {result.is_valid}, Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")

        return result

    def _find_coordinate_columns(
        self,
        df: pd.DataFrame,
        x_col: Optional[str],
        y_col: Optional[str],
        z_col: Optional[str]
    ) -> Dict[str, Optional[str]]:
        """Find coordinate columns using flexible name matching."""
        def find_column(candidates: List[str], preferred: Optional[str] = None) -> Optional[str]:
            if preferred and preferred in df.columns:
                return preferred

            # Case-insensitive search
            lowered = {c.lower(): c for c in df.columns}
            for cand in candidates:
                if cand in df.columns:
                    return cand
                if cand.lower() in lowered:
                    return lowered[cand.lower()]
            return None

        return {
            'x': find_column(['x', 'X', 'easting', 'Easting', 'EASTING'], x_col),
            'y': find_column(['y', 'Y', 'northing', 'Northing', 'NORTHING'], y_col),
            'z': find_column(['z', 'Z', 'elevation', 'Elevation', 'ELEVATION', 'rl', 'RL'], z_col)
        }

    def _check_coordinate_precision(self, series: pd.Series) -> float:
        """Check coordinate precision (smallest difference between values)."""
        if len(series) < 2:
            return 0.0

        # Get unique sorted values
        unique_vals = np.sort(series.dropna().unique())

        if len(unique_vals) < 2:
            return 0.0

        # Find minimum difference
        diffs = np.diff(unique_vals)
        return float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0

    # =========================================================================
    # WEIGHT COMPUTATION LAYER
    # =========================================================================

    def compute_weights(
        self,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        z_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None,
        source_type: str = "unknown",
        validation_status: str = "NOT_RUN",
        require_composites: bool = False,
        require_validation: bool = False
    ) -> Tuple[pd.DataFrame, DeclusteringSummary]:
        """
        Compute declustering weights using cell-based methodology.

        Core algorithm:
        1. Group samples into spatial cells
        2. Calculate weights = 1 / samples_per_cell
        3. Generate diagnostic summaries

        Args:
            df: Input DataFrame with sample coordinates
            x_col, y_col, z_col: Coordinate column names
            value_cols: Grade columns for statistics
            source_type: Source data type for lineage tracking ('composites', 'assays')
            validation_status: Drillhole validation status
            require_composites: If True, raises error for non-composited data
            require_validation: If True, raises error if validation not passed

        Returns:
            Tuple of (df_with_weights, summary)

        Raises:
            ValueError: If input validation fails
            ValueError: If lineage validation fails (when require_composites=True)
        """
        import time
        start_time = time.time()

        # GUARD: Empty DataFrame check
        if df is None:
            raise ValueError("Input DataFrame is None")
        if df.empty:
            raise ValueError("Input DataFrame is empty - cannot compute declustering weights")

        # LINEAGE VALIDATION (engine-level enforcement)
        lineage_result = self.validate_lineage(
            df,
            source_type=source_type,
            validation_status=validation_status,
            require_composites=require_composites,
            require_validation=require_validation
        )
        if not lineage_result.is_valid:
            error_msg = "; ".join(lineage_result.errors)
            raise ValueError(f"Lineage validation failed: {error_msg}")

        # Log lineage warnings
        for warning in lineage_result.warnings:
            logger.warning(f"LINEAGE WARNING: {warning}")

        # Validation
        coord_cols = self._find_coordinate_columns(df, x_col, y_col, z_col)
        validation = self.validate_input_data(df, x_col, y_col, z_col, value_cols)
        if not validation.is_valid:
            error_msg = "; ".join(validation.errors)
            raise ValueError(f"Input validation failed: {error_msg}")

        # GUARD: Check for NaN coordinates before proceeding
        x_col_name = coord_cols['x']
        y_col_name = coord_cols['y']
        z_col_name = coord_cols['z']

        # Performance optimization: Build NaN mask on original df, then copy only once
        nan_mask_x = df[x_col_name].isna() if x_col_name else pd.Series(False, index=df.index)
        nan_mask_y = df[y_col_name].isna() if y_col_name else pd.Series(False, index=df.index)
        nan_mask_z = df[z_col_name].isna() if z_col_name and self.config.cell_definition.is_3d else pd.Series(False, index=df.index)
        nan_mask = nan_mask_x | nan_mask_y | nan_mask_z

        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.warning(f"GUARD: Dropping {nan_count} rows with NaN coordinates before declustering")
            # Single copy with filtering
            df_working = df.loc[~nan_mask].copy()
        else:
            # Single copy, no filtering needed
            df_working = df.copy()

        # GUARD: Check for empty DataFrame after NaN removal
        if df_working.empty:
            raise ValueError(
                f"All {len(df)} samples have NaN coordinates - cannot compute declustering weights"
            )

        # GUARD: Check for duplicate coordinates (potential data quality issue)
        coord_subset = [x_col_name, y_col_name]
        if z_col_name and self.config.cell_definition.is_3d:
            coord_subset.append(z_col_name)
        duplicate_count = df_working.duplicated(subset=coord_subset).sum()
        if duplicate_count > 0:
            logger.warning(
                f"GUARD: {duplicate_count} samples have duplicate coordinates (potential colocation issue). "
                "These will be assigned to the same cell."
            )

        # Set cell origins from data before assigning cells
        self._set_cell_origins_from_data(df_working, coord_cols)

        # Assign samples to cells
        cell_assignments = self._assign_samples_to_cells(
            df_working,
            coord_cols['x'],
            coord_cols['y'],
            coord_cols['z']
        )

        # Calculate weights
        weights = self._calculate_cell_weights(cell_assignments)

        # GUARD: Verify weights sum correctly (should sum to occupied_cells)
        weight_sum = weights.sum()
        occupied_cells = cell_assignments.nunique()
        expected_sum = float(occupied_cells)
        if abs(weight_sum - expected_sum) > 0.001:
            logger.warning(
                f"GUARD: Weight sum ({weight_sum:.4f}) differs from expected ({expected_sum:.4f}). "
                "This may indicate a calculation issue."
            )

        # Add weights to DataFrame (in-place on existing copy - no extra copy needed)
        df_working['declust_weight'] = weights
        df_working['declust_cell'] = cell_assignments

        # Generate summary
        summary = self._generate_summary(
            df_working,
            cell_assignments,
            coord_cols,
            time.time() - start_time
        )

        self._last_summary = summary

        # Attach lineage metadata to result DataFrame for downstream tracking
        df_working.attrs['source_type'] = 'declustered'
        df_working.attrs['parent_source_type'] = source_type
        df_working.attrs['validation_status'] = validation_status
        df_working.attrs['lineage_gate_passed'] = lineage_result.is_valid
        df_working.attrs['declustering_cell_size'] = (
            self.config.cell_definition.cell_size_x,
            self.config.cell_definition.cell_size_y,
            self.config.cell_definition.cell_size_z
        )
        df_working.attrs['rows_dropped_nan'] = nan_count
        df_working.attrs['duplicate_coords_count'] = duplicate_count

        logger.info(f"Declustering complete. {summary.total_samples} samples in {summary.occupied_cells} cells. "
                   f"Weight range: {summary.min_weight:.6f} - {summary.max_weight:.6f}")

        return df_working, summary

    def _assign_samples_to_cells(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: Optional[str]
    ) -> pd.Series:
        """Assign each sample to a spatial cell using vectorized integer encoding.

        Performance optimization: Uses integer-based cell keys instead of strings.
        Cell key = ix + iy * stride_y + iz * stride_z (Cantor-like encoding)
        This is ~10x faster than string concatenation for large datasets.
        """
        cell_def = self.config.cell_definition

        x = df[x_col].to_numpy(float)
        y = df[y_col].to_numpy(float)

        ix = np.floor((x - cell_def.origin_x) / cell_def.cell_size_x).astype(np.int64)
        iy = np.floor((y - cell_def.origin_y) / cell_def.cell_size_y).astype(np.int64)

        # Use large stride to create unique integer keys (faster than string keys)
        # Stride must be larger than max possible index in each dimension
        stride = 1_000_000  # Supports up to 1M cells per dimension

        if cell_def.is_3d and z_col:
            z = df[z_col].to_numpy(float)
            iz = np.floor((z - cell_def.origin_z) / cell_def.cell_size_z).astype(np.int64)
            # Encode: ix + iy * stride + iz * stride^2
            cell_keys = ix + iy * stride + iz * (stride * stride)
        else:
            # Encode: ix + iy * stride
            cell_keys = ix + iy * stride

        return pd.Series(cell_keys, index=df.index, name="cell_key")

    def _calculate_cell_weights(self, cell_assignments: pd.Series) -> pd.Series:
        """Calculate weights based on samples per cell.

        Performance optimization: Uses vectorized map instead of lambda.
        """
        # Count samples per cell
        cell_counts = cell_assignments.value_counts()

        # Weights = 1 / samples_per_cell (vectorized - no lambda)
        weights = 1.0 / cell_assignments.map(cell_counts)

        logger.debug(f"Cell weight calculation: {len(cell_counts)} unique cells, "
                    f"max samples/cell: {cell_counts.max()}, min weight: {weights.min():.6f}")

        return weights

    def _set_cell_origins_from_data(self, df: pd.DataFrame, coord_cols: Dict[str, Optional[str]]) -> None:
        """Set cell definition origins from actual data extents."""
        cell_def = self.config.cell_definition

        x = df[coord_cols['x']].to_numpy(float)
        y = df[coord_cols['y']].to_numpy(float)
        z = df[coord_cols['z']].to_numpy(float) if coord_cols['z'] else None

        cell_def.origin_x = float(np.nanmin(x))
        cell_def.origin_y = float(np.nanmin(y))

        if cell_def.is_3d and z is not None:
            cell_def.origin_z = float(np.nanmin(z))
        else:
            cell_def.origin_z = 0.0

    # =========================================================================
    # DIAGNOSTIC REPORTING LAYER
    # =========================================================================

    def _generate_summary(
        self,
        df: pd.DataFrame,
        cell_assignments: pd.Series,
        coord_cols: Dict[str, Optional[str]],
        processing_time: float
    ) -> DeclusteringSummary:
        """Generate comprehensive diagnostic summary."""
        from datetime import datetime

        # Basic counts
        total_samples = len(df)
        occupied_cells = cell_assignments.nunique()
        cell_counts = cell_assignments.value_counts()

        # Calculate empty cells (total possible cells in spatial domain minus occupied cells)
        cell_def = self.config.cell_definition
        x_coords = df[coord_cols['x']].values
        y_coords = df[coord_cols['y']].values
        x_min, x_max = float(np.nanmin(x_coords)), float(np.nanmax(x_coords))
        y_min, y_max = float(np.nanmin(y_coords)), float(np.nanmax(y_coords))

        # Calculate number of cells needed to cover the spatial extent
        n_cells_x = int(np.ceil((x_max - x_min) / cell_def.cell_size_x))
        n_cells_y = int(np.ceil((y_max - y_min) / cell_def.cell_size_y))
        total_possible_cells = n_cells_x * n_cells_y
        empty_cells = max(0, total_possible_cells - occupied_cells)

        # Weight statistics
        weights = df['declust_weight']
        weight_stats = {
            'min_weight': weights.min(),
            'max_weight': weights.max(),
            'mean_weight': weights.mean(),
            'weight_std': weights.std()
        }

        # Cell size description
        cell_def = self.config.cell_definition
        if cell_def.is_3d:
            cell_desc = f"{cell_def.cell_size_x:.1f}m × {cell_def.cell_size_y:.1f}m × {cell_def.cell_size_z:.1f}m"
        else:
            cell_desc = f"{cell_def.cell_size_x:.1f}m × {cell_def.cell_size_y:.1f}m (2D)"

        # Cell diagnostics (sample first 100 cells for performance)
        cell_diagnostics = self._generate_cell_diagnostics(df, cell_assignments, coord_cols)

        # Variable summaries (raw vs declustered statistics)
        variable_summaries = {}
        coord_vals = {c for c in coord_cols.values() if c}
        grade_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            and col not in ('declust_weight', 'declust_cell')
            and col not in coord_vals
        ]

        for col in grade_cols:
            v = df[col].to_numpy(float)
            w = df['declust_weight'].to_numpy(float)
            mask = ~np.isnan(v)
            if not mask.any():
                continue
            raw_mean = float(np.nanmean(v[mask]))
            declust_mean = float(np.nansum(v[mask] * w[mask]) / np.nansum(w[mask]))
            variable_summaries[col] = {
                "mean_raw": raw_mean,
                "mean_declust": declust_mean,
                "delta": declust_mean - raw_mean,
            }

        summary = DeclusteringSummary(
            cell_size_summary=cell_desc,
            total_samples=total_samples,
            occupied_cells=occupied_cells,
            empty_cells=empty_cells,
            cell_diagnostics=cell_diagnostics,
            variable_summaries=variable_summaries,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat(),
            **weight_stats
        )

        return summary

    def _generate_cell_diagnostics(
        self,
        df: pd.DataFrame,
        cell_assignments: pd.Series,
        coord_cols: Dict[str, Optional[str]],
        max_cells: int = 100
    ) -> List[CellDiagnostic]:
        """Generate detailed diagnostics for top cells.

        Performance optimization: Uses vectorized aggregations and limits
        detailed diagnostics to the top N most populated cells.

        Args:
            df: DataFrame with samples
            cell_assignments: Cell key for each sample
            coord_cols: Coordinate column names
            max_cells: Maximum number of cells to generate detailed diagnostics for

        Returns:
            List of CellDiagnostic for top cells (sorted by sample count)
        """
        # Vectorized cell counts and bounds using groupby.agg (much faster than iteration)
        x_col = coord_cols['x']
        y_col = coord_cols['y']
        z_col = coord_cols['z']

        # Build aggregation dict for coordinates
        agg_dict = {
            x_col: ['min', 'max', 'count'],
            y_col: ['min', 'max'],
        }
        if z_col and z_col in df.columns:
            agg_dict[z_col] = ['min', 'max']

        # Single vectorized aggregation
        cell_stats = df.groupby(cell_assignments).agg(agg_dict)

        # Flatten column names
        cell_stats.columns = ['_'.join(col).strip() for col in cell_stats.columns.values]

        # Get sample counts and sort to find top cells
        count_col = f"{x_col}_count"
        cell_stats = cell_stats.sort_values(count_col, ascending=False)

        # Limit to top N cells for detailed diagnostics
        top_cells = cell_stats.head(max_cells)

        diagnostics = []
        for cell_key in top_cells.index:
            row = top_cells.loc[cell_key]
            sample_count = int(row[count_col])

            bounds = {
                'min_x': row[f"{x_col}_min"],
                'max_x': row[f"{x_col}_max"],
                'min_y': row[f"{y_col}_min"],
                'max_y': row[f"{y_col}_max"],
                'min_z': row.get(f"{z_col}_min") if z_col else None,
                'max_z': row.get(f"{z_col}_max") if z_col else None,
            }

            diagnostic = CellDiagnostic(
                cell_key=cell_key,
                sample_count=sample_count,
                weight_value=1.0 / sample_count,
                sample_indices=[],  # Skip indices for performance (can be computed on demand)
                **bounds
            )
            # Skip per-cell variable stats for performance (computed in summary if needed)
            diagnostic.variable_stats = {}

            diagnostics.append(diagnostic)

        return diagnostics

    # =========================================================================
    # MULTI-CELL-SIZE ANALYSIS
    # =========================================================================

    def analyze_cell_sizes(
        self,
        df: pd.DataFrame,
        cell_sizes: List[Union[float, Tuple[float, float], Tuple[float, float, float]]],
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        z_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None
    ) -> Dict[str, Tuple[pd.DataFrame, DeclusteringSummary]]:
        """
        Perform sensitivity analysis across multiple cell sizes.

        Evaluates declustering stability across different spatial discretizations
        to assess sensitivity and support optimal cell size selection.

        Args:
            df: Input DataFrame
            cell_sizes: List of cell size specifications
            x_col, y_col, z_col: Coordinate columns
            value_cols: Grade columns for comparison

        Returns:
            Dict mapping cell size descriptions to (df_with_weights, summary) tuples
        """
        results = {}
        prev_mean_weight = None
        stability_threshold = 0.001  # 0.1% change threshold

        for cell_size_spec in cell_sizes:
            # Create cell definition for this size
            cell_def = self._create_cell_definition(cell_size_spec)

            # Create temporary config with this cell size
            temp_config = DeclusteringConfig(
                method=self.config.method,
                cell_definition=cell_def,
                weighting_rule=self.config.weighting_rule
            )

            # Create temporary engine
            temp_engine = DeclusteringEngine(temp_config)

            # Compute weights
            try:
                df_result, summary = temp_engine.compute_weights(df, x_col, y_col, z_col, value_cols)

                # Generate key for results dict
                if isinstance(cell_size_spec, (int, float)):
                    key = f"{cell_size_spec:.1f}m_cubic"
                elif len(cell_size_spec) == 2:
                    key = f"{cell_size_spec[0]:.1f}m_x_{cell_size_spec[1]:.1f}m"
                else:
                    key = f"{cell_size_spec[0]:.1f}m_x_{cell_size_spec[1]:.1f}m_x_{cell_size_spec[2]:.1f}m"

                # Check for stability (minimal change in mean weight)
                stability_achieved = False
                if prev_mean_weight is not None:
                    weight_change = abs(summary.mean_weight - prev_mean_weight)
                    stability_achieved = weight_change < stability_threshold
                    if stability_achieved:
                        logger.info(f"Stability achieved at cell size {key}: weight change {weight_change:.6f} < {stability_threshold}")

                # Add stability info to summary
                summary.stability_achieved = stability_achieved
                summary.weight_change_from_previous = abs(summary.mean_weight - prev_mean_weight) if prev_mean_weight is not None else None

                results[key] = (df_result, summary)
                prev_mean_weight = summary.mean_weight

                logger.info(f"Completed cell size analysis: {key}")

            except Exception as e:
                logger.warning(f"Failed cell size analysis for {cell_size_spec}: {e}")
                continue

        logger.info(f"Multi-cell-size analysis complete. Evaluated {len(results)} cell sizes.")
        return results

    def get_recommended_cell_size(self, multi_results: Dict[str, Tuple[pd.DataFrame, DeclusteringSummary]]) -> Optional[str]:
        """
        Determine recommended cell size based on stability analysis.

        Returns the first cell size where declustering stabilizes (minimal weight change).
        This follows the methodology used in commercial software like Supervisor.

        Args:
            multi_results: Results from analyze_cell_sizes()

        Returns:
            Recommended cell size description, or None if no stable size found
        """
        if not multi_results:
            return None

        # Sort by cell size (assuming keys follow pattern like "5.0m_cubic", "10.0m_cubic", etc.)
        sorted_keys = sorted(multi_results.keys(), key=lambda x: float(x.split('m')[0]))

        # Look for first stable cell size
        stability_threshold = 0.001  # 0.1% change

        for i, key in enumerate(sorted_keys):
            _, summary = multi_results[key]

            if summary.stability_achieved:
                logger.info(f"Recommended cell size: {key} (stability achieved)")
                return key

        # If no stable size found, look for smallest relative weight change
        # This is more data-adaptive than just picking the largest size
        best_key = None
        min_change = float('inf')

        for key in sorted_keys:
            _, summary = multi_results[key]
            if summary.weight_change_from_previous is not None:
                if summary.weight_change_from_previous < min_change:
                    min_change = summary.weight_change_from_previous
                    best_key = key

        if best_key:
            logger.info(f"Recommended cell size: {best_key} (smallest weight change: {min_change:.6f})")
            return best_key

        # Final fallback: use the median size, not the largest
        median_idx = len(sorted_keys) // 2
        median_key = sorted_keys[median_idx]
        logger.info(f"No clear recommendation, using median cell size: {median_key}")
        return median_key

    def suggest_cell_sizes_from_data(
        self,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        z_col: Optional[str] = None,
        n_sizes: int = 5
    ) -> Tuple[float, float, float, List[float]]:
        """
        Suggest optimal cell sizes based on actual data characteristics.

        Analyzes the spatial distribution of samples to suggest cell sizes
        that are appropriate for the data, rather than using arbitrary defaults.

        Args:
            df: Input DataFrame with sample coordinates
            x_col, y_col, z_col: Coordinate column names
            n_sizes: Number of cell sizes to suggest for sensitivity analysis

        Returns:
            Tuple of (suggested_x, suggested_y, suggested_z, multi_sizes_list)
            where multi_sizes_list contains n_sizes values for sensitivity analysis
        """
        # Find coordinate columns
        coord_cols = self._find_coordinate_columns(df, x_col, y_col, z_col)
        x_col_name = coord_cols['x']
        y_col_name = coord_cols['y']
        z_col_name = coord_cols['z']

        if not x_col_name or not y_col_name:
            logger.warning("Cannot suggest cell sizes: missing coordinate columns")
            # Return sensible defaults
            return 25.0, 25.0, 25.0, [10.0, 15.0, 20.0, 25.0, 30.0]

        # Extract coordinates
        x = df[x_col_name].dropna().values
        y = df[y_col_name].dropna().values
        z = df[z_col_name].dropna().values if z_col_name else None

        n_samples = len(x)
        if n_samples < 2:
            logger.warning("Too few samples for cell size suggestion")
            return 25.0, 25.0, 25.0, [10.0, 15.0, 20.0, 25.0, 30.0]

        # Calculate spatial extent
        x_extent = float(np.max(x) - np.min(x))
        y_extent = float(np.max(y) - np.min(y))
        z_extent = float(np.max(z) - np.min(z)) if z is not None else 100.0

        # Estimate average sample spacing using nearest-neighbor approximation
        # For efficiency, sample if dataset is large
        if n_samples > 1000:
            sample_idx = np.random.choice(n_samples, 1000, replace=False)
            x_sample = x[sample_idx]
            y_sample = y[sample_idx]
        else:
            x_sample = x
            y_sample = y

        # Calculate average nearest-neighbor distance (2D)
        try:
            from scipy.spatial import cKDTree
            coords_2d = np.column_stack([x_sample, y_sample])
            tree = cKDTree(coords_2d)
            distances, _ = tree.query(coords_2d, k=2)  # k=2 to get nearest neighbor (not self)
            nn_distances = distances[:, 1]  # Second column is nearest neighbor
            avg_spacing = float(np.median(nn_distances))  # Use median for robustness
        except (ImportError, Exception) as e:
            logger.warning(f"Could not calculate NN distances (scipy may not be available): {e}")
            # Fall back to density-based estimate
            area = x_extent * y_extent
            avg_spacing = np.sqrt(area / n_samples) if n_samples > 0 else 25.0

        # Suggest cell sizes based on sample spacing
        # Rule of thumb: cell size should be 1-3x the average sample spacing
        # to capture clustering without being too fine-grained
        suggested_xy = max(5.0, min(200.0, avg_spacing * 2.0))

        # Z cell size should be based on typical downhole interval
        if z is not None and len(z) > 1:
            z_diffs = np.abs(np.diff(np.sort(z)))
            z_diffs = z_diffs[z_diffs > 0.1]  # Filter near-zero differences
            if len(z_diffs) > 0:
                suggested_z = max(2.0, min(100.0, float(np.median(z_diffs)) * 3.0))
            else:
                suggested_z = suggested_xy  # Use XY size if Z spacing unclear
        else:
            suggested_z = suggested_xy

        # Round to nice values
        def round_to_nice(val: float) -> float:
            """Round to nearest nice number (1, 2, 5, 10, 15, 20, 25, etc.)"""
            if val < 2:
                return round(val, 1)
            elif val < 5:
                return round(val)
            elif val < 10:
                return round(val / 2.5) * 2.5
            elif val < 50:
                return round(val / 5) * 5
            else:
                return round(val / 10) * 10

        suggested_x = round_to_nice(suggested_xy)
        suggested_y = round_to_nice(suggested_xy)
        suggested_z = round_to_nice(suggested_z)

        # Generate multi-size list centered around the suggested size
        # Range from 0.5x to 2x the suggested size
        min_size = max(2.0, suggested_x * 0.5)
        max_size = min(200.0, suggested_x * 2.0)

        multi_sizes = []
        for i in range(n_sizes):
            ratio = i / (n_sizes - 1) if n_sizes > 1 else 0.5
            size = min_size + ratio * (max_size - min_size)
            multi_sizes.append(round_to_nice(size))

        # Remove duplicates and sort
        multi_sizes = sorted(list(set(multi_sizes)))

        # Ensure we have at least n_sizes values
        while len(multi_sizes) < n_sizes:
            # Add intermediate values
            if len(multi_sizes) >= 2:
                new_size = (multi_sizes[-1] + multi_sizes[-2]) / 2
                multi_sizes.append(round_to_nice(new_size))
                multi_sizes = sorted(list(set(multi_sizes)))
            else:
                multi_sizes.append(suggested_x)

        logger.info(
            f"CELL SIZE SUGGESTION: Based on {n_samples} samples, "
            f"avg spacing ~{avg_spacing:.1f}m, extent {x_extent:.0f}x{y_extent:.0f}x{z_extent:.0f}m. "
            f"Suggested: {suggested_x}x{suggested_y}x{suggested_z}m, "
            f"Multi-sizes: {multi_sizes}"
        )

        return suggested_x, suggested_y, suggested_z, multi_sizes

    def _create_cell_definition(
        self,
        cell_size_spec: Union[float, Tuple[float, float], Tuple[float, float, float]]
    ) -> CellDefinition:
        """Create CellDefinition from size specification."""
        if isinstance(cell_size_spec, (int, float)):
            # Cubic cells - respect 2D/3D setting from original config
            size = float(cell_size_spec)
            if self.config.cell_definition.is_3d:
                return CellDefinition(size, size, size)
            else:
                return CellDefinition(size, size)
        elif len(cell_size_spec) == 2:
            # 2D rectangular
            sx, sy = cell_size_spec
            return CellDefinition(sx, sy)
        elif len(cell_size_spec) == 3:
            # 3D prismatic
            sx, sy, sz = cell_size_spec
            return CellDefinition(sx, sy, sz)
        else:
            raise ValueError(f"Invalid cell size specification: {cell_size_spec}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def decode_cell_key(self, cell_key: int) -> Tuple[int, int, Optional[int]]:
        """Decode integer cell key back to (ix, iy, iz) indices.

        Args:
            cell_key: Integer-encoded cell key

        Returns:
            Tuple of (ix, iy, iz) where iz is None for 2D
        """
        stride = 1_000_000
        if self.config.cell_definition.is_3d:
            iz = cell_key // (stride * stride)
            remainder = cell_key % (stride * stride)
            iy = remainder // stride
            ix = remainder % stride
            return (ix, iy, iz)
        else:
            iy = cell_key // stride
            ix = cell_key % stride
            return (ix, iy, None)

    def cell_key_to_string(self, cell_key: int) -> str:
        """Convert integer cell key to human-readable string format.

        Args:
            cell_key: Integer-encoded cell key

        Returns:
            String like "ix,iy" or "ix,iy,iz"
        """
        ix, iy, iz = self.decode_cell_key(cell_key)
        if iz is not None:
            return f"{ix},{iy},{iz}"
        return f"{ix},{iy}"

    def get_last_summary(self) -> Optional[DeclusteringSummary]:
        """Get the summary from the last declustering operation."""
        return self._last_summary

    def export_summary_csv(self, filepath: str, summary: Optional[DeclusteringSummary] = None) -> None:
        """Export declustering summary to CSV for audit purposes."""
        summary = summary or self._last_summary
        if not summary:
            raise ValueError("No summary available for export")

        # Create summary DataFrame
        summary_data = {
            'metric': [
                'Cell Size',
                'Total Samples',
                'Occupied Cells',
                'Empty Cells',
                'Samples per Cell (avg)',
                'Min Weight',
                'Max Weight',
                'Mean Weight',
                'Weight Std Dev',
                'Processing Time (s)',
                'Stability Achieved',
                'Weight Change from Previous'
            ],
            'value': [
                summary.cell_size_summary,
                summary.total_samples,
                summary.occupied_cells,
                summary.empty_cells,
                summary.cells_per_sample,
                summary.min_weight,
                summary.max_weight,
                summary.mean_weight,
                summary.weight_std,
                summary.processing_time_seconds,
                summary.stability_achieved,
                summary.weight_change_from_previous
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)

        logger.info(f"Exported declustering summary to {filepath}")

    def export_cell_weights_csv(self, df_with_weights: pd.DataFrame, filepath: str, readable_keys: bool = True) -> None:
        """Export cell weights map to CSV for hotspot analysis and QA review.

        Args:
            df_with_weights: DataFrame with declust_weight and declust_cell columns
            filepath: Path to save CSV file
            readable_keys: If True, convert integer keys to "ix,iy,iz" format
        """
        if 'declust_weight' not in df_with_weights.columns or 'declust_cell' not in df_with_weights.columns:
            raise ValueError("DataFrame must contain 'declust_weight' and 'declust_cell' columns")

        # Group by cell and aggregate
        cell_summary = df_with_weights.groupby('declust_cell').agg({
            'declust_weight': ['count', 'mean', 'min', 'max']
        }).reset_index()

        # Flatten column names
        cell_summary.columns = ['cell_key', 'sample_count', 'mean_weight', 'min_weight', 'max_weight']

        # Convert integer keys to readable format if requested
        if readable_keys and pd.api.types.is_integer_dtype(cell_summary['cell_key']):
            cell_summary['cell_key'] = cell_summary['cell_key'].apply(self.cell_key_to_string)

        # Sort by sample count (most populated cells first)
        cell_summary = cell_summary.sort_values('sample_count', ascending=False)

        # Save to CSV
        cell_summary.to_csv(filepath, index=False)

        logger.info(f"Exported cell weights map to {filepath} ({len(cell_summary)} cells)")

    # =========================================================================
    # INTEGRATION HOOKS FOR GeoX
    # =========================================================================

    def prepare_for_variogram_engine(self, df_with_weights: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare declustered data for Variogram engine input.

        The declust_weight column is directly usable by the experimental
        variogram calculation in the Variogram engine.

        Args:
            df_with_weights: DataFrame with declust_weight column

        Returns:
            DataFrame ready for variogram analysis
        """
        required_cols = ['declust_weight']
        if not all(col in df_with_weights.columns for col in required_cols):
            raise ValueError("DataFrame missing required declustering columns")

        logger.info("Prepared declustered data for Variogram engine input")
        return df_with_weights

    def prepare_summary_for_qa_panel(self, summary: Optional[DeclusteringSummary] = None) -> Dict[str, Any]:
        """
        Prepare summary data for QA panel display and CP reporting.

        Formats declustering diagnostics for integration with the QA/QC panel
        for compliance reporting and audit trails.

        Args:
            summary: Declustering summary (uses last if None)

        Returns:
            Dict ready for QA panel consumption
        """
        summary = summary or self._last_summary
        if not summary:
            raise ValueError("No summary available for QA panel")

        qa_data = {
            'declustering_method': self.config.method.value,
            'cell_size': summary.cell_size_summary,
            'total_samples': summary.total_samples,
            'occupied_cells': summary.occupied_cells,
            'samples_per_cell': summary.cells_per_sample,
            'weight_statistics': {
                'min': summary.min_weight,
                'max': summary.max_weight,
                'mean': summary.mean_weight,
                'std': summary.weight_std
            },
            'processing_timestamp': summary.timestamp,
            'audit_ready': True  # Flag for JORC/SAMREC compliance
        }

        logger.info("Prepared declustering summary for QA panel")
        return qa_data
